//! Utilities for analyzing places: children, aliases, etc.

use std::{ops::ControlFlow, rc::Rc};

use indexical::ToIndex;
use rustc_borrowck::consumers::BodyWithBorrowckFacts;
use rustc_data_structures::fx::FxHashSet;
use rustc_hir::def_id::DefId;
use rustc_middle::{
  mir::*,
  ty::{
    Region, RegionKind, RegionVid, Ty, TyCtxt, TyKind, TypeSuperVisitable, TypeVisitable,
    TypeVisitor, TypingEnv, fold_regions,
  },
};
use rustc_utils::{
  BodyExt, MutabilityExt, PlaceExt, block_timer,
  cache::{Cache, CopyCache},
  mir::{
    location_or_arg::{
      LocationOrArg,
      index::{LocationOrArgDomain, LocationOrArgIndex},
    },
    place::UNKNOWN_REGION,
  },
};
use smallvec::SmallVec;

use super::{aliases::Aliases, utils::PlaceSet};
use crate::extensions::{MutabilityMode, is_extension_active};

/// Utilities for analyzing places: children, aliases, etc.
pub struct PlaceInfo<'a, 'tcx> {
  pub(crate) tcx: TyCtxt<'tcx>,
  pub(crate) body: &'a Body<'tcx>,
  pub(crate) def_id: DefId,
  location_domain: Rc<LocationOrArgDomain>,

  // Core computed data structure
  aliases: Aliases<'a, 'tcx>,

  // Caching for derived analysis
  normalized_cache: CopyCache<Place<'tcx>, Place<'tcx>>,
  aliases_cache: Cache<Place<'tcx>, PlaceSet<'tcx>>,
  conflicts_cache: Cache<Place<'tcx>, PlaceSet<'tcx>>,
  reachable_cache: Cache<(Place<'tcx>, Mutability), PlaceSet<'tcx>>,
}

impl<'a, 'tcx> PlaceInfo<'a, 'tcx> {
  fn build_location_arg_domain(body: &Body) -> Rc<LocationOrArgDomain> {
    let all_locations = body.all_locations().map(LocationOrArg::Location);
    let all_locals = body.args_iter().map(LocationOrArg::Arg);
    let domain = all_locations.chain(all_locals).collect::<Vec<_>>();
    Rc::new(LocationOrArgDomain::from_iter(domain))
  }

  /// Computes all the metadata about places used within the infoflow analysis.
  pub fn build(
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
    body_with_facts: &'a BodyWithBorrowckFacts<'tcx>,
  ) -> Self {
    block_timer!("aliases");
    let body = &body_with_facts.body;
    let location_domain = Self::build_location_arg_domain(body);
    let aliases = Aliases::build(tcx, def_id, body_with_facts);

    PlaceInfo {
      aliases,
      tcx,
      body,
      def_id,
      location_domain,
      aliases_cache: Cache::default(),
      normalized_cache: CopyCache::default(),
      conflicts_cache: Cache::default(),
      reachable_cache: Cache::default(),
    }
  }

  /// Normalizes a place via [`PlaceExt::normalize`] (cached).
  ///
  /// See the `PlaceExt` documentation for details on how normalization works.
  pub fn normalize(&self, place: Place<'tcx>) -> Place<'tcx> {
    self
      .normalized_cache
      .get(&place, |place| place.normalize(self.tcx, self.def_id))
  }

  /// Computes the aliases of a place (cached).
  ///
  /// For example, if `x = &y`, then `*x` aliases `y`.
  /// Note that an alias is NOT guaranteed to be of the same type as `place`!
  pub fn aliases(&self, place: Place<'tcx>) -> &PlaceSet<'tcx> {
    // note: important that aliases are computed on the unnormalized place
    // which contains region information
    self
      .aliases_cache
      .get(&self.normalize(place), move |_| self.aliases.aliases(place))
  }

  pub fn influences(&self, place: Place<'tcx>) -> SmallVec<[Place<'tcx>; 8]> {
    let conflicts = self
      .aliases(place)
      .iter()
      .flat_map(|alias| self.conflicts(*alias));
    let provenance =
      place
        .refs_in_projection(self.body, self.tcx)
        .flat_map(|(place_ref, _)| {
          self.aliases(Place::from_ref(place_ref, self.tcx)).iter()
        });
    conflicts.chain(provenance).copied().collect()
  }

  pub fn is_approximate_loan(&self, place: Place<'tcx>) -> bool {
    self
      .aliases
      .approximate_loans
      .contains(&self.normalize(place))
  }

  /// Returns all reachable fields of `place` without going through references.
  ///
  /// For example, if `x = (0, 1)` then `children(x) = {x, x.0, x.1}`.
  pub fn children(&self, place: Place<'tcx>) -> PlaceSet<'tcx> {
    PlaceSet::from_iter(place.interior_places(self.tcx, self.body, self.def_id))
  }

  /// Returns all places that *directly* conflict with `place`, i.e. that a mutation to `place`
  /// would also be a mutation to the conflicting place.
  ///
  /// For example, if `x = ((0, 1), 2)` then `conflicts(x.0) = {x, x.0, x.0.0, x.0.1}`, but not `x.1`.
  ///
  /// For indirect places, this function follows conflicting parents up until a reference point.
  /// So if `x = (0, &(box 1, 2))` then conflicts(*(*(x.1).0)) = {*(*(x.1).0), *(x.1).0, *(x.1)}
  pub fn conflicts(&self, place: Place<'tcx>) -> &PlaceSet<'tcx> {
    self.conflicts_cache.get(&place, |place| {
      let children = self.children(place);
      let parents = place
        .projection
        .iter()
        .enumerate()
        .map(|(i, elem)| {
          let place = PlaceRef {
            local: place.local,
            projection: &place.projection[.. i],
          };
          (place, elem)
        })
        .take_while(|(place, elem)| {
          place.ty(self.body.local_decls(), self.tcx).ty.is_box()
            || !matches!(elem, PlaceElem::Deref)
        })
        .map(|(place_ref, _)| Place::from_ref(place_ref, self.tcx));
      children.into_iter().chain(parents).collect()
    })
  }

  /// Returns all [direct](PlaceExt::is_direct) places that are reachable from `place`
  /// and can be used at the provided level of [`Mutability`] (cached).
  ///
  /// For example, if `x = 0` and `y = (0, &x)`, then `reachable_values(y, Mutability::Not)`
  /// is `{y, x}`. With `Mutability::Mut`, then the output is `{y}` (no `x`).
  pub fn reachable_values(
    &self,
    place: Place<'tcx>,
    mutability: Mutability,
  ) -> &PlaceSet<'tcx> {
    self.reachable_cache.get(&(place, mutability), |_| {
      let ty = place.ty(self.body.local_decls(), self.tcx).ty;
      let loans = self.collect_loans(ty, mutability);
      loans
        .into_iter()
        .chain([place])
        .filter(|place| {
          if let Some((place, _)) = place.refs_in_projection(self.body, self.tcx).last() {
            let ty = place.ty(self.body.local_decls(), self.tcx).ty;
            if ty.is_box() || ty.is_raw_ptr() {
              return true;
            }
          }
          place.is_direct(self.body, self.tcx)
        })
        .collect()
    })
  }

  fn collect_loans(&self, ty: Ty<'tcx>, mutability: Mutability) -> PlaceSet<'tcx> {
    let mut collector = LoanCollector {
      aliases: &self.aliases,
      tcx: self.tcx,
      body: self.body,
      typing_env: self.body.typing_env(self.tcx),
      unknown_region: Region::new_var(self.tcx, UNKNOWN_REGION),
      target_mutability: mutability,
      stack: vec![],
      loans: PlaceSet::default(),
    };
    let _ = collector.visit_ty(ty);
    collector.loans
  }

  /// Returns all [direct](PlaceExt::is_direct) places reachable from arguments
  /// to the current body.
  pub fn all_args(
    &'a self,
  ) -> impl Iterator<Item = (Place<'tcx>, LocationOrArgIndex)> + 'a {
    self.body.args_iter().flat_map(|local| {
      let location = local.to_index(&self.location_domain);
      let place = Place::from_local(local, self.tcx);
      let ptrs = place
        .interior_pointers(self.tcx, self.body, self.def_id)
        .into_values()
        .flat_map(|ptrs| {
          ptrs
            .into_iter()
            .filter(|(ptr, _)| ptr.projection.len() <= 2)
            .map(|(ptr, _)| self.tcx.mk_place_deref(ptr))
        });
      ptrs
        .chain([place])
        .flat_map(|place| place.interior_places(self.tcx, self.body, self.def_id))
        .map(move |place| (place, location))
    })
  }

  /// Returns the [`LocationOrArgDomain`] for the current body.
  pub fn location_domain(&self) -> &Rc<LocationOrArgDomain> {
    &self.location_domain
  }
}

// TODO: this visitor shares some structure with the PlaceCollector in mir utils.
// Can we consolidate these?
struct LoanCollector<'a, 'tcx> {
  aliases: &'a Aliases<'a, 'tcx>,
  tcx: TyCtxt<'tcx>,
  body: &'a Body<'tcx>,
  typing_env: TypingEnv<'tcx>,
  unknown_region: Region<'tcx>,
  target_mutability: Mutability,
  stack: Vec<Mutability>,
  loans: PlaceSet<'tcx>,
}

// `Freeze` does not depend on regions, and `Ty::is_freeze` panics on a region
// bound outside `ty` (the `'a` of `Formatter<'a>` inside `for<'a> fn(&'a mut
// Formatter<'a>)`), so every such region is erased first.
fn is_freeze<'tcx>(tcx: TyCtxt<'tcx>, typing_env: TypingEnv<'tcx>, ty: Ty<'tcx>) -> bool {
  fold_regions(tcx, ty, |_, _| tcx.lifetimes.re_erased).is_freeze(tcx, typing_env)
}

// A shared borrow of data with interior mutability (`&cell`, `&vec_of_cells`,
// `&rc`) can mutate that data, so it is treated as a mutable loan.
pub fn may_contain_unsafe_cell<'tcx>(
  tcx: TyCtxt<'tcx>,
  typing_env: TypingEnv<'tcx>,
  ty: Ty<'tcx>,
) -> bool {
  struct UnsafeCellFinder<'tcx> {
    tcx: TyCtxt<'tcx>,
    typing_env: TypingEnv<'tcx>,
    visited: FxHashSet<Ty<'tcx>>,
  }

  impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for UnsafeCellFinder<'tcx> {
    type Result = ControlFlow<()>;

    fn visit_ty(&mut self, ty: Ty<'tcx>) -> Self::Result {
      if !self.visited.insert(ty) {
        return ControlFlow::Continue(());
      }
      if !is_freeze(self.tcx, self.typing_env, ty) {
        return ControlFlow::Break(());
      }
      // `Freeze` is shallow, so the search goes through pointers. The explicit
      // cases are the constituent types that the generic arguments do not show.
      match ty.kind() {
        TyKind::Adt(adt, args) => {
          for field in adt.all_fields() {
            field.ty(self.tcx, args).visit_with(self)?;
          }
        }
        TyKind::Coroutine(..) | TyKind::CoroutineWitness(..) => {
          return ControlFlow::Break(());
        }
        _ => {}
      }
      ty.super_visit_with(self)
    }
  }

  let mut finder = UnsafeCellFinder {
    tcx,
    typing_env,
    visited: FxHashSet::default(),
  };
  finder.visit_ty(ty).is_break()
}

impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for LoanCollector<'_, 'tcx> {
  type Result = ControlFlow<()>;

  fn visit_ty(&mut self, ty: Ty<'tcx>) -> Self::Result {
    match ty.kind() {
      TyKind::Ref(_, _, mutability) => {
        self.stack.push(*mutability);
        ty.super_visit_with(self)?;
        self.stack.pop();
      }
      _ if ty.is_box() || ty.is_raw_ptr() => {
        self.visit_region(self.unknown_region)?;
        ty.super_visit_with(self)?;
      }
      _ => ty.super_visit_with(self)?,
    };

    ControlFlow::Continue(())
  }

  fn visit_region(&mut self, region: Region<'tcx>) -> Self::Result {
    let region = match region.kind() {
      RegionKind::ReVar(region) => region,
      RegionKind::ReStatic => RegionVid::from_usize(0),
      // TODO: do we need to handle bound regions?
      // e.g. shows up with closures, for<'a> ...
      RegionKind::ReErased | RegionKind::ReBound(..) => {
        return ControlFlow::Continue(());
      }
      _ => unreachable!("{region:?}"),
    };
    if let Some(loans) = self.aliases.loans.get(&region) {
      let under_immut_ref = self.stack.contains(&Mutability::Not);
      let ignore_mut =
        is_extension_active(|mode| mode.mutability_mode == MutabilityMode::IgnoreMut);
      self
        .loans
        .extend(loans.iter().filter_map(|(place, mutability)| {
          if ignore_mut {
            return Some(place);
          }
          let loan_ty = place.ty(self.body.local_decls(), self.tcx).ty;
          let loan_mutability =
            if may_contain_unsafe_cell(self.tcx, self.typing_env, loan_ty) {
              Mutability::Mut
            } else if under_immut_ref {
              Mutability::Not
            } else {
              *mutability
            };
          self
            .target_mutability
            .is_permissive_as(loan_mutability)
            .then_some(place)
        }))
    }

    ControlFlow::Continue(())
  }
}

#[cfg(test)]
mod test {
  use rustc_utils::{
    hashset,
    test_utils::{Placer, compare_sets},
  };

  use super::*;
  use crate::test_utils;

  fn placeinfo_harness(
    input: &str,
    f: impl for<'tcx> FnOnce(TyCtxt<'tcx>, &Body<'tcx>, PlaceInfo<'_, 'tcx>) + Send,
  ) {
    test_utils::compile_body(input, |tcx, body_id, body_with_facts| {
      let body = &body_with_facts.body;
      let def_id = tcx.hir_body_owner_def_id(body_id);
      let place_info = PlaceInfo::build(tcx, def_id.to_def_id(), body_with_facts);

      f(tcx, body, place_info)
    });
  }

  #[test]
  fn test_placeinfo_basic() {
    let input = r#"
fn main() {
  let a = 0;
  let mut b = 1;
  let c = ((0, &a), &mut b);
  let d = 0;
  let e = &d;
  let f = &e;
}
    "#;
    placeinfo_harness(input, |tcx, body, place_info| {
      let p = Placer::new(tcx, body);
      let c = p.local("c");
      compare_sets(place_info.children(c.mk()), hashset! {
        c.mk(),
        c.field(0).mk(),
        c.field(0).field(0).mk(),
        c.field(0).field(1).mk(),
        c.field(1).mk(),
      });

      compare_sets(place_info.conflicts(c.field(0).mk()), &hashset! {
        c.mk(),
        c.field(0).mk(),
        c.field(0).field(0).mk(),
        c.field(0).field(1).mk(),
        // c.field(1) not part of the set
      });

      // a and b are reachable at immut-level
      compare_sets(
        place_info.reachable_values(c.mk(), Mutability::Not),
        &hashset! {
          c.mk(),
          p.local("a").mk(),
          p.local("b").mk()
        },
      );

      // only b is reachable at mut-level
      compare_sets(
        place_info.reachable_values(c.mk(), Mutability::Mut),
        &hashset! {
          c.mk(),
          p.local("b").mk()
        },
      );

      // handles transitive references
      compare_sets(
        place_info.reachable_values(p.local("f").mk(), Mutability::Not),
        &hashset! {
          p.local("f").mk(),
          p.local("e").mk(),
          p.local("d").mk()
        },
      )
    });
  }
}
