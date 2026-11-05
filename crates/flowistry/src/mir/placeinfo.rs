//! Utilities for analyzing places: children, aliases, etc.

use std::{cell::OnceCell, ops::ControlFlow, rc::Rc};

use indexical::ToIndex;
use rustc_borrowck::consumers::BodyWithBorrowckFacts;
use rustc_data_structures::fx::FxHashSet;
use rustc_hir::def_id::DefId;
use rustc_middle::{
  mir::*,
  ty::{
    Region, RegionKind, RegionVid, Ty, TyCtxt, TyKind, TypeSuperVisitable, TypeVisitable,
    TypeVisitor, TypingEnv, Unnormalized, fold_regions,
  },
};
use rustc_utils::{
  AdtDefExt, BodyExt, MutabilityExt, PlaceExt, block_timer,
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
  typing_env: TypingEnv<'tcx>,
  location_domain: Rc<LocationOrArgDomain>,

  // Core computed data structure
  aliases: Aliases<'a, 'tcx>,

  // Caching for derived analysis
  normalized_cache: CopyCache<Place<'tcx>, Place<'tcx>>,
  normalized_projections: CopyCache<PlaceElem<'tcx>, Option<PlaceElem<'tcx>>>,
  leaf_cache: CopyCache<Ty<'tcx>, bool>,
  aliases_cache: Cache<Place<'tcx>, PlaceSet<'tcx>>,
  conflicts_cache: Cache<Place<'tcx>, PlaceSet<'tcx>>,
  reachable_cache: Cache<(Place<'tcx>, Mutability), PlaceSet<'tcx>>,
  argument_places: OnceCell<Vec<(Place<'tcx>, LocationOrArgIndex)>>,
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
    typing_env: TypingEnv<'tcx>,
    body_with_facts: &'a BodyWithBorrowckFacts<'tcx>,
  ) -> Self {
    block_timer!("aliases");
    let aliases = Aliases::build(tcx, def_id, typing_env, body_with_facts);
    Self::new(tcx, def_id, typing_env, &body_with_facts.body, aliases)
  }

  /// Computes the metadata about places of a `body` whose region constraints are
  /// `subset_base`.
  pub fn build_from_subset_base(
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
    typing_env: TypingEnv<'tcx>,
    body: &'a Body<'tcx>,
    subset_base: Vec<(RegionVid, RegionVid)>,
  ) -> Self {
    block_timer!("aliases");
    let aliases =
      Aliases::build_from_subset_base(tcx, def_id, typing_env, body, subset_base);
    Self::new(tcx, def_id, typing_env, body, aliases)
  }

  fn new(
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
    typing_env: TypingEnv<'tcx>,
    body: &'a Body<'tcx>,
    aliases: Aliases<'a, 'tcx>,
  ) -> Self {
    let location_domain = Self::build_location_arg_domain(body);

    PlaceInfo {
      aliases,
      tcx,
      body,
      def_id,
      typing_env,
      location_domain,
      aliases_cache: Cache::default(),
      normalized_cache: CopyCache::default(),
      normalized_projections: CopyCache::default(),
      leaf_cache: CopyCache::default(),
      conflicts_cache: Cache::default(),
      reachable_cache: Cache::default(),
      argument_places: OnceCell::new(),
    }
  }

  /// Normalizes a place and its shared projection elements (cached).
  pub fn normalize(&self, place: Place<'tcx>) -> Place<'tcx> {
    self.normalized_cache.get(&place, |place| {
      let Some(changed) = place
        .projection
        .iter()
        .position(|elem| self.normalize_projection(elem) != Some(elem))
      else {
        return place;
      };
      let mut projection = Vec::with_capacity(place.projection.len());
      projection.extend_from_slice(&place.projection[.. changed]);
      projection.extend(
        place.projection[changed ..]
          .iter()
          .copied()
          .filter_map(|elem| self.normalize_projection(elem)),
      );
      Place::make(place.local, &projection, self.tcx)
    })
  }

  fn normalize_projection(&self, elem: PlaceElem<'tcx>) -> Option<PlaceElem<'tcx>> {
    self.normalized_projections.get(&elem, |elem| match elem {
      ProjectionElem::Index(_) | ProjectionElem::ConstantIndex { .. } => {
        Some(ProjectionElem::Index(Local::from_usize(0)))
      }
      ProjectionElem::Subslice { .. } => None,
      elem => Some(
        self
          .tcx
          .normalize_erasing_regions(self.typing_env, Unnormalized::new_wip(elem)),
      ),
    })
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

  pub fn is_parameter_loan(&self, place: Place<'tcx>) -> bool {
    self
      .aliases
      .parameter_loans
      .contains(&self.normalize(place))
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

  /// Whether `children(place)` contains only `place` (cached by type).
  pub fn is_leaf(&self, place: Place<'tcx>) -> bool {
    let ty = place.ty(self.body.local_decls(), self.tcx).ty;
    // Each traversal starts with an empty type stack. Its relative paths depend
    // only on the type and the visibility scope, which is fixed for this instance.
    self.leaf_cache.get(&ty, |_| {
      // A proper child is enough to disprove singleton membership. References
      // stop the original traversal; boxes and raw pointers do not.
      let has_child = |child| child != ty;
      match ty.kind() {
        _ if ty.is_box() => !has_child(ty.boxed_ty().unwrap()),
        TyKind::Ref(..) => true,
        TyKind::RawPtr(child, _) | TyKind::Array(child, _) | TyKind::Slice(child) => {
          !has_child(*child)
        }
        TyKind::Tuple(fields) => !fields.iter().any(has_child),
        TyKind::Adt(adt, args) if adt.is_struct() => !adt
          .all_visible_fields(self.def_id, self.tcx)
          .any(|field| has_child(field.ty(self.tcx, args))),
        TyKind::Adt(adt, args) if adt.is_enum() => !adt
          .all_fields()
          .any(|field| has_child(field.ty(self.tcx, args))),
        TyKind::Closure(_, args) => !args.as_closure().upvar_tys().iter().any(has_child),
        _ => self.children(place).len() == 1,
      }
    })
  }

  pub fn contains(&self, outer: Place<'tcx>, inner: Place<'tcx>) -> bool {
    inner.local == outer.local
      && inner.projection.starts_with(outer.projection)
      && self.stays_in_prefix(inner, outer.projection.len())
  }

  pub fn may_overlap(&self, left: Place<'tcx>, right: Place<'tcx>) -> bool {
    if left.local != right.local {
      return false;
    }
    let common = left
      .projection
      .iter()
      .zip(right.projection)
      .take_while(|(left, right)| left == right)
      .count();
    if !self.stays_in_prefix(left, common) || !self.stays_in_prefix(right, common) {
      return false;
    }
    if common == left.projection.len() || common == right.projection.len() {
      return true;
    }
    let shared = Place::ty_from(
      left.local,
      &left.projection[.. common],
      self.body.local_decls(),
      self.tcx,
    )
    .ty;
    shared.is_union() || shared.is_enum()
  }

  pub fn is_one_location(&self, place: Place<'tcx>) -> bool {
    place.iter_projections().all(|(prefix, elem)| match elem {
      ProjectionElem::Index(_)
      | ProjectionElem::ConstantIndex { .. }
      | ProjectionElem::Subslice { .. } => false,
      ProjectionElem::Deref => {
        !prefix.ty(self.body.local_decls(), self.tcx).ty.is_raw_ptr()
      }
      _ => true,
    })
  }

  fn stays_in_prefix(&self, place: Place<'tcx>, length: usize) -> bool {
    place.iter_projections().skip(length).all(|(prefix, elem)| {
      !matches!(elem, ProjectionElem::Deref)
        || prefix.ty(self.body.local_decls(), self.tcx).ty.is_box()
    })
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
      children.into_iter().chain(self.parents(place)).collect()
    })
  }

  pub fn parents(&self, place: Place<'tcx>) -> impl Iterator<Item = Place<'tcx>> + '_ {
    place
      .iter_projections()
      .take_while(|(place, elem)| {
        place.ty(self.body.local_decls(), self.tcx).ty.is_box()
          || !matches!(elem, PlaceElem::Deref)
      })
      .map(|(place_ref, _)| Place::from_ref(place_ref, self.tcx))
  }

  pub fn containing_places(
    &self,
    place: Place<'tcx>,
  ) -> impl Iterator<Item = PlaceRef<'tcx>> + '_ {
    place
      .iter_projections()
      .rev()
      .take_while(|(prefix, elem)| {
        !matches!(elem, PlaceElem::Deref)
          || prefix.ty(self.body.local_decls(), self.tcx).ty.is_box()
      })
      .map(|(prefix, _)| prefix)
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
      def_id: self.def_id,
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
    self
      .argument_places
      .get_or_init(|| {
        self
          .body
          .args_iter()
          .flat_map(|local| {
            let location = local.to_index(&self.location_domain);
            let place = Place::from_local(local, self.tcx);
            let ptrs = self
              .aliases
              .argument_pointees
              .iter()
              .filter(move |pointee| pointee.local == local)
              .copied();
            ptrs
              .chain([place])
              .flat_map(|place| place.interior_places(self.tcx, self.body, self.def_id))
              .map(move |place| (place, location))
          })
          .collect()
      })
      .iter()
      .copied()
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
  def_id: DefId,
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
      // A raw pointer may point anywhere.
      _ if ty.is_raw_ptr() => {
        self
          .loans
          .extend(self.body.all_places(self.tcx, self.def_id));
      }
      _ if ty.is_box() => {
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
      let place_info = PlaceInfo::build(
        tcx,
        def_id.to_def_id(),
        tcx.typing_env_normalized_for_post_analysis(def_id),
        body_with_facts,
      );

      f(tcx, body, place_info)
    });
  }

  #[test]
  fn cached_place_queries_preserve_the_original_traversal() {
    let input = r#"
mod hidden {
  pub struct Opaque(u32);
  pub struct Visible(pub u32);
}
struct Empty;
enum EmptyEnum {}
enum UnitVariants { A, B }
enum Enum { Empty, Value(u32), Pair(u32, bool) }
union Union { integer: u32, boolean: bool }
struct Node { value: u32, next: Option<Box<Node>> }
fn fixture(
  scalar: u32, unit: (), tuple: (u32, (u32, bool)), reference: &u32,
  raw: *const u32, boxed: Box<u32>, array: [u32; 0], empty: Empty,
  never: EmptyEnum, enumeration: Enum, union: Union,
  opaque: hidden::Opaque, visible: hidden::Visible, node: Node,
  units: UnitVariants, empty_box: Box<()>, empty_raw: *const (),
  function: (for<'a> fn(&'a u32),),
) {
  let captured = || scalar;
  let uncaptured = || 1;
}
    "#;
    preborrowck_mir::test_utils::Compile::source(input).run(|tcx| {
      let owner = tcx
        .hir_body_owners()
        .find(|&owner| tcx.item_name(owner).as_str() == "fixture")
        .unwrap();
      let facts = preborrowck_mir::borrowck_facts(tcx, owner);
      let body = &facts.body;
      let place_info = PlaceInfo::build(
        tcx,
        owner.to_def_id(),
        tcx.typing_env_normalized_for_post_analysis(owner),
        facts,
      );
      let places: Vec<_> = body
        .local_decls
        .indices()
        .flat_map(|local| place_info.children(Place::from_local(local, tcx)))
        .collect();
      let expected_arguments: FxHashSet<_> = body
        .args_iter()
        .flat_map(|local| {
          let location = local.to_index(&place_info.location_domain);
          let place = Place::from_local(local, tcx);
          let pointers = place
            .interior_pointers(tcx, body, place_info.def_id)
            .into_values()
            .flatten()
            .filter(|(pointer, _)| pointer.projection.len() <= 3)
            .map(|(pointer, _)| tcx.mk_place_deref(pointer));
          pointers
            .chain([place])
            .flat_map(|place| place.interior_places(tcx, body, place_info.def_id))
            .map(move |place| (place, location))
        })
        .collect();
      assert_eq!(
        place_info.all_args().collect::<FxHashSet<_>>(),
        expected_arguments
      );
      for &place in places.iter().chain(places.iter().rev()) {
        assert_eq!(
          place_info.normalize(place),
          crate::mir::utils::normalize_place(tcx, place_info.typing_env, place),
          "normalization of {place:?}"
        );
        assert_eq!(
          place_info.is_leaf(place),
          place_info.children(place).len() == 1,
          "{place:?}"
        );
      }
      let p = Placer::new(tcx, body);
      assert!(place_info.is_leaf(p.local("reference").mk()));
      assert!(place_info.is_leaf(p.local("opaque").mk()));
      assert!(!place_info.is_leaf(p.local("visible").mk()));
      assert!(!place_info.is_leaf(p.local("array").mk()));
      assert!(place_info.is_leaf(p.local("units").mk()));
      assert!(!place_info.is_leaf(p.local("empty_box").mk()));
      assert!(!place_info.is_leaf(p.local("empty_raw").mk()));
    });
  }

  #[test]
  fn leaf_cache_respects_the_visibility_scope() {
    let input = r#"
mod hidden {
  pub struct Opaque(u32);
  pub fn fixture(value: Opaque) {}
}
    "#;
    placeinfo_harness(input, |tcx, body, place_info| {
      let p = Placer::new(tcx, body);
      let value = p.local("value").mk();
      assert!(!place_info.is_leaf(value));
      assert_eq!(
        place_info.is_leaf(value),
        place_info.children(value).len() == 1
      );
    });
  }

  #[test]
  fn overlap_follows_the_memory_of_places() {
    let input = r#"
union Union { integer: u32, boolean: bool }
fn main() {
  let tuple = (0u32, (0u32, false));
  let reference = &tuple;
  let raw = &tuple as *const (u32, (u32, bool));
  let boxed = Box::new((0u32, 0u32));
  let option = Some((0u32, 0u32));
  let union = Union { integer: 0 };
  let array = [0u32; 2];
}
    "#;
    placeinfo_harness(input, |tcx, body, place_info| {
      let p = Placer::new(tcx, body);
      let tuple = p.local("tuple");
      let reference = p.local("reference");
      let raw = p.local("raw");
      let boxed = p.local("boxed");
      let option = p.local("option");
      let union = p.local("union");
      let array = p.local("array");

      assert!(place_info.contains(tuple.mk(), tuple.field(1).field(0).mk()));
      assert!(!place_info.contains(tuple.field(1).mk(), tuple.mk()));
      assert!(place_info.contains(boxed.mk(), boxed.deref().field(0).mk()));
      assert!(!place_info.contains(reference.mk(), reference.deref().mk()));
      assert!(!place_info.contains(raw.mk(), raw.deref().mk()));
      assert!(place_info.contains(raw.deref().mk(), raw.deref().field(0).mk()));

      assert!(place_info.may_overlap(tuple.field(1).mk(), tuple.mk()));
      assert!(place_info.may_overlap(tuple.mk(), tuple.field(1).field(1).mk()));
      assert!(!place_info.may_overlap(tuple.field(0).mk(), tuple.field(1).mk()));
      assert!(place_info.may_overlap(boxed.mk(), boxed.deref().field(1).mk()));
      assert!(!place_info.may_overlap(reference.mk(), reference.deref().field(0).mk()));
      assert!(!place_info.may_overlap(raw.mk(), raw.deref().field(0).mk()));
      assert!(place_info.may_overlap(union.field(0).mk(), union.field(1).mk()));
      assert!(
        place_info.may_overlap(option.downcast(0).mk(), option.downcast(1).field(0).mk())
      );
      assert!(!place_info.may_overlap(
        option.downcast(1).field(0).field(0).mk(),
        option.downcast(1).field(0).field(1).mk()
      ));
      assert!(place_info.may_overlap(array.index(0).mk(), array.index(0).mk()));

      assert!(place_info.is_one_location(tuple.field(1).field(0).mk()));
      assert!(place_info.is_one_location(boxed.deref().field(0).mk()));
      assert!(place_info.is_one_location(reference.deref().field(0).mk()));
      assert!(!place_info.is_one_location(raw.deref().mk()));
      assert!(!place_info.is_one_location(array.index(0).mk()));
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
