use log::warn;
use rustc_abi::{FieldIdx, VariantIdx};
use rustc_data_structures::fx::FxHashMap as HashMap;
use rustc_hir::def_id::DefId;
use rustc_middle::{
  mir::{Body, HasLocalDecls, Local, Mutability, Place, PlaceElem, ProjectionElem},
  ty::{self, Region, RegionKind, RegionVid, Ty, TyCtxt, TyKind, TypingEnv},
};
use rustc_utils::{PlaceExt, mir::place::UNKNOWN_REGION};

use super::{placeinfo::contains_unsafe_cell, utils::PlaceSet};

pub fn interior_places<'tcx>(
  place: Place<'tcx>,
  tcx: TyCtxt<'tcx>,
  body: &Body<'tcx>,
  def_id: DefId,
) -> PlaceSet<'tcx> {
  Interior::visit(
    place,
    tcx,
    body,
    def_id,
    StoppingCondition::BeforeRefs,
    None,
  )
  .places
}

pub fn interior_places_with_unsafe_cell<'tcx>(
  place: Place<'tcx>,
  tcx: TyCtxt<'tcx>,
  body: &Body<'tcx>,
  def_id: DefId,
) -> PlaceSet<'tcx> {
  Interior::visit(
    place,
    tcx,
    body,
    def_id,
    StoppingCondition::BeforeRefs,
    Some(body.typing_env(tcx)),
  )
  .places_with_unsafe_cell
}

pub fn interior_paths<'tcx>(
  place: Place<'tcx>,
  tcx: TyCtxt<'tcx>,
  body: &Body<'tcx>,
  def_id: DefId,
) -> PlaceSet<'tcx> {
  Interior::visit(place, tcx, body, def_id, StoppingCondition::None, None).places
}

pub fn interior_pointers<'tcx>(
  place: Place<'tcx>,
  tcx: TyCtxt<'tcx>,
  body: &Body<'tcx>,
  def_id: DefId,
) -> HashMap<RegionVid, Vec<(Place<'tcx>, Mutability)>> {
  Interior::visit(place, tcx, body, def_id, StoppingCondition::None, None).pointers
}

#[derive(Clone, Copy)]
enum StoppingCondition {
  None,
  BeforeRefs,
}

struct Interior<'tcx> {
  tcx: TyCtxt<'tcx>,
  def_id: DefId,
  local: Local,
  place_stack: Vec<PlaceElem<'tcx>>,
  ty_stack: Vec<Ty<'tcx>>,
  stop_at: StoppingCondition,
  typing_env: Option<TypingEnv<'tcx>>,
  places: PlaceSet<'tcx>,
  places_with_unsafe_cell: PlaceSet<'tcx>,
  pointers: HashMap<RegionVid, Vec<(Place<'tcx>, Mutability)>>,
}

impl<'tcx> Interior<'tcx> {
  fn visit(
    place: Place<'tcx>,
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    def_id: DefId,
    stop_at: StoppingCondition,
    typing_env: Option<TypingEnv<'tcx>>,
  ) -> Self {
    let mut interior = Interior {
      tcx,
      def_id,
      local: place.local,
      place_stack: place.projection.to_vec(),
      ty_stack: Vec::new(),
      stop_at,
      typing_env,
      places: PlaceSet::default(),
      places_with_unsafe_cell: PlaceSet::default(),
      pointers: HashMap::default(),
    };
    interior.visit_ty(place.ty(body.local_decls(), tcx).ty);
    interior
  }

  fn contains_unsafe_cell(&self, ty: Ty<'tcx>) -> bool {
    self
      .typing_env
      .is_some_and(|typing_env| contains_unsafe_cell(self.tcx, typing_env, ty))
  }

  fn visit_projection(&mut self, elem: PlaceElem<'tcx>, ty: Ty<'tcx>) -> bool {
    self.place_stack.push(elem);
    let visited = self.visit_ty(ty);
    self.place_stack.pop();
    visited
  }

  fn visit_child(
    &mut self,
    elem: PlaceElem<'tcx>,
    ty: Ty<'tcx>,
    unsafe_cell_in_children: &mut bool,
    unsafe_cell_outside_children: &mut bool,
  ) {
    let visited = self.visit_projection(elem, ty);
    let contains = self.contains_unsafe_cell(ty);
    if visited {
      *unsafe_cell_in_children |= contains;
    } else {
      *unsafe_cell_outside_children |= contains;
    }
  }

  fn visit_ty(&mut self, ty: Ty<'tcx>) -> bool {
    let tcx = self.tcx;
    if self.ty_stack.contains(&ty) {
      return false;
    }
    self.ty_stack.push(ty);
    let mut unsafe_cell_in_children = false;
    let mut unsafe_cell_outside_children = false;

    match ty.kind() {
      TyKind::Adt(_, args) if ty.is_box() => {
        let boxed = ty.boxed_ty().expect("a box has a boxed type");
        self.visit_child(
          ProjectionElem::Deref,
          boxed,
          &mut unsafe_cell_in_children,
          &mut unsafe_cell_outside_children,
        );
        for allocator in args.types().skip(1) {
          unsafe_cell_outside_children |= self.contains_unsafe_cell(allocator);
        }
      }

      TyKind::Tuple(fields) => {
        for (i, field) in fields.iter().enumerate() {
          self.visit_child(
            ProjectionElem::Field(FieldIdx::from_usize(i), field),
            field,
            &mut unsafe_cell_in_children,
            &mut unsafe_cell_outside_children,
          );
        }
      }

      TyKind::Adt(adt_def, args) => match adt_def.adt_kind() {
        ty::AdtKind::Struct => {
          for (i, field) in adt_def.all_fields().enumerate() {
            let ty = field.ty(tcx, args);
            if !field.vis.is_accessible_from(self.def_id, tcx) {
              unsafe_cell_outside_children |= self.contains_unsafe_cell(ty);
              continue;
            }
            self.visit_child(
              ProjectionElem::Field(FieldIdx::from_usize(i), ty),
              ty,
              &mut unsafe_cell_in_children,
              &mut unsafe_cell_outside_children,
            );
          }
        }
        ty::AdtKind::Union => {}
        ty::AdtKind::Enum => {
          for (i, variant) in adt_def.variants().iter().enumerate() {
            let variant_index = VariantIdx::from_usize(i);
            self.place_stack.push(PlaceElem::Downcast(
              Some(variant.ident(tcx).name),
              variant_index,
            ));
            for (j, field) in variant.fields.iter().enumerate() {
              let ty = field.ty(tcx, args);
              self.visit_child(
                ProjectionElem::Field(FieldIdx::from_usize(j), ty),
                ty,
                &mut unsafe_cell_in_children,
                &mut unsafe_cell_outside_children,
              );
            }
            self.place_stack.pop();
          }
        }
      },

      TyKind::Array(elem_ty, _) | TyKind::Slice(elem_ty) => {
        self.visit_child(
          ProjectionElem::Index(Local::from_usize(0)),
          *elem_ty,
          &mut unsafe_cell_in_children,
          &mut unsafe_cell_outside_children,
        );
      }

      TyKind::Ref(region, elem_ty, _) => match self.stop_at {
        StoppingCondition::None => {
          self.visit_region(*region);
          self.visit_projection(ProjectionElem::Deref, *elem_ty);
        }
        StoppingCondition::BeforeRefs => {}
      },

      TyKind::Closure(_, args) => {
        let upvars = args.as_closure().tupled_upvars_ty();
        let visited = self.visit_ty(upvars);
        let contains = self.contains_unsafe_cell(upvars);
        if visited {
          unsafe_cell_in_children |= contains;
        } else {
          unsafe_cell_outside_children |= contains;
        }
      }

      TyKind::Coroutine(_, args) => {
        unsafe_cell_outside_children |= self.contains_unsafe_cell(ty);
        self.visit_ty(args.as_closure().tupled_upvars_ty());
      }

      TyKind::RawPtr(..) if self.typing_env.is_some() => {}

      TyKind::RawPtr(ty, _) => {
        self.visit_region(Region::new_var(tcx, UNKNOWN_REGION));
        self.visit_projection(ProjectionElem::Deref, *ty);
      }

      TyKind::FnDef(..)
      | TyKind::FnPtr(..)
      | TyKind::Foreign(..)
      | TyKind::Dynamic(..)
      | TyKind::Param(..)
      | TyKind::Never => {}

      _ if ty.is_primitive_ty() => {}

      _ => warn!("unimplemented {ty:?} ({:?})", ty.kind()),
    }

    let place = Place::make(self.local, &self.place_stack, tcx);
    if self.contains_unsafe_cell(ty)
      && (unsafe_cell_outside_children || !unsafe_cell_in_children)
    {
      self.places_with_unsafe_cell.insert(place);
    }
    self.places.insert(place);
    self.ty_stack.pop();
    true
  }

  fn visit_region(&mut self, region: Region<'tcx>) {
    let region = match region.kind() {
      RegionKind::ReVar(region) => region,
      RegionKind::ReStatic => RegionVid::from_usize(0),
      RegionKind::ReErased | RegionKind::ReLateParam(_) => return,
      _ => unreachable!("{:?}: {region:?}", self.ty_stack.first()),
    };
    let mutability = if self
      .ty_stack
      .iter()
      .any(|ty| ty.ref_mutability() == Some(Mutability::Not))
    {
      Mutability::Not
    } else {
      Mutability::Mut
    };
    let place = Place::make(self.local, &self.place_stack, self.tcx);
    self
      .pointers
      .entry(region)
      .or_default()
      .push((place, mutability));
    self.places.insert(self.tcx.mk_place_deref(place));
  }
}
