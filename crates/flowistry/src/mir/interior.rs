use log::warn;
use rustc_abi::{FieldIdx, VariantIdx};
use rustc_data_structures::fx::FxHashMap as HashMap;
use rustc_hir::def_id::DefId;
use rustc_middle::{
  mir::{Body, HasLocalDecls, Local, Mutability, Place, PlaceElem, ProjectionElem},
  ty::{self, Region, RegionKind, RegionVid, Ty, TyCtxt, TyKind},
};
use rustc_utils::{PlaceExt, mir::place::UNKNOWN_REGION};

use super::utils::PlaceSet;

pub fn interior_places<'tcx>(
  place: Place<'tcx>,
  tcx: TyCtxt<'tcx>,
  body: &Body<'tcx>,
  def_id: DefId,
) -> PlaceSet<'tcx> {
  Interior::visit(place, tcx, body, def_id, StoppingCondition::BeforeRefs).places
}

pub fn interior_paths<'tcx>(
  place: Place<'tcx>,
  tcx: TyCtxt<'tcx>,
  body: &Body<'tcx>,
  def_id: DefId,
) -> PlaceSet<'tcx> {
  Interior::visit(place, tcx, body, def_id, StoppingCondition::None).places
}

pub fn interior_pointers<'tcx>(
  place: Place<'tcx>,
  tcx: TyCtxt<'tcx>,
  body: &Body<'tcx>,
  def_id: DefId,
) -> HashMap<RegionVid, Vec<(Place<'tcx>, Mutability)>> {
  Interior::visit(place, tcx, body, def_id, StoppingCondition::None).pointers
}

pub fn all_places<'tcx>(
  body: &Body<'tcx>,
  tcx: TyCtxt<'tcx>,
  def_id: DefId,
) -> impl Iterator<Item = Place<'tcx>> {
  body.local_decls.indices().flat_map(move |local| {
    interior_paths(Place::from_local(local, tcx), tcx, body, def_id)
  })
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
  places: PlaceSet<'tcx>,
  pointers: HashMap<RegionVid, Vec<(Place<'tcx>, Mutability)>>,
}

impl<'tcx> Interior<'tcx> {
  fn visit(
    place: Place<'tcx>,
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    def_id: DefId,
    stop_at: StoppingCondition,
  ) -> Self {
    let mut interior = Interior {
      tcx,
      def_id,
      local: place.local,
      place_stack: place.projection.to_vec(),
      ty_stack: Vec::new(),
      stop_at,
      places: PlaceSet::default(),
      pointers: HashMap::default(),
    };
    interior.visit_ty(place.ty(body.local_decls(), tcx).ty);
    interior
  }

  fn visit_projection(&mut self, elem: PlaceElem<'tcx>, ty: Ty<'tcx>) {
    self.place_stack.push(elem);
    self.visit_ty(ty);
    self.place_stack.pop();
  }

  fn visit_ty(&mut self, ty: Ty<'tcx>) {
    let tcx = self.tcx;
    if self.ty_stack.contains(&ty) {
      return;
    }
    self.ty_stack.push(ty);

    match ty.kind() {
      _ if ty.is_box() => {
        let boxed = ty.boxed_ty().expect("a box has a boxed type");
        self.visit_projection(ProjectionElem::Deref, boxed);
      }

      TyKind::Tuple(fields) => {
        for (i, field) in fields.iter().enumerate() {
          self.visit_projection(
            ProjectionElem::Field(FieldIdx::from_usize(i), field),
            field,
          );
        }
      }

      TyKind::Adt(adt_def, args) => match adt_def.adt_kind() {
        ty::AdtKind::Struct => {
          for (i, field) in adt_def.all_fields().enumerate() {
            if !field.vis.is_accessible_from(self.def_id, tcx) {
              continue;
            }
            let ty = field.ty(tcx, args);
            self.visit_projection(ProjectionElem::Field(FieldIdx::from_usize(i), ty), ty);
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
              self
                .visit_projection(ProjectionElem::Field(FieldIdx::from_usize(j), ty), ty);
            }
            self.place_stack.pop();
          }
        }
      },

      TyKind::Array(elem_ty, _) | TyKind::Slice(elem_ty) => {
        self.visit_projection(ProjectionElem::Index(Local::from_usize(0)), *elem_ty);
      }

      TyKind::Ref(region, elem_ty, _) => match self.stop_at {
        StoppingCondition::None => {
          self.visit_region(*region);
          self.visit_projection(ProjectionElem::Deref, *elem_ty);
        }
        StoppingCondition::BeforeRefs => {}
      },

      TyKind::Closure(_, args) | TyKind::Coroutine(_, args) => {
        self.visit_ty(args.as_closure().tupled_upvars_ty());
      }

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

    self
      .places
      .insert(Place::make(self.local, &self.place_stack, tcx));
    self.ty_stack.pop();
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
