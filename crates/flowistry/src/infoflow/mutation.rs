//! Identifies the mutated places in a MIR instruction via modular approximation based on types.

use log::debug;
use rustc_abi::FieldIdx;
use rustc_middle::{
  mir::{visit::Visitor, *},
  ty::{AdtKind, TyKind},
};
use rustc_utils::{AdtDefExt, OperandExt, mir::place::PlaceCollector};

use crate::mir::{
  placeinfo::PlaceInfo,
  utils::{self, AsyncHack},
};

/// Indicator of certainty about whether a place is being mutated.
/// Used to determine whether an update should be strong or weak.
#[derive(Clone, Copy, Debug)]
pub enum MutationStatus {
  /// A place is definitely mutated, e.g. `x = y` definitely mutates `x`.
  Definitely,

  /// A place is possibly mutated, e.g. `f(&mut x)` possibly mutates `x`.
  Possibly,
}

/// Information about a particular mutation.
#[derive(Debug)]
pub struct Mutation<'tcx> {
  /// The place that is being mutated.
  pub mutated: Place<'tcx>,

  /// The set of inputs to the mutating operation.
  pub inputs: Vec<Place<'tcx>>,

  /// The certainty of whether the mutation is happening.
  pub status: MutationStatus,
}

/// MIR visitor that invokes a callback for every [`Mutation`] in the visited object.
///
/// Construct the visitor with [`ModularMutationVisitor::new`], then call one of the
/// MIR [`Visitor`] methods.
pub struct ModularMutationVisitor<'a, 'tcx, F>
where
  F: FnMut(Location, Vec<Mutation<'tcx>>),
{
  f: F,
  place_info: &'a PlaceInfo<'a, 'tcx>,
}

impl<'a, 'tcx, F> ModularMutationVisitor<'a, 'tcx, F>
where
  F: FnMut(Location, Vec<Mutation<'tcx>>),
{
  /// Constructs a new visitor.
  pub fn new(place_info: &'a PlaceInfo<'a, 'tcx>, f: F) -> Self {
    ModularMutationVisitor { place_info, f }
  }

  /// Returns the fields of a tuple or of a struct whose fields are all visible.
  fn fields(&self, place: Place<'tcx>) -> Vec<PlaceElem<'tcx>> {
    let tcx = self.place_info.tcx;
    match place.ty(&self.place_info.body.local_decls, tcx).ty.kind() {
      TyKind::Tuple(tys) => tys
        .iter()
        .enumerate()
        .map(|(i, ty)| PlaceElem::Field(FieldIdx::from_usize(i), ty))
        .collect(),
      TyKind::Closure(_, args) => args
        .as_closure()
        .upvar_tys()
        .iter()
        .enumerate()
        .map(|(i, ty)| PlaceElem::Field(FieldIdx::from_usize(i), ty))
        .collect(),
      TyKind::Adt(adt_def, substs)
        if adt_def.is_struct()
          && adt_def
            .all_visible_fields(self.place_info.def_id, tcx)
            .count()
            == adt_def.all_fields().count() =>
      {
        adt_def
          .all_fields()
          .enumerate()
          .map(|(i, field_def)| {
            PlaceElem::Field(FieldIdx::from_usize(i), field_def.ty(tcx, substs))
          })
          .collect()
      }
      _ => Vec::new(),
    }
  }

  /// Destructures `mutated = input` into mutations of the innermost fields.
  fn destructure(&self, mutated: Place<'tcx>, input: Place<'tcx>) -> Vec<Mutation<'tcx>> {
    let tcx = self.place_info.tcx;
    let fields = self.fields(input);
    if fields.is_empty() {
      return vec![Mutation {
        mutated,
        inputs: vec![input],
        status: MutationStatus::Definitely,
      }];
    }
    fields
      .into_iter()
      .flat_map(|field| {
        self.destructure(
          mutated.project_deeper(&[field], tcx),
          input.project_deeper(&[field], tcx),
        )
      })
      .collect()
  }
}

impl<'tcx, F> Visitor<'tcx> for ModularMutationVisitor<'_, 'tcx, F>
where
  F: FnMut(Location, Vec<Mutation<'tcx>>),
{
  fn visit_assign(
    &mut self,
    mutated: &Place<'tcx>,
    rvalue: &Rvalue<'tcx>,
    location: Location,
  ) {
    debug!("Checking {location:?}: {mutated:?} = {rvalue:?}");
    let body = self.place_info.body;
    let tcx = self.place_info.tcx;

    match rvalue {
      // In the case of _1 = aggregate { field1: op1, field2: op2, ... },
      // then destructure this into a series of mutations like
      // _1.field1 = op1, _1.field2 = op2, and so on, each destructured further.
      Rvalue::Aggregate(agg_kind, ops) => {
        let info = match &**agg_kind {
          AggregateKind::Adt(def_id, idx, substs, _, _) => {
            let adt_def = tcx.adt_def(*def_id);
            let variant = adt_def.variant(*idx);
            let mutated = match adt_def.adt_kind() {
              AdtKind::Enum => mutated.project_deeper(
                &[ProjectionElem::Downcast(Some(variant.name), *idx)],
                tcx,
              ),
              AdtKind::Struct | AdtKind::Union => *mutated,
            };
            let fields = variant.fields.iter();
            let tys = fields
              .map(|field| field.ty(tcx, substs))
              .collect::<Vec<_>>();
            Some((mutated, tys))
          }
          AggregateKind::Tuple => {
            let ty = rvalue.ty(body.local_decls(), tcx);
            Some((*mutated, ty.tuple_fields().to_vec()))
          }
          AggregateKind::Closure(_, args) => {
            Some((*mutated, args.as_closure().upvar_tys().to_vec()))
          }
          _ => None,
        };

        if let Some((fields_place, tys)) = info
          && tys.len() > 0
        {
          let discriminant_write = matches!(
            **agg_kind,
            AggregateKind::Adt(def_id, ..) if tcx.adt_def(def_id).is_enum()
          )
          .then(|| Mutation {
            mutated: *mutated,
            inputs: Vec::new(),
            status: MutationStatus::Definitely,
          });
          let fields = tys.into_iter().enumerate().zip(ops.iter()).flat_map(
            |((i, ty), input_op)| {
              let field = PlaceElem::Field(FieldIdx::from_usize(i), ty);
              let mutated = fields_place.project_deeper(&[field], tcx);
              match input_op.as_place() {
                Some(input) => self.destructure(mutated, input),
                None => vec![Mutation {
                  mutated,
                  inputs: Vec::new(),
                  status: MutationStatus::Definitely,
                }],
              }
            },
          );
          let mutations = discriminant_write
            .into_iter()
            .chain(fields)
            .collect::<Vec<_>>();
          (self.f)(location, mutations);
          return;
        }
      }

      // In the case of _1 = _2 where _2 : struct Foo { x: T, y: S, .. },
      // then destructure this into a series of mutations like
      // _1.x = _2.x, _1.y = _2.y, and so on, each destructured further.
      Rvalue::Use(Operand::Move(place) | Operand::Copy(place)) => {
        let mutations = self.destructure(*mutated, *place);
        (self.f)(location, mutations);
        return;
      }

      _ => {}
    }

    let mut collector = PlaceCollector::default();
    collector.visit_rvalue(rvalue, location);
    (self.f)(location, vec![Mutation {
      mutated: *mutated,
      inputs: collector.0,
      status: MutationStatus::Definitely,
    }]);
  }

  fn visit_terminator(&mut self, terminator: &Terminator<'tcx>, location: Location) {
    debug!("Checking {location:?}: {:?}", terminator.kind);
    let tcx = self.place_info.tcx;

    match &terminator.kind {
      TerminatorKind::Call {
        func,
        args,
        destination,
        ..
      } => {
        let async_hack = AsyncHack::new(
          self.place_info.tcx,
          self.place_info.body,
          self.place_info.def_id,
        );
        let arg_places = utils::arg_places(args)
          .into_iter()
          .map(|(_, place)| place)
          .filter(|place| !async_hack.ignore_place(*place))
          .collect::<Vec<_>>();
        let arg_inputs = arg_places.clone();

        let ret_is_unit = destination
          .ty(self.place_info.body.local_decls(), tcx)
          .ty
          .is_unit();
        let inputs = if ret_is_unit {
          Vec::new()
        } else {
          arg_inputs.clone()
        };

        if let Some((def_id, _)) = func.const_fn_def()
          && tcx.def_path_str(def_id) == "std::boxed::box_assume_init_into_vec_unsafe"
        {
          // TODO: we need to fix this case to fix vec_read...
        }

        let mut mutations = vec![Mutation {
          mutated: *destination,
          inputs,
          status: MutationStatus::Definitely,
        }];

        for arg in arg_places {
          for arg_mut in self.place_info.reachable_values(arg, Mutability::Mut) {
            mutations.push(Mutation {
              mutated: *arg_mut,
              inputs: arg_inputs.clone(),
              status: MutationStatus::Possibly,
            });
          }
        }

        (self.f)(location, mutations);
      }

      _ => {}
    }
  }
}
