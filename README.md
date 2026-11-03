# Flowistry fork

This fork of [Flowistry](https://github.com/willcrichton/flowistry) gives a sound alias
and mutation analysis of any MIR body whose region constraints the caller gives. The
upstream README is in [README.upstream.md](README.upstream.md).

## Differences from upstream

**Region constraints are an input.** `PlaceInfo::build_from_subset_base` takes a MIR
body whose regions are region variables, the outlives constraints between them, and a
typing env. Upstream takes only the borrowck facts of a local function. So the fork also
analyzes bodies that rustc does not borrowck, such as monomorphized instances.

**Aliases and mutations are sound.** The fork keeps two kinds that upstream drops:

- Writes through a shared reference to data that may contain an `UnsafeCell`, such as
  `&Cell<T>`, `&RefCell<T>`, and `&Rc<T>`. Only these types lose precision.
- Loans of an argument region that no reference place has. They are approximated by the
  innermost places of the argument whose type mentions the region.
