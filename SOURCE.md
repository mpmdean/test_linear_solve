# Source provenance

Upstream: [EDRIXS/edrixs](https://github.com/EDRIXS/edrixs)

Production source revision:
[`83bd2fb29eebd3ffbfb469a8c5d8ea271f1ce3ba`](https://github.com/EDRIXS/edrixs/tree/83bd2fb29eebd3ffbfb469a8c5d8ea271f1ce3ba).

The following files are unmodified copies from that revision:

- `src/m_constants.f90`
- `src/m_control.f90`
- `src/m_types.f90`
- `src/spmv.f90`
- `src/linsys.f90`
- `LICENSE`

The Pu O45 parameter helper reproduces the EDRIXS example's parameter setup,
including its original Slater-integral rescaling behavior. The Python runner
and `pminres_test_bridge.f90` are test harness code added for this comparison.
The bridge converts PETSc local CSR data into EDRIXS's distributed CSR blocks
and calls the original solver; it does not implement a replacement MINRES.

The source hashes used for the reference measurement are recorded in
[`results/reference/commands.json`](results/reference/commands.json).
Its commands have been normalized to remove machine-specific paths; its
Fortran source hashes and numerical measurements are unchanged.

The repository is distributed under the included GNU GPL version 3 license.
