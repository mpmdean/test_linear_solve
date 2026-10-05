# Pu O45 linear-solver comparison

Compare EDRIXS's actual Fortran `pminres_csr` with PETSc `KSP` GMRES on the
**same complex Pu O45 intermediate-state matrix and right-hand side**.
This runs one incident energy, not a complete XAS/RIXS calculation.

The runner compiles the unchanged Fortran solver and distributed CSR matvec,
builds the physical system with PETSc+Numba, and checks that PETSc and Fortran
matrix-vector products agree before measuring either solve.

## Setup

Use a conda-forge environment with complex PETSc/SLEPc, their Python bindings,
Numba, MPI, and a Fortran compiler. `environment.yml` records the package
versions used for the reference run; exact builds vary by platform. The
standalone package was tested using the existing macOS environment, rather
than a fresh solve of this environment file.

```sh
conda env create -f environment.yml
conda activate pu-o45-linear-solvers
```

EDRIXS is an external dependency. Use a compatible `petsc_edrixs` source
checkout, not the older PyPI release. This commit supplies the APIs used here:

```sh
git clone --branch petsc_edrixs https://github.com/EDRIXS/edrixs.git ../edrixs-linear-test
git -C ../edrixs-linear-test checkout 83bd2fb29eebd3ffbfb469a8c5d8ea271f1ce3ba
```

You do not need to build the EDRIXS Fortran extension: the runner compiles the
five bundled production Fortran files itself. Use Python, `mpirun`, `mpifort`,
PETSc, and mpi4py from a compatible MPI environment. Tools are looked up beside
the Python interpreter first, then on `PATH`. The build command targets a GNU
Fortran-compatible compiler; macOS and Linux linker flags are selected automatically.

## Run

From this repository, invoke the parent runner once. It launches MPI itself:

```sh
python test_pu_o45_linear_solvers.py \
  --edrixs-root ../edrixs-linear-test \
  --output-dir runs/pu-o45-8rank \
  --ranks 8 --repeats 3 --energy-index 4
```

The output directory must not exist. If a compatible EDRIXS is already
importable, omit `--edrixs-root`. Choose `--ranks 1` for a serial smoke test.
`--energy-index` selects an index from the original ten-point incident grid
(0–9); the default index 4 is 112.511111111 eV.

The defaults are:

- Matrix dimension: 34,320; core-hole broadening: 0.2 eV.
- Eight MPI ranks, one BLAS/OpenMP/Numba thread per rank.
- Zero initial guesses; absolute tolerance `1e-8`; iteration cap 5,000 for both.
- GMRES restart 200, no preconditioner, relative tolerance zero.
- Three solves per method, alternating which method runs first.

The script writes `report.md`, `solves.csv`, `results.json`, `commands.json`,
`build.log`, and `run.log`, plus local build artifacts. An independent
`||Ax-b||` check is required for every solve; failed residual/convergence checks
produce a nonzero exit status after saving the results. Fortran's `info` output
is not a sufficient convergence check, because the production routine always
sets it to zero. Iteration counts come from its original diagnostic output.

## Reference result

On an Apple M2 Pro with eight MPI ranks:

| Solver | Median solve time | Iterations | True absolute residual |
| :--- | ---: | ---: | ---: |
| Fortran `pminres_csr` | 0.172 s | 361 | 6.32e-9 |
| PETSc GMRES | 1.654 s | 1,147 | 9.98e-9 |

Fortran was **9.62 times faster for this system**. Both methods passed the
`1e-8` residual threshold in all three repetitions. Their paired solutions
differed by at most `4.11e-8` in relative 2-norm. The independently checked
matrix-action difference was `2.15e-16`.

[Reference report](results/reference/report.md) ·
[Raw timings](results/reference/solves.csv) ·
[Raw results](results/reference/results.json)

## Timing interpretation

Timings are maximum elapsed solve times across ranks. Model construction,
first-use Numba compilation, ground-state calculation, CSR conversion, and
independent residual checks are excluded. There is no untimed solver warm-up.
Fresh PETSc KSP objects include lazy setup within `KSPSolve`; Fortran includes
temporary allocation, its diagnostic output, and its own final residual check.

PETSc and Fortran have different native matvec and communication
implementations. This compares implementations, not recurrence arithmetic
alone. One energy and three repetitions do not establish a universal ranking
or a full-workload speedup.

## Source and license

The bundled production Fortran files are copied unchanged from EDRIXS and
retain its GPLv3 license. See [SOURCE.md](SOURCE.md) for attribution and the
pinned source revision, and [LICENSE](LICENSE) for the license text.
Compiled binaries and generated runs are not committed.

## Performance investigation and faster PETSc solvers

[Investigation, measured speedups, and recommendations](PERFORMANCE.md).
The [mathematical note](output/pdf/shifted_pminres_math.pdf)
([LaTeX source](output/pdf/shifted_pminres_math.tex)) derives the recurrence,
maps the equations to the code, and explains why the true residual must be checked.

The compiled PETSc short recurrence reduced median solve time from 1.854 s
to 0.153 s (12.1×) in seven repetitions, with every residual check passing.
Native PETSc BiCG required only a solver-setting change and achieved 0.303 s
(6.1×). The original Fortran median was 0.168 s in that comparison.

The runner now accepts `--methods` to compare built-in PETSc alternatives and
two experimental ports of the EDRIXS shifted short recurrence. The original
two-method comparison remains the default.

```sh
python test_pu_o45_linear_solvers.py \
  --output-dir runs/optimized --ranks 8 --repeats 7 \
  --methods pminres_csr gmres gmres400 bicg petsc_pminres_c
```

- `gmres400` increases the restart to 400; `gmres800` tests 800.
- `bicg` uses native PETSc BiCG without preconditioning. It needs the adjoint
  matrix product, which the assembled AIJ matrix provides.
- `petsc_pminres` ports the bundled Fortran recurrence to Python with PETSc
  distributed Mat/Vec operations.
- `petsc_pminres_c` runs that recurrence in C with PETSc Mat/Vec operations and
  fused vector updates. It requires `mpicc` and `pkg-config` with `PETSc.pc`.
  The runner compiles it automatically using the active environment.
- `gmres_jacobi`, `gmres_bjacobi`, `gmres_asm`, `bcgs`, `bcgsl`, `tfqmr`, and
  `cg_symmetric` are exploratory alternatives. Some fail for this system; the
  runner saves the results and exits nonzero if any solve fails.

Add `--profile --matvec-repeats 500` for per-solve PETSc event profiles and
separate matrix-product timings. Use an unprofiled run for final timing.
Preconditioner construction remains inside the solve timer. Subdomain options
remain available until lazy preconditioner setup completes.

The experimental recurrence requires **A = H + i gamma I with H Hermitian**,
no preconditioner, and zero initial guesses. The benchmark verifies this
structure numerically. This is not PETSc's standard Hermitian MINRES, and
should not be selected for a general nonsymmetric operator. Both ports check
the true residual before declaring convergence; the runner checks it again
independently. The unchanged production Fortran remains the reference.

Run the small numerical tests after a compiled benchmark:

```sh
PMINRES_TEST_LIBRARY="$PWD/runs/optimized/build/libpetsc_pminres.dylib" \
  python -m unittest -v test_shifted_pminres
```

The tests cover genuinely complex Hermitian matrices with imaginary shifts,
comparison to a dense direct solution, zero RHS, exact one-step termination,
and iteration-limit failure. Without the environment variable, only the
Python-port tests run.
