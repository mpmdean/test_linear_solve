# PETSc performance investigation — 2026-10-05

PETSc can match the Fortran solver on this problem. The original comparison
mostly measures the cost of restarted GMRES versus a short recurrence,
rather than a Fortran-versus-PETSc kernel-speed difference.

## Measured result

Same 34,320-dimensional system, incident energy index 4, eight MPI ranks, one
thread per rank, PETSc 3.25.5 with complex double precision and Open MPI 5.0.10.
These are medians of **seven unprofiled solves per method**, reversing order
on even repetitions. Fresh solvers, zero guesses, absolute tolerance `1e-8`,
and maximum 5,000 iterations. Every solve below passed the independent
`||Ax-b||` check. Preconditioner/solver setup inside `KSPSolve` is included.

| Method | Median seconds | Iterations | Speedup over original PETSc | Worst true residual |
| :--- | ---: | ---: | ---: | ---: |
| Original Fortran PMINRES | 0.167717 | 361 | 11.05× | 6.318e-9 |
| Original PETSc GMRES(200) | 1.853958 | 1,147 | 1.00× | 9.985e-9 |
| PETSc GMRES(400) | 0.868269 | 358 | 2.14× | 8.717e-9 |
| PETSc BiCG | 0.302603 | 362 | 6.13× | 6.863e-9 |
| Shifted short recurrence in C using PETSc | **0.152891** | 361 | **12.13×** | 6.864e-9 |

The optimized PETSc implementation is competitive with Fortran. Its median
is about 9% lower here, but individual timings overlap and fluctuate; the
strong conclusion is removal of the roughly tenfold gap, not a universal
9% advantage. The largest paired relative solution difference between the
compiled PETSc recurrence and Fortran was about `1.8e-10`.

[Full timing report](results/investigation/optimized/report.md) ·
[Every solve](results/investigation/optimized/solves.csv) ·
[Commands and source hashes](results/investigation/optimized/commands.json)

The unmodified original test was also run first: Fortran 0.157447 s, GMRES
1.663795 s, all six solves passing. All bundled production Fortran files and
the original C-interoperability bridge still match the reference SHA-256 hashes.

## Why the original PETSc configuration loses

1. **Restarting loses useful progress.** GMRES(200) takes 1,147 iterations.
   With restart 400 it converges in 358 iterations, before the first restart.
   Restart 800 also takes 358 iterations and gives no useful improvement.
   The Fortran recurrence retains its progress using a fixed number of vectors
   and takes 361 iterations.

2. **GMRES orthogonalization is expensive.** In the exploratory profile,
   `KSPGMRESOrthog` took 1.0914 s within a 1.6505 s `KSPSolve`: about 66%.
   `MatMult` took 0.4991 s; `KSPSetUp` only 0.000493 s. The multi-dot and
   multi-AXPY operations account for most of the orthogonalization cost.
   These are nested event timings; do not add them to their parent events.
   GMRES compares each new basis vector with all retained vectors, unlike the
   constant-work short recurrence. This is consistent with PETSc's
   [restart documentation](https://petsc.org/release/manualpages/KSP/KSPGMRESSetRestart/).

3. **PETSc's matrix kernel is already competitive.** A separate 500-product
   measurement gave 0.329 ms/product for PETSc versus 0.405 ms for the Fortran
   distributed CSR routine. A prior 200-product measurement also favored
   PETSc, 0.339 versus 0.389 ms. These include each implementation's native
   communication and the Python calling boundary. They rule out a tenfold
   slower PETSc matrix product on this machine.

4. **The problem has exploitable structure.** Here `A = H + 0.2i I`, with
   `H` Hermitian. The measured relative Hermitian defect after removing the
   imaginary shift is `1.37e-17`. The matrix also happens to be transpose
   symmetric to `1.37e-17`, but it is not Hermitian with the shift included.
   The bundled Fortran routine exploits a shifted-Hermitian short recurrence;
   switching blindly to ordinary PETSc MINRES would not reproduce it.

[Exploratory PETSc profile](results/investigation/sweep/petsc-profile.txt) ·
[Profile of the implemented recurrence](results/investigation/profile/petsc-profile.txt)

## What to use

**Smallest integration change: native PETSc BiCG, no preconditioner.** It was
6.13× faster than baseline GMRES in the final repeated comparison:

```python
ksp.setType('bicg')
ksp.getPC().setType('none')
ksp.setPCSide(PETSc.PC.Side.LEFT)
ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
ksp.setTolerances(rtol=0, atol=1e-8, max_it=5000)
ksp.setOperators(matrix)
ksp.solve(rhs, solution)
```

Keep the independent residual check and a GMRES fallback if BiCG breaks down.
[PETSc BiCG](https://petsc.org/release/manualpages/KSP/KSPBICG/) uses both the
operator and its adjoint in the complex build. The assembled matrix supports
this; a matrix-free replacement must supply the appropriate adjoint operation.
Its roughly two matrix products per iteration explain much of the remaining
gap to the one-product short recurrence.

**If staying with GMRES: use restart 400.** This gives a 2.14× improvement here,
at the cost of more basis storage. No preconditioner, same tolerance. A
restart of 800 was unnecessary for this particular solve.

**For Fortran-level performance: use the experimental shifted recurrence.**
`petsc_shifted_pminres.c` ports the existing recurrence to PETSc `MatMult`,
`VecDot`, `VecNorm`, and fused `VecAXPBYPCZ` updates. It uses five work vectors
instead of retaining a GMRES basis. `petsc_shifted_pminres.py` supplies its
KSP Python adapter, with one C call for the entire iteration loop, plus a
readable Python implementation for comparison. Both versions verify the
actual residual before accepting convergence and report breakdown/iteration
failure. They retain the original zero guess, tolerance, and iteration limit.

The pure Python recurrence was already competitive in the five-repeat profile
(0.157496 s versus 0.157374 s compiled). Compiling the loop is not the main
source of the speedup: choosing the right recurrence is.

This is a benchmark prototype specialized to shifted-Hermitian matrices with
no preconditioner. The harness verifies that structure before using it. It is
not a general KSP replacement, and does not implement the full monitor and
custom-convergence callback interface of native PETSc solvers. Further
production work should preserve the structure check and true-residual check,
and provide a fallback for recurrence breakdown.

## Alternatives that failed

The one-repeat exploratory sweep also tested Jacobi, block-Jacobi with local
ILU(0), and overlapping ASM with local ILU(0), all with GMRES(200) and right
preconditioning. All hit 5,000 iterations with residuals around 1.6. They
should not be recommended as improvements for this system. Complex-symmetric
CG stopped with an indefinite-matrix reason; TFQMR stopped with a nonfinite
arithmetic reason. Their short times are **failures, not speedups**.

BiCGStab and BiCGStab(l) passed but were slower than BiCG in that sweep.
The sweep intentionally returns nonzero because some candidates fail; its
saved results retain the failures. Subdomain options in this early sweep
matched PETSc's defaults; the final harness keeps options available through
lazy setup so future nondefault subdomain settings also take effect.

## Reproduce

Use the environment described in README.md, with `mpicc`, `pkg-config`, and
PETSc development headers available for the compiled recurrence:

```sh
python test_pu_o45_linear_solvers.py \
  --output-dir runs/my-optimized --ranks 8 --repeats 7 \
  --methods pminres_csr gmres gmres400 bicg petsc_pminres_c

python test_pu_o45_linear_solvers.py \
  --output-dir runs/my-profile --ranks 8 --repeats 3 \
  --methods pminres_csr gmres gmres400 bicg petsc_pminres petsc_pminres_c \
  --profile --matvec-repeats 500

PMINRES_TEST_LIBRARY="$PWD/runs/my-optimized/build/libpetsc_pminres.dylib" \
  python -m unittest -v test_shifted_pminres
```

Eight numerical tests passed: both ports against a dense solve for a genuinely
complex Hermitian matrix with an imaginary shift, zero RHS, single-eigenvalue
termination, and an enforced iteration-limit failure. Physical-system
validation retains the independent matrix-action and solution-residual checks.

The low- and high-energy endpoints also passed three repetitions each for
Fortran, BiCG, and the compiled PETSc recurrence (all 18 additional solves):

| Incident energy | Fortran median | BiCG median | PETSc recurrence median | Fortran / PETSc recurrence iterations |
| :--- | ---: | ---: | ---: | ---: |
| 101.4 eV, index 0 | 0.041207 s | 0.085501 s | 0.040368 s | 74 / 74 |
| 126.4 eV, index 9 | 0.070419 s | 0.115390 s | 0.075571 s | 128 / 128 |

[Low-energy results](results/investigation/energy0/report.md) ·
[High-energy results](results/investigation/energy9/report.md).
These support comparable speed and accuracy at three sampled energies; they
do not establish robustness for every Hamiltonian, broadening, or RHS.

Model construction, Numba compilation, the ground-state solve, matrix
conversion/structure checks, and external residual checks are outside solve
timing. This study measures individual linear solves, not full RIXS wall time.
