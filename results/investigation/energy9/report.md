# Pu O45: Fortran PMINRES versus PETSc solvers

One 34320 × 34320 intermediate-state system, 8 MPI ranks, one thread/rank; incident grid index 9, omega=126.400000000 eV, broadening 0.2 eV.

All methods receive identical CSR coefficients and RHS, use a zero initial guess, absolute tolerance 1e-8 and a common 5,000-iteration cap. The gmres baseline uses restart=200 and no preconditioner; variant options are recorded in results.json. GMRES variants use right preconditioning and an unpreconditioned residual norm. The actual production Fortran sources are compiled unchanged with -O3 and called through a test-only C interface. CSR conversion preserves PETSc row ownership and uses the production Fortran MPI matvec.

| Solver | Median solve (s) | Range (s) | Iterations | Worst true residual | Passes |
| :--- | ---: | ---: | ---: | ---: | ---: |
| pminres_csr | 0.070419 | 0.062952–0.093478 | 128–128 | 7.725e-09 | 3/3 |
| bicg | 0.115390 | 0.108510–0.162601 | 130–130 | 7.968e-09 | 3/3 |
| petsc_pminres_c | 0.075571 | 0.052554–0.111703 | 128–128 | 7.717e-09 | 3/3 |

Interpret speed only together with the independently checked residuals.

Fortran/PETSc matrix-action relative difference: 2.471e-16.

3 measurements per solver, reversing order on even repeats. Model setup, JIT, matrix conversion, and independent residual checks are outside timing. Maximum elapsed time across ranks is reported. Fresh KSP objects include lazy solver setup in KSPSolve; Fortran includes allocation, its diagnostic output and its own final residual check. No untimed solver warm-up. Optional matrix-product measurements run before the solvers; profiling adds instrumentation overhead. This is one incident energy, not a full-workload or general solver ranking. Native matvec/communication implementations differ, so this measures implementations rather than recurrence cost alone.

Per-solve residuals and solution differences: [solves.csv](solves.csv). Production source hashes and commands: [commands.json](commands.json).

Maximum relative solution difference from the first method in each repeat: 4.164e-08.
