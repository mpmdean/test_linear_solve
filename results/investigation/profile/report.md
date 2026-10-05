# Pu O45: Fortran PMINRES versus PETSc solvers

One 34320 × 34320 intermediate-state system, 8 MPI ranks, one thread/rank; incident grid index 4, omega=112.511111111 eV, broadening 0.2 eV.

All methods receive identical CSR coefficients and RHS, use a zero initial guess, absolute tolerance 1e-8 and a common 5,000-iteration cap. The gmres baseline uses restart=200 and no preconditioner; variant options are recorded in results.json. GMRES variants use right preconditioning and an unpreconditioned residual norm. The actual production Fortran sources are compiled unchanged with -O3 and called through a test-only C interface. CSR conversion preserves PETSc row ownership and uses the production Fortran MPI matvec.

| Solver | Median solve (s) | Range (s) | Iterations | Worst true residual | Passes |
| :--- | ---: | ---: | ---: | ---: | ---: |
| pminres_csr | 0.172131 | 0.147143–0.174961 | 361–361 | 6.318e-09 | 5/5 |
| gmres | 2.009769 | 1.776621–2.364638 | 1147–1147 | 9.985e-09 | 5/5 |
| gmres400 | 0.962197 | 0.816184–1.061697 | 358–358 | 8.717e-09 | 5/5 |
| bicg | 0.381581 | 0.303175–0.503301 | 362–362 | 6.863e-09 | 5/5 |
| petsc_pminres | 0.157496 | 0.147239–0.173672 | 361–361 | 5.496e-09 | 5/5 |
| petsc_pminres_c | 0.157374 | 0.146838–0.168105 | 361–361 | 6.864e-09 | 5/5 |

GMRES/PMINRES median time ratio: **11.676**.

Interpret speed only together with the independently checked residuals.

Fortran/PETSc matrix-action relative difference: 2.151e-16.

5 measurements per solver, reversing order on even repeats. Model setup, JIT, matrix conversion, and independent residual checks are outside timing. Maximum elapsed time across ranks is reported. Fresh KSP objects include lazy solver setup in KSPSolve; Fortran includes allocation, its diagnostic output and its own final residual check. No untimed solver warm-up. Optional matrix-product measurements run before the solvers; profiling adds instrumentation overhead. This is one incident energy, not a full-workload or general solver ranking. Native matvec/communication implementations differ, so this measures implementations rather than recurrence cost alone.

Per-solve residuals and solution differences: [solves.csv](solves.csv). Production source hashes and commands: [commands.json](commands.json).

Maximum relative solution difference from the first method in each repeat: 4.110e-08.

Isolated matrix products (seconds/product, maximum rank elapsed):

- fortran: 0.000404534
- petsc: 0.00032858
