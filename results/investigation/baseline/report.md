# Pu O45: Fortran PMINRES versus PETSc GMRES

One 34320 × 34320 intermediate-state system, 8 MPI ranks, one thread/rank; incident grid index 4, omega=112.511111111 eV, broadening 0.2 eV.

Both methods receive identical CSR coefficients and RHS, use a zero initial guess, absolute tolerance 1e-8 and a common 5,000-iteration cap. GMRES restart=200, no preconditioner. The actual production Fortran sources are compiled unchanged with -O3 and called through a test-only C interface. CSR conversion preserves PETSc row ownership and uses the production Fortran MPI matvec.

| Solver | Median solve (s) | Range (s) | Iterations | Worst true residual | Passes |
| :--- | ---: | ---: | ---: | ---: | ---: |
| pminres_csr | 0.157447 | 0.152708–0.169003 | 361–361 | 6.318e-09 | 3/3 |
| gmres | 1.663795 | 1.633522–1.931024 | 1147–1147 | 9.985e-09 | 3/3 |

GMRES/PMINRES median time ratio: **10.567**. Interpret speed only together with the independently checked residuals.

Fortran/PETSc matrix-action relative difference: 2.151e-16.

3 measurements per solver in alternating order. Setup, JIT, matrix conversion, and independent residual checks are outside timing. Maximum elapsed time across ranks is reported. Fresh KSP objects include lazy solver setup in KSPSolve; Fortran includes allocation, its diagnostic output and its own final residual check. No untimed solver warm-up. This is one incident energy, not a full-workload or general solver ranking. Native matvec/communication implementations differ, so this measures implementations rather than recurrence cost alone.

Per-solve residuals and solution differences: [solves.csv](solves.csv). Production source hashes and commands: [commands.json](commands.json).

Maximum paired relative solution difference: 4.109e-08.
