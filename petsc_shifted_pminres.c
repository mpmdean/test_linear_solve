/* GPL-3.0: PETSc port of the EDRIXS recurrence in src/linsys.f90.
 * Requires A = H + i*gamma*I, H Hermitian, PCNONE and a zero initial guess.
 * This is a benchmark implementation, not PETSc's standard KSPMINRES.
 */
#include <petscksp.h>

PetscErrorCode shifted_pminres(Mat A, Vec b, Vec x, PetscReal tol,
                              PetscInt maxits, PetscInt *iterations, PetscInt *reason)
{
    Vec v, vp, w, wp, work, swap;
    PetscReal beta, next, estimate, actual;
    PetscScalar eta, alpha, delta, rho1, rho2, rho3;
    PetscScalar gamma0 = 1, gamma1 = 1, sigma0 = 0, sigma1 = 0, oldgamma, oldsigma;
    PetscFunctionBeginUser;
    PetscCall(VecDuplicate(b, &v));
    PetscCall(VecDuplicate(b, &vp));
    PetscCall(VecDuplicate(b, &w));
    PetscCall(VecDuplicate(b, &wp));
    PetscCall(VecDuplicate(b, &work));
    PetscCall(VecCopy(b, v));
    PetscCall(VecSet(vp, 0));
    PetscCall(VecSet(w, 0));
    PetscCall(VecSet(wp, 0));
    PetscCall(VecSet(x, 0));
    PetscCall(VecNorm(b, NORM_2, &beta));
    eta = estimate = beta;
    *iterations = 0;
    *reason = beta <= tol ? KSP_CONVERGED_ATOL : KSP_DIVERGED_ITS;
    while (beta > 0 && *iterations < maxits && *reason < 0) {
        ++*iterations;
        PetscCall(VecScale(v, 1.0 / beta));
        PetscCall(MatMult(A, v, work));
        /* PETSc conjugates the second dot-product argument. */
        PetscCall(VecDot(work, v, &alpha));
        PetscCall(VecAXPBYPCZ(vp, 1.0, -alpha, -beta, work, v));
        PetscCall(VecNorm(vp, NORM_2, &next));
        delta = gamma0 * alpha - gamma1 * sigma0 * beta;
        rho1 = PetscSqrtScalar(delta * delta + next * next);
        if (PetscAbsScalar(rho1) == 0 || PetscIsInfOrNanScalar(rho1)) {
            *reason = KSP_DIVERGED_BREAKDOWN;
            break;
        }
        rho2 = sigma0 * alpha + gamma1 * gamma0 * beta;
        rho3 = sigma1 * beta;
        oldgamma = gamma0;
        oldsigma = sigma0;
        gamma0 = delta / rho1;
        sigma0 = next / rho1;
        gamma1 = oldgamma;
        sigma1 = oldsigma;
        PetscCall(VecAXPBYPCZ(wp, 1.0 / rho1, -rho2 / rho1, -rho3 / rho1, v, w));
        PetscCall(VecAXPY(x, gamma0 * eta, wp));
        estimate *= PetscAbsScalar(sigma0);
        eta *= -sigma0;
        if (estimate <= tol || next == 0) {
            PetscCall(MatMult(A, x, work));
            PetscCall(VecAXPY(work, -1, b));
            PetscCall(VecNorm(work, NORM_2, &actual));
            if (actual <= tol) {
                *reason = KSP_CONVERGED_ATOL;
                break;
            }
        }
        if (next == 0) {
            *reason = KSP_DIVERGED_BREAKDOWN;
            break;
        }
        beta = next;
        swap = v; v = vp; vp = swap;
        swap = w; w = wp; wp = swap;
    }
    PetscCall(VecDestroy(&v));
    PetscCall(VecDestroy(&vp));
    PetscCall(VecDestroy(&w));
    PetscCall(VecDestroy(&wp));
    PetscCall(VecDestroy(&work));
    PetscFunctionReturn(PETSC_SUCCESS);
}
