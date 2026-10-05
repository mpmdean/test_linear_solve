"""Experimental EDRIXS short recurrence using PETSc distributed Mat/Vec kernels.

Ported from the GPLv3 src/linsys.f90 bundled in this repository. This is NOT
PETSc's Hermitian KSPMINRES. It requires A = H + i*gamma*I with H Hermitian,
no preconditioner and a zero initial guess. The benchmark checks this structure
before selecting it. The recurrence's residual estimate is always checked
against the actual residual before returning convergence.
"""

import cmath

from petsc4py import PETSc


class CompiledShiftedPMINRES:
    """One Python call per solve; all iterations run in C using PETSc kernels."""

    def __init__(self, library):
        import ctypes as ct
        import numpy as np
        self.ct = ct
        self.integer = ct.c_int32 if np.dtype(PETSc.IntType).itemsize == 4 else ct.c_int64
        self.library = ct.CDLL(str(library))
        self.function = self.library.shifted_pminres
        self.function.argtypes = [ct.c_void_p]*3 + [ct.c_double, self.integer,
                                                  ct.POINTER(self.integer), ct.POINTER(self.integer)]
        self.function.restype = ct.c_int

    def solve(self, ksp, b, x):
        if ksp.getPC().getType() != 'none' or ksp.getInitialGuessNonzero():
            raise ValueError('ShiftedPMINRES needs PCNONE and a zero initial guess')
        matrix, _ = ksp.getOperators()
        rtol, atol, _, maxits = ksp.getTolerances()
        if rtol != 0:
            raise ValueError('Compiled benchmark recurrence requires an absolute tolerance')
        iterations, reason = self.integer(), self.integer()
        error = self.function(matrix.handle, b.handle, x.handle, atol, maxits,
                              self.ct.byref(iterations), self.ct.byref(reason))
        if error:
            raise PETSc.Error(error)
        residual = b.duplicate()
        try:
            matrix.mult(x, residual)
            residual.axpy(-1, b)
            ksp.setResidualNorm(residual.norm())
        finally:
            residual.destroy()
        ksp.setIterationNumber(iterations.value)
        ksp.setConvergedReason(reason.value)


class ShiftedPMINRES:
    def solve(self, ksp, b, x):
        if ksp.getPC().getType() != 'none' or ksp.getInitialGuessNonzero():
            raise ValueError('ShiftedPMINRES needs PCNONE and a zero initial guess')
        matrix, _ = ksp.getOperators()
        rtol, atol, _, maxits = ksp.getTolerances()
        vectors = [b.duplicate() for _ in range(5)]
        v, vprev, w, wprev, work = vectors
        try:
            for vec in vectors:
                vec.set(0)
            x.set(0)
            b.copy(v)
            beta = b.norm()
            threshold = max(atol, rtol * beta)
            eta = estimate = beta
            gamma0 = gamma1 = 1.0 + 0j
            sigma0 = sigma1 = 0j
            reason = PETSc.KSP.ConvergedReason.DIVERGED_MAX_IT
            iterations = 0
            if beta <= threshold:
                reason = PETSc.KSP.ConvergedReason.CONVERGED_ATOL
            else:
                for iterations in range(1, maxits + 1):
                    v.scale(1 / beta)
                    matrix.mult(v, work)
                    # PETSc conjugates its SECOND dot-product argument.
                    alpha = work.dot(v)
                    vprev.scale(-beta)
                    vprev.axpy(1, work)
                    vprev.axpy(-alpha, v)
                    beta_next = vprev.norm()
                    delta = gamma0 * alpha - gamma1 * sigma0 * beta
                    rho1 = cmath.sqrt(delta * delta + beta_next * beta_next)
                    if abs(rho1) == 0:
                        reason = PETSc.KSP.ConvergedReason.DIVERGED_BREAKDOWN
                        break
                    rho2 = sigma0 * alpha + gamma1 * gamma0 * beta
                    rho3 = sigma1 * beta
                    gamma1, sigma1 = gamma0, sigma0
                    gamma0, sigma0 = delta / rho1, beta_next / rho1
                    wprev.scale(-rho3)
                    wprev.axpy(1, v)
                    wprev.axpy(-rho2, w)
                    wprev.scale(1 / rho1)
                    x.axpy(gamma0 * eta, wprev)
                    estimate *= abs(sigma0)
                    eta *= -sigma0
                    if estimate <= threshold or beta_next == 0:
                        matrix.mult(x, work)
                        work.axpy(-1, b)
                        if work.norm() <= threshold:
                            reason = PETSc.KSP.ConvergedReason.CONVERGED_ATOL
                            break
                    if beta_next == 0:
                        reason = PETSc.KSP.ConvergedReason.DIVERGED_BREAKDOWN
                        break
                    beta = beta_next
                    v, vprev = vprev, v
                    w, wprev = wprev, w
            matrix.mult(x, work)
            work.axpy(-1, b)
            ksp.setIterationNumber(iterations)
            ksp.setResidualNorm(work.norm())
            ksp.setConvergedReason(reason)
        finally:
            for vec in vectors:
                vec.destroy()
