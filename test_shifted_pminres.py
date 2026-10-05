"""Numerical checks for the experimental PETSc recurrence (run with unittest)."""

import unittest
import os

import numpy as np
from petsc4py import PETSc

from petsc_shifted_pminres import ShiftedPMINRES, CompiledShiftedPMINRES


class ShiftedPMINRESTest(unittest.TestCase):
    def context(self):
        return ShiftedPMINRES()

    def solve_system(self, coefficients, rhs, maxits=200):
        n = len(rhs)
        matrix = PETSc.Mat().createDense([n, n], comm=PETSc.COMM_SELF)
        matrix.setUp()
        matrix.setValues(range(n), range(n), coefficients)
        matrix.assemble()
        b, x = matrix.createVecs()
        b.getArray()[:] = rhs
        ksp = PETSc.KSP().createPython(self.context(), comm=PETSc.COMM_SELF)
        ksp.getPC().setType('none')
        ksp.setOperators(matrix)
        ksp.setTolerances(rtol=0, atol=1e-10, max_it=maxits)
        try:
            ksp.solve(b, x)
            return (x.getArray().copy(), ksp.getConvergedReason(),
                    ksp.getIterationNumber(), ksp.getResidualNorm())
        finally:
            for obj in (ksp, matrix, b, x):
                obj.destroy()

    def test_complex_hermitian_shift(self):
        # Complex off-diagonal entries catch accidental conjugation reversal.
        rng = np.random.default_rng(581)
        raw = rng.normal(size=(12, 12)) + 1j*rng.normal(size=(12, 12))
        matrix = (raw + raw.conj().T)/2 + .2j*np.eye(12)
        rhs = rng.normal(size=12) + 1j*rng.normal(size=12)
        solution, reason, _, residual = self.solve_system(matrix, rhs)
        self.assertGreater(reason, 0)
        self.assertLessEqual(np.linalg.norm(matrix @ solution - rhs), 1e-10)
        self.assertLessEqual(residual, 1e-10)
        np.testing.assert_allclose(solution, np.linalg.solve(matrix, rhs), atol=1e-9)

    def test_zero_rhs(self):
        solution, reason, iterations, residual = self.solve_system(np.eye(3)*(.5+.2j), np.zeros(3))
        self.assertGreater(reason, 0)
        self.assertEqual(iterations, 0)
        self.assertEqual(residual, 0)
        np.testing.assert_array_equal(solution, 0)

    def test_single_eigenvalue(self):
        matrix = np.eye(3)*(2+.2j)
        rhs = np.array([1., 0., 0.])
        solution, reason, iterations, _ = self.solve_system(matrix, rhs)
        self.assertGreater(reason, 0)
        self.assertEqual(iterations, 1)
        np.testing.assert_allclose(matrix @ solution, rhs, atol=1e-10)

    def test_iteration_limit_is_failure(self):
        matrix = np.diag(np.arange(1, 6) + .2j)
        _, reason, iterations, residual = self.solve_system(matrix, np.ones(5), maxits=1)
        self.assertEqual(reason, PETSc.KSP.ConvergedReason.DIVERGED_MAX_IT)
        self.assertEqual(iterations, 1)
        self.assertGreater(residual, 1e-10)


@unittest.skipUnless(os.environ.get('PMINRES_TEST_LIBRARY'), 'set PMINRES_TEST_LIBRARY to the compiled library')
class CompiledShiftedPMINRESTest(ShiftedPMINRESTest):
    def context(self):
        return CompiledShiftedPMINRES(os.environ['PMINRES_TEST_LIBRARY'])


if __name__ == '__main__':
    unittest.main()
