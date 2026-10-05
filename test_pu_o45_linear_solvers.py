#!/usr/bin/env python3
"""Compare actual Fortran PMINRES with PETSc solvers on one Pu O45 system.

Run using the PETSc+Numba environment's Python. Defaults: eight ranks, grid
index 4 of the ten Pu O45 incident energies, three solves per method, alternating
order. Model/Numba setup, CSR conversion and independent residual checks are
outside the solve timers. Production sources are compiled unchanged.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import shutil
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

# Each variant still uses the same absolute tolerance and independent residual
# check. Right preconditioning keeps GMRES's monitored norm unpreconditioned.
PETSC_METHODS = {
    'petsc_pminres': {'ksp_type': 'python', 'pc_type': 'none'},
    'petsc_pminres_c': {'ksp_type': 'python', 'pc_type': 'none'},
    'gmres': {'ksp_type': 'gmres', 'ksp_gmres_restart': 200, 'pc_type': 'none'},
    'gmres400': {'ksp_type': 'gmres', 'ksp_gmres_restart': 400, 'pc_type': 'none'},
    'gmres800': {'ksp_type': 'gmres', 'ksp_gmres_restart': 800, 'pc_type': 'none'},
    'cg_symmetric': {'ksp_type': 'cg', 'ksp_cg_type': 'symmetric', 'pc_type': 'none'},
    'gmres_jacobi': {'ksp_type': 'gmres', 'ksp_gmres_restart': 200, 'pc_type': 'jacobi'},
    'gmres_bjacobi': {'ksp_type': 'gmres', 'ksp_gmres_restart': 200,
                      'pc_type': 'bjacobi', 'sub_ksp_type': 'preonly',
                      'sub_pc_type': 'ilu', 'sub_pc_factor_levels': 0},
    'gmres_asm': {'ksp_type': 'gmres', 'ksp_gmres_restart': 200,
                  'pc_type': 'asm', 'pc_asm_overlap': 1, 'sub_ksp_type': 'preonly',
                  'sub_pc_type': 'ilu', 'sub_pc_factor_levels': 0},
    **{name: {'ksp_type': name, 'pc_type': 'none'}
       for name in ('bcgs', 'bcgsl', 'tfqmr', 'bicg')},
}


def worker(args):
    import ctypes as ct
    import numpy as np
    from mpi4py import MPI
    from petsc4py import PETSc
    import edrixs
    from edrixs.petsc_backend.petsc_backend import (
        _rixs_polarization_vectors, _apply_trans_combination,
    )
    from pu_o45_parameters import build_physical_parameters

    comm = MPI.COMM_WORLD
    if args.profile:
        PETSc.Log.begin()
    assert comm.size == args.ranks == PETSc.COMM_WORLD.getSize()
    assert np.dtype(PETSc.ScalarType) == np.dtype(np.complex128)
    params = build_physical_parameters()
    shift_om = params['om_shift']
    *problem, shift = edrixs.model_1v1c(
        ('f', 'd'), shell_level=(0, -shift_om), v_soc=params['v_soc'],
        c_soc=params['c_soc'], v_noccu=params['occupancy'],
        slater=params['slater'], sparse_U=True,
    )
    hi, hn, transitions = edrixs.get_ops(*problem, backend='petsc', use_numba=True,
                                       backend_kws={'comm': PETSc.COMM_WORLD})
    energies, states = edrixs.ed(
        hi, shift=shift, num_evals=30, backend='petsc',
        backend_kws={'ncv': 60, 'eigval_tol': 1e-8, 'maxiter': 500},
    )
    incoming, _ = _rixs_polarization_vectors(
        3, np.pi/4, np.pi/4, 0, 'linear', 0., 'linear', 0., np.eye(3))
    rhs = _apply_trans_combination(transitions, incoming, states[0])
    grid = np.linspace(shift_om-5, shift_om+20, 10)
    omega, omega_ref, energy_ref = grid[args.energy_index], grid[0], energies[0]
    # Exactly the centering used in rixs_petsc: avoid cancellation of large energies.
    matrix = hn.copy()
    matrix.shift(-energy_ref)
    matrix.shift(-omega_ref)
    matrix.scale(-1)
    matrix.shift(complex(omega-omega_ref, .2))
    assert matrix.getSize() == (34320, 34320)
    rows, cols, values = matrix.getValuesCSR()
    rows = np.asarray(rows, dtype=np.int32)
    cols = np.asarray(cols, dtype=np.int32)
    values = np.asarray(values, dtype=np.complex128)
    offsets = np.asarray(matrix.getOwnershipRanges(), dtype=np.int32)
    n = rhs.getLocalSize()
    lib = ct.CDLL(str(args.output_dir/'build/libpminres_test.dylib'))
    int_array = np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
    complex_array = np.ctypeslib.ndpointer(dtype=np.complex128, flags='C_CONTIGUOUS')
    lib.setup.argtypes = [ct.c_int, ct.c_int, int_array, int_array, complex_array,
                         int_array, ct.c_int, ct.c_double, ct.c_int]
    lib.setup.restype = lib.solve.restype = lib.multiply.restype = lib.cleanup.restype = None
    lib.solve.argtypes = lib.multiply.argtypes = [complex_array, complex_array]
    lib.cleanup.argtypes = []
    lib.setup(n, len(values), rows, cols, values, offsets, comm.py2f(), 1e-8, 5000)
    # Verify global column indexing, CSR blocks, communication and complex ABI.
    probe = rhs.duplicate()
    start, end = probe.getOwnershipRange()
    index = np.arange(start, end)
    probe.getArray()[:] = np.sin(index*.13) + 1j*np.cos(index*.17)
    reference = rhs.duplicate()
    matrix.mult(probe, reference)
    probe_array = np.array(probe.getArray(readonly=True), copy=True)
    converted = np.empty(n, dtype=np.complex128)
    lib.multiply(probe_array, converted)
    conversion_error = np.sqrt(comm.allreduce(float(np.linalg.norm(
        converted-reference.getArray(readonly=True))**2), op=MPI.SUM)) / reference.norm()
    if conversion_error > 1e-13:
        raise RuntimeError(f'Fortran/PETSc matrix-action mismatch: {conversion_error}')
    b = np.array(rhs.getArray(readonly=True), copy=True)
    rhs_norm = rhs.norm()
    residual = rhs.duplicate()
    solution = rhs.duplicate()
    previous = rhs.duplicate()
    difference_vec = rhs.duplicate()
    # Check the structure instead of assuming that a complex shift is Hermitian.
    centered = matrix.copy()
    centered.shift(-.2j)
    adjoint = centered.copy()
    adjoint.hermitianTranspose()
    adjoint.axpy(-1, centered)
    transpose = matrix.copy()
    transpose.transpose()
    transpose.axpy(-1, matrix)
    structure = {'shift_removed_hermitian_defect': adjoint.norm()/centered.norm(),
                 'transpose_symmetry_defect': transpose.norm()/matrix.norm()}
    adjoint.destroy()
    transpose.destroy()
    centered.destroy()
    if comm.rank == 0:
        print('MATRIX_STRUCTURE '+json.dumps(structure), flush=True)
    matvec_times = {}
    if args.matvec_repeats:
        for method in ('fortran', 'petsc'):
            comm.Barrier()
            begin = MPI.Wtime()
            for _ in range(args.matvec_repeats):
                if method == 'fortran':
                    lib.multiply(probe_array, converted)
                else:
                    matrix.mult(probe, reference)
            matvec_times[method] = comm.allreduce(MPI.Wtime()-begin, op=MPI.MAX) / args.matvec_repeats
    records = []
    for repeat in range(1, args.repeats+1):
        order = args.methods if repeat % 2 else list(reversed(args.methods))
        for position, method in enumerate(order):
            if comm.rank == 0:
                print(f'TRIAL {repeat} {method}', flush=True)
            solution.set(0)
            ksp = None
            if method != 'pminres_csr':
                if method == 'cg_symmetric' and structure['transpose_symmetry_defect'] > 1e-13:
                    raise ValueError('Complex symmetric CG requires A = A^T')
                ksp = PETSc.KSP().create(matrix.getComm())
                prefix = 'bench_'
                ksp.setOptionsPrefix(prefix)
                options = PETSc.Options()
                for key, value in PETSC_METHODS[method].items():
                    options[prefix+key] = value
                ksp.setFromOptions()
                if method in ('petsc_pminres', 'petsc_pminres_c'):
                    from petsc_shifted_pminres import ShiftedPMINRES, CompiledShiftedPMINRES
                    if structure['shift_removed_hermitian_defect'] > 1e-13:
                        raise ValueError('Shifted PMINRES requires A - 0.2j*I to be Hermitian')
                    context = (ShiftedPMINRES() if method == 'petsc_pminres' else
                               CompiledShiftedPMINRES(args.output_dir/'build/libpetsc_pminres.dylib'))
                    ksp.setPythonContext(context)
                if ksp.getType() not in ('bicg', 'cg', 'python'):
                    ksp.setPCSide(PETSc.PC.Side.RIGHT)
                ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
                ksp.setTolerances(rtol=0, atol=1e-8, max_it=5000)
                ksp.setOperators(matrix)
            stage = PETSc.Log.Stage(f'{method}_{repeat}') if args.profile else None
            output = solution.getArray()
            comm.Barrier()
            if stage is not None:
                stage.push()
            begin = MPI.Wtime()
            if ksp is None:
                lib.solve(b, output)
            else:
                ksp.solve(rhs, solution)
            elapsed = comm.allreduce(MPI.Wtime()-begin, op=MPI.MAX)
            if stage is not None:
                stage.pop()
            del output
            matrix.mult(solution, residual)
            residual.axpy(-1, rhs)
            true_residual = residual.norm()
            difference = None
            if position == 0:
                solution.copy(previous)
            else:
                previous.copy(difference_vec)
                difference_vec.axpy(-1, solution)
                difference = difference_vec.norm()/solution.norm()
            record = {'repeat': repeat, 'method': method, 'wall_s': elapsed,
                      'iterations': ksp.getIterationNumber() if ksp is not None else None,
                      'reason': ksp.getConvergedReason() if ksp is not None else None,
                      'true_absolute_residual': true_residual,
                      'true_relative_residual': true_residual/rhs_norm,
                      'passes_atol': bool(true_residual <= 1e-8),
                      'solution_comparison_method': order[0] if position else None,
                      'relative_solution_difference': difference}
            records.append(record)
            if comm.rank == 0:
                print('RESULT '+json.dumps(record), flush=True)
            if ksp is not None:
                ksp.destroy()
                # Subdomain KSPs read their options during lazy PC setup.
                for key in PETSC_METHODS[method]:
                    del options[prefix+key]
    if args.profile:
        viewer = PETSc.Viewer().createASCII(str(args.output_dir/'petsc-profile.txt'), comm=matrix.getComm())
        PETSc.Log.view(viewer)
        viewer.destroy()
    if comm.rank == 0:
        (args.output_dir/'results.json').write_text(json.dumps({
            'ranks': comm.size, 'dimension': matrix.getSize()[0],
            'energy_index': args.energy_index, 'omega_eV': float(omega), 'gamma_eV': .2,
            'ground_energy_eV': float(energy_ref), 'rhs_norm': rhs_norm,
            'matrix_action_relative_error': conversion_error,
            'atol': 1e-8, 'maxiter_both': 5000, 'gmres_restart': 200,
            'petsc_version': PETSc.Sys.getVersion(), 'mpi': MPI.Get_library_version(),
            'methods': {m: PETSC_METHODS.get(m) for m in args.methods},
            'matrix_structure': structure, 'matvec_seconds': matvec_times,
            'profile_enabled': args.profile, 'matvec_repeats': args.matvec_repeats,
            'runs': records,
        }, indent=2)+'\n')
    lib.cleanup()
    for obj in [probe, reference, residual, solution, previous, difference_vec, matrix, rhs,
                *states, hi, hn, *transitions]:
        obj.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--ranks', type=int, default=8)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--energy-index', type=int, default=4, choices=range(10))
    parser.add_argument('--methods', nargs='+', default=['pminres_csr', 'gmres'],
                        choices=['pminres_csr', *PETSC_METHODS])
    parser.add_argument('--profile', action='store_true', help='write PETSc event timings by solve')
    parser.add_argument('--matvec-repeats', type=int, default=0, help='time isolated matrix products')
    parser.add_argument('--edrixs-root', type=Path, help='path to a compatible EDRIXS source checkout')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    if args.ranks < 1 or args.repeats < 1 or args.matvec_repeats < 0:
        parser.error('ranks and repeats must be positive; matvec-repeats must be nonnegative')
    if len(set(args.methods)) != len(args.methods):
        parser.error('methods must be distinct')
    if args.worker:
        try:
            worker(args)
        except BaseException:
            import traceback
            traceback.print_exc()
            from mpi4py import MPI
            MPI.COMM_WORLD.Abort(1)
        return
    args.output_dir.mkdir(parents=True, exist_ok=False)
    build = args.output_dir/'build'
    build.mkdir()
    prefix = Path(sys.executable).parent
    env = os.environ.copy()
    env['PATH'] = str(prefix)+os.pathsep+env.get('PATH','')
    if args.edrixs_root is not None:
        edrixs_root = args.edrixs_root.resolve()
        if not (edrixs_root/'edrixs/petsc_backend/petsc_backend.py').is_file():
            parser.error('--edrixs-root must contain edrixs/petsc_backend/petsc_backend.py')
        env['PYTHONPATH'] = str(edrixs_root)+os.pathsep+env.get('PYTHONPATH','')
    def executable(name):
        local = prefix/name
        found = str(local) if local.exists() else shutil.which(name, path=env['PATH'])
        if not found:
            raise RuntimeError(f'{name} is required in the Python environment or PATH')
        return found
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','NUMBA_NUM_THREADS'):
        env[name] = '1'
    env['MPLCONFIGDIR'] = str(args.output_dir/'mpl_cache')
    sources = [ROOT/'src'/name for name in ('m_constants.f90','m_control.f90','m_types.f90',
                                           'spmv.f90','linsys.f90')]
    sources.append(Path(__file__).with_name('pminres_test_bridge.f90'))
    compile_cmd = [executable('mpifort'), '-O3', '-fPIC', '-ffree-line-length-none',
                   '-dynamiclib' if sys.platform == 'darwin' else '-shared',
                   *map(str,sources), '-o', str(build/'libpminres_test.dylib')]
    with (args.output_dir/'build.log').open('w') as stream:
        subprocess.run(compile_cmd, cwd=build, env=env, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)
    petsc_compile_cmd = None
    if 'petsc_pminres_c' in args.methods:
        env['PKG_CONFIG_PATH'] = str(prefix.parent/'lib/pkgconfig')+os.pathsep+env.get('PKG_CONFIG_PATH', '')
        flags = subprocess.check_output([executable('pkg-config'), '--cflags', '--libs', 'PETSc'],
                                        env=env, text=True)
        petsc_compile_cmd = [executable('mpicc'), '-O3', '-fPIC',
                             '-dynamiclib' if sys.platform == 'darwin' else '-shared',
                             str(ROOT/'petsc_shifted_pminres.c'), *shlex.split(flags),
                             '-o', str(build/'libpetsc_pminres.dylib')]
        with (args.output_dir/'build.log').open('a') as stream:
            subprocess.run(petsc_compile_cmd, cwd=build, env=env, stdout=stream,
                           stderr=subprocess.STDOUT, check=True)
    command = [executable('mpirun'), '-n', str(args.ranks), sys.executable, '-u',
               str(Path(__file__).resolve()), '--worker', '--output-dir', str(args.output_dir),
               '--ranks', str(args.ranks), '--repeats', str(args.repeats),
               '--energy-index', str(args.energy_index)]
    command += ['--methods', *args.methods, '--matvec-repeats', str(args.matvec_repeats)]
    if args.profile:
        command.append('--profile')
    (args.output_dir/'commands.json').write_text(json.dumps({
        'compile': compile_cmd, 'petsc_compile': petsc_compile_cmd, 'run': command,
        'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [*sources, Path(__file__), ROOT/'pu_o45_parameters.py',
                                    ROOT/'petsc_shifted_pminres.py', ROOT/'petsc_shifted_pminres.c']},
    }, indent=2)+'\n')
    with (args.output_dir/'run.log').open('w') as stream:
        subprocess.run(command, cwd=args.output_dir, env=env, stdout=stream,
                       stderr=subprocess.STDOUT, check=True, timeout=600)
    result = json.loads((args.output_dir/'results.json').read_text())
    # Read the iteration count printed by the unchanged Fortran routine.
    for block in (args.output_dir/'run.log').read_text().split('TRIAL ')[1:]:
        header = block.splitlines()[0].split()
        if header[1] != 'pminres_csr':
            continue
        counts = re.findall(r'PMINRES iteration:\s+(\d+)', block)
        assert counts, 'Missing Fortran iteration diagnostic'
        for record in result['runs']:
            if record['repeat'] == int(header[0]) and record['method'] == 'pminres_csr':
                record['iterations'] = int(counts[-1])
    (args.output_dir/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    with (args.output_dir/'solves.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result['runs'][0]))
        writer.writeheader()
        writer.writerows(result['runs'])
    lines = ['# Pu O45: Fortran PMINRES versus PETSc solvers', '',
             f"One {result['dimension']} × {result['dimension']} intermediate-state system, "
             f"{args.ranks} MPI ranks, one thread/rank; incident grid index {args.energy_index}, "
             f"omega={result['omega_eV']:.9f} eV, broadening 0.2 eV.", '',
             'All methods receive identical CSR coefficients and RHS, use a zero initial guess, '
             'absolute tolerance 1e-8 and a common 5,000-iteration cap. The gmres baseline uses '
             'restart=200 and no preconditioner; variant options are recorded in results.json. '
             'GMRES variants use right preconditioning and an unpreconditioned residual norm. '
             'The actual production Fortran sources are compiled unchanged '
             'with -O3 and called through a test-only C interface. CSR conversion preserves '
             'PETSc row ownership and uses the production Fortran MPI matvec.', '',
             '| Solver | Median solve (s) | Range (s) | Iterations | Worst true residual | Passes |',
             '| :--- | ---: | ---: | ---: | ---: | ---: |']
    medians = {}
    for method in args.methods:
        runs = [r for r in result['runs'] if r['method'] == method]
        times = [r['wall_s'] for r in runs]
        medians[method] = statistics.median(times)
        lines.append(f"| {method} | {medians[method]:.6f} | {min(times):.6f}–{max(times):.6f} | "
                     f"{min(r['iterations'] for r in runs)}–{max(r['iterations'] for r in runs)} | "
                     f"{max(r['true_absolute_residual'] for r in runs):.3e} | "
                     f"{sum(r['passes_atol'] and (r['reason'] is None or r['reason'] > 0) for r in runs)}/{len(runs)} |")
    if 'gmres' in medians and 'pminres_csr' in medians:
        ratio = medians['gmres']/medians['pminres_csr']
        lines += ['', f'GMRES/PMINRES median time ratio: **{ratio:.3f}**.']
    lines += ['', 'Interpret speed only together with the independently checked residuals.', '',
              f"Fortran/PETSc matrix-action relative difference: {result['matrix_action_relative_error']:.3e}.", '',
              f'{args.repeats} measurements per solver, reversing order on even repeats. Model setup, JIT, matrix '
              'conversion, and independent residual checks are outside timing. Maximum elapsed '
              'time across ranks is reported. Fresh KSP objects include lazy solver setup in '
              'KSPSolve; Fortran includes allocation, its diagnostic output and its own final '
              'residual check. No untimed solver warm-up. Optional matrix-product measurements '
              'run before the solvers; profiling adds instrumentation overhead. This is one incident energy, not a '
              'full-workload or general solver ranking. Native matvec/communication implementations '
              'differ, so this measures implementations rather than recurrence cost alone.', '',
              'Per-solve residuals and solution differences: [solves.csv](solves.csv). '
              'Production source hashes and commands: [commands.json](commands.json).']
    differences = [r['relative_solution_difference'] for r in result['runs']
                   if r['relative_solution_difference'] is not None]
    if differences:
        lines += ['', f'Maximum relative solution difference from the first method in each repeat: {max(differences):.3e}.']
    if result['matvec_seconds']:
        lines += ['', 'Isolated matrix products (seconds/product, maximum rank elapsed):', '',
                  *[f'- {name}: {seconds:.6g}' for name, seconds in result['matvec_seconds'].items()]]
    (args.output_dir/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))
    if not all(r['passes_atol'] and (r['reason'] is None or r['reason'] > 0) for r in result['runs']):
        raise SystemExit('One or more solves failed the independent residual/convergence check; see report.')


if __name__ == '__main__':
    main()
