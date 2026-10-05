"""Physical parameters from the EDRIXS Pu O45 example (GPL-3.0)."""

import edrixs


def build_physical_parameters() -> dict:
    """Build the exact parameters from Pu_O45/run_rixs_fsolver.py."""
    occupancy = 6
    atom = edrixs.get_atom_data(
        "Pu", v_name="5f", v_noccu=occupancy, edge="O45"
    )

    _, slater_i_atomic = [list(values) for values in zip(*atom["slater_i"])]
    _, slater_n_atomic = [list(values) for values in zip(*atom["slater_n"])]

    slater_i = edrixs.rescale(
        slater_i_atomic,
        ([1, 2, 3], [0.77, 0.77, 0.77]),
    )
    slater_i[0] = edrixs.get_F0(
        "f", slater_i[1], slater_i[2], slater_i[3]
    )

    # Preserve the reference example exactly. It supplies eight indices but
    # seven factors, and edrixs.rescale uses zip(), so index 9 (G5_fd) remains
    # unscaled while indices 5, 6, 7, and 8 are scaled by 0.6.
    slater_n = edrixs.rescale(
        slater_n_atomic,
        (
            [1, 2, 3, 5, 6, 7, 8, 9],
            [0.77, 0.77, 0.77, 0.6, 0.6, 0.6, 0.6],
        ),
    )
    slater_n[0] = edrixs.get_F0(
        "f", slater_n[1], slater_n[2], slater_n[3]
    )
    slater_n[4] = edrixs.get_F0(
        "fd", slater_n[7], slater_n[8], slater_n[9]
    )

    zeta_f_i = atom["v_soc_i"][0] * 0.9
    zeta_f_n = atom["v_soc_n"][0] * 0.9
    zeta_d_n = (atom["edge_ene"][0] - atom["edge_ene"][1]) / 2.5
    om_shift = (
        2.0 * atom["edge_ene"][0] + 3.0 * atom["edge_ene"][1]
    ) / 5.0

    return {
        "occupancy": occupancy,
        "slater": (slater_i, slater_n),
        "v_soc": (zeta_f_i, zeta_f_n),
        "c_soc": zeta_d_n,
        "om_shift": om_shift,
    }
