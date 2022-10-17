from QCDSFGamma import *
import numpy as np
from pathlib import Path
import pickle
from formatting import err_brackets


def F3_fnc(Gamm, oper, p, pp):
    """Calculates the value of the F_3 function.
    Gamm is the projection matrix
    oper is the operator
    p and pp are the initial and final momentum respectively
    """
    Ep = p.Energy()
    Epp = pp.Energy()
    pdg = p.px * gamma1 + p.py * gamma2 + p.pz * gamma3
    ppdg = pp.px * gamma1 + pp.py * gamma2 + pp.pz * gamma3
    term2 = -1.0j * pdg + gamma4 * Ep + gammaI * (p.m)
    term1 = -1.0j * ppdg + gamma4 * Epp + gammaI * (pp.m)
    intG = 0.25 * np.matmul(np.matmul(Gamm, term1), np.matmul(oper, term2))
    return intG.trace()


class pmom:
    def __init__(self, px, py, pz, m):
        self.px = px
        self.py = py
        self.pz = pz
        self.m = m

    def Energy(self):
        return np.sqrt(self.px**2 + self.py**2 + self.pz**2 + self.m**2)

    def vector(self):
        return np.array([self.px, self.py, self.pz, 1.0j * self.Energy()])


def main():
    """Calculate the trace of the spinors to get the value of the factor which multiplies each form factor for every combination of polarization, operator and real/imaginary. Use these factors to then solve for the three form factors of the transition matrix element"""

    resultsdir = Path.home() / Path(
        "Dropbox/PhD/analysis_code/transition_3pt_function/"
    )
    datadir = resultsdir / Path("data/")

    mN = 0.4382953145893229
    mS = 0.4751146506255126

    L = 32
    T = 64
    a = 0.074
    nboot = 500

    # What else depends on the number of momenta listed here?
    # momenta_pickle = ["p+0+0+0", "p+1+0+0", "p+1+1+0"]
    # momenta = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]]) * 2 * np.pi / L
    momenta_pickle = ["p+1+0+0", "p+1+1+0"]
    polarizations_pickle = ["UNPOL", "POL"]
    operators_pickle = [
        "g0",
        "g1",
        "g2",
        "g3",
    ]
    momenta = np.array([[1, 0, 0], [1, 1, 0]]) * 2 * np.pi / L
    polarizations = [GammaUP, GammaZ]
    operators_F1 = [
        gamma1,
        gamma2,
        gamma3,
        gamma4,
    ]
    operators_F2 = [
        [sigma12, sigma13, sigma14],
        [sigma21, sigma23, sigma24],
        [sigma31, sigma32, sigma34],
        [sigma41, sigma42, sigma43],
    ]
    operators_F3 = [
        gammaI,
        gammaI,
        gammaI,
        gammaI,
    ]

    form_factors_zeromom(datadir)

    FF_factors_all, matrix_elements_all, form_factor_values = form_factors_n2sig(
        mN,
        mS,
        nboot,
        datadir,
        momenta_pickle,
        operators_pickle,
        polarizations_pickle,
        momenta,
        polarizations,
        operators_F1,
        operators_F2,
        operators_F3,
    )

    FF_factors_all, matrix_elements_all, form_factor_values = form_factors_sig2n(
        mN,
        mS,
        nboot,
        datadir,
        momenta_pickle,
        operators_pickle,
        polarizations_pickle,
        momenta,
        polarizations,
        operators_F1,
        operators_F2,
        operators_F3,
    )
    return


def form_factors_zeromom(datadir):
    """Construct the prefactors for the neutron to sigma transition
    Saves the data in a pickle file and returns
    an array of all the prefactors,
    an array of the matrix element values,
    an array of the values of the three form factors for each momentum, and all bootstraps
    """

    datafile_ratio = datadir / Path(
        f"p+0+0+0_g3_UNPOL_nr_real_double_3pt_ratio_fit.pkl"
    )
    with open(datafile_ratio, "rb") as file_in:
        fit_data = pickle.load(file_in)
    matrix_element_zeromom = fit_data[0][:, 0]

    datafile = datadir / Path(f"matrix_element_3pt_fit_zeromom.pkl")
    with open(datafile, "wb") as file_out:
        pickle.dump(matrix_element_zeromom, file_out)

    return matrix_element_zeromom


def form_factors_n2sig(
    mN,
    mS,
    nboot,
    datadir,
    momenta_pickle,
    operators_pickle,
    polarizations_pickle,
    momenta,
    polarizations,
    operators_F1,
    operators_F2,
    operators_F3,
):
    """Construct the prefactors for the neutron to sigma transition
    Saves the data in a pickle file and returns
    an array of all the prefactors,
    an array of the matrix element values,
    an array of the values of the three form factors for each momentum, and all bootstraps
    """

    pp = pmom(0, 0, 0, mS)  # sigma

    num_of_ff = 3
    FF_factors_all = np.zeros(
        (len(momenta), len(operators_F1), len(polarizations), num_of_ff),
        dtype="complex",
    )

    for imom, mom in enumerate(momenta):
        print(f"\n\n\n{mom=}")
        p = pmom(*mom, mN)
        print(f"Neutron energy = {p.Energy()}")
        prefactor = 1 / (mS * np.sqrt(2 * p.Energy() * (p.Energy() + mN)))
        factors_F1 = [
            prefactor,
            prefactor,
            prefactor,
            prefactor,
        ]
        factors_F2 = [
            [
                prefactor * p.vector()[1] / (mN + mS),
                prefactor * p.vector()[2] / (mN + mS),
                prefactor * p.vector()[3] / (mN + mS),
            ],
            [
                prefactor * p.vector()[0] / (mN + mS),
                prefactor * p.vector()[2] / (mN + mS),
                prefactor * p.vector()[3] / (mN + mS),
            ],
            [
                prefactor * p.vector()[0] / (mN + mS),
                prefactor * p.vector()[1] / (mN + mS),
                prefactor * p.vector()[3] / (mN + mS),
            ],
            [
                prefactor * p.vector()[0] / (mN + mS),
                prefactor * p.vector()[1] / (mN + mS),
                prefactor * p.vector()[2] / (mN + mS),
            ],
        ]

        factors_F3 = [
            prefactor * 1.0j * p.vector()[0] / (mN + mS),
            prefactor * 1.0j * p.vector()[1] / (mN + mS),
            prefactor * 1.0j * p.vector()[2] / (mN + mS),
            prefactor * 1.0j * p.vector()[3] / (mN + mS),
        ]

        for iop, op in enumerate(operators_F1):
            # print(f"\n{iop=}")
            for ipol, pol in enumerate(polarizations):
                # print(f"\n{ipol=}")
                # Calculate the value of the F_3 function for each operator in front of form factor F_i
                # Then multiply this by the appropriate kinematic factors
                F3_F1 = F3_fnc(pol, operators_F1[iop], p, pp)
                F3_F2 = np.array([F3_fnc(pol, jop, p, pp) for jop in operators_F2[iop]])
                F3_F3 = F3_fnc(pol, operators_F3[iop], p, pp)

                F1_factor = F3_F1 * factors_F1[iop]
                F2_factor = np.dot(F3_F2, factors_F2[iop])
                F3_factor = F3_F3 * factors_F3[iop]

                FF_factors = np.array([F1_factor, F2_factor, F3_factor])
                FF_factors_all[imom, iop, ipol] = FF_factors

    # Read the values of the matrix element from fits to the three-point functions
    transition = "n2sig"
    matrix_elements_all = read_fit_data(
        nboot,
        datadir,
        operators_pickle,
        polarizations_pickle,
        momenta_pickle,
        transition,
    )
    # print(f"{np.shape(matrix_elements_all)=}")
    # # print(f"{matrix_elements_all=}")

    # print(
    #     np.real(FF_factors_all[0, 3, 0, :]),
    #     np.average(matrix_elements_all[0, 3, 0, 0, :]),
    # )
    # print(
    #     np.imag(FF_factors_all[0, 3, 0, :]),
    #     np.average(matrix_elements_all[0, 3, 0, 1, :]),
    # )

    # The shapes of the matrices are:
    # np.shape(FF_factors_all)=(3, 4, 2, 3)
    #    = (momenta, operators, polarizations, num_of_ff) complex numbers
    # np.shape(matrix_elements_all)=(3, 4, 2, 2, 500)
    #    = (momenta, operators, polarizations, real/imag, nboot)

    print("solving for the form factors")
    form_factor_values = np.zeros((len(momenta), num_of_ff, nboot))
    for solve_mom, mom_val in enumerate(momenta):
        # print(f"\n\n{mom_val}")
        # This only does the real matrix elements, should extend to include the imaginary
        FF_factors_real = np.real(FF_factors_all[solve_mom, :, :, :]).reshape(
            4 * 2, num_of_ff
        )
        A_inv = np.linalg.pinv(FF_factors_real)

        for iboot in range(nboot):
            # Reshape the matrix element values and solve for the form factors
            matrix_elements_real = matrix_elements_all[
                solve_mom, :, :, 0, iboot
            ].reshape(4 * 2)
            x = np.matmul(A_inv, matrix_elements_real)
            form_factor_values[solve_mom, :, iboot] = x

    datafile = datadir / Path(f"form_factors_3pt_fit_n2sig.pkl")
    with open(datafile, "wb") as file_out:
        pickle.dump(form_factor_values, file_out)

    # print(f"{np.shape(ff_avg)=}")
    # print(f"{np.shape(mom0_factors)=}")
    # print(f"{ff_avg=}")
    # print(f"{ff_std=}")

    gamma4_factors = np.real(FF_factors_all[:, 3, 0, :])

    matrix_element_gamma4 = np.einsum("ij,ijk->ijk", gamma4_factors, form_factor_values)
    test2_ = np.einsum("ijk->ik", matrix_element_gamma4)
    print(f"{np.average(test2_, axis=1)=}")

    # test = gamma4_factors * ff_avg
    # test2 = np.sum(test * ff_avg, axis=1)
    # print(f"{test=}")
    # print(f"{test2=}")

    datafile = datadir / Path(f"matrix_element_3pt_fit_n2sig.pkl")
    with open(datafile, "wb") as file_out:
        pickle.dump(test2_, file_out)

    ff_avg = np.average(form_factor_values, axis=2)
    ff_std = np.std(form_factor_values, axis=2)
    for i, ff in enumerate(ff_avg):
        print([err_brackets(ff[j], ff_std[i, j]) for j in range(len(ff))])

    return FF_factors_all, matrix_elements_all, form_factor_values


def form_factors_sig2n(
    mN,
    mS,
    nboot,
    datadir,
    momenta_pickle,
    operators_pickle,
    polarizations_pickle,
    momenta,
    polarizations,
    operators_F1,
    operators_F2,
    operators_F3,
):
    """Construct the prefactors for the sigma to neutron transition
    Saves the data in a pickle file and returns
    an array of all the prefactors,
    an array of the matrix element values,
    an array of the values of the three form factors for each momentum, and all bootstraps
    """

    pp = pmom(0, 0, 0, mN)  # Neutron at the sink, always at rest

    num_of_ff = 3
    FF_factors_all = np.zeros(
        (len(momenta), len(operators_F1), len(polarizations), num_of_ff),
        dtype="complex",
    )

    for imom, mom in enumerate(momenta):
        print(f"\n\n\n{mom=}")
        p = pmom(*mom, mS)  # Sigma at the source, varying momentum
        print(f"Sigma energy = {p.Energy()}")
        prefactor = 1 / (mN * np.sqrt(2 * p.Energy() * (p.Energy() + mS)))
        factors_F1 = [
            prefactor,
            prefactor,
            prefactor,
            prefactor,
        ]
        factors_F2 = [
            [
                prefactor * p.vector()[1] / (mN + mS),
                prefactor * p.vector()[2] / (mN + mS),
                prefactor * p.vector()[3] / (mN + mS),
            ],
            [
                prefactor * p.vector()[0] / (mN + mS),
                prefactor * p.vector()[2] / (mN + mS),
                prefactor * p.vector()[3] / (mN + mS),
            ],
            [
                prefactor * p.vector()[0] / (mN + mS),
                prefactor * p.vector()[1] / (mN + mS),
                prefactor * p.vector()[3] / (mN + mS),
            ],
            [
                prefactor * p.vector()[0] / (mN + mS),
                prefactor * p.vector()[1] / (mN + mS),
                prefactor * p.vector()[2] / (mN + mS),
            ],
        ]

        factors_F3 = [
            prefactor * 1.0j * p.vector()[0] / (mN + mS),
            prefactor * 1.0j * p.vector()[1] / (mN + mS),
            prefactor * 1.0j * p.vector()[2] / (mN + mS),
            prefactor * 1.0j * p.vector()[3] / (mN + mS),
        ]

        for iop, op in enumerate(operators_F1):
            # print(f"\n{iop=}")
            for ipol, pol in enumerate(polarizations):
                # print(f"\n{ipol=}")
                # Calculate the value of the F_3 function for each operator in front of form factor F_i
                # Then multiply this by the appropriate kinematic factors
                F3_F1 = F3_fnc(pol, operators_F1[iop], p, pp)
                F3_F2 = np.array([F3_fnc(pol, jop, p, pp) for jop in operators_F2[iop]])
                F3_F3 = F3_fnc(pol, operators_F3[iop], p, pp)

                F1_factor = F3_F1 * factors_F1[iop]
                F2_factor = np.dot(F3_F2, factors_F2[iop])
                F3_factor = F3_F3 * factors_F3[iop]

                FF_factors = np.array([F1_factor, F2_factor, F3_factor])
                FF_factors_all[imom, iop, ipol] = FF_factors

    # Read the values of the matrix element from fits to the three-point functions
    transition = "sig2n"
    matrix_elements_all = read_fit_data(
        nboot,
        datadir,
        operators_pickle,
        polarizations_pickle,
        momenta_pickle,
        transition,
    )

    # The shapes of the matrices are:
    # np.shape(FF_factors_all)=(3, 4, 2, 3)
    #    = (momenta, operators, polarizations, num_of_ff) complex numbers
    # np.shape(matrix_elements_all)=(3, 4, 2, 2, 500)
    #    = (momenta, operators, polarizations, real/imag, nboot)
    print("solving for the form factors")
    form_factor_values = np.zeros((len(momenta), num_of_ff, nboot))
    for solve_mom, mom_val in enumerate(momenta):
        print(f"\n\n{mom_val}")
        # This only does the real matrix elements, should extend to include the imaginary
        FF_factors_real = np.real(FF_factors_all[solve_mom, :, :, :]).reshape(
            4 * 2, num_of_ff
        )
        A_inv = np.linalg.pinv(FF_factors_real)

        for iboot in range(nboot):
            # Reshape the matrix element values and solve for the form factors
            matrix_elements_real = matrix_elements_all[
                solve_mom, :, :, 0, iboot
            ].reshape(4 * 2)
            x = np.matmul(A_inv, matrix_elements_real)
            form_factor_values[solve_mom, :, iboot] = x

    datafile = datadir / Path(f"form_factors_3pt_fit_sig2n.pkl")
    with open(datafile, "wb") as file_out:
        pickle.dump(form_factor_values, file_out)

    # Combine the form factors in the same combination as the Feynman-Hellmann results
    gamma4_factors = np.real(FF_factors_all[:, 3, 0, :])
    matrix_element_gamma4 = np.einsum("ij,ijk->ijk", gamma4_factors, form_factor_values)
    test2_ = np.einsum("ijk->ik", matrix_element_gamma4)
    print(f"{np.average(test2_, axis=1)=}")

    datafile = datadir / Path(f"matrix_element_3pt_fit_sig2n.pkl")
    with open(datafile, "wb") as file_out:
        pickle.dump(test2_, file_out)

    ff_avg = np.average(form_factor_values, axis=2)
    ff_std = np.std(form_factor_values, axis=2)
    for i, ff in enumerate(ff_avg):
        print([err_brackets(ff[j], ff_std[i, j]) for j in range(len(ff))])

    return FF_factors_all, matrix_elements_all, form_factor_values


def read_fit_data(nboot, datadir, operators, polarizations, momenta, transition):
    """Read the data from the two-exponential fits to the three-point functions for each momentum, polarization and operator"""

    src_snk_times = np.array([10, 13, 16])
    rel = "nr"

    matrix_elements_all = np.zeros(
        (len(momenta), len(operators), len(polarizations), 2, nboot)
    )

    for imom, mom in enumerate(momenta):
        # print(f"\n{mom}")
        for iop, operator in enumerate(operators):
            # print(f"\n{operator}")
            for ipol, pol in enumerate(polarizations):
                # print(f"\n{pol}")
                for ir, reim in enumerate(["real", "imag"]):
                    # print(reim)
                    datafile = datadir / Path(
                        f"{mom}_{operator}_{pol}_{rel}_{reim}_3pt_ratio_fit_{transition}.pkl"
                    )
                    with open(datafile, "rb") as file_in:
                        fit_data = pickle.load(file_in)

                    matrix_elements_all[imom, iop, ipol, ir] = fit_data[0][:, 0]
    return matrix_elements_all


if __name__ == "__main__":
    main()
