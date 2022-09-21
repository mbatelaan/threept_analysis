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
    resultsdir = Path.home() / Path(
        "Dropbox/PhD/analysis_code/transition_3pt_function/"
    )
    datadir = resultsdir / Path("data/")

    # mN = 4.179255e-01
    # mS = 4.641829e-01

    mN = 0.4382953145893229
    mS = 0.4751146506255126

    L = 32
    T = 64
    a = 0.074
    nboot = 500

    momenta_pickle = ["p+0+0+0", "p+1+0+0", "p+1+1+0"]
    operators_pickle = [
        "g0",
        "g1",
        "g2",
        "g3",
    ]

    momenta = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]]) * 2 * np.pi / L
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
    nucl_mom = 0

    FF_factors_all = np.zeros(
        (len(momenta), len(operators_F1), len(polarizations), 3), dtype="complex"
    )
    # print(f"{np.shape(FF_factors_all)=}")

    pp = pmom(0, 0, 0, mS)  # sigma
    for imom, mom in enumerate(momenta):
        # print(f"\n\n\n{mom=}")
        p = pmom(*mom, mN)
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
                F3_F1 = F3_fnc(pol, operators_F1[iop], p, pp)
                F3_F2 = np.array([F3_fnc(pol, jop, p, pp) for jop in operators_F2[iop]])
                F3_F3 = F3_fnc(pol, operators_F3[iop], p, pp)

                F1_factor = F3_F1 * factors_F1[iop]
                F2_factor = np.dot(F3_F2, factors_F2[iop])
                F3_factor = F3_F3 * factors_F3[iop]

                FF_factors = np.array([F1_factor, F2_factor, F3_factor])
                # print(f"{FF_factors=}")

                FF_factors_all[imom, iop, ipol] = FF_factors

                # print(f"{F1_factor=}")
                # print(f"{F2_factor=}")
                # print(f"{F3_factor=}")

    print(f"{np.shape(FF_factors_all)=}")
    # print(f"{FF_factors_all}")

    matrix_elements_all = read_fit_data(nboot, datadir)
    print(f"{np.shape(matrix_elements_all)=}")
    # print(f"{matrix_elements_all=}")

    print(
        np.real(FF_factors_all[0, 3, 0, :]),
        np.average(matrix_elements_all[0, 3, 0, 0, :]),
    )
    print(
        np.imag(FF_factors_all[0, 3, 0, :]),
        np.average(matrix_elements_all[0, 3, 0, 1, :]),
    )

    # np.shape(FF_factors_all)=(3, 4, 2, 3)
    # np.shape(matrix_elements_all)=(3, 4, 2, 2, 500)

    # solve_mom = 0
    print("solving for the form factors")
    form_factor_values = np.zeros((len(momenta), 3, nboot))
    for solve_mom, mom_val in enumerate(momenta):
        print(f"\n\n{mom_val}")
        mom0_factors = np.real(FF_factors_all[solve_mom, :, :, :]).reshape(4 * 2, 3)
        # mom0_matrix_elements = np.average(matrix_elements_all, axis=4)[
        #     solve_mom, :, :, 0
        # ].reshape(4 * 2)
        A_inv = np.linalg.pinv(mom0_factors)

        # form_factor_values = np.zeros((3, nboot))

        for iboot in range(nboot):
            mom0_matrix_elements = matrix_elements_all[
                solve_mom, :, :, 0, iboot
            ].reshape(4 * 2)
            x = np.matmul(A_inv, mom0_matrix_elements)
            form_factor_values[solve_mom, :, iboot] = x

    datafile = datadir / Path(f"form_factors_3pt_fit.pkl")
    with open(datafile, "wb") as file_out:
        pickle.dump(form_factor_values, file_out)

    ff_avg = np.average(form_factor_values, axis=2)
    ff_std = np.std(form_factor_values, axis=2)

    print(f"{np.shape(ff_avg)=}")
    print(f"{np.shape(mom0_factors)=}")
    print(f"{ff_avg=}")
    print(f"{ff_std=}")

    gamma4_factors = np.real(FF_factors_all[:, 3, 0, :])
    print(f"{np.shape(gamma4_factors)=}")
    print(f"{gamma4_factors=}")
    print(f"{ff_avg=}")

    test_ = np.einsum("ij,ijk->ijk", gamma4_factors, form_factor_values)
    test2_ = np.einsum("ijk->ik", test_)
    print(f"{np.average(test2_, axis=1)=}")

    # test = gamma4_factors * ff_avg
    # test2 = np.sum(test * ff_avg, axis=1)
    # print(f"{test=}")
    # print(f"{test2=}")

    datafile = datadir / Path(f"matrix_element_3pt_fit.pkl")
    with open(datafile, "wb") as file_out:
        pickle.dump(test2_, file_out)

    for i, ff in enumerate(ff_avg):
        print([err_brackets(ff[j], ff_std[i, j]) for j in range(len(ff))])

    return


def read_fit_data(nboot, datadir):
    """Read the data from the two-exponential fits to the three-point functions for each momentum, polarization and operator"""
    operators = [
        "g0",
        "g1",
        "g2",
        "g3",
    ]
    polarizations = ["UNPOL", "POL"]
    momenta = ["p+0+0+0", "p+1+0+0", "p+1+1+0"]
    src_snk_times = np.array([10, 13, 16])
    rel = "nr"
    # nboot = 500

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
                        f"{mom}_{operator}_{pol}_{rel}_{reim}_3pt_ratio_fit.pkl"
                    )
                    with open(datafile, "rb") as file_in:
                        fit_data = pickle.load(file_in)

                    matrix_elements_all[imom, iop, ipol, ir] = fit_data[0][:, 0]
    return matrix_elements_all


if __name__ == "__main__":
    main()
