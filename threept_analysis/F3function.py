from QCDSFGamma import *
import numpy as np

# from Mom import SingleThreeMom

# def Ffnc(Gamm, oper, p, pp):
#     """Calculates the value of the F_3 function.
#     Gamm is the projection matrix
#     oper is the operator
#     p and pp are the initial and final momentum respectively
#     """
#     Ep = p.Energy()
#     Epp = pp.Energy()
#     pdg = p.px * gamma1 + p.py * gamma2 + p.pz * gamma3
#     ppdg = pp.px * gamma1 + pp.py * gamma2 + pp.pz * gamma3
#     term2 = gamma4 - pdg * (1.0j / Ep) + gammaI * (p.m / Ep)
#     term1 = gamma4 - ppdg * (1.0j / Epp) + gammaI * (pp.m / Epp)
#     intG = np.matmul( np.matmul(Gamm,term1), np.matmul(oper,term2))
#     #    print intG
#     return 0.25 * intG.trace()


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
    # term2 = -1.0j * pdg + gamma4 * Ep + gammaI * (p.m / Ep)
    # term1 = -1.0j * ppdg + gamma4 * Epp + gammaI * (pp.m / Epp)
    # term2 = gamma4 - pdg * (1.0j / Ep) + gammaI * (p.m / Ep)
    # term1 = gamma4 - ppdg * (1.0j / Epp) + gammaI * (pp.m / Epp)
    intG = np.matmul(np.matmul(Gamm, term1), np.matmul(oper, term2))
    return intG.trace()


def F2FH_fnc(Gamm, p):
    """Calculates the value of the F_2 function for FH.
    Gamm is the projection matrix
    p is the momentum
    """
    Ep = p.Energy()
    pdg = p.px * gamma1 + p.py * gamma2 + p.pz * gamma3
    term1 = -1.0j * pdg + gamma4 * Ep + gammaI * (p.m)
    intG = np.matmul(Gamm, term1)
    return 0.25 * intG.trace()


class pmom:
    def __init__(self, px, py, pz, m):
        self.px = px
        self.py = py
        self.pz = pz
        self.m = m

    def Energy(self):
        return np.sqrt(self.px**2 + self.py**2 + self.pz**2 + self.m**2)


def main():
    mN = 4.179255e-01
    mS = 4.641829e-01
    L = 32
    T = 64
    a = 0.074

    nucl_mom_units = 0
    pp = pmom(0, 0, 0, mS)  # sigma
    p = pmom(nucl_mom_units, 0, 0, mN)

    Pol = GammaUP  # UnPolarized
    Oper = gamma4
    F3_gamma4 = F3_fnc(Pol, Oper, p, pp)
    print("\n")
    print(f"{F3_gamma4=}")

    # Oper1 = sigma41
    Oper1 = 1.0j / 2 * (np.matmul(gamma4, gamma1) - np.matmul(gamma1, gamma4))
    F3_sigma41 = F3_fnc(Pol, Oper1, p, pp)
    print(f"{F3_sigma41=}")

    # Oper2 = sigma42
    Oper2 = 1.0j / 2 * (np.matmul(gamma4, gamma2) - np.matmul(gamma2, gamma4))
    F3_sigma42 = F3_fnc(Pol, Oper2, p, pp)
    print(f"{F3_sigma42=}")

    # Oper3 = sigma43
    Oper3 = 1.0j / 2 * (np.matmul(gamma4, gamma3) - np.matmul(gamma3, gamma4))
    F3_sigma43 = F3_fnc(Pol, Oper3, p, pp)
    print(f"{F3_sigma43=}")

    Oper4 = gammaI
    F3_Identity = F3_fnc(Pol, Oper4, p, pp)
    print(f"{F3_Identity=}")

    print("\n")
    ### Calculate the prefactors for each form factor
    prefacF1 = F3_gamma4 / ((2 * p.Energy()) * (2 * (pp.Energy() + pp.m)))
    prefacF2 = (
        (F3_sigma41 * pp.px + F3_sigma42 * pp.py + F3_sigma43 * pp.pz)
        / (pp.m + p.m)
        / ((2 * p.Energy()) * (2 * (pp.Energy() + pp.m)))
    )


def old_main():
    # mass = 0.4622
    mN = 4.179255e-01
    mS = 4.641829e-01
    L = 32
    T = 64
    a = 0.074
    theta = 0.4832569416636752

    p = pmom(0, 0, 0, mS)
    pp = pmom(1 * 2 * np.pi / L, theta * np.pi / L, 0, mN)
    print(f"{p.Energy()=}")
    print(f"{pp.Energy()=}")
    print(f"{p.m=}")
    print(f"{pp.m=}")

    # Qsqrd = np.array([ np.dot(q,q)*(2*np.pi/pars.L)**2*(0.1973**2)/(pars.a**2) for q in pars.qval ])
    # Qsq   = np.array([ np.dot(q,q)*(2*np.pi/pars.L)**2 for q in pars.qval[1:] ])
    qsqvalue = (pp.px**2 + pp.py**2 + pp.pz**2) * (0.1973**2) / (a**2)
    print(f"{qsqvalue=}")

    Pol = GammaUP  # UnPolarized
    Oper = gamma4
    F3_gamma4 = F3_fnc(Pol, Oper, p, pp)
    print("\n")
    print(f"{F3_gamma4=}")

    # Oper1 = sigma41
    Oper1 = 1.0j / 2 * (np.matmul(gamma4, gamma1) - np.matmul(gamma1, gamma4))
    F3_sigma41 = F3_fnc(Pol, Oper1, p, pp)
    print(f"{F3_sigma41=}")

    # Oper2 = sigma42
    Oper2 = 1.0j / 2 * (np.matmul(gamma4, gamma2) - np.matmul(gamma2, gamma4))
    F3_sigma42 = F3_fnc(Pol, Oper2, p, pp)
    print(f"{F3_sigma42=}")

    # Oper3 = sigma43
    Oper3 = 1.0j / 2 * (np.matmul(gamma4, gamma3) - np.matmul(gamma3, gamma4))
    F3_sigma43 = F3_fnc(Pol, Oper3, p, pp)
    print(f"{F3_sigma43=}")

    Oper4 = gammaI
    F3_Identity = F3_fnc(Pol, Oper4, p, pp)
    print(f"{F3_Identity=}")

    print("\n")
    ### Calculate the prefactors for each form factor
    prefacF1 = F3_gamma4 / ((2 * p.Energy()) * (2 * (pp.Energy() + pp.m)))
    prefacF2 = (
        (F3_sigma41 * pp.px + F3_sigma42 * pp.py + F3_sigma43 * pp.pz)
        / (pp.m + p.m)
        / ((2 * p.Energy()) * (2 * (pp.Energy() + pp.m)))
    )
    # prefacF2 = (
    #     (F3_sigma41 * pp.px + F3_sigma42 * pp.py + F3_sigma43 * pp.pz)
    #     / (pp.m + p.m)
    #     / (2 * p.Energy() * 2 * pp.Energy())
    # )
    prefacF3 = 0
    print(f"{prefacF1=}")
    print(f"{prefacF2=}")
    print(f"{prefacF3=}")

    ### Along Chambers' thesis:
    F2_unpol = F2FH_fnc(Pol, p)
    print(f"{F2_unpol=}")
    prefacF1_ = F3_gamma4 / (2 * p.Energy() * F2_unpol)
    print(f"{prefacF1_=}")

    Energy = 0.5 * (pp.Energy() + p.Energy())
    print("\n")
    # ### Prefactors by hand
    # prefacF1H = 1
    # prefacF2H = (
    #     -1 * (pp.px ** 2 + pp.py ** 2 + pp.pz ** 2) / ((pp.Energy() + pp.m) * (pp.m + p.m))
    # )
    # prefacF3H = pp.Energy() / (pp.m + p.m)
    # print(f"{prefacF1H=}")
    # print(f"{prefacF2H=}")
    # print(f"{prefacF3H=}")

    # print(pp.Energy(), p.Energy())
    # print(pp.m, p.m)
    # Energy = 0.5 * (pp.Energy() + p.Energy())
    # # ### Prefactors by hand
    # # prefacF1H = np.sqrt(2 * Energy * (Energy + pp.m))
    # # prefacF2H = (
    # #     np.sqrt(2 * Energy * (Energy + pp.m))
    # #     * (pp.px ** 2 + pp.py ** 2 + pp.pz ** 2)
    # #     / (Energy + pp.m) ** 2
    # # )
    # # prefacF3H = 0
    # # print(f"{prefacF1H=}")
    # # print(f"{prefacF2H=}")
    # # print(f"{prefacF3H=}")

    # ### Prefactors by hand
    # prefacF1H = np.sqrt(2 * Energy * (Energy + pp.m)) / (2 * Energy)
    # prefacF2H = (
    #     np.sqrt(2 * p.m * (p.m + pp.m))
    #     * (pp.px ** 2 + pp.py ** 2 + pp.pz ** 2)
    #     / (p.m + pp.m)
    #     / (2 * Energy)
    # )
    # prefacF3H = 0
    # print(f"{prefacF1H=}")
    # print(f"{prefacF2H=}")
    # print(f"{prefacF3H=}")

    ### Prefactors by hand
    prefacF1H = np.sqrt((Energy + pp.m) / (8 * pp.m * Energy * p.m))
    prefacF2H = (
        np.sqrt((Energy + pp.m) / (8 * pp.m * Energy * p.m))
        * (pp.px**2 + pp.py**2 + pp.pz**2)
        / (Energy + +pp.m)
    )
    prefacF3H = 0
    print(f"{prefacF1H=}")
    print(f"{prefacF2H=}")
    print(f"{prefacF3H=}")

    ### Prefactors by hand again
    prefacF1H = 1
    prefacF2H = (pp.px**2 + pp.py**2 + pp.pz**2) / (
        (Energy + pp.m) * (pp.m + p.m)
    )
    prefacF3H = 0
    print(f"{prefacF1H=}")
    print(f"{prefacF2H=}")
    print(f"{prefacF3H=}")


if __name__ == "__main__":
    main()
