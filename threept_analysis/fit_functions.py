import numpy as np


def threeptBS(X, B):
    B00, B10, B01, B11 = B
    tau, t, A_E0i, A_E0f, E0i, E0f, Delta_E01i, Delta_E01f = X

    return (
        np.sqrt(A_E0i * A_E0f)
        * np.exp(-E0f * t)
        * np.exp(-(E0i - E0f) * tau)
        * (
            B00
            + B10 * np.exp(-Delta_E01i * tau)
            + B01 * np.exp(-Delta_E01f * (t - tau))
            + B11 * np.exp(-Delta_E01f * t) * np.exp(-(Delta_E01i - Delta_E01f) * tau)
        )
    )


def threept_ratio(X, B):
    """
    The fitfunction of the three-point function mulitplied by a factor of twopoint functions
    All functions are expressed as a sum of two exponentials
    """
    B00, B10, B01, B11 = B
    tau, t, A_E0i, A_E0f, A_E1i, A_E1f, E0i, E0f, Delta_E01i, Delta_E01f = X

    def twoexp_(t, p):
        return p[0] * np.exp(-p[1] * t) + p[2] * np.exp(-p[3] * t)

    twopt_factor = (
        1
        / twoexp_(t, [A_E0f, E0f, A_E1f, E0f + Delta_E01f])
        * np.sqrt(
            (
                twoexp_(tau, [A_E0f, E0f, A_E1f, E0f + Delta_E01f])
                * twoexp_(t, [A_E0f, E0f, A_E1f, E0f + Delta_E01f])
                * twoexp_(t - tau, [A_E0i, E0i, A_E1i, E0i + Delta_E01i])
            )
            / (
                twoexp_(tau, [A_E0i, E0i, A_E1i, E0i + Delta_E01i])
                * twoexp_(t, [A_E0i, E0i, A_E1i, E0i + Delta_E01i])
                * twoexp_(t - tau, [A_E0f, E0f, A_E1f, E0f + Delta_E01f])
            )
        )
    )

    threept_function = (
        np.sqrt(A_E0i * A_E0f)
        * np.exp(-E0f * t)
        * np.exp(-(E0i - E0f) * tau)
        * (
            B00
            + B10 * np.exp(-Delta_E01i * tau)
            + B01 * np.exp(-Delta_E01f * (t - tau))
            + B11 * np.exp(-Delta_E01f * t) * np.exp(-(Delta_E01i - Delta_E01f) * tau)
        )
    )

    return threept_function * twopt_factor


def double_threept_ratio(X, B):
    """
    The fitfunction of the product of two three-point functions divided by two two-point functions
    All functions are expressed as a sum of two exponentials
    """
    B00, B10, B01, B11 = B
    tau, t, A_E0i, A_E0f, A_E1i, A_E1f, E0i, E0f, Delta_E01i, Delta_E01f = X

    def twoexp_(t, p):
        return p[0] * np.exp(-p[1] * t) + p[2] * np.exp(-p[3] * t)

    twopt_factor = np.sqrt(
        1
        / (
            twoexp_(t, [A_E0f, E0f, A_E1f, E0f + Delta_E01f])
            * twoexp_(t, [A_E0i, E0i, A_E1i, E0i + Delta_E01i])
        )
    )

    threept_function = np.sqrt(
        (
            np.sqrt(A_E0i * A_E0f)
            * np.exp(-E0f * t)
            * np.exp(-(E0i - E0f) * tau)
            * (
                B00
                + B10 * np.exp(-Delta_E01i * tau)
                + B01 * np.exp(-Delta_E01f * (t - tau))
                + B11
                * np.exp(-Delta_E01f * t)
                * np.exp(-(Delta_E01i - Delta_E01f) * tau)
            )
        )
        * (
            np.sqrt(A_E0f * A_E0i)
            * np.exp(-E0i * t)
            * np.exp(-(E0f - E0i) * tau)
            * (
                B00
                + B10 * np.exp(-Delta_E01f * tau)
                + B01 * np.exp(-Delta_E01i * (t - tau))
                + B11
                * np.exp(-Delta_E01i * t)
                * np.exp(-(Delta_E01f - Delta_E01i) * tau)
            )
        )
    )

    return threept_function * twopt_factor


def threept_ratio_3exp(X, B):
    """
    The fitfunction of the three-point function mulitplied by a factor of twopoint functions
    All functions are expressed as a sum of two exponentials
    """
    B00, B10, B01, B11, B20, B02, B21, B12, B22 = B
    tau, t, A_E0i, A_E0f, A_E1i, A_E1f, A_E2i, A_E2f, E0i, E0f, E1i, E1f, E2i, E2f = X

    def twopt_3exp(t, p):
        return (
            p[0] * np.exp(-p[1] * t)
            + p[2] * np.exp(-p[3] * t)
            + p[4] * np.exp(-p[5] * t)
        )

    twopt_factor = (
        1
        / twopt_3exp(t, [A_E0f, E0f, A_E1f, E1f, A_E2f, E2f])
        * np.sqrt(
            (
                twopt_3exp(tau, [A_E0f, E0f, A_E1f, E1f, A_E2f, E2f])
                * twopt_3exp(t, [A_E0f, E0f, A_E1f, E1f, A_E2f, E2f])
                * twopt_3exp(t - tau, [A_E0i, E0i, A_E1i, E1i, A_E2i, E2i])
            )
            / (
                twopt_3exp(tau, [A_E0i, E0i, A_E1i, E1i, A_E2i, E2i])
                * twopt_3exp(t, [A_E0i, E0i, A_E1i, E1i, A_E2i, E2i])
                * twopt_3exp(t - tau, [A_E0f, E0f, A_E1f, E1f, A_E2f, E2f])
            )
        )
    )

    threept_function = np.sqrt(A_E0i * A_E0f) * (
        B00 * np.exp(-E0i * tau) * np.exp(-E0f * (t - tau))
        + B10 * np.exp(-E1i * tau) * np.exp(-E0f * (t - tau))
        + B01 * np.exp(-E0i * tau) * np.exp(-E1f * (t - tau))
        + B11 * np.exp(-E1i * tau) * np.exp(-E1f * (t - tau))
        + B20 * np.exp(-E2i * tau) * np.exp(-E0f * (t - tau))
        + B02 * np.exp(-E0i * tau) * np.exp(-E2f * (t - tau))
        + B21 * np.exp(-E2i * tau) * np.exp(-E1f * (t - tau))
        + B12 * np.exp(-E1i * tau) * np.exp(-E2f * (t - tau))
        + B22 * np.exp(-E2i * tau) * np.exp(-E2f * (t - tau))
    )

    return threept_function * twopt_factor
