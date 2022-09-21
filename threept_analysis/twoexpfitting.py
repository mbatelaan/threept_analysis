import numpy as np
from pathlib import Path

import pickle
import csv
import scipy.optimize as syopt
import matplotlib.pyplot as plt

from plot_utils import save_plot
from formatting import err_brackets
from analysis.evxptreaders import evxptdata
from analysis.bootstrap import bootstrap
from analysis import stats
from analysis import fitfunc

_metadata = {"Author": "Mischa Batelaan", "Creator": __file__}

_colors = [
    (0, 0, 0),
    (0.9, 0.6, 0),
    (0.35, 0.7, 0.9),
    (0, 0.6, 0.5),
    (0.95, 0.9, 0.25),
    (0, 0.45, 0.7),
    (0.8, 0.4, 0),
    (0.8, 0.6, 0.7),
]

_fmts = ["s", "^", "o", ".", "p", "v", "P", ",", "*"]


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


def fit_3ptfn_2exp(
    threeptfn_list,
    twoptfn_list,
    fit_data_list,
    src_snk_times,
    delta_t,
    datadir,
    ireim=0,
):
    """Fit to the three-point function with a two-exponential function, which includes parameters from the two-point functions
    ireim=0: real
    ireim=1: imaginary
    """

    chisq_tol = 0.01
    fitweights_n = np.array([fit["weight"] for fit in fit_data_list[0]])
    fitweights_n = np.where(fitweights_n > chisq_tol, fitweights_n, 0)
    fitweights_n = fitweights_n / sum(fitweights_n)
    fitparams_n = np.array([fit["param"] for fit in fit_data_list[0]])
    best_fit_n = fit_data_list[0][np.argmax(fitweights_n)]
    weighted_fit_n = np.einsum("i,ijk->jk", fitweights_n, fitparams_n)
    # print(f"{fitweights_n=}")
    # print(np.where(fitweights_n > chisq_tol, fitweights_n, 0))
    print(f"{best_fit_n['x']=}")
    print(f"{best_fit_n['paramavg'][1]=}")
    # print(f"{[i for i in best_fit_n]=}")
    # print(f"{np.average(best_fit_n['param'],axis=0)=}")
    # print(f"{np.average(weighted_fit_n,axis=0)=}")

    fitweights_s = np.array([fit["weight"] for fit in fit_data_list[1]])
    # print(np.where(fitweights_s > chisq_tol, fitweights_s, 0))
    fitweights_s = np.where(fitweights_s > chisq_tol, fitweights_s, 0)
    # print(f"{sum(fitweights_s)=}")
    fitweights_s = fitweights_s / sum(fitweights_s)
    fitparams_s = np.array([fit["param"] for fit in fit_data_list[1]])
    best_fit_s = fit_data_list[1][np.argmax(fitweights_s)]
    weighted_fit_s = np.einsum("i,ijk->jk", fitweights_s, fitparams_s)
    print(f"{best_fit_s['paramavg'][1]=}")

    # Set the parameters from the twoptfn
    A_E0i = weighted_fit_n[:, 0]
    A_E0f = weighted_fit_s[:, 0]
    E0i = weighted_fit_n[:, 1]
    E0f = weighted_fit_s[:, 1]
    Delta_E01i = weighted_fit_n[:, 3] - weighted_fit_n[:, 1]
    Delta_E01f = weighted_fit_s[:, 3] - weighted_fit_s[:, 1]

    fitfnc_2exp = threeptBS

    # Create the fit data
    fitdata = np.concatenate(
        (
            threeptfn_list[0][:, delta_t : src_snk_times[0] + 1 - delta_t, ireim],
            threeptfn_list[1][:, delta_t : src_snk_times[1] + 1 - delta_t, ireim],
            threeptfn_list[2][:, delta_t : src_snk_times[2] + 1 - delta_t, ireim],
        ),
        axis=1,
    )
    # print(f"{np.shape(fitdata)=}")
    t_values = np.concatenate(
        (
            [src_snk_times[0]] * (src_snk_times[0] + 1 - 2 * delta_t),
            [src_snk_times[1]] * (src_snk_times[1] + 1 - 2 * delta_t),
            [src_snk_times[2]] * (src_snk_times[2] + 1 - 2 * delta_t),
        )
    )
    tau_values = np.concatenate(
        (
            np.arange(src_snk_times[0] + 1)[delta_t:-delta_t],
            np.arange(src_snk_times[1] + 1)[delta_t:-delta_t],
            np.arange(src_snk_times[2] + 1)[delta_t:-delta_t],
        )
    )

    # Fit to the average of the data
    x_avg = [
        tau_values,
        t_values,
        np.average(A_E0i),
        np.average(A_E0f),
        np.average(E0i),
        np.average(E0f),
        np.average(Delta_E01i),
        np.average(Delta_E01f),
    ]
    p0 = [1, 1, 1, 1]
    fitdata_avg = np.average(fitdata, axis=0)
    fitdata_std = np.std(fitdata, axis=0)
    cvinv = np.linalg.inv(np.cov(fitdata.T))
    var_inv = np.diag(1 / (fitdata_std**2))
    resavg = syopt.minimize(
        fitfunc.chisqfn,
        p0,
        args=(fitfnc_2exp, x_avg, fitdata_avg, var_inv),
        method="Nelder-Mead",
        options={"disp": False},
    )
    # print(f"{resavg=}")
    fit_param_avg = resavg.x
    threept_fit_avg = fitfnc_2exp(x_avg, fit_param_avg)
    chisq = fitfunc.chisqfn(resavg.x, fitfnc_2exp, x_avg, fitdata_avg, cvinv)
    redchisq = chisq / (len(fitdata_avg) - len(p0))
    # print(f"{redchisq=}")

    # Fit to each bootstrap
    p0 = fit_param_avg
    nboot = np.shape(threeptfn_list[0])[0]
    fit_param_boot = []
    threept_fit_boot = []
    for iboot in np.arange(nboot):
        x = [
            tau_values,
            t_values,
            A_E0i[iboot],
            A_E0f[iboot],
            E0i[iboot],
            E0f[iboot],
            Delta_E01i[iboot],
            Delta_E01f[iboot],
        ]
        res = syopt.minimize(
            fitfunc.chisqfn,
            p0,
            args=(fitfnc_2exp, x, fitdata[iboot], var_inv),
            method="Nelder-Mead",
            options={"disp": False},
        )
        fit_param_boot.append(res.x)
        threept_fit_boot.append(fitfnc_2exp(x, res.x))
    threept_fit_boot = np.array(threept_fit_boot)
    fit_param_boot = np.array(fit_param_boot)

    twopt_denominator = np.concatenate(
        (
            [twoptfn_list[1][:, src_snk_times[0], 0]]
            * (src_snk_times[0] + 1 - 2 * delta_t),
            [twoptfn_list[1][:, src_snk_times[1], 0]]
            * (src_snk_times[1] + 1 - 2 * delta_t),
            [twoptfn_list[1][:, src_snk_times[2], 0]]
            * (src_snk_times[2] + 1 - 2 * delta_t),
        )
    )
    twopt_denominator = np.moveaxis(twopt_denominator, 0, 1)
    # twopt_denominator = np.einsum(
    #     "i,ij->ij",
    #     weighted_fit_s[:, 0],
    #     np.exp(-np.einsum("i,j->ij", weighted_fit_s[:, 1], x_avg[1])),
    # ) + np.einsum(
    #     "i,ij->ij",
    #     weighted_fit_s[:, 2],
    #     np.exp(-np.einsum("i,j->ij", weighted_fit_s[:, 3], x_avg[1])),
    # )

    fit_ratio_boot = threept_fit_boot / twopt_denominator

    return (
        fit_param_boot,
        fit_ratio_boot,
        threept_fit_boot,
        fit_param_avg,
        redchisq,
        best_fit_n,
        best_fit_s,
    )


def fit_ratio_2exp(
    ratio_list,
    twoptfn_list,
    fit_data_list,
    src_snk_times,
    delta_t,
    datadir,
):
    """Fit to the three-point function with a two-exponential function, which includes parameters from the two-point functions"""

    chisq_tol = 0.01
    fitweights_n = np.array([fit["weight"] for fit in fit_data_list[0]])
    fitweights_n = np.where(fitweights_n > chisq_tol, fitweights_n, 0)
    fitweights_n = fitweights_n / sum(fitweights_n)
    fitparams_n = np.array([fit["param"] for fit in fit_data_list[0]])
    best_fit_n = fit_data_list[0][np.argmax(fitweights_n)]
    best2_fit_n = fit_data_list[0][np.argsort(fitweights_n)[1]]
    weighted_fit_n = np.einsum("i,ijk->jk", fitweights_n, fitparams_n)
    # print(f"{fitweights_n=}")
    # print(np.where(fitweights_n > chisq_tol, fitweights_n, 0))
    print(f"{best_fit_n['x']=}")
    print(f"{best_fit_n['redchisq']=}")
    print(f"{best_fit_n['paramavg'][1]=}")
    print(f"{best2_fit_n['x']=}")
    print(f"{best2_fit_n['redchisq']=}")
    print(f"{best2_fit_n['paramavg'][1]=}")
    # print(f"{[i for i in best_fit_n]=}")
    # print(f"{np.average(best_fit_n['param'],axis=0)=}")
    # print(f"{np.average(weighted_fit_n,axis=0)=}")

    fitweights_s = np.array([fit["weight"] for fit in fit_data_list[1]])
    fitweights_s = np.where(fitweights_s > chisq_tol, fitweights_s, 0)
    fitweights_s = fitweights_s / sum(fitweights_s)
    fitparams_s = np.array([fit["param"] for fit in fit_data_list[1]])
    best_fit_s = fit_data_list[1][np.argmax(fitweights_s)]
    weighted_fit_s = np.einsum("i,ijk->jk", fitweights_s, fitparams_s)

    # Set the parameters from the twoptfn
    A_E0i = weighted_fit_n[:, 0]
    A_E0f = weighted_fit_s[:, 0]
    A_E1i = weighted_fit_n[:, 2]
    A_E1f = weighted_fit_s[:, 2]
    E0i = weighted_fit_n[:, 1]
    E0f = weighted_fit_s[:, 1]
    Delta_E01i = weighted_fit_n[:, 3] - weighted_fit_n[:, 1]
    Delta_E01f = weighted_fit_s[:, 3] - weighted_fit_s[:, 1]

    fitfnc_2exp = threept_ratio

    # Create the fit data
    fitdata = np.concatenate(
        (
            ratio_list[0][:, delta_t : src_snk_times[0] + 1 - delta_t],
            ratio_list[1][:, delta_t : src_snk_times[1] + 1 - delta_t],
            ratio_list[2][:, delta_t : src_snk_times[2] + 1 - delta_t],
        ),
        axis=1,
    )
    t_values = np.concatenate(
        (
            [src_snk_times[0]] * (src_snk_times[0] + 1 - 2 * delta_t),
            [src_snk_times[1]] * (src_snk_times[1] + 1 - 2 * delta_t),
            [src_snk_times[2]] * (src_snk_times[2] + 1 - 2 * delta_t),
        )
    )
    tau_values = np.concatenate(
        (
            np.arange(src_snk_times[0] + 1)[delta_t:-delta_t],
            np.arange(src_snk_times[1] + 1)[delta_t:-delta_t],
            np.arange(src_snk_times[2] + 1)[delta_t:-delta_t],
        )
    )

    # Fit to the average of the data
    x_avg = [
        tau_values,
        t_values,
        np.average(A_E0i),
        np.average(A_E0f),
        np.average(A_E1i),
        np.average(A_E1f),
        np.average(E0i),
        np.average(E0f),
        np.average(Delta_E01i),
        np.average(Delta_E01f),
    ]
    p0 = [1, 1, 1, 1]
    fitdata_avg = np.average(fitdata, axis=0)
    fitdata_std = np.std(fitdata, axis=0)
    cvinv = np.linalg.pinv(np.cov(fitdata.T))
    var_inv = np.diag(1 / (fitdata_std**2))
    resavg = syopt.minimize(
        fitfunc.chisqfn,
        p0,
        args=(fitfnc_2exp, x_avg, fitdata_avg, var_inv),
        method="Nelder-Mead",
        options={"disp": False},
    )

    fit_param_avg = resavg.x
    # ratio_fit_avg = fitfnc_2exp(x_avg, fit_param_avg)
    chisq = fitfunc.chisqfn(resavg.x, fitfnc_2exp, x_avg, fitdata_avg, cvinv)
    redchisq = chisq / (len(fitdata_avg) - len(p0))

    # Fit to each bootstrap
    p0 = fit_param_avg
    nboot = np.shape(ratio_list[0])[0]
    fit_param_boot = []
    ratio_fit_boot = []
    for iboot in np.arange(nboot):
        x = [
            tau_values,
            t_values,
            A_E0i[iboot],
            A_E0f[iboot],
            A_E1i[iboot],
            A_E1f[iboot],
            E0i[iboot],
            E0f[iboot],
            Delta_E01i[iboot],
            Delta_E01f[iboot],
        ]
        res = syopt.minimize(
            fitfunc.chisqfn,
            p0,
            args=(fitfnc_2exp, x, fitdata[iboot], var_inv),
            method="Nelder-Mead",
            options={"disp": False},
        )
        fit_param_boot.append(res.x)
        ratio_fit_boot.append(fitfnc_2exp(x, res.x))
    ratio_fit_boot = np.array(ratio_fit_boot)
    fit_param_boot = np.array(fit_param_boot)

    # chisq = fitfunc.chisqfn(
    #     np.average(fit_param_boot, axis=0), fitfnc_2exp, x_avg, fitdata_avg, cvinv
    # )
    # redchisq = chisq / (len(fitdata_avg) - len(p0))

    return (
        fit_param_boot,
        ratio_fit_boot,
        fit_param_avg,
        redchisq,
        best_fit_n,
        best_fit_s,
    )


def make_mat_elements(fit_params, datadir):
    kappa_combs = [
        "kp121040kp121040",
        "kp121040kp120620",
        "kp120620kp121040",
        "kp120620kp120620",
    ]
    momenta = ["p+0+0+0", "p+1+0+0", "p+1+1+0"]
    datafile_n = datadir / Path(f"{kappa_combs[0]}_p+0+0+0_fitlist_2pt.pkl")
    # datafile_n = datadir / Path(f"{kappa_combs[0]}_{mom}_fitlist_2pt.pkl")
    with open(datafile_n, "rb") as file_in:
        fit_data_n = pickle.load(file_in)
    datafile_s = datadir / Path(f"{kappa_combs[1]}_p+0+0+0_fitlist_2pt.pkl")
    with open(datafile_s, "rb") as file_in:
        fit_data_s = pickle.load(file_in)

    chisq_tol = 0.01
    fitweights_n = np.array([fit["weight"] for fit in fit_data_n])
    fitweights_n = np.where(fitweights_n > chisq_tol, fitweights_n, 0)
    fitweights_n = fitweights_n / sum(fitweights_n)
    fitparams_n = np.array([fit["param"] for fit in fit_data_n])
    weighted_fit_n = np.einsum("i,ijk->jk", fitweights_n, fitparams_n)

    fitweights_s = np.array([fit["weight"] for fit in fit_data_s])
    fitweights_s = np.where(fitweights_s > chisq_tol, fitweights_s, 0)
    fitweights_s = fitweights_s / sum(fitweights_s)
    fitparams_s = np.array([fit["param"] for fit in fit_data_s])
    weighted_fit_s = np.einsum("i,ijk->jk", fitweights_s, fitparams_s)

    # Set the parameters from the twoptfn
    neutron_mass = weighted_fit_n[:, 1]
    sigma_mass = weighted_fit_s[:, 1]

    mat_element_list = []
    for imom, mom in enumerate(momenta):
        datafile_n = datadir / Path(f"{kappa_combs[0]}_{mom}_fitlist_2pt.pkl")
        with open(datafile_n, "rb") as file_in:
            fit_data_n = pickle.load(file_in)
        fitweights_n = np.array([fit["weight"] for fit in fit_data_n])
        fitweights_n = np.where(fitweights_n > chisq_tol, fitweights_n, 0)
        fitweights_n = fitweights_n / sum(fitweights_n)
        fitparams_n = np.array([fit["param"] for fit in fit_data_n])
        weighted_fit_n = np.einsum("i,ijk->jk", fitweights_n, fitparams_n)
        neutron_energy = weighted_fit_n[:, 1]

        # factor = (neutron_energy + neutron_mass) * sigma_mass
        # mat_element = fit_params[imom] * factor

        factor = np.sqrt(2) * np.sqrt(neutron_energy / (neutron_energy + neutron_mass))
        mat_element = fit_params[imom] * factor

        # print(f"{np.average(factor)=}")
        # print(f"{np.average(mat_element)=}")
        mat_element_list.append(mat_element)

    return np.array(mat_element_list)


def read_pickle(filename, nboot=200, nbin=1):
    """Get the data from the pickle file and output a bootstrapped numpy array.

    The output is a numpy matrix with:
    axis=0: bootstraps
    axis=2: time axis
    axis=3: real & imaginary parts
    """
    with open(filename, "rb") as file_in:
        data = pickle.load(file_in)
    bsdata = bootstrap(data, config_ax=0, nboot=nboot, nbin=nbin)
    return bsdata


def plot_3ptfn(threepoint, src_snk_time=10):
    time = np.arange(64)
    # time_eff = np.arange(63)
    ydata = np.average(threepoint, axis=0)
    yerror = np.std(threepoint, axis=0)
    # threepoint_eff = stats.bs_effmass(threepoint, time_axis=1, spacing=1)
    # ydata_eff = np.average(threepoint_eff, axis=0)
    # yerror_eff = np.std(threepoint_eff, axis=0)

    plt.figure()
    plt.errorbar(
        time,
        ydata,
        yerror,
        elinewidth=1,
        color=_colors[0],
        fmt="s",
        markerfacecolor="none",
    )
    # plt.semilogy()

    plt.show()
    plt.close()


def plot_all_3ptfn(threepoint_fns, src_snk_times, plotdir, title=""):
    time = np.arange(64)
    time_eff = np.arange(63)
    # labels = ["t10", "t13", "t16"]
    labels = [
        r"$t_{\mathrm{sep}}=10$",
        r"$t_{\mathrm{sep}}=13$",
        r"$t_{\mathrm{sep}}=16$",
    ]

    f, axarr = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(16, 9))
    for icorr, corr in enumerate(threepoint_fns):
        plot_time = time[: src_snk_times[icorr]] - src_snk_times[icorr] / 2
        # threepoint_eff = stats.bs_effmass(corr, time_axis=1, spacing=1)
        ydata = np.average(corr, axis=0)
        yerror = np.std(corr, axis=0)
        axarr[icorr].errorbar(
            plot_time,
            ydata[: src_snk_times[icorr]],
            yerror[: src_snk_times[icorr]],
            capsize=4,
            elinewidth=1,
            color=_colors[icorr],
            fmt=_fmts[icorr],
            # markerfacecolor="none",
            label=labels[icorr],
        )
        axarr[icorr].grid(True)
        axarr[icorr].legend(fontsize=15, loc="upper left")
        axarr[icorr].set_xlabel(r"$\tau-t_{\mathrm{sep}}/2$", labelpad=14, fontsize=18)
        axarr[icorr].set_ylabel(
            r"$G_3(\vec{p}\, ; t_{\mathrm{sep}}, \tau)$",
            labelpad=5,
            fontsize=18,
        )
        axarr[icorr].label_outer()

    f.suptitle(
        r"3-point function with $\hat{\mathcal{O}}=\gamma_4$, $\vec{q}\, =(0,0,0)$ for $t_{\mathrm{sep}}=10,13,16$"
    )
    savefile = plotdir / Path(f"{title}.pdf")
    savefile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(savefile)
    plt.close()
    return


def loop_all_3pt(latticedir, plotdir):
    """Loop over all the available three-point functions, read in the data from the evxpt file and plot the three different source-sink separations for each momentum, operator and polarization choice"""
    current_names = []
    currents = [
        "vector-0",
        "vector-1",
        "vector-2",
        "vector-3",
        "axial-0",
        "axial-1",
        "axial-2",
        "axial-3",
        "tensor-01",
        "tensor-02",
        "tensor-03",
        "tensor-12",
        "tensor-13",
        "tensor-23",
    ]
    polarizations = ["unpol", "pol_3"]
    # momenta = ["q+0+0+0", "q+1+0+0", "q+1+1+0", "q+1+1+1", "q+2+0+0", "q+2+1+0"]
    momenta = ["q+0+0+0", "q+1+0+0", "q+1+1+0"]
    current_choice = 0
    pol_choice = 0
    mom_choice = 0

    for current_choice, current in enumerate(currents):
        for pol_choice, pol in enumerate(polarizations):
            for mom_choice, momentum in enumerate(momenta):
                evxptres_t10 = latticedir / Path(
                    f"b5p50kp121040kp120620c2p6500-32x64_t10/d0/point/vector/{currents[current_choice]}/kp121040kp120620/p+0+0+0/{momenta[mom_choice]}/d_quark/{polarizations[pol_choice]}/dump/dump.res"
                )
                threepoint_t10 = evxptdata(evxptres_t10, numbers=[0], nboot=500, nbin=1)
                threepoint_t10_real = threepoint_t10[:, 0, :, 0]

                evxptres_t13 = latticedir / Path(
                    f"b5p50kp121040kp120620c2p6500-32x64_t13/d0/point/vector/{currents[current_choice]}/kp121040kp120620/p+0+0+0/{momenta[mom_choice]}/d_quark/{polarizations[pol_choice]}/dump/dump.res"
                )
                threepoint_t13 = evxptdata(evxptres_t13, numbers=[0], nboot=500, nbin=1)
                threepoint_t13_real = threepoint_t13[:, 0, :, 0]

                evxptres_t16 = latticedir / Path(
                    f"b5p50kp121040kp120620c2p6500-32x64_t16/d0/point/vector/{currents[current_choice]}/kp121040kp120620/p+0+0+0/{momenta[mom_choice]}/d_quark/{polarizations[pol_choice]}/dump/dump.res"
                )
                threepoint_t16 = evxptdata(evxptres_t16, numbers=[0], nboot=500, nbin=1)
                threepoint_t16_real = threepoint_t16[:, 0, :, 0]

                threepoint_fns = np.array(
                    [threepoint_t10_real, threepoint_t13_real, threepoint_t16_real]
                )
                src_snk_times = np.array([10, 13, 16])
                plot_all_3ptfn(
                    threepoint_fns,
                    src_snk_times,
                    plotdir,
                    title=f"_{current}_{momentum}_{pol}",
                )
    return


def plot_all_ratios(ratios, src_snk_times, plotdir, plotparam, title=""):
    time = np.arange(64)
    labels = [
        r"$t_{\mathrm{sep}}=10$",
        r"$t_{\mathrm{sep}}=13$",
        r"$t_{\mathrm{sep}}=16$",
    ]

    f, axarr = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(16, 9))
    for icorr, corr in enumerate(ratios):
        plot_time = time[: src_snk_times[icorr]] - src_snk_times[icorr] / 2
        plot_time2 = time - src_snk_times[icorr] / 2
        # print(f"{plot_time2=}")
        ydata = np.average(corr, axis=0)
        yerror = np.std(corr, axis=0)

        axarr[icorr].errorbar(
            plot_time2[1 : src_snk_times[icorr]],
            ydata[1 : src_snk_times[icorr]],
            yerror[1 : src_snk_times[icorr]],
            capsize=4,
            elinewidth=1,
            color=_colors[icorr],
            fmt=_fmts[icorr],
            # markerfacecolor="none",
            label=labels[icorr],
        )
        axarr[icorr].grid(True)
        axarr[icorr].legend(fontsize=15, loc="upper left")
        axarr[icorr].set_xlabel(r"$\tau-t_{\mathrm{sep}}/2$", labelpad=14, fontsize=18)
        axarr[icorr].set_ylabel(
            r"$R(\vec{p}\, ; t_{\mathrm{sep}}, \tau)$", labelpad=5, fontsize=18
        )
        axarr[icorr].label_outer()
        # axarr[icorr].set_xlim(-0.5, src_snk_times[icorr] + 0.5)
        axarr[icorr].set_xlim(
            plot_time2[0] - 0.5, plot_time2[src_snk_times[icorr]] + 0.5
        )

    f.suptitle(
        rf"full 3-point function ratio with $\hat{{\mathcal{{O}}}}=${plotparam[1]}, $\Gamma = ${plotparam[2]}, $\vec{{q}}\, ={plotparam[0][1:]}$"
    )
    # f.suptitle(
    #     r"3-point function with $\hat{\mathcal{O}}=\gamma_4$, $\vec{q}\, =(0,0,0)$ for $t_{\mathrm{sep}}=10,13,16$"
    # )
    savefile = plotdir / Path(f"{title}.pdf")
    savefile.parent.mkdir(parents=True, exist_ok=True)
    # print(f"{savefile=}")
    plt.savefig(savefile)
    # plt.show()
    plt.close()
    return


def plot_all_fitratios(
    ratios,
    fit_result,
    delta_t,
    src_snk_times,
    redchisq,
    fit_param_boot,
    plotdir,
    plotparam,
    title="",
):
    time = np.arange(64)
    labels = [
        r"$t_{\mathrm{sep}}=10$",
        r"$t_{\mathrm{sep}}=13$",
        r"$t_{\mathrm{sep}}=16$",
    ]

    f, axarr = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(16, 9))
    for icorr, corr in enumerate(ratios):
        # plot_time = time[: src_snk_times[icorr]] - (src_snk_times[icorr] - 1) / 2
        # print(f"{plot_time=}")
        plot_time2 = time - (src_snk_times[icorr]) / 2
        ydata = np.average(corr, axis=0)
        yerror = np.std(corr, axis=0)
        tau_values = (
            np.arange(src_snk_times[icorr] + 1)[delta_t:-delta_t]
            - (src_snk_times[icorr]) / 2
        )

        axarr[icorr].errorbar(
            plot_time2[1 : src_snk_times[icorr]],
            ydata[1 : src_snk_times[icorr]],
            yerror[1 : src_snk_times[icorr]],
            capsize=4,
            elinewidth=1,
            color=_colors[icorr],
            fmt=_fmts[icorr],
            # markerfacecolor="none",
            label=labels[icorr],
        )
        step_indices = [
            0,
            src_snk_times[0] + 1 - (2 * delta_t),
            src_snk_times[0] + 1 + src_snk_times[1] + 1 - (4 * delta_t),
            src_snk_times[0]
            + 1
            + src_snk_times[1]
            + 1
            + src_snk_times[2]
            + 1
            - (6 * delta_t),
        ]
        axarr[icorr].plot(
            tau_values,
            np.average(
                fit_result[:, step_indices[icorr] : step_indices[icorr + 1]], axis=0
            ),
            color=_colors[3],
        )
        axarr[icorr].fill_between(
            tau_values,
            np.average(
                fit_result[:, step_indices[icorr] : step_indices[icorr + 1]], axis=0
            )
            - np.std(
                fit_result[:, step_indices[icorr] : step_indices[icorr + 1]], axis=0
            ),
            np.average(
                fit_result[:, step_indices[icorr] : step_indices[icorr + 1]], axis=0
            )
            + np.std(
                fit_result[:, step_indices[icorr] : step_indices[icorr + 1]], axis=0
            ),
            alpha=0.3,
            linewidth=0,
            color=_colors[3],
        )
        axarr[icorr].axhline(
            np.average(fit_param_boot[:, 0]),
            color=_colors[5],
            label=rf"fit = {err_brackets(np.average(fit_param_boot[:, 0]), np.std(fit_param_boot[:, 0]))}",
        )
        # err_brackets(nucleon_avg[i], nucleon_err[i])
        plot_time3 = np.array([-20, 20])
        axarr[icorr].fill_between(
            plot_time3,
            [np.average(fit_param_boot[:, 0]) - np.std(fit_param_boot[:, 0])]
            * len(plot_time3),
            [np.average(fit_param_boot[:, 0]) + np.std(fit_param_boot[:, 0])]
            * len(plot_time3),
            alpha=0.3,
            linewidth=0,
            color=_colors[5],
        )

        axarr[icorr].grid(True)
        axarr[icorr].legend(fontsize=15, loc="upper left")
        axarr[icorr].set_xlabel(r"$\tau-t_{\mathrm{sep}}/2$", labelpad=14, fontsize=18)
        axarr[icorr].set_ylabel(
            r"$R(\vec{p}\, ; t_{\mathrm{sep}}, \tau)$", labelpad=5, fontsize=18
        )
        axarr[icorr].label_outer()
        # axarr[icorr].set_xlim(-src_snk_times[-1] - 1, src_snk_times[-1] + 1)
        axarr[icorr].set_xlim(plot_time2[0] - 0.5, plot_time2[src_snk_times[-1]] + 0.5)

    f.suptitle(
        rf"{plotparam[3]} 3-point function ratio with $\hat{{\mathcal{{O}}}}=${plotparam[1]}, $\Gamma = ${plotparam[2]}, $\vec{{q}}\, ={plotparam[0][1:]}$ with two-state fit $\chi^2_{{\mathrm{{dof}}}}={redchisq:.2f}$"
    )
    savefile = plotdir / Path(f"{title}.pdf")
    savefile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(savefile)
    # plt.show()
    plt.close()
    return


def plot_full_fitratios(
    ratios,
    threept_fit,
    twopt_fit,
    delta_t,
    src_snk_times,
    redchisq,
    fit_param_boot,
    plotdir,
    plotparam,
    title="",
):
    time = np.arange(64)
    labels = [
        r"$t_{\mathrm{sep}}=10$",
        r"$t_{\mathrm{sep}}=13$",
        r"$t_{\mathrm{sep}}=16$",
    ]

    f, axarr = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(16, 9))
    for icorr, corr in enumerate(ratios):

        plot_time2 = time - (src_snk_times[icorr]) / 2
        ydata = np.average(corr, axis=0)
        yerror = np.std(corr, axis=0)
        plot_x_values = (
            np.arange(src_snk_times[icorr] + 1)[delta_t:-delta_t]
            - (src_snk_times[icorr]) / 2
        )
        tau_values = np.arange(src_snk_times[icorr] + 1)[delta_t:-delta_t]
        t_values = np.array([src_snk_times[icorr]] * len(tau_values))
        print(f"{tau_values=}")
        print(f"{t_values=}")

        step_indices = [
            0,
            src_snk_times[0] + 1 - (2 * delta_t),
            src_snk_times[0] + 1 + src_snk_times[1] + 1 - (4 * delta_t),
            src_snk_times[0]
            + 1
            + src_snk_times[1]
            + 1
            + src_snk_times[2]
            + 1
            - (6 * delta_t),
        ]

        nboot = 500
        twoexp = fitfunc.initffncs("Twoexp").eval

        twoptfn_sigma_t = np.array(
            [twoexp(t_values, twopt_fit[1]["param"][iboot]) for iboot in range(nboot)]
        )
        twoptfn_sigma_tau = np.array(
            [twoexp(tau_values, twopt_fit[1]["param"][iboot]) for iboot in range(nboot)]
        )
        twoptfn_sigma_tmtau = np.array(
            [
                twoexp(t_values - tau_values, twopt_fit[1]["param"][iboot])
                for iboot in range(nboot)
            ]
        )

        twoptfn_neutron_t = np.array(
            [twoexp(t_values, twopt_fit[0]["param"][iboot]) for iboot in range(nboot)]
        )
        twoptfn_neutron_tau = np.array(
            [twoexp(tau_values, twopt_fit[0]["param"][iboot]) for iboot in range(nboot)]
        )
        twoptfn_neutron_tmtau = np.array(
            [
                twoexp(t_values - tau_values, twopt_fit[0]["param"][iboot])
                for iboot in range(nboot)
            ]
        )

        ratio_factor = (
            1
            / twoptfn_sigma_t
            * np.sqrt(
                (twoptfn_sigma_tau * twoptfn_sigma_t * twoptfn_neutron_tmtau)
                / (twoptfn_neutron_tau * twoptfn_neutron_t * twoptfn_sigma_tmtau)
            )
        )

        ratio_ = (
            threept_fit[:, step_indices[icorr] : step_indices[icorr + 1]] * ratio_factor
        )

        axarr[icorr].errorbar(
            plot_time2[1 : src_snk_times[icorr]],
            ydata[1 : src_snk_times[icorr]],
            yerror[1 : src_snk_times[icorr]],
            capsize=4,
            elinewidth=1,
            color=_colors[icorr],
            fmt=_fmts[icorr],
            label=labels[icorr],
        )
        axarr[icorr].plot(
            plot_x_values,
            np.average(ratio_, axis=0),
            color=_colors[3],
        )
        axarr[icorr].fill_between(
            plot_x_values,
            np.average(ratio_, axis=0) - np.std(ratio_, axis=0),
            np.average(ratio_, axis=0) + np.std(ratio_, axis=0),
            alpha=0.3,
            linewidth=0,
            color=_colors[3],
        )
        axarr[icorr].axhline(
            np.average(fit_param_boot[:, 0]),
            color=_colors[5],
            label=rf"fit = {err_brackets(np.average(fit_param_boot[:, 0]), np.std(fit_param_boot[:, 0]))}",
        )

        plot_time3 = np.array([-20, 20])
        axarr[icorr].fill_between(
            plot_time3,
            [np.average(fit_param_boot[:, 0]) - np.std(fit_param_boot[:, 0])]
            * len(plot_time3),
            [np.average(fit_param_boot[:, 0]) + np.std(fit_param_boot[:, 0])]
            * len(plot_time3),
            alpha=0.3,
            linewidth=0,
            color=_colors[5],
        )

        axarr[icorr].grid(True)
        axarr[icorr].legend(fontsize=15, loc="upper left")
        axarr[icorr].set_xlabel(r"$\tau-t_{\mathrm{sep}}/2$", labelpad=14, fontsize=18)
        axarr[icorr].set_ylabel(
            r"$R(\vec{p}\, ; t_{\mathrm{sep}}, \tau)$", labelpad=5, fontsize=18
        )
        axarr[icorr].label_outer()
        # axarr[icorr].set_xlim(-src_snk_times[-1] - 1, src_snk_times[-1] + 1)
        axarr[icorr].set_xlim(plot_time2[0] - 0.5, plot_time2[src_snk_times[-1]] + 0.5)

    f.suptitle(
        rf"{plotparam[3]} 3-point function ratio with $\hat{{\mathcal{{O}}}}=${plotparam[1]}, $\Gamma = ${plotparam[2]}, $\vec{{q}}\, ={plotparam[0][1:]}$ with two-state fit $\chi^2_{{\mathrm{{dof}}}}={redchisq:.2f}$"
    )
    savefile = plotdir / Path(f"{title}_full.pdf")
    savefile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(savefile)
    # plt.show()
    plt.close()
    return


def plot_ratio_fit(
    ratios,
    ratio_fit,
    delta_t,
    src_snk_times,
    redchisq,
    fit_param_boot,
    plotdir,
    plotparam,
    title="",
):
    time = np.arange(64)
    labels = [
        r"$t_{\mathrm{sep}}=10$",
        r"$t_{\mathrm{sep}}=13$",
        r"$t_{\mathrm{sep}}=16$",
    ]

    f, axarr = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(16, 9))
    for icorr, corr in enumerate(ratios):
        plot_time2 = time - (src_snk_times[icorr]) / 2
        ydata = np.average(corr, axis=0)
        yerror = np.std(corr, axis=0)
        plot_x_values = (
            np.arange(src_snk_times[icorr] + 1)[delta_t:-delta_t]
            - (src_snk_times[icorr]) / 2
        )
        tau_values = np.arange(src_snk_times[icorr] + 1)[delta_t:-delta_t]
        t_values = np.array([src_snk_times[icorr]] * len(tau_values))

        step_indices = [
            0,
            src_snk_times[0] + 1 - (2 * delta_t),
            src_snk_times[0] + 1 + src_snk_times[1] + 1 - (4 * delta_t),
            src_snk_times[0]
            + 1
            + src_snk_times[1]
            + 1
            + src_snk_times[2]
            + 1
            - (6 * delta_t),
        ]

        axarr[icorr].errorbar(
            plot_time2[1 : src_snk_times[icorr]],
            ydata[1 : src_snk_times[icorr]],
            yerror[1 : src_snk_times[icorr]],
            capsize=4,
            elinewidth=1,
            color=_colors[icorr],
            fmt=_fmts[icorr],
            label=labels[icorr],
        )
        axarr[icorr].plot(
            plot_x_values,
            np.average(
                ratio_fit[:, step_indices[icorr] : step_indices[icorr + 1]], axis=0
            ),
            color=_colors[3],
        )
        axarr[icorr].fill_between(
            plot_x_values,
            np.average(
                ratio_fit[:, step_indices[icorr] : step_indices[icorr + 1]], axis=0
            )
            - np.std(
                ratio_fit[:, step_indices[icorr] : step_indices[icorr + 1]], axis=0
            ),
            np.average(
                ratio_fit[:, step_indices[icorr] : step_indices[icorr + 1]], axis=0
            )
            + np.std(
                ratio_fit[:, step_indices[icorr] : step_indices[icorr + 1]], axis=0
            ),
            alpha=0.3,
            linewidth=0,
            color=_colors[3],
        )
        axarr[icorr].axhline(
            np.average(fit_param_boot[:, 0]),
            color=_colors[5],
            label=rf"fit = {err_brackets(np.average(fit_param_boot[:, 0]), np.std(fit_param_boot[:, 0]))}",
        )

        plot_time3 = np.array([-20, 20])
        axarr[icorr].fill_between(
            plot_time3,
            [np.average(fit_param_boot[:, 0]) - np.std(fit_param_boot[:, 0])]
            * len(plot_time3),
            [np.average(fit_param_boot[:, 0]) + np.std(fit_param_boot[:, 0])]
            * len(plot_time3),
            alpha=0.3,
            linewidth=0,
            color=_colors[5],
        )

        axarr[icorr].grid(True)
        axarr[icorr].legend(fontsize=15, loc="upper left")
        axarr[icorr].set_xlabel(r"$\tau-t_{\mathrm{sep}}/2$", labelpad=14, fontsize=18)
        axarr[icorr].set_ylabel(
            r"$R(\vec{p}\, ; t_{\mathrm{sep}}, \tau)$", labelpad=5, fontsize=18
        )
        axarr[icorr].label_outer()
        # axarr[icorr].set_xlim(-src_snk_times[-1] - 1, src_snk_times[-1] + 1)
        axarr[icorr].set_xlim(plot_time2[0] - 0.5, plot_time2[src_snk_times[-1]] + 0.5)

    f.suptitle(
        rf"{plotparam[3]} 3-point function ratio with $\hat{{\mathcal{{O}}}}=${plotparam[1]}, $\Gamma = ${plotparam[2]}, $\vec{{q}}\, ={plotparam[0][1:]}$ with two-state fit $\chi^2_{{\mathrm{{dof}}}}={redchisq:.2f}$"
    )
    savefile = plotdir / Path(f"{title}_full.pdf")
    savefile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(savefile)
    # plt.show()
    plt.close()
    return


def make_full_ratio(threeptfn, twoptfn_sigma_real, twoptfn_neutron_real, src_snk_time):
    """Make the ratio of two-point and three-point functions which produces the plateau"""
    sqrt_factor = np.sqrt(
        (
            twoptfn_sigma_real[:, : src_snk_time + 1]
            * twoptfn_neutron_real[:, src_snk_time::-1]
        )
        / (
            twoptfn_neutron_real[:, : src_snk_time + 1]
            * twoptfn_sigma_real[:, src_snk_time::-1]
        )
    )
    prefactor_full = np.einsum(
        "ij,i->ij",
        sqrt_factor,
        np.sqrt(
            twoptfn_sigma_real[:, src_snk_time] / twoptfn_neutron_real[:, src_snk_time]
        )
        / twoptfn_sigma_real[:, src_snk_time],
    )
    ratio = np.einsum("ijk,ij->ijk", threeptfn[:, : src_snk_time + 1], prefactor_full)
    return ratio


def main():
    plt.style.use("./mystyle.txt")
    plt.rc("text.latex", preamble=r"\usepackage{physics}")

    # --- directories ---
    latticedir = Path.home() / Path(
        "Documents/PhD/lattice_results/transition_3pt_function/"
    )
    resultsdir = Path.home() / Path(
        "Dropbox/PhD/analysis_code/transition_3pt_function/"
    )
    plotdir = resultsdir / Path("plots/")
    plotdir2 = plotdir / Path("twopoint/")
    datadir = resultsdir / Path("data/")

    plotdir.mkdir(parents=True, exist_ok=True)
    plotdir2.mkdir(parents=True, exist_ok=True)
    datadir.mkdir(parents=True, exist_ok=True)

    # # ======================================================================
    # # Make plots of the evxpt data to compare
    # evxpt_data(latticedir, resultsdir, plotdir, datadir)
    # # ======================================================================

    # ======================================================================
    # Read in the three point function data
    operators_tex = [
        "$\gamma_1$",
        "$\gamma_2$",
        "$\gamma_3$",
        "$\gamma_4$",
        # "g5",
        # "g51",
        # "g53",
        # "g01",
        # "g02",
        # "g03",
        # "g05",
        # "g12",
        # "g13",
        # "g23",
        # "g25",
        # "gI",
    ]
    operators = [
        "g0",
        "g1",
        "g2",
        "g3",
        # "g5",
        # "g51",
        # "g53",
        # "g01",
        # "g02",
        # "g03",
        # "g05",
        # "g12",
        # "g13",
        # "g23",
        # "g25",
        # "gI",
    ]
    # polarizations = ["UNPOL"]
    polarizations = ["UNPOL", "POL"]
    momenta = ["p+0+0+0", "p+1+0+0", "p+1+1+0"]
    # momenta = ["p+0+0+0", "p-1+0+0", "p-1-1+0"]
    src_snk_times = np.array([10, 13, 16])
    rel = "nr"  #'rel'
    delta_t = 4

    for imom, mom in enumerate(momenta):
        print(f"\n{mom}")
        # ======================================================================
        # Read the two-point function and fit a two-exponential function to it
        twoptfn_filename_sigma = latticedir / Path(
            f"mass_spectrum/baryon_qcdsf/barspec/32x64/unpreconditioned_slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/barspec_nucleon_{rel}_500cfgs.pickle"
        )
        twoptfn_filename_neutron = latticedir / Path(
            f"mass_spectrum/baryon_qcdsf/barspec/32x64/unpreconditioned_slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/{mom}/barspec_nucleon_{rel}_500cfgs.pickle"
        )
        twoptfn_sigma = read_pickle(twoptfn_filename_sigma, nboot=500, nbin=1)
        twoptfn_neutron = read_pickle(twoptfn_filename_neutron, nboot=500, nbin=1)

        twoptfn_sigma_real = twoptfn_sigma[:, :, 0]
        twoptfn_neutron_real = twoptfn_neutron[:, :, 0]

        for iop, operator in enumerate(operators):
            print(f"\n{operator}")
            for ipol, pol in enumerate(polarizations):
                print(f"\n{pol}")
                # Read in the 3pt function data
                threeptfn_pickle_t10 = latticedir / Path(
                    f"bar3ptfn_t10/bar3ptfn/32x64/unpreconditioned_slrc/kp121040tkp120620_kp121040/NUCL_D_{pol}_NONREL_gI_t10/sh_gij_p21_90-sh_gij_p21_90/{mom}/bar3ptfn_{operator}_500cfgs.pickle"
                )
                threeptfn_pickle_t13 = latticedir / Path(
                    f"bar3ptfn_t13/bar3ptfn/32x64/unpreconditioned_slrc/kp121040tkp120620_kp121040/NUCL_D_{pol}_NONREL_gI_t13/sh_gij_p21_90-sh_gij_p21_90/{mom}/bar3ptfn_{operator}_500cfgs.pickle"
                )
                threeptfn_pickle_t16 = latticedir / Path(
                    f"bar3ptfn_t16/bar3ptfn/32x64/unpreconditioned_slrc/kp121040tkp120620_kp121040/NUCL_D_{pol}_NONREL_gI_t16/sh_gij_p21_90-sh_gij_p21_90/{mom}/bar3ptfn_{operator}_500cfgs.pickle"
                )
                threeptfn_t10 = read_pickle(threeptfn_pickle_t10, nboot=500, nbin=1)
                threeptfn_t13 = read_pickle(threeptfn_pickle_t13, nboot=500, nbin=1)
                threeptfn_t16 = read_pickle(threeptfn_pickle_t16, nboot=500, nbin=1)

                # ======================================================================
                # Construct the simple ratio of 3pt and 2pt functions
                ratio_t10 = np.einsum(
                    "ijk,i->ijk", threeptfn_t10, twoptfn_sigma_real[:, 10] ** (-1)
                )
                ratio_t13 = np.einsum(
                    "ijk,i->ijk", threeptfn_t13, twoptfn_sigma_real[:, 13] ** (-1)
                )
                ratio_t16 = np.einsum(
                    "ijk,i->ijk", threeptfn_t16, twoptfn_sigma_real[:, 16] ** (-1)
                )
                # ratio_list_real = [
                #     ratio_t10[:, :, 0],
                #     ratio_t13[:, :, 0],
                #     ratio_t16[:, :, 0],
                # ]
                simple_ratio_list = np.array([ratio_t10, ratio_t13, ratio_t16])

                # ======================================================================
                # Construct the full ratio of 3pt and 2pt functions
                ratio_full_t10 = make_full_ratio(
                    threeptfn_t10, twoptfn_sigma_real, twoptfn_neutron_real, 10
                )
                ratio_full_t13 = make_full_ratio(
                    threeptfn_t13, twoptfn_sigma_real, twoptfn_neutron_real, 13
                )
                ratio_full_t16 = make_full_ratio(
                    threeptfn_t16, twoptfn_sigma_real, twoptfn_neutron_real, 16
                )

                # ======================================================================
                # plot the real part of the 3pt fn ratio
                full_ratio_list_reim = [
                    [
                        ratio_full_t10[:, :, 0],
                        ratio_full_t13[:, :, 0],
                        ratio_full_t16[:, :, 0],
                    ],
                    [
                        ratio_full_t10[:, :, 1],
                        ratio_full_t13[:, :, 1],
                        ratio_full_t16[:, :, 1],
                    ],
                ]
                full_ratio_list_real = [
                    ratio_full_t10[:, :, 0],
                    ratio_full_t13[:, :, 0],
                    ratio_full_t16[:, :, 0],
                ]
                full_ratio_list = [
                    ratio_full_t10,
                    ratio_full_t13,
                    ratio_full_t16,
                ]
                # Fit to the full ratio of 3pt and 2pt functions
                # (
                #     fit_param_boot,
                #     fit_ratio_boot,
                #     fit_param_avg,
                #     redchisq,
                # ) = fit_full_ratio(
                #     full_ratio_list,
                #     # np.array([threeptfn_t10, threeptfn_t13, threeptfn_t16]),
                #     # np.array([twoptfn_neutron, twoptfn_sigma]),
                #     np.array([fit_data_n, fit_data_s]),
                #     src_snk_times,
                #     delta_t,
                #     datadir,
                # )

                plot_all_ratios(
                    full_ratio_list_real,
                    src_snk_times,
                    plotdir,
                    [mom, operators_tex[iop], pol],
                    title=f"{mom}/{pol}/full_ratios_real_{operator}",
                )

                # ======================================================================
                # Read the results of the fit to the two-point functions

                kappa_combs = [
                    "kp121040kp121040",
                    "kp121040kp120620",
                    "kp120620kp121040",
                    "kp120620kp120620",
                ]
                datafile_n = datadir / Path(
                    f"{kappa_combs[0]}_{mom}_{rel}_fitlist_2pt.pkl"
                )
                with open(datafile_n, "rb") as file_in:
                    fit_data_n = pickle.load(file_in)
                datafile_s = datadir / Path(
                    f"{kappa_combs[1]}_p+0+0+0_{rel}_fitlist_2pt.pkl"
                )
                with open(datafile_s, "rb") as file_in:
                    fit_data_s = pickle.load(file_in)

                for ir, reim in enumerate(["real", "imag"]):
                    print(reim)
                    # ======================================================================
                    # fit to the three-point function with a two-exponential function
                    # (
                    #     fit_param_boot,
                    #     fit_ratio_boot,
                    #     threept_fit_boot,
                    #     fit_param_avg,
                    #     redchisq,
                    #     best_fit_n,
                    #     best_fit_s,
                    # ) = fit_3ptfn_2exp(
                    #     np.array([threeptfn_t10, threeptfn_t13, threeptfn_t16]),
                    #     np.array([twoptfn_neutron, twoptfn_sigma]),
                    #     np.array([fit_data_n, fit_data_s]),
                    #     src_snk_times,
                    #     delta_t,
                    #     datadir,
                    #     ir,
                    # )
                    # fit_params = [
                    #     fit_param_boot,
                    #     fit_ratio_boot,
                    #     fit_param_avg,
                    #     redchisq,
                    # ]

                    # # Save the fit results to pickle files
                    # datafile = datadir / Path(
                    #     f"{mom}_{operator}_{pol}_{rel}_{reim}_3pt_fit.pkl"
                    # )
                    # with open(datafile, "wb") as file_out:
                    #     pickle.dump(fit_params, file_out)

                    # print(f"{np.average(fit_param_boot, axis=0)=}")
                    # print(f"{np.shape(fit_param_boot)=}")

                    # ======================================================================
                    # fit to the ratio of 3pt and 2pt functions with a two-exponential function
                    (
                        fit_param_ratio_boot,
                        ratio_fit_boot,
                        fit_param_ratio_avg,
                        redchisq_ratio,
                        best_fit_n,
                        best_fit_s,
                    ) = fit_ratio_2exp(
                        full_ratio_list_reim[ir],
                        np.array([twoptfn_neutron, twoptfn_sigma]),
                        np.array([fit_data_n, fit_data_s]),
                        src_snk_times,
                        delta_t,
                        datadir,
                    )
                    fit_params_ratio = [
                        fit_param_ratio_boot,
                        ratio_fit_boot,
                        fit_param_ratio_avg,
                        redchisq_ratio,
                    ]

                    # Save the fit results to pickle files
                    datafile_ratio = datadir / Path(
                        f"{mom}_{operator}_{pol}_{rel}_{reim}_3pt_ratio_fit.pkl"
                    )
                    with open(datafile_ratio, "wb") as file_out:
                        pickle.dump(fit_params_ratio, file_out)

                    # ======================================================================
                    # Plot the results of the fit to the 3pt function against a ratio of the data.
                    # plot_all_fitratios(
                    #     simple_ratio_list[:, :, :, ir],
                    #     fit_ratio_boot,
                    #     delta_t,
                    #     src_snk_times,
                    #     redchisq,
                    #     fit_param_boot,
                    #     plotdir,
                    #     [mom, operators_tex[iop], pol, reim],
                    #     title=f"{mom}/{pol}/fit_ratios_{reim}_{operator}",
                    # )

                    # plot_full_fitratios(
                    #     full_ratio_list_reim[ir],
                    #     threept_fit_boot,
                    #     [best_fit_n, best_fit_s],
                    #     delta_t,
                    #     src_snk_times,
                    #     redchisq,
                    #     fit_param_boot,
                    #     plotdir,
                    #     [mom, operators_tex[iop], pol, reim],
                    #     title=f"{mom}/{pol}/fit_ratios_{reim}_{operator}",
                    # )

                    # ======================================================================
                    # Plot the results of the fit to the ratio
                    plot_ratio_fit(
                        full_ratio_list_reim[ir],
                        ratio_fit_boot,
                        delta_t,
                        src_snk_times,
                        redchisq_ratio,
                        fit_param_ratio_boot,
                        plotdir,
                        [mom, operators_tex[iop], pol, reim],
                        title=f"{mom}/{pol}/ratio_fit_{reim}_{operator}",
                    )


def evxpt_data(latticedir, resultsdir, plotdir, datadir):
    """Using the evxpt result files, construct and plot a ratio of three-point and two-point functions"""
    # ======================================================================
    # Read the two-point function and fit a two-exponential function to it
    twoptfn_filename_sigma = latticedir / Path(
        "mass_spectrum/baryon_qcdsf/barspec/32x64/unpreconditioned_slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/barspec_nucleon_rel_500cfgs.pickle"
    )
    twoptfn_sigma = read_pickle(twoptfn_filename_sigma, nboot=500, nbin=1)
    twoptfn_sigma_real = twoptfn_sigma[:, :, 0]

    twoptfn_filename_neutron = latticedir / Path(
        "mass_spectrum/baryon_qcdsf/barspec/32x64/unpreconditioned_slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/barspec_nucleon_rel_500cfgs.pickle"
    )
    twoptfn_neutron = read_pickle(twoptfn_filename_neutron, nboot=500, nbin=1)
    twoptfn_neutron_real = twoptfn_neutron[:, :, 0]

    # ======================================================================
    # Read the three-point function data from the evxpt files
    currents = [
        "vector-0",
        "vector-1",
        "vector-2",
        "vector-3",
        "axial-0",
        "axial-1",
        "axial-2",
        "axial-3",
        "tensor-01",
        "tensor-02",
        "tensor-03",
        "tensor-12",
        "tensor-13",
        "tensor-23",
    ]
    polarizations = ["unpol", "pol_3"]
    momenta = ["q+0+0+0", "q+1+0+0", "q+1+1+0", "q+1+1+1", "q+2+0+0", "q+2+1+0"]
    # momenta = ["q+0+0+0", "q-1+0+0", "q-1-1+0", "q-1-1-1", "q-2+0+0", "q-2-1+0"]
    current_choice = 0
    pol_choice = 1
    mom_choice = 0

    evxptres_t10 = latticedir / Path(
        f"b5p50kp121040kp120620c2p6500-32x64_t10/d0/point/vector/{currents[current_choice]}/kp121040kp120620/p+0+0+0/{momenta[mom_choice]}/d_quark/{polarizations[pol_choice]}/dump/dump.res"
    )
    threepoint_t10 = evxptdata(evxptres_t10, numbers=[0], nboot=500, nbin=1)
    threepoint_t10_real = threepoint_t10[:, 0, :, 0]

    evxptres_t13 = latticedir / Path(
        f"b5p50kp121040kp120620c2p6500-32x64_t13/d0/point/vector/{currents[current_choice]}/kp121040kp120620/p+0+0+0/{momenta[mom_choice]}/d_quark/{polarizations[pol_choice]}/dump/dump.res"
    )
    threepoint_t13 = evxptdata(evxptres_t13, numbers=[0], nboot=500, nbin=1)
    threepoint_t13_real = threepoint_t13[:, 0, :, 0]

    evxptres_t16 = latticedir / Path(
        f"b5p50kp121040kp120620c2p6500-32x64_t16/d0/point/vector/{currents[current_choice]}/kp121040kp120620/p+0+0+0/{momenta[mom_choice]}/d_quark/{polarizations[pol_choice]}/dump/dump.res"
    )
    threepoint_t16 = evxptdata(evxptres_t16, numbers=[0], nboot=500, nbin=1)
    threepoint_t16_real = threepoint_t16[:, 0, :, 0]

    # ======================================================================
    # Plot all the 3pt functions
    # plot_3ptfn(threepoint_t10_real, src_snk_time=10)
    # plot_3ptfn(threepoint_t13_real, src_snk_time=13)
    # plot_3ptfn(threepoint_t16_real, src_snk_time=16)
    threepoint_fns = np.array(
        [threepoint_t10_real, threepoint_t13_real, threepoint_t16_real]
    )
    src_snk_times = np.array([10, 13, 16])
    # plot_all_3ptfn(threepoint_fns, src_snk_times, plotdir)

    # ======================================================================
    # Construct the simple ratio of 3pt and 2pt functions
    ratio_unpol_t10 = np.einsum(
        "ij,i->ij", threepoint_t10_real, twoptfn_sigma_real[:, 10] ** (-1)
    )
    ratio_unpol_t13 = np.einsum(
        "ij,i->ij", threepoint_t13_real, twoptfn_sigma_real[:, 13] ** (-1)
    )
    ratio_unpol_t16 = np.einsum(
        "ij,i->ij", threepoint_t16_real, twoptfn_sigma_real[:, 16] ** (-1)
    )

    print(f"{threepoint_t10_real=}")
    print(twoptfn_sigma_real[:, 10] ** (-1))

    ratio_unpol_t10 = (
        -1 * threepoint_t10_real / np.array([twoptfn_sigma_real[:, 10]] * 64).T
    )
    ratio_unpol_t13 = (
        -1 * threepoint_t13_real / np.array([twoptfn_sigma_real[:, 13]] * 64).T
    )
    ratio_unpol_t16 = (
        -1 * threepoint_t16_real / np.array([twoptfn_sigma_real[:, 16]] * 64).T
    )

    ratio_list = np.array([ratio_unpol_t10, ratio_unpol_t13, ratio_unpol_t16])
    src_snk_times = np.array([10, 13, 16])
    mom = momenta[mom_choice]
    operator = currents[current_choice]
    pol = polarizations[pol_choice]
    plot_all_ratios(
        ratio_list,
        src_snk_times,
        plotdir,
        [mom, operator, pol],
        title=f"{mom}_{pol}_full_ratios_real_{operator}",
    )
    return


if __name__ == "__main__":
    main()
