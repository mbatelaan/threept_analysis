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
import fit_functions as ff


def fit_3ptfn_2exp(
    threeptfn_list,
    twoptfn_list,
    fit_data_list,
    src_snk_times,
    delta_t,
    tmin_choice,
    datadir,
    ireim=0,
):
    """Fit to the three-point function with a two-exponential function, which includes parameters from the two-point functions
    ireim=0: real
    ireim=1: imaginary
    """
    tmin_choice_sigma = 4
    weight_tol = 0.01
    fitweights_n = np.array([fit["weight"] for fit in fit_data_list[0]])
    fitweights_n = np.where(fitweights_n > weight_tol, fitweights_n, 0)
    fitweights_n = fitweights_n / sum(fitweights_n)
    fitparams_n = np.array([fit["param"] for fit in fit_data_list[0]])
    fit_times_n = [fit["x"] for fit in fit_data_list[0]]
    chosen_time = np.where([times[0] == tmin_choice for times in fit_times_n])[0][0]
    best_fit_n = fit_data_list[0][chosen_time]
    weighted_fit_n = best_fit_n["param"]

    fitweights_s = np.array([fit["weight"] for fit in fit_data_list[1]])
    fitweights_s = np.where(fitweights_s > weight_tol, fitweights_s, 0)
    fitweights_s = fitweights_s / sum(fitweights_s)
    fitparams_s = np.array([fit["param"] for fit in fit_data_list[1]])
    fit_times_s = [fit["x"] for fit in fit_data_list[1]]
    chosen_time = np.where([times[0] == tmin_choice_sigma for times in fit_times_s])[0][
        0
    ]
    best_fit_s = fit_data_list[1][chosen_time]
    weighted_fit_s = best_fit_s["param"]

    # Set the parameters from the twoptfn
    A_E0i = weighted_fit_n[:, 0]
    A_E0f = weighted_fit_s[:, 0]
    E0i = weighted_fit_n[:, 1]
    E0f = weighted_fit_s[:, 1]
    Delta_E01i = np.exp(weighted_fit_n[:, 3])
    Delta_E01f = np.exp(weighted_fit_s[:, 3])
    # Delta_E01i = weighted_fit_n[:, 3] - weighted_fit_n[:, 1]
    # Delta_E01f = weighted_fit_s[:, 3] - weighted_fit_s[:, 1]

    fitfnc_2exp = ff.threeptBS

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


def fit_ratio_3exp(
    ratio_list,
    twoptfn_list,
    fit_data_list,
    src_snk_times,
    delta_t,
    tmin_choice,
    datadir,
):
    """Fit to the three-point function with a two-exponential function, which includes parameters from the two-point functions"""
    tmin_choice_sigma = 5
    weight_tol = 0.01
    fitweights_n = np.array([fit["weight"] for fit in fit_data_list[0]])
    fitweights_n = np.where(fitweights_n > weight_tol, fitweights_n, 0)
    fitweights_n = fitweights_n / sum(fitweights_n)
    fitparams_n = np.array([fit["param"] for fit in fit_data_list[0]])
    fit_times_n = [fit["x"] for fit in fit_data_list[0]]
    chosen_time = np.where([times[0] == tmin_choice for times in fit_times_n])[0][0]
    best_fit_n = fit_data_list[0][chosen_time]
    weighted_fit_n = best_fit_n["param"]

    fitweights_s = np.array([fit["weight"] for fit in fit_data_list[1]])
    fitweights_s = np.where(fitweights_s > weight_tol, fitweights_s, 0)
    fitweights_s = fitweights_s / sum(fitweights_s)
    fitparams_s = np.array([fit["param"] for fit in fit_data_list[1]])
    fit_times_s = [fit["x"] for fit in fit_data_list[1]]
    chosen_time = np.where([times[0] == tmin_choice_sigma for times in fit_times_s])[0][
        0
    ]
    best_fit_s = fit_data_list[1][chosen_time]
    weighted_fit_s = best_fit_s["param"]

    # Set the parameters from the twoptfn
    A_E0i = weighted_fit_n[:, 0]
    A_E0f = weighted_fit_s[:, 0]
    A_E1i = weighted_fit_n[:, 0] * weighted_fit_n[:, 2]
    A_E1f = weighted_fit_s[:, 0] * weighted_fit_s[:, 2]
    A_E2i = weighted_fit_n[:, 0] * weighted_fit_n[:, 4]
    A_E2f = weighted_fit_s[:, 0] * weighted_fit_s[:, 4]
    E0i = weighted_fit_n[:, 1]
    E0f = weighted_fit_s[:, 1]
    E1i = weighted_fit_n[:, 1] + np.exp(weighted_fit_n[:, 3])
    E1f = weighted_fit_s[:, 1] + np.exp(weighted_fit_s[:, 3])
    E2i = (
        weighted_fit_n[:, 1]
        + np.exp(weighted_fit_n[:, 3])
        + np.exp(weighted_fit_n[:, 5])
    )
    E2f = (
        weighted_fit_s[:, 1]
        + np.exp(weighted_fit_s[:, 3])
        + np.exp(weighted_fit_s[:, 5])
    )

    fitfnc_2exp = ff.threept_ratio_3exp

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
        np.average(A_E2i),
        np.average(A_E2f),
        np.average(E0i),
        np.average(E0f),
        np.average(E1i),
        np.average(E1f),
        np.average(E2i),
        np.average(E2f),
    ]
    p0 = [1, 1, 1, 1, 1, 1, 1, 1, 1]
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
    chisq = fitfunc.chisqfn(resavg.x, fitfnc_2exp, x_avg, fitdata_avg, cvinv)
    redchisq = chisq / (len(fitdata_avg) - len(p0))
    print(f"{redchisq=}")
    print(f"{resavg.fun/(len(fitdata_avg) - len(p0))=}")

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
            A_E2i[iboot],
            A_E2f[iboot],
            E0i[iboot],
            E0f[iboot],
            E1i[iboot],
            E1f[iboot],
            E2i[iboot],
            E2f[iboot],
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

    chisq_ = fitfunc.chisqfn(
        np.average(fit_param_boot, axis=0), fitfnc_2exp, x_avg, fitdata_avg, cvinv
    )
    redchisq_ = chisq_ / (len(fitdata_avg) - len(p0))
    print(f"{redchisq_=}")

    return (
        fit_param_boot,
        ratio_fit_boot,
        fit_param_avg,
        redchisq,
        best_fit_n,
        best_fit_s,
    )


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
