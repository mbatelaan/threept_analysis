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


def select_2pt_fit(
    fit_data_list, tmin_choice_nucl, tmin_choice_sigma, datadir, mom, transition
):
    """Sort through the fits to the two-point function and pick out one fit result that has the chosen tmin and tmax"""
    # Nucleon
    fit_times_n = [fit["x"] for fit in fit_data_list[0]]
    chosen_time = np.where([times[0] == tmin_choice_nucl for times in fit_times_n])[0][
        0
    ]
    best_fit_n = fit_data_list[0][chosen_time]
    fit_params_n = best_fit_n["param"]

    # Sigma
    fit_times_s = [fit["x"] for fit in fit_data_list[1]]
    chosen_time = np.where([times[0] == tmin_choice_sigma for times in fit_times_s])[0][
        0
    ]
    best_fit_s = fit_data_list[1][chosen_time]
    fit_params_s = best_fit_s["param"]

    # Save the chosen fit results to pickle files
    datafile_ = datadir / Path(f"{mom}_{transition}_2pt_fit_choice.pkl")
    with open(datafile_, "wb") as file_out:
        pickle.dump([fit_params_n, fit_params_s], file_out)

    return best_fit_n, best_fit_s, fit_params_n, fit_params_s


def fit_ratio_2exp(
    ratio_list,
    twoptfn_list,
    fit_data_list,
    src_snk_times,
    delta_t,
    tmin_choice_nucl,
    tmin_choice_sigma,
    datadir,
    fitfnc_2exp,
):
    """Fit to the three-point function with a two-exponential function, which includes parameters from the two-point functions"""
    # tmin_choice_sigma = 5
    twopt_fit_params_n, twopt_fit_params_s = fit_data_list

    # Set the parameters from the twoptfn
    A_E0i = twopt_fit_params_n[:, 0]
    A_E0f = twopt_fit_params_s[:, 0]
    A_E1i = twopt_fit_params_n[:, 0] * twopt_fit_params_n[:, 2]
    A_E1f = twopt_fit_params_s[:, 0] * twopt_fit_params_s[:, 2]
    E0i = twopt_fit_params_n[:, 1]
    E0f = twopt_fit_params_s[:, 1]
    Delta_E01i = np.exp(twopt_fit_params_n[:, 3])
    Delta_E01f = np.exp(twopt_fit_params_s[:, 3])

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
    # cvinv = np.linalg.pinv(np.cov(fitdata.T))
    cvinv = np.linalg.inv(np.cov(fitdata.T))
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
    )


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

        # axarr[icorr].grid(True)
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


def plot_ratio_fit_paper(
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

    # f, axarr = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(11, 5))
    f, axarr = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(7, 4))
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
            # label=labels[icorr],
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
            # label=rf"fit = {err_brackets(np.average(fit_param_boot[:, 0]), np.std(fit_param_boot[:, 0]))}",
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

        # axarr[icorr].grid(True, alpha=0.4)
        # axarr[icorr].legend(fontsize=15, loc="upper left")
        axarr[icorr].set_title(labels[icorr])

        axarr[icorr].set_xlabel(r"$\tau-t_{\mathrm{sep}}/2$", labelpad=14, fontsize=18)
        axarr[icorr].set_ylabel(
            r"$R(\vec{p}\, ; t_{\mathrm{sep}}, \tau)$", labelpad=5, fontsize=18
        )
        axarr[icorr].label_outer()
        # axarr[icorr].set_xlim(-src_snk_times[-1] - 1, src_snk_times[-1] + 1)
        axarr[icorr].set_xlim(plot_time2[0] - 0.5, plot_time2[src_snk_times[-1]] + 0.5)
        # axarr[icorr].set_ylim(1.104, 1.181)

    # f.suptitle(
    #     rf"{plotparam[3]} 3-point function ratio with $\hat{{\mathcal{{O}}}}=${plotparam[1]}, $\Gamma = ${plotparam[2]}, $\vec{{q}}\, ={plotparam[0][1:]}$ with two-state fit $\chi^2_{{\mathrm{{dof}}}}={redchisq:.2f}$"
    # )
    savefile = plotdir / Path(f"{title}.pdf")
    # savefile_ylim = plotdir / Path(f"{title}_ylim.pdf")
    savefile2 = plotdir / Path(f"{title}.png")
    savefile3 = plotdir / Path(f"{title}_small.png")
    savefile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(savefile)
    # plt.savefig(savefile2, dpi=500)
    plt.savefig(savefile3, dpi=100)
    # plt.ylim(1.104, 1.181)
    # plt.savefig(savefile_ylim)
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


def make_double_ratio(
    threeptfn_n2sig,
    threeptfn_sig2n,
    twoptfn_sigma_real,
    twoptfn_neutron_real,
    src_snk_time,
):
    """Make the ratio of two-point and three-point functions which produces the plateau
    This ratio uses both the nucleon to sigma transition and the sigma to nucleon transition. Referenced in Flynn 2007 paper."""

    three_point_product = np.einsum(
        "ijk,ijk->ijk",
        threeptfn_n2sig[:, : src_snk_time + 1],
        threeptfn_sig2n[:, : src_snk_time + 1],
    )
    denominator = (
        twoptfn_sigma_real[:, src_snk_time] * twoptfn_neutron_real[:, src_snk_time]
    ) ** (-1)
    ratio = np.sqrt(np.einsum("ijk,i->ijk", three_point_product, denominator))
    return ratio


def fit_3point_loop_n2sig(
    latticedir,
    resultsdir,
    plotdir,
    datadir,
    operators,
    operators_tex,
    polarizations,
    momenta,
    delta_t_list,
    tmin_choice,
    tmin_choice_zero,
):
    """Loop over the operators and momenta"""
    transition = "n2sig"
    src_snk_times = np.array([10, 13, 16])
    rel = "nr"
    for imom, mom in enumerate(momenta):
        print(f"\n{mom}")
        # ======================================================================
        # Read the two-point function data
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

        # ======================================================================
        # Read the results of the fit to the two-point functions
        kappa_combs = ["kp121040kp121040", "kp121040kp120620"]
        datafile_n = datadir / Path(
            f"{kappa_combs[0]}_{mom}_{rel}_fitlist_2pt_2exp.pkl"
        )
        with open(datafile_n, "rb") as file_in:
            fit_data_n = pickle.load(file_in)
        datafile_s = datadir / Path(
            f"{kappa_combs[1]}_p+0+0+0_{rel}_fitlist_2pt_2exp.pkl"
        )
        with open(datafile_s, "rb") as file_in:
            fit_data_s = pickle.load(file_in)

        # Pick out the chosen 2pt fn fits
        best_fit_n, best_fit_s, fit_params_n, fit_params_s = select_2pt_fit(
            [fit_data_n, fit_data_s],
            tmin_choice[imom],
            tmin_choice_zero,
            datadir,
            mom,
            transition,
        )

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

                for ir, reim in enumerate(["real", "imag"]):
                    # if False:
                    print(reim)
                    # ======================================================================
                    # fit to the ratio of 3pt and 2pt functions with a two-exponential function
                    fitfnc_2exp = ff.threept_ratio
                    (
                        fit_param_ratio_boot,
                        ratio_fit_boot,
                        fit_param_ratio_avg,
                        redchisq_ratio,
                    ) = fit_ratio_2exp(
                        full_ratio_list_reim[ir],
                        np.array([twoptfn_neutron, twoptfn_sigma]),
                        # [fit_data_n, fit_data_s],
                        [fit_params_n, fit_params_s],
                        src_snk_times,
                        delta_t_list[imom],
                        tmin_choice[imom],
                        tmin_choice_zero,
                        datadir,
                        fitfnc_2exp,
                    )
                    fit_params_ratio = [
                        fit_param_ratio_boot,
                        ratio_fit_boot,
                        fit_param_ratio_avg,
                        redchisq_ratio,
                        best_fit_n,
                        best_fit_s,
                    ]

                    # Save the fit results to pickle files
                    datafile_ratio = datadir / Path(
                        f"{mom}_{operator}_{pol}_{rel}_{reim}_3pt_ratio_fit_n2sig.pkl"
                    )
                    with open(datafile_ratio, "wb") as file_out:
                        pickle.dump(fit_params_ratio, file_out)

    return


def fit_3point_loop_sig2n(
    latticedir,
    resultsdir,
    plotdir,
    datadir,
    operators,
    operators_tex,
    polarizations,
    momenta,
    delta_t_list,
    tmin_choice,
    tmin_choice_zero,
):
    """Loop over the operators and momenta"""
    transition = "sig2n"
    src_snk_times = np.array([10, 13, 16])
    rel = "nr"
    for imom, mom in enumerate(momenta):
        print(f"\n{mom}")
        # ======================================================================
        # Read the two-point function data
        twoptfn_filename_sigma = latticedir / Path(
            f"mass_spectrum/baryon_qcdsf/barspec/32x64/unpreconditioned_slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/{mom}/barspec_nucleon_{rel}_500cfgs.pickle"
        )
        twoptfn_filename_neutron = latticedir / Path(
            f"mass_spectrum/baryon_qcdsf/barspec/32x64/unpreconditioned_slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/barspec_nucleon_{rel}_500cfgs.pickle"
        )
        twoptfn_sigma = read_pickle(twoptfn_filename_sigma, nboot=500, nbin=1)
        twoptfn_neutron = read_pickle(twoptfn_filename_neutron, nboot=500, nbin=1)

        twoptfn_sigma_real = twoptfn_sigma[:, :, 0]
        twoptfn_neutron_real = twoptfn_neutron[:, :, 0]

        # ======================================================================
        # Read the results of the fit to the two-point functions
        kappa_combs = ["kp121040kp121040", "kp121040kp120620"]
        datafile_n = datadir / Path(
            f"{kappa_combs[0]}_p+0+0+0_{rel}_fitlist_2pt_2exp.pkl"
        )
        with open(datafile_n, "rb") as file_in:
            fit_data_n = pickle.load(file_in)
        datafile_s = datadir / Path(
            f"{kappa_combs[1]}_{mom}_{rel}_fitlist_2pt_2exp.pkl"
        )
        with open(datafile_s, "rb") as file_in:
            fit_data_s = pickle.load(file_in)

        # Pick out the chosen 2pt fn fits
        best_fit_n, best_fit_s, fit_params_n, fit_params_s = select_2pt_fit(
            [fit_data_n, fit_data_s],
            tmin_choice[imom],
            tmin_choice_zero,
            datadir,
            mom,
            transition,
        )

        for iop, operator in enumerate(operators):
            print(f"\n{operator}")
            for ipol, pol in enumerate(polarizations):
                print(f"\n{pol}")
                # Read in the 3pt function data
                threeptfn_pickle_t10 = latticedir / Path(
                    f"sig2n/bar3ptfn_t10/bar3ptfn/32x64/unpreconditioned_slrc/kp120620tkp121040_kp121040/NUCL_D_{pol}_NONREL_gI_t10/sh_gij_p21_90-sh_gij_p21_90/{mom}/bar3ptfn_{operator}_500cfgs.pickle"
                )
                threeptfn_pickle_t13 = latticedir / Path(
                    f"sig2n/bar3ptfn_t13/bar3ptfn/32x64/unpreconditioned_slrc/kp120620tkp121040_kp121040/NUCL_D_{pol}_NONREL_gI_t13/sh_gij_p21_90-sh_gij_p21_90/{mom}/bar3ptfn_{operator}_500cfgs.pickle"
                )
                threeptfn_pickle_t16 = latticedir / Path(
                    f"sig2n/bar3ptfn_t16/bar3ptfn/32x64/unpreconditioned_slrc/kp120620tkp121040_kp121040/NUCL_D_{pol}_NONREL_gI_t16/sh_gij_p21_90-sh_gij_p21_90/{mom}/bar3ptfn_{operator}_500cfgs.pickle"
                )
                threeptfn_t10 = read_pickle(threeptfn_pickle_t10, nboot=500, nbin=1)
                threeptfn_t13 = read_pickle(threeptfn_pickle_t13, nboot=500, nbin=1)
                threeptfn_t16 = read_pickle(threeptfn_pickle_t16, nboot=500, nbin=1)

                # ======================================================================
                # Construct the full ratio of 3pt and 2pt functions
                ratio_full_t10 = make_full_ratio(
                    threeptfn_t10, twoptfn_neutron_real, twoptfn_sigma_real, 10
                )
                ratio_full_t13 = make_full_ratio(
                    threeptfn_t13, twoptfn_neutron_real, twoptfn_sigma_real, 13
                )
                ratio_full_t16 = make_full_ratio(
                    threeptfn_t16, twoptfn_neutron_real, twoptfn_sigma_real, 16
                )
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

                for ir, reim in enumerate(["real", "imag"]):
                    # if False:
                    print(reim)
                    # ======================================================================
                    # fit to the ratio of 3pt and 2pt functions with a two-exponential function
                    fitfnc_2exp = ff.threept_ratio
                    (
                        fit_param_ratio_boot,
                        ratio_fit_boot,
                        fit_param_ratio_avg,
                        redchisq_ratio,
                    ) = fit_ratio_2exp(
                        full_ratio_list_reim[ir],
                        np.array([twoptfn_sigma, twoptfn_neutron]),
                        [fit_params_n, fit_params_s],
                        src_snk_times,
                        delta_t_list[imom],
                        tmin_choice[imom],
                        tmin_choice_zero,
                        datadir,
                        fitfnc_2exp,
                    )
                    fit_params_ratio = [
                        fit_param_ratio_boot,
                        ratio_fit_boot,
                        fit_param_ratio_avg,
                        redchisq_ratio,
                        best_fit_n,
                        best_fit_s,
                    ]

                    # Save the fit results to pickle files
                    datafile_ratio = datadir / Path(
                        f"{mom}_{operator}_{pol}_{rel}_{reim}_3pt_ratio_fit_sig2n.pkl"
                    )
                    with open(datafile_ratio, "wb") as file_out:
                        pickle.dump(fit_params_ratio, file_out)

    return


def fit_3point_zeromom(
    latticedir,
    resultsdir,
    plotdir,
    datadir,
    operators,
    operators_tex,
    polarizations,
    momenta,
    delta_t_list,
    tmin_choice,
):
    """Loop over the operators and momenta"""

    src_snk_times = np.array([10, 13, 16])
    rel = "nr"
    for imom, mom in enumerate(momenta):
        print(f"\n{mom}")
        # ======================================================================
        # Read the two-point function data
        twoptfn_filename_sigma = latticedir / Path(
            f"mass_spectrum/baryon_qcdsf/barspec/32x64/unpreconditioned_slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/{mom}/barspec_nucleon_{rel}_500cfgs.pickle"
        )
        twoptfn_filename_neutron = latticedir / Path(
            f"mass_spectrum/baryon_qcdsf/barspec/32x64/unpreconditioned_slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/{mom}/barspec_nucleon_{rel}_500cfgs.pickle"
        )
        twoptfn_sigma = read_pickle(twoptfn_filename_sigma, nboot=500, nbin=1)
        twoptfn_neutron = read_pickle(twoptfn_filename_neutron, nboot=500, nbin=1)

        twoptfn_sigma_real = twoptfn_sigma[:, :, 0]
        twoptfn_neutron_real = twoptfn_neutron[:, :, 0]

        # ======================================================================
        # Read the results of the fit to the two-point functions
        kappa_combs = ["kp121040kp121040", "kp121040kp120620"]
        datafile_n = datadir / Path(
            f"{kappa_combs[0]}_{mom}_{rel}_fitlist_2pt_2exp.pkl"
        )
        with open(datafile_n, "rb") as file_in:
            fit_data_n = pickle.load(file_in)
        datafile_s = datadir / Path(
            f"{kappa_combs[1]}_{mom}_{rel}_fitlist_2pt_2exp.pkl"
        )
        with open(datafile_s, "rb") as file_in:
            fit_data_s = pickle.load(file_in)

        # Pick out the chosen 2pt fn fits
        best_fit_n, best_fit_s, fit_params_n, fit_params_s = select_2pt_fit(
            [fit_data_n, fit_data_s],
            tmin_choice[imom],
            tmin_choice[imom],
            datadir,
            mom,
            "zeromom",
        )

        for iop, operator in enumerate(operators):
            print(f"\n{operator}")
            for ipol, pol in enumerate(polarizations):
                print(f"\n{pol}")
                # Read in the 3pt function data
                threeptfn_n2sig_pickle_t10 = latticedir / Path(
                    f"bar3ptfn_t10/bar3ptfn/32x64/unpreconditioned_slrc/kp121040tkp120620_kp121040/NUCL_D_{pol}_NONREL_gI_t10/sh_gij_p21_90-sh_gij_p21_90/{mom}/bar3ptfn_{operator}_500cfgs.pickle"
                )
                threeptfn_n2sig_pickle_t13 = latticedir / Path(
                    f"bar3ptfn_t13/bar3ptfn/32x64/unpreconditioned_slrc/kp121040tkp120620_kp121040/NUCL_D_{pol}_NONREL_gI_t13/sh_gij_p21_90-sh_gij_p21_90/{mom}/bar3ptfn_{operator}_500cfgs.pickle"
                )
                threeptfn_n2sig_pickle_t16 = latticedir / Path(
                    f"bar3ptfn_t16/bar3ptfn/32x64/unpreconditioned_slrc/kp121040tkp120620_kp121040/NUCL_D_{pol}_NONREL_gI_t16/sh_gij_p21_90-sh_gij_p21_90/{mom}/bar3ptfn_{operator}_500cfgs.pickle"
                )
                threeptfn_n2sig_t10 = read_pickle(
                    threeptfn_n2sig_pickle_t10, nboot=500, nbin=1
                )
                threeptfn_n2sig_t13 = read_pickle(
                    threeptfn_n2sig_pickle_t13, nboot=500, nbin=1
                )
                threeptfn_n2sig_t16 = read_pickle(
                    threeptfn_n2sig_pickle_t16, nboot=500, nbin=1
                )

                threeptfn_sig2n_pickle_t10 = latticedir / Path(
                    f"sig2n/bar3ptfn_t10/bar3ptfn/32x64/unpreconditioned_slrc/kp120620tkp121040_kp121040/NUCL_D_{pol}_NONREL_gI_t10/sh_gij_p21_90-sh_gij_p21_90/{mom}/bar3ptfn_{operator}_500cfgs.pickle"
                )
                threeptfn_sig2n_pickle_t13 = latticedir / Path(
                    f"sig2n/bar3ptfn_t13/bar3ptfn/32x64/unpreconditioned_slrc/kp120620tkp121040_kp121040/NUCL_D_{pol}_NONREL_gI_t13/sh_gij_p21_90-sh_gij_p21_90/{mom}/bar3ptfn_{operator}_500cfgs.pickle"
                )
                threeptfn_sig2n_pickle_t16 = latticedir / Path(
                    f"sig2n/bar3ptfn_t16/bar3ptfn/32x64/unpreconditioned_slrc/kp120620tkp121040_kp121040/NUCL_D_{pol}_NONREL_gI_t16/sh_gij_p21_90-sh_gij_p21_90/{mom}/bar3ptfn_{operator}_500cfgs.pickle"
                )
                threeptfn_sig2n_t10 = read_pickle(
                    threeptfn_sig2n_pickle_t10, nboot=500, nbin=1
                )
                threeptfn_sig2n_t13 = read_pickle(
                    threeptfn_sig2n_pickle_t13, nboot=500, nbin=1
                )
                threeptfn_sig2n_t16 = read_pickle(
                    threeptfn_sig2n_pickle_t16, nboot=500, nbin=1
                )

                # ======================================================================
                # Construct the full ratio of 3pt and 2pt functions
                ratio_full_t10 = make_double_ratio(
                    threeptfn_n2sig_t10,
                    threeptfn_sig2n_t10,
                    twoptfn_sigma_real,
                    twoptfn_neutron_real,
                    10,
                )
                ratio_full_t13 = make_double_ratio(
                    threeptfn_n2sig_t13,
                    threeptfn_sig2n_t13,
                    twoptfn_sigma_real,
                    twoptfn_neutron_real,
                    13,
                )
                ratio_full_t16 = make_double_ratio(
                    threeptfn_n2sig_t16,
                    threeptfn_sig2n_t16,
                    twoptfn_sigma_real,
                    twoptfn_neutron_real,
                    16,
                )
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

                for ir, reim in enumerate(["real"]):
                    print(reim)
                    # ======================================================================
                    # fit to the ratio of 3pt and 2pt functions with a two-exponential function
                    fitfnc_2exp = ff.double_threept_ratio
                    (
                        fit_param_ratio_boot,
                        ratio_fit_boot,
                        fit_param_ratio_avg,
                        redchisq_ratio,
                    ) = fit_ratio_2exp(
                        full_ratio_list_reim[ir],
                        np.array([twoptfn_neutron, twoptfn_sigma]),
                        [fit_params_n, fit_params_s],
                        src_snk_times,
                        delta_t_list[imom],
                        tmin_choice[imom],
                        tmin_choice[imom],
                        datadir,
                        fitfnc_2exp,
                    )
                    fit_params_ratio = [
                        fit_param_ratio_boot,
                        ratio_fit_boot,
                        fit_param_ratio_avg,
                        redchisq_ratio,
                        best_fit_n,
                        best_fit_s,
                    ]

                    # Save the fit results to pickle files
                    # print(f"{fit_params_ratio[0][:,0]=}")
                    datafile_ratio = datadir / Path(
                        f"{mom}_{operator}_{pol}_{rel}_{reim}_double_3pt_ratio_fit.pkl"
                    )
                    with open(datafile_ratio, "wb") as file_out:
                        pickle.dump(fit_params_ratio, file_out)

                    # ======================================================================
                    # Plot the results of the fit to the ratio
                    plot_ratio_fit(
                        full_ratio_list_reim[ir],
                        ratio_fit_boot,
                        delta_t_list[imom],
                        src_snk_times,
                        redchisq_ratio,
                        fit_param_ratio_boot,
                        plotdir,
                        [mom, operators_tex[iop], pol, reim],
                        title=f"{mom}/{pol}/double_ratio_fit_{reim}_{operator}",
                    )
                    # ======================================================================
                    # Plot the results of the fit to the ratio
                    plot_ratio_fit_paper(
                        full_ratio_list_reim[ir],
                        ratio_fit_boot,
                        delta_t_list[imom],
                        src_snk_times,
                        redchisq_ratio,
                        fit_param_ratio_boot,
                        plotdir,
                        [mom, operators_tex[iop], pol, reim],
                        title=f"{mom}/{pol}/double_ratio_fit_{reim}_{operator}_{mom}_paper",
                    )
    return


def plot_3point_loop_n2sig(
    latticedir,
    resultsdir,
    plotdir,
    datadir,
    operators,
    operators_tex,
    polarizations,
    momenta,
    delta_t_list,
    tmin_choice,
    tmin_choice_zero,
):
    """Loop over the operators and momenta"""

    src_snk_times = np.array([10, 13, 16])
    rel = "nr"
    for imom, mom in enumerate(momenta):
        print(f"\n{mom}")
        # ======================================================================
        # Read the two-point function data
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

                for ir, reim in enumerate(["real", "imag"]):
                    print(reim)
                    # ======================================================================
                    # read the fit results to pickle files
                    datafile_ratio = datadir / Path(
                        f"{mom}_{operator}_{pol}_{rel}_{reim}_3pt_ratio_fit_n2sig.pkl"
                    )
                    with open(datafile_ratio, "rb") as file_in:
                        fit_params_ratio = pickle.load(file_in)

                    (
                        fit_param_ratio_boot,
                        ratio_fit_boot,
                        fit_param_ratio_avg,
                        redchisq_ratio,
                        best_fit_n,
                        best_fit_s,
                    ) = fit_params_ratio

                    # ======================================================================
                    # Plot the results of the fit to the ratio
                    plot_ratio_fit(
                        full_ratio_list_reim[ir],
                        ratio_fit_boot,
                        delta_t_list[imom],
                        src_snk_times,
                        redchisq_ratio,
                        fit_param_ratio_boot,
                        plotdir,
                        [mom, operators_tex[iop], pol, reim],
                        title=f"{mom}/{pol}/ratio_fit_{reim}_{operator}_n2sig",
                    )
                    plot_ratio_fit_paper(
                        full_ratio_list_reim[ir],
                        ratio_fit_boot,
                        delta_t_list[imom],
                        src_snk_times,
                        redchisq_ratio,
                        fit_param_ratio_boot,
                        plotdir,
                        [mom, operators_tex[iop], pol, reim],
                        title=f"{mom}/{pol}/ratio_fit_{reim}_{operator}_{mom}_n2sig_paper",
                    )

    return


def plot_3point_loop_sig2n(
    latticedir,
    resultsdir,
    plotdir,
    datadir,
    operators,
    operators_tex,
    polarizations,
    momenta,
    delta_t_list,
    tmin_choice,
    tmin_choice_zero,
):
    """Loop over the operators and momenta"""

    src_snk_times = np.array([10, 13, 16])
    rel = "nr"
    for imom, mom in enumerate(momenta):
        print(f"\n{mom}")
        # ======================================================================
        # Read the two-point function data
        twoptfn_filename_sigma = latticedir / Path(
            f"mass_spectrum/baryon_qcdsf/barspec/32x64/unpreconditioned_slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/{mom}/barspec_nucleon_{rel}_500cfgs.pickle"
        )
        twoptfn_filename_neutron = latticedir / Path(
            f"mass_spectrum/baryon_qcdsf/barspec/32x64/unpreconditioned_slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/barspec_nucleon_{rel}_500cfgs.pickle"
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
                    f"sig2n/bar3ptfn_t10/bar3ptfn/32x64/unpreconditioned_slrc/kp120620tkp121040_kp121040/NUCL_D_{pol}_NONREL_gI_t10/sh_gij_p21_90-sh_gij_p21_90/{mom}/bar3ptfn_{operator}_500cfgs.pickle"
                )
                threeptfn_pickle_t13 = latticedir / Path(
                    f"sig2n/bar3ptfn_t13/bar3ptfn/32x64/unpreconditioned_slrc/kp120620tkp121040_kp121040/NUCL_D_{pol}_NONREL_gI_t13/sh_gij_p21_90-sh_gij_p21_90/{mom}/bar3ptfn_{operator}_500cfgs.pickle"
                )
                threeptfn_pickle_t16 = latticedir / Path(
                    f"sig2n/bar3ptfn_t16/bar3ptfn/32x64/unpreconditioned_slrc/kp120620tkp121040_kp121040/NUCL_D_{pol}_NONREL_gI_t16/sh_gij_p21_90-sh_gij_p21_90/{mom}/bar3ptfn_{operator}_500cfgs.pickle"
                )
                threeptfn_t10 = read_pickle(threeptfn_pickle_t10, nboot=500, nbin=1)
                threeptfn_t13 = read_pickle(threeptfn_pickle_t13, nboot=500, nbin=1)
                threeptfn_t16 = read_pickle(threeptfn_pickle_t16, nboot=500, nbin=1)

                # ======================================================================
                # Construct the full ratio of 3pt and 2pt functions
                ratio_full_t10 = make_full_ratio(
                    threeptfn_t10, twoptfn_neutron_real, twoptfn_sigma_real, 10
                )
                ratio_full_t13 = make_full_ratio(
                    threeptfn_t13, twoptfn_neutron_real, twoptfn_sigma_real, 13
                )
                ratio_full_t16 = make_full_ratio(
                    threeptfn_t16, twoptfn_neutron_real, twoptfn_sigma_real, 16
                )
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

                for ir, reim in enumerate(["real", "imag"]):
                    print(reim)
                    # ======================================================================
                    # read the fit results to pickle files
                    datafile_ratio = datadir / Path(
                        f"{mom}_{operator}_{pol}_{rel}_{reim}_3pt_ratio_fit_sig2n.pkl"
                    )
                    with open(datafile_ratio, "rb") as file_in:
                        fit_params_ratio = pickle.load(file_in)

                    (
                        fit_param_ratio_boot,
                        ratio_fit_boot,
                        fit_param_ratio_avg,
                        redchisq_ratio,
                        best_fit_n,
                        best_fit_s,
                    ) = fit_params_ratio

                    # ======================================================================
                    # Plot the results of the fit to the ratio
                    plot_ratio_fit(
                        full_ratio_list_reim[ir],
                        ratio_fit_boot,
                        delta_t_list[imom],
                        src_snk_times,
                        redchisq_ratio,
                        fit_param_ratio_boot,
                        plotdir,
                        [mom, operators_tex[iop], pol, reim],
                        title=f"{mom}/{pol}/ratio_fit_{reim}_{operator}_sig2n",
                    )
                    plot_ratio_fit_paper(
                        full_ratio_list_reim[ir],
                        ratio_fit_boot,
                        delta_t_list[imom],
                        src_snk_times,
                        redchisq_ratio,
                        fit_param_ratio_boot,
                        plotdir,
                        [mom, operators_tex[iop], pol, reim],
                        title=f"{mom}/{pol}/ratio_fit_{reim}_{operator}_{mom}_sig2n_paper",
                    )

    return


def plot_3point_zeromom(
    latticedir,
    resultsdir,
    plotdir,
    datadir,
    operators,
    operators_tex,
    polarizations,
    momenta,
    delta_t_list,
    tmin_choice,
):
    """Loop over the operators and momenta"""

    src_snk_times = np.array([10, 13, 16])
    rel = "nr"
    for imom, mom in enumerate(momenta):
        print(f"\n{mom}")
        # ======================================================================
        # Read the two-point function data
        twoptfn_filename_sigma = latticedir / Path(
            f"mass_spectrum/baryon_qcdsf/barspec/32x64/unpreconditioned_slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/{mom}/barspec_nucleon_{rel}_500cfgs.pickle"
        )
        twoptfn_filename_neutron = latticedir / Path(
            f"mass_spectrum/baryon_qcdsf/barspec/32x64/unpreconditioned_slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/{mom}/barspec_nucleon_{rel}_500cfgs.pickle"
        )
        twoptfn_sigma = read_pickle(twoptfn_filename_sigma, nboot=500, nbin=1)
        twoptfn_neutron = read_pickle(twoptfn_filename_neutron, nboot=500, nbin=1)

        twoptfn_sigma_real = twoptfn_sigma[:, :, 0]
        twoptfn_neutron_real = twoptfn_neutron[:, :, 0]

        # ======================================================================
        # Read the results of the fit to the two-point functions
        kappa_combs = ["kp121040kp121040", "kp121040kp120620"]
        datafile_n = datadir / Path(
            f"{kappa_combs[0]}_{mom}_{rel}_fitlist_2pt_2exp.pkl"
        )
        with open(datafile_n, "rb") as file_in:
            fit_data_n = pickle.load(file_in)
        datafile_s = datadir / Path(
            f"{kappa_combs[1]}_{mom}_{rel}_fitlist_2pt_2exp.pkl"
        )
        with open(datafile_s, "rb") as file_in:
            fit_data_s = pickle.load(file_in)

        # Pick out the chosen 2pt fn fits
        best_fit_n, best_fit_s, fit_params_n, fit_params_s = select_2pt_fit(
            [fit_data_n, fit_data_s],
            tmin_choice[imom],
            tmin_choice[imom],
            datadir,
            mom,
            "zeromom",
        )

        for iop, operator in enumerate(operators):
            print(f"\n{operator}")
            for ipol, pol in enumerate(polarizations):
                print(f"\n{pol}")
                # Read in the 3pt function data
                threeptfn_n2sig_pickle_t10 = latticedir / Path(
                    f"bar3ptfn_t10/bar3ptfn/32x64/unpreconditioned_slrc/kp121040tkp120620_kp121040/NUCL_D_{pol}_NONREL_gI_t10/sh_gij_p21_90-sh_gij_p21_90/{mom}/bar3ptfn_{operator}_500cfgs.pickle"
                )
                threeptfn_n2sig_pickle_t13 = latticedir / Path(
                    f"bar3ptfn_t13/bar3ptfn/32x64/unpreconditioned_slrc/kp121040tkp120620_kp121040/NUCL_D_{pol}_NONREL_gI_t13/sh_gij_p21_90-sh_gij_p21_90/{mom}/bar3ptfn_{operator}_500cfgs.pickle"
                )
                threeptfn_n2sig_pickle_t16 = latticedir / Path(
                    f"bar3ptfn_t16/bar3ptfn/32x64/unpreconditioned_slrc/kp121040tkp120620_kp121040/NUCL_D_{pol}_NONREL_gI_t16/sh_gij_p21_90-sh_gij_p21_90/{mom}/bar3ptfn_{operator}_500cfgs.pickle"
                )
                threeptfn_n2sig_t10 = read_pickle(
                    threeptfn_n2sig_pickle_t10, nboot=500, nbin=1
                )
                threeptfn_n2sig_t13 = read_pickle(
                    threeptfn_n2sig_pickle_t13, nboot=500, nbin=1
                )
                threeptfn_n2sig_t16 = read_pickle(
                    threeptfn_n2sig_pickle_t16, nboot=500, nbin=1
                )

                threeptfn_sig2n_pickle_t10 = latticedir / Path(
                    f"sig2n/bar3ptfn_t10/bar3ptfn/32x64/unpreconditioned_slrc/kp120620tkp121040_kp121040/NUCL_D_{pol}_NONREL_gI_t10/sh_gij_p21_90-sh_gij_p21_90/{mom}/bar3ptfn_{operator}_500cfgs.pickle"
                )
                threeptfn_sig2n_pickle_t13 = latticedir / Path(
                    f"sig2n/bar3ptfn_t13/bar3ptfn/32x64/unpreconditioned_slrc/kp120620tkp121040_kp121040/NUCL_D_{pol}_NONREL_gI_t13/sh_gij_p21_90-sh_gij_p21_90/{mom}/bar3ptfn_{operator}_500cfgs.pickle"
                )
                threeptfn_sig2n_pickle_t16 = latticedir / Path(
                    f"sig2n/bar3ptfn_t16/bar3ptfn/32x64/unpreconditioned_slrc/kp120620tkp121040_kp121040/NUCL_D_{pol}_NONREL_gI_t16/sh_gij_p21_90-sh_gij_p21_90/{mom}/bar3ptfn_{operator}_500cfgs.pickle"
                )
                threeptfn_sig2n_t10 = read_pickle(
                    threeptfn_sig2n_pickle_t10, nboot=500, nbin=1
                )
                threeptfn_sig2n_t13 = read_pickle(
                    threeptfn_sig2n_pickle_t13, nboot=500, nbin=1
                )
                threeptfn_sig2n_t16 = read_pickle(
                    threeptfn_sig2n_pickle_t16, nboot=500, nbin=1
                )

                # ======================================================================
                # Construct the full ratio of 3pt and 2pt functions
                ratio_full_t10 = make_double_ratio(
                    threeptfn_n2sig_t10,
                    threeptfn_sig2n_t10,
                    twoptfn_sigma_real,
                    twoptfn_neutron_real,
                    10,
                )
                ratio_full_t13 = make_double_ratio(
                    threeptfn_n2sig_t13,
                    threeptfn_sig2n_t13,
                    twoptfn_sigma_real,
                    twoptfn_neutron_real,
                    13,
                )
                ratio_full_t16 = make_double_ratio(
                    threeptfn_n2sig_t16,
                    threeptfn_sig2n_t16,
                    twoptfn_sigma_real,
                    twoptfn_neutron_real,
                    16,
                )
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

                for ir, reim in enumerate(["real"]):
                    print(reim)
                    # ======================================================================
                    # fit to the ratio of 3pt and 2pt functions with a two-exponential function
                    fitfnc_2exp = ff.double_threept_ratio
                    (
                        fit_param_ratio_boot,
                        ratio_fit_boot,
                        fit_param_ratio_avg,
                        redchisq_ratio,
                    ) = fit_ratio_2exp(
                        full_ratio_list_reim[ir],
                        np.array([twoptfn_neutron, twoptfn_sigma]),
                        [fit_params_n, fit_params_s],
                        src_snk_times,
                        delta_t_list[imom],
                        tmin_choice[imom],
                        tmin_choice[imom],
                        datadir,
                        fitfnc_2exp,
                    )
                    fit_params_ratio = [
                        fit_param_ratio_boot,
                        ratio_fit_boot,
                        fit_param_ratio_avg,
                        redchisq_ratio,
                        best_fit_n,
                        best_fit_s,
                    ]

                    # Save the fit results to pickle files
                    # print(f"{fit_params_ratio[0][:,0]=}")
                    datafile_ratio = datadir / Path(
                        f"{mom}_{operator}_{pol}_{rel}_{reim}_double_3pt_ratio_fit.pkl"
                    )
                    with open(datafile_ratio, "wb") as file_out:
                        pickle.dump(fit_params_ratio, file_out)

                    # ======================================================================
                    # Plot the results of the fit to the ratio
                    plot_ratio_fit(
                        full_ratio_list_reim[ir],
                        ratio_fit_boot,
                        delta_t_list[imom],
                        src_snk_times,
                        redchisq_ratio,
                        fit_param_ratio_boot,
                        plotdir,
                        [mom, operators_tex[iop], pol, reim],
                        title=f"{mom}/{pol}/double_ratio_fit_{reim}_{operator}",
                    )
                    # ======================================================================
                    # Plot the results of the fit to the ratio
                    plot_ratio_fit_paper(
                        full_ratio_list_reim[ir],
                        ratio_fit_boot,
                        delta_t_list[imom],
                        src_snk_times,
                        redchisq_ratio,
                        fit_param_ratio_boot,
                        plotdir,
                        [mom, operators_tex[iop], pol, reim],
                        title=f"{mom}/{pol}/double_ratio_fit_{reim}_{operator}_{mom}_paper",
                    )
    return


def get_Qsquared_values(datadir, tmin_choices_nucl, tmin_choices_sigm):
    """Loop over the momenta and read the two-point functions, then get the energies from them and use those to calculate the Q^2 values."""

    a = 0.074
    L = 32
    src_snk_times = np.array([10, 13, 16])
    rel = "nr"
    kappa_combs = ["kp121040kp121040", "kp121040kp120620"]
    momenta = ["p+0+0+0", "p+1+0+0", "p+1+1+0"]
    momenta_values = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]])

    # Load the fit values for the Q^2=0 correlators
    datafile_n = datadir / Path(
        f"{kappa_combs[0]}_{momenta[0]}_{rel}_fitlist_2pt_2exp.pkl"
    )
    with open(datafile_n, "rb") as file_in:
        fit_data_n_mom0 = pickle.load(file_in)
    datafile_s = datadir / Path(
        f"{kappa_combs[1]}_{momenta[0]}_{rel}_fitlist_2pt_2exp.pkl"
    )
    with open(datafile_s, "rb") as file_in:
        fit_data_s_mom0 = pickle.load(file_in)

    # Extract the chosen fit's energy from the data
    # Neutron
    fit_times_n = [fit["x"] for fit in fit_data_n_mom0]
    chosen_time_n = np.where(
        [times[0] == tmin_choices_nucl[0] for times in fit_times_n]
    )[0][0]
    best_fit_n_mom0 = fit_data_n_mom0[chosen_time_n]
    energy_n_mom0 = np.average(best_fit_n_mom0["param"][:, 1])
    # Sigma
    fit_times_s = [fit["x"] for fit in fit_data_s_mom0]
    chosen_time_s = np.where(
        [times[0] == tmin_choices_sigm[0] for times in fit_times_s]
    )[0][0]
    best_fit_s_mom0 = fit_data_s_mom0[chosen_time_s]
    energy_s_mom0 = np.average(best_fit_s_mom0["param"][:, 1])

    qsquared_sig2n_list = []
    qsquared_n2sig_list = []
    for imom, mom in enumerate(momenta):
        # Load the fit values for correlators at the given momentum
        datafile_n = datadir / Path(
            f"{kappa_combs[0]}_{mom}_{rel}_fitlist_2pt_2exp.pkl"
        )
        with open(datafile_n, "rb") as file_in:
            fit_data_n = pickle.load(file_in)
        datafile_s = datadir / Path(
            f"{kappa_combs[1]}_{mom}_{rel}_fitlist_2pt_2exp.pkl"
        )
        with open(datafile_s, "rb") as file_in:
            fit_data_s = pickle.load(file_in)

        # Extract the chosen fit's energy from the data
        # Neutron
        fit_times_n = [fit["x"] for fit in fit_data_n]
        chosen_time_n = np.where(
            [times[0] == tmin_choices_nucl[imom] for times in fit_times_n]
        )[0][0]
        best_fit_n = fit_data_n[chosen_time_n]
        energy_n = np.average(best_fit_n["param"][:, 1])
        # Sigma
        fit_times_s = [fit["x"] for fit in fit_data_s]
        chosen_time_s = np.where(
            [times[0] == tmin_choices_sigm[imom] for times in fit_times_s]
        )[0][0]
        best_fit_s = fit_data_s[chosen_time_s]
        energy_s = np.average(best_fit_s["param"][:, 1])

        sig2n_Qsq = Q_squared_energies(
            energy_s, energy_n_mom0, momenta_values[imom], momenta_values[0], L, a
        )
        n2sig_Qsq = Q_squared_energies(
            energy_n, energy_s_mom0, momenta_values[imom], momenta_values[0], L, a
        )
        qsquared_sig2n_list.append(sig2n_Qsq)
        qsquared_n2sig_list.append(n2sig_Qsq)
        print(f"{sig2n_Qsq=}")
        print(f"{n2sig_Qsq=}")

    qsquared_sig2n_list = np.array(qsquared_sig2n_list)
    qsquared_n2sig_list = np.array(qsquared_n2sig_list)

    # Save the data to a file
    datafile_sig2n = datadir / Path(f"Qsquared_sig2n.pkl")
    datafile_n2sig = datadir / Path(f"Qsquared_n2sig.pkl")
    with open(datafile_sig2n, "wb") as file_out:
        pickle.dump(qsquared_sig2n_list, file_out)
    with open(datafile_n2sig, "wb") as file_out:
        pickle.dump(qsquared_n2sig_list, file_out)

    return qsquared_sig2n_list, qsquared_n2sig_list


def Q_squared_energies(E1, E2, n1, n2, L, a):
    """Returns Q^2 between two particles with momentum and twisted BC's
    n1, n2 are arrays which contain the fourier momenta for the first and second particle.
    L is the spatial lattice extent
    a is the lattice spacing
    """
    energydiff = np.sqrt(E2**2) - np.sqrt(E1**2)
    qvector_diff = ((2 * n2) - (2 * n1)) * (np.pi / L)
    Qsquared = (
        -1
        * (energydiff**2 - np.dot(qvector_diff, qvector_diff))
        * (0.1973**2)
        / (a**2)
    )
    return Qsquared


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

    # ======================================================================
    # Read in the three point function data
    operators_tex = [
        "$\gamma_1$",
        "$\gamma_2$",
        "$\gamma_3$",
        "$\gamma_4$",
    ]
    operators = [
        "g0",
        "g1",
        "g2",
        "g3",
    ]
    polarizations = ["UNPOL", "POL"]
    momenta = ["p+0+0+0", "p+1+0+0", "p+1+1+0"]
    delta_t_list = [5, 5, 5]
    # tmin_choice = [5, 5, 5]
    tmin_choice = [4, 4, 4]
    # tmin_choice = [7, 7, 7]

    # ======================================================================
    # Calculate the Q^2 values for each of the n2sig and sig2n transitions and momenta
    # Then save these to a file.
    Qsquared_values_sig2n, Qsquared_values_n2sig = get_Qsquared_values(
        datadir, tmin_choice, tmin_choice
    )

    fit = False
    if fit:
        # ======================================================================
        # Construct a ratio with two 3pt functions and fit it for the zero momentum transfer case
        operators_0 = ["g3"]
        operators_tex_0 = ["$\gamma_4$"]
        polarizations_0 = ["UNPOL"]
        momenta_0 = ["p+0+0+0"]
        fit_3point_zeromom(
            latticedir,
            resultsdir,
            plotdir,
            datadir,
            operators_0,
            operators_tex_0,
            polarizations_0,
            momenta_0,
            delta_t_list,
            tmin_choice,
        )

        # ======================================================================
        # Construct a ratio with 3pt and 2pt functions and fit it for the sigma to neutron transition
        # Only for non-zero momentum
        fit_3point_loop_sig2n(
            latticedir,
            resultsdir,
            plotdir,
            datadir,
            operators,
            operators_tex,
            polarizations,
            momenta[1:],
            delta_t_list[1:],
            tmin_choice[1:],
            tmin_choice[0],
        )

        # ======================================================================
        # Construct a ratio with 3pt and 2pt functions and fit it for the neutron to sigma transition
        # Only for non-zero momentum
        fit_3point_loop_n2sig(
            latticedir,
            resultsdir,
            plotdir,
            datadir,
            operators,
            operators_tex,
            polarizations,
            momenta[1:],
            delta_t_list[1:],
            tmin_choice[1:],
            tmin_choice[0],
        )
    else:
        # ======================================================================
        # plot the results of the three-point fn ratio fits
        plot_3point_loop_sig2n(
            latticedir,
            resultsdir,
            plotdir,
            datadir,
            operators,
            operators_tex,
            polarizations,
            momenta[1:],
            delta_t_list[1:],
            tmin_choice[1:],
            tmin_choice[0],
        )
        plot_3point_loop_n2sig(
            latticedir,
            resultsdir,
            plotdir,
            datadir,
            operators,
            operators_tex,
            polarizations,
            momenta[1:],
            delta_t_list[1:],
            tmin_choice[1:],
            tmin_choice[0],
        )
        plot_3point_zeromom(
            latticedir,
            resultsdir,
            plotdir,
            datadir,
            operators_0,
            operators_tex_0,
            polarizations_0,
            momenta_0,
            delta_t_list,
            tmin_choice,
        )

    return


if __name__ == "__main__":
    main()
