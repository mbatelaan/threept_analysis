import numpy as np
from pathlib import Path

# from BootStrap3 import bootstrap
# import scipy.optimize as syopt
# from scipy.optimize import curve_fit
import pickle
import csv

from plot_utils import save_plot

import matplotlib.pyplot as plt

from formatting import err_brackets

from analysis.evxptreaders import evxptdata
from analysis.bootstrap import bootstrap
from analysis import stats
from analysis import fitfunc

# from threept_analysis.twoexpfitting import fit_2ptfn
from threept_analysis.twoexpfitting import read_pickle

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


def fit_2ptfn_2exp(latticedir, plotdir, datadir, rel="rel"):
    """Read the two-point function and fit a two-exponential function to it over a range of fit windows, then save the fit data to pickle files."""

    kappa_combs = [
        "kp121040kp121040",
        "kp121040kp120620",
    ]
    momenta = ["p+0+0+0", "p+1+0+0", "p+1+1+0"]
    fitfunction = fitfunc.initffncs("Twoexp_log")
    time_limits = [
        # [[1, 12], [26, 26]],
        [[1, 12], [22, 22]],
        [[1, 9], [20, 20]],
        [[1, 8], [17, 17]],
    ]

    for ikappa, kappa in enumerate(kappa_combs):
        print(f"\n{kappa}")
        for imom, mom in enumerate(momenta):
            # if imom < 2:
            #     continue
            print(f"\n{mom}")
            twopointfn_filename = latticedir / Path(
                f"mass_spectrum/baryon_qcdsf/barspec/32x64/unpreconditioned_slrc/{kappa}/sh_gij_p21_90-sh_gij_p21_90/{mom}/barspec_nucleon_{rel}_500cfgs.pickle"
            )
            twopoint_fn = read_pickle(twopointfn_filename, nboot=500, nbin=1)

            # Plot the effective mass of the two-point function
            twopoint_fn_real = twopoint_fn[:, :, 0]
            fitlist_2pt = stats.fit_loop_bayes(
                twopoint_fn_real,
                fitfunction,
                time_limits[imom],
                plot=False,
                disp=True,
                time=False,
                weights_=True,
                timeslice=14,
            )

            datafile = datadir / Path(f"{kappa}_{mom}_{rel}_fitlist_2pt_2exp.pkl")
            with open(datafile, "wb") as file_out:
                pickle.dump(fitlist_2pt, file_out)
    return


def fit_2ptfn_3exp(latticedir, plotdir, datadir, rel="rel"):
    """Read the two-point function and fit a two-exponential function to it over a range of fit windows, then save the fit data to pickle files."""

    kappa_combs = [
        "kp121040kp121040",
        "kp121040kp120620",
    ]
    momenta = ["p+0+0+0", "p+1+0+0", "p+1+1+0"]
    fitfunction = fitfunc.initffncs("Threeexp_log")
    time_limits = [
        [[1, 12], [20, 20]],
        [[1, 9], [20, 20]],
        [[1, 8], [19, 19]],
    ]
    # time_limits = [
    #     [[1, 12], [25, 25]],
    #     [[1, 9], [21, 21]],
    #     [[1, 8], [18, 18]],
    # ]

    for ikappa, kappa in enumerate(kappa_combs):
        print(f"\n{kappa}")
        for imom, mom in enumerate(momenta):
            print(f"\n{mom}")
            twopointfn_filename = latticedir / Path(
                f"mass_spectrum/baryon_qcdsf/barspec/32x64/unpreconditioned_slrc/{kappa}/sh_gij_p21_90-sh_gij_p21_90/{mom}/barspec_nucleon_{rel}_500cfgs.pickle"
            )
            twopoint_fn = read_pickle(twopointfn_filename, nboot=500, nbin=1)

            # Plot the effective mass of the two-point function
            twopoint_fn_real = twopoint_fn[:, :, 0]
            # stats.bs_effmass(
            #     twopoint_fn_real,
            #     time_axis=1,
            #     plot=True,
            #     show=False,
            #     savefile=plotdir / Path(f"twopoint/{kappa}_{mom}_effmass_2pt_fn.pdf"),
            # )
            fitlist_2pt = stats.fit_loop_bayes(
                twopoint_fn_real,
                fitfunction,
                time_limits[imom],
                plot=False,
                disp=True,
                time=False,
                weights_=True,
                timeslice=17,
            )

            datafile = datadir / Path(f"{kappa}_{mom}_{rel}_fitlist_2pt_3exp.pkl")
            with open(datafile, "wb") as file_out:
                pickle.dump(fitlist_2pt, file_out)
    return


def plot_2pt_2exp_fit(
    fit_data_list,
    tmin_choice,
    datadir,
    plotdir,
    title,
):
    """Plot the effective energy of the twopoint functions and their fits"""

    fitfunction = fitfunc.initffncs("Twoexp_log").eval
    tmin_choice_sigma = 4
    weight_tol = 0.01
    # print([i for i in fit_data_list[0][0]])
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

    time = np.arange(64)
    efftime = np.arange(63)

    # ======================================================================
    # Plot the fit parameters for neutron
    fit_tmin_n = [fit["x"][0] for fit in fit_data_list[0]]
    energies_n = np.array([fit["param"][:, 1::2] for fit in fit_data_list[0]])
    energies_n_avg = np.average(energies_n, axis=1)
    energies_n_std = np.std(energies_n, axis=1)
    energy_1_n = energies_n[:, :, 0] + np.exp(energies_n[:, :, 1])

    priors = best_fit_n["prior"][1::2]
    priors_std = best_fit_n["priorsigma"][1::2]
    prior_1_n = priors[0] + np.exp(priors[1])
    prior_1_n_min = priors[0] + np.exp(priors[1] - priors_std[1])
    prior_1_n_max = priors[0] + np.exp(priors[1] + priors_std[1])

    plt.figure(figsize=(6, 5))
    plt.errorbar(
        fit_tmin_n,
        energies_n_avg[:, 0],
        energies_n_std[:, 0],
        elinewidth=1,
        capsize=4,
        color=_colors[0],
        fmt="s",
        label=r"$E_0$",
    )
    plt.errorbar(
        fit_tmin_n,
        np.average(energy_1_n, axis=1),
        np.std(energy_1_n, axis=1),
        elinewidth=1,
        capsize=4,
        color=_colors[1],
        fmt="o",
        label=r"$E_1$",
    )
    plt.fill_between(
        fit_tmin_n,
        np.array([priors[0]] * len(fit_tmin_n))
        - np.array([priors_std[0]] * len(fit_tmin_n)),
        np.array([priors[0]] * len(fit_tmin_n))
        + np.array([priors_std[0]] * len(fit_tmin_n)),
        alpha=0.3,
        linewidth=0,
        color=_colors[0],
    )
    plt.fill_between(
        fit_tmin_n,
        np.array([prior_1_n_min] * len(fit_tmin_n)),
        np.array([prior_1_n_max] * len(fit_tmin_n)),
        alpha=0.3,
        linewidth=0,
        color=_colors[1],
    )
    plt.legend()
    # plt.ylim(0.3, 1.5)
    plt.ylim(0.38, 0.8)
    plt.xlabel(r"$t_{\textrm{min}}$")
    plt.ylabel(r"$E_i$")
    savefile = plotdir / Path(f"twopoint/energies_{title}_n_2exp.pdf")
    savefile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(savefile)
    # plt.show()
    plt.close()

    # ======================================================================
    # Plot the fit parameters for sigma
    fit_tmin_s = [fit["x"][0] for fit in fit_data_list[1]]
    energies_s = np.array([fit["param"][:, 1::2] for fit in fit_data_list[1]])
    energies_s_avg = np.average(energies_s, axis=1)
    energies_s_std = np.std(energies_s, axis=1)
    energy_1_s = energies_s[:, :, 0] + np.exp(energies_s[:, :, 1])

    priors = best_fit_s["prior"][1::2]
    priors_std = best_fit_s["priorsigma"][1::2]
    prior_1_s = priors[0] + np.exp(priors[1])
    prior_1_s_min = priors[0] + np.exp(priors[1] - priors_std[1])
    prior_1_s_max = priors[0] + np.exp(priors[1] + priors_std[1])

    plt.figure(figsize=(6, 5))
    plt.errorbar(
        fit_tmin_s,
        energies_s_avg[:, 0],
        energies_s_std[:, 0],
        elinewidth=1,
        capsize=4,
        color=_colors[0],
        fmt="s",
        label=r"$E_0$",
    )
    plt.errorbar(
        fit_tmin_s,
        np.average(energy_1_s, axis=1),
        np.std(energy_1_s, axis=1),
        elinewidth=1,
        capsize=4,
        color=_colors[1],
        fmt="o",
        label=r"$E_1$",
    )
    plt.fill_between(
        fit_tmin_s,
        np.array([priors[0]] * len(fit_tmin_s))
        - np.array([priors_std[0]] * len(fit_tmin_s)),
        np.array([priors[0]] * len(fit_tmin_s))
        + np.array([priors_std[0]] * len(fit_tmin_s)),
        alpha=0.3,
        linewidth=0,
        color=_colors[0],
    )
    plt.fill_between(
        fit_tmin_s,
        np.array([prior_1_s_min] * len(fit_tmin_s)),
        np.array([prior_1_s_max] * len(fit_tmin_s)),
        alpha=0.3,
        linewidth=0,
        color=_colors[1],
    )
    plt.legend()
    # plt.ylim(0.3, 1.5)
    plt.ylim(0.38, 0.8)
    plt.xlabel(r"$t_{\textrm{min}}$")
    plt.ylabel(r"$E_i$")
    savefile = plotdir / Path(f"twopoint/energies_{title}_s_2exp.pdf")
    savefile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(savefile)
    # plt.show()
    plt.close()

    # ======================================================================
    # nucleon fit
    eff_energy_n = stats.bs_effmass(best_fit_n["y"], time_axis=1, spacing=1)
    fit_result = np.array(
        [fitfunction(best_fit_n["x"], fitparam) for fitparam in best_fit_n["param"]]
    )
    fit_result_eff_energy_n = stats.bs_effmass(fit_result)

    plt.figure(figsize=(6, 4))
    plt.errorbar(
        efftime,
        np.average(eff_energy_n, axis=0),
        np.std(eff_energy_n, axis=0),
        elinewidth=1,
        capsize=4,
        color=_colors[0],
        fmt="s",
    )
    plt.xlim(0, 26)
    plt.ylim(0, 1)
    print(f"{best_fit_n['x']=}")
    plt.plot(
        best_fit_n["x"][:-1],
        np.average(fit_result_eff_energy_n, axis=0),
        color=_colors[1],
    )
    plt.fill_between(
        best_fit_n["x"][:-1],
        np.average(fit_result_eff_energy_n, axis=0)
        - np.std(fit_result_eff_energy_n, axis=0),
        np.average(fit_result_eff_energy_n, axis=0)
        + np.std(fit_result_eff_energy_n, axis=0),
        label=rf"$\chi^2_{{\textrm{{dof}}}} = {best_fit_n['redchisq']:.2f}$",
        color=_colors[1],
        alpha=0.3,
        linewidth=0,
    )
    plt.xlabel(r"$t$")
    plt.ylabel(r"$E_{\textrm{eff}}$")
    plt.legend()
    plt.grid(True, alpha=0.3)
    savefile = plotdir / Path(f"twopoint/bestfit_{title}_n_2exp.pdf")
    savefile2 = plotdir / Path(f"twopoint/bestfit_{title}_n_2exp.png")
    savefile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(savefile)
    plt.savefig(savefile2, dpi=500)
    plt.close()

    # ======================================================================
    # sigma fit
    eff_energy_s = stats.bs_effmass(best_fit_s["y"], time_axis=1, spacing=1)
    fit_result = np.array(
        [fitfunction(best_fit_s["x"], fitparam) for fitparam in best_fit_s["param"]]
    )
    fit_result_eff_energy_s = stats.bs_effmass(fit_result)

    plt.figure(figsize=(6, 4))
    plt.errorbar(
        efftime,
        np.average(eff_energy_s, axis=0),
        np.std(eff_energy_s, axis=0),
        elinewidth=1,
        capsize=4,
        color=_colors[0],
        fmt="s",
    )
    plt.xlim(0, 26)
    plt.ylim(0, 1)
    plt.plot(
        best_fit_s["x"][:-1],
        np.average(fit_result_eff_energy_s, axis=0),
        color=_colors[1],
    )
    plt.fill_between(
        best_fit_s["x"][:-1],
        np.average(fit_result_eff_energy_s, axis=0)
        - np.std(fit_result_eff_energy_s, axis=0),
        np.average(fit_result_eff_energy_s, axis=0)
        + np.std(fit_result_eff_energy_s, axis=0),
        label=rf"$\chi^2_{{\textrm{{dof}}}} = {best_fit_n['redchisq']:.2f}$",
        color=_colors[1],
        alpha=0.3,
        linewidth=0,
    )
    plt.legend()
    plt.xlabel(r"$t$")
    plt.ylabel(r"$E_{\textrm{eff}}$")
    savefile = plotdir / Path(f"twopoint/bestfit_{title}_s_2exp.pdf")
    savefile2 = plotdir / Path(f"twopoint/bestfit_{title}_s_2exp.png")
    savefile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(savefile)
    plt.savefig(savefile2, dpi=500)
    plt.close()

    return


def plot_fit_loop_energies(
    fit_data_list,
    tmin_choice,
    datadir,
    plotdir,
    title,
):
    """Plot the effective energy of the twopoint functions and their fits"""

    # fitfunction = fitfunc.initffncs("Twoexp_log").eval
    # tmin_choice_sigma = 5
    # weight_tol = 0.01

    fitparams = np.array([fit["param"] for fit in fit_data_list])
    fit_times = [fit["x"] for fit in fit_data_list]
    # chosen_time = np.where([times[0] == tmin_choice for times in fit_times_n])[0][0]
    # best_fit_n = fit_data_list[0][chosen_time]
    # weighted_fit_n = best_fit_n["param"]

    time = np.arange(len(fit_data_list[0]["x"]))
    efftime = time[:-1]

    # ======================================================================
    # Plot the energies
    fit_tmin = [fit["x"][0] for fit in fit_data_list[0]]
    energies = np.array([fit["param"][:, 1::2] for fit in fit_data_list[0]])
    energies_avg = np.average(energies, axis=1)
    energies_std = np.std(energies, axis=1)
    energy_1 = energies[:, :, 0] + np.exp(energies[:, :, 1])

    priors = best_fit["prior"][1::2]
    priors_std = best_fit["priorsigma"][1::2]
    prior_1 = priors[0] + np.exp(priors[1])
    prior_1_min = priors[0] + np.exp(priors[1] - priors_std[1])
    prior_1_max = priors[0] + np.exp(priors[1] + priors_std[1])

    plt.figure(figsize=(6, 5))
    plt.errorbar(
        fit_tmin,
        energies_avg[:, 0],
        energies_std[:, 0],
        elinewidth=1,
        capsize=4,
        color=_colors[0],
        fmt="s",
        label=r"$E_0$",
    )
    plt.errorbar(
        fit_tmin,
        np.average(energy_1, axis=1),
        np.std(energy_1, axis=1),
        elinewidth=1,
        capsize=4,
        color=_colors[1],
        fmt="o",
        label=r"$E_1$",
    )
    plt.fill_between(
        fit_tmin,
        np.array([priors[0]] * len(fit_tmin))
        - np.array([priors_std[0]] * len(fit_tmin)),
        np.array([priors[0]] * len(fit_tmin))
        + np.array([priors_std[0]] * len(fit_tmin)),
        alpha=0.3,
        linewidth=0,
        color=_colors[0],
    )
    plt.fill_between(
        fit_tmin,
        np.array([prior_1_min] * len(fit_tmin)),
        np.array([prior_1_max] * len(fit_tmin)),
        alpha=0.3,
        linewidth=0,
        color=_colors[1],
    )
    plt.legend()
    # plt.ylim(0.3, 1.5)
    plt.ylim(0.38, 0.8)
    plt.xlabel(r"$t_{\textrm{min}}$")
    plt.ylabel(r"$E_i$")
    savefile = plotdir / Path(f"twopoint/energies_{title}_2exp.pdf")
    savefile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(savefile)
    # plt.show()
    plt.close()

    # ======================================================================
    # nucleon fit
    eff_energy_n = stats.bs_effmass(best_fit_n["y"], time_axis=1, spacing=1)
    fit_result = np.array(
        [fitfunction(best_fit_n["x"], fitparam) for fitparam in best_fit_n["param"]]
    )
    fit_result_eff_energy_n = stats.bs_effmass(fit_result)

    plt.figure(figsize=(6, 5))
    plt.errorbar(
        efftime,
        np.average(eff_energy_n, axis=0),
        np.std(eff_energy_n, axis=0),
        elinewidth=1,
        capsize=4,
        color=_colors[0],
        fmt="s",
    )
    plt.xlim(0, 30)
    plt.ylim(0, 1)
    plt.plot(
        best_fit_n["x"][:-1],
        np.average(fit_result_eff_energy_n, axis=0),
        color=_colors[1],
    )
    plt.fill_between(
        best_fit_n["x"][:-1],
        np.average(fit_result_eff_energy_n, axis=0)
        - np.std(fit_result_eff_energy_n, axis=0),
        np.average(fit_result_eff_energy_n, axis=0)
        + np.std(fit_result_eff_energy_n, axis=0),
        label=rf"$\chi^2_{{\textrm{{dof}}}} = {best_fit_n['redchisq']:.2f}$",
        color=_colors[1],
        alpha=0.3,
        linewidth=0,
    )
    plt.legend()
    savefile = plotdir / Path(f"twopoint/bestfit_{title}_n_2exp.pdf")
    savefile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(savefile)
    plt.close()

    return


def plot_2pt_3exp_fit(
    fit_data_list,
    tmin_choice,
    datadir,
    plotdir,
    title,
):
    """Plot the effective energy of the twopoint functions and their fits"""

    fitfunction = fitfunc.initffncs("Threeexp_log").eval
    tmin_choice_sigma = 5
    weight_tol = 0.01
    # print([i for i in fit_data_list[0][0]])
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

    time = np.arange(64)
    efftime = np.arange(63)

    # ======================================================================
    # Plot the fit parameters for neutron
    fit_tmin_n = [fit["x"][0] for fit in fit_data_list[0]]
    energies_n = np.array([fit["param"][:, 1::2] for fit in fit_data_list[0]])
    energies_n_avg = np.average(energies_n, axis=1)
    energies_n_std = np.std(energies_n, axis=1)
    energy_1_n = energies_n[:, :, 0] + np.exp(energies_n[:, :, 1])
    energy_2_n = (
        energies_n[:, :, 0] + np.exp(energies_n[:, :, 1]) + np.exp(energies_n[:, :, 2])
    )

    priors = best_fit_n["prior"][1::2]
    priors_std = best_fit_n["priorsigma"][1::2]

    prior_1_n = priors[0] + np.exp(priors[1])
    prior_1_n_min = priors[0] + np.exp(priors[1] - priors_std[1])
    prior_1_n_max = priors[0] + np.exp(priors[1] + priors_std[1])
    # prior_1_n_min = priors[0] + np.exp(priors[1] - 0.5)
    # prior_1_n_max = priors[0] + np.exp(priors[1] + 0.5)

    plt.figure(figsize=(6, 4))
    plt.errorbar(
        fit_tmin_n,
        energies_n_avg[:, 0],
        energies_n_std[:, 0],
        elinewidth=1,
        capsize=4,
        color=_colors[0],
        fmt="s",
        label=r"$E_0$",
    )
    plt.errorbar(
        fit_tmin_n,
        np.average(energy_1_n, axis=1),
        np.std(energy_1_n, axis=1),
        elinewidth=1,
        capsize=4,
        color=_colors[1],
        fmt="o",
        label=r"$E_1$",
    )
    plt.errorbar(
        fit_tmin_n,
        np.average(energy_2_n, axis=1),
        np.std(energy_2_n, axis=1),
        elinewidth=1,
        capsize=4,
        color=_colors[2],
        fmt="^",
        label=r"$E_2$",
    )
    plt.fill_between(
        fit_tmin_n,
        np.array([priors[0]] * len(fit_tmin_n))
        - np.array([priors_std[0]] * len(fit_tmin_n)),
        np.array([priors[0]] * len(fit_tmin_n))
        + np.array([priors_std[0]] * len(fit_tmin_n)),
        alpha=0.3,
        linewidth=0,
        color=_colors[0],
    )
    plt.fill_between(
        fit_tmin_n,
        np.array([prior_1_n_min] * len(fit_tmin_n)),
        np.array([prior_1_n_max] * len(fit_tmin_n)),
        alpha=0.3,
        linewidth=0,
        color=_colors[1],
    )
    plt.legend()
    # plt.ylim(0, 1.5)
    plt.ylim(0.38, 1.1)
    plt.xlabel(r"$t_{\textrm{min}}$")
    plt.ylabel(r"$E_i$")
    savefile = plotdir / Path(f"twopoint/energies_{title}_n_3exp.pdf")
    savefile2 = plotdir / Path(f"twopoint/energies_{title}_n_3exp.png")
    savefile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(savefile)
    plt.savefig(savefile2, dpi=500)
    # plt.show()
    plt.close()

    # ======================================================================
    # Plot the fit parameters for sigma
    fit_tmin_s = [fit["x"][0] for fit in fit_data_list[1]]
    energies_s = np.array([fit["param"][:, 1::2] for fit in fit_data_list[1]])
    energies_s_avg = np.average(energies_s, axis=1)
    energies_s_std = np.std(energies_s, axis=1)
    energy_1_s = energies_s[:, :, 0] + np.exp(energies_s[:, :, 1])
    energy_2_s = (
        energies_s[:, :, 0] + np.exp(energies_s[:, :, 1]) + np.exp(energies_s[:, :, 2])
    )

    priors = best_fit_s["prior"][1::2]
    priors_std = best_fit_s["priorsigma"][1::2]

    prior_1_s = priors[0] + np.exp(priors[1])
    prior_1_s_min = priors[0] + np.exp(priors[1] - priors_std[1])
    prior_1_s_max = priors[0] + np.exp(priors[1] + priors_std[1])
    # prior_1_s_min = priors[0] + np.exp(priors[1] - 0.5)
    # prior_1_s_max = priors[0] + np.exp(priors[1] + 0.5)

    plt.figure(figsize=(6, 4))
    plt.errorbar(
        fit_tmin_s,
        energies_s_avg[:, 0],
        energies_s_std[:, 0],
        elinewidth=1,
        capsize=4,
        color=_colors[0],
        fmt="s",
        label=r"$E_0$",
    )
    plt.errorbar(
        fit_tmin_s,
        np.average(energy_1_s, axis=1),
        np.std(energy_1_s, axis=1),
        elinewidth=1,
        capsize=4,
        color=_colors[1],
        fmt="o",
        label=r"$E_1$",
    )
    plt.errorbar(
        fit_tmin_s,
        np.average(energy_2_s, axis=1),
        np.std(energy_2_s, axis=1),
        elinewidth=1,
        capsize=4,
        color=_colors[2],
        fmt="^",
        label=r"$E_2$",
    )
    plt.fill_between(
        fit_tmin_s,
        np.array([priors[0]] * len(fit_tmin_s))
        - np.array([priors_std[0]] * len(fit_tmin_s)),
        np.array([priors[0]] * len(fit_tmin_s))
        + np.array([priors_std[0]] * len(fit_tmin_s)),
        alpha=0.3,
        linewidth=0,
        color=_colors[0],
    )
    plt.fill_between(
        fit_tmin_s,
        np.array([prior_1_s_min] * len(fit_tmin_s)),
        np.array([prior_1_s_max] * len(fit_tmin_s)),
        alpha=0.3,
        linewidth=0,
        color=_colors[1],
    )
    plt.legend(fontsize="x-small")
    # plt.ylim(0, 1.5)
    plt.ylim(0.38, 1.1)
    # plt.ylim(0.38, 1.0)
    plt.xlabel(r"$t_{\textrm{min}}$")
    plt.ylabel(r"$E_i$")
    savefile = plotdir / Path(f"twopoint/energies_{title}_s_3exp.pdf")
    savefile2 = plotdir / Path(f"twopoint/energies_{title}_s_3exp.png")
    savefile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(savefile)
    plt.savefig(savefile2, dpi=500)
    # plt.show()
    plt.close()

    # ======================================================================
    # nucleon fit
    eff_energy_n = stats.bs_effmass(best_fit_n["y"], time_axis=1, spacing=1)
    fit_result = np.array(
        [fitfunction(best_fit_n["x"], fitparam) for fitparam in best_fit_n["param"]]
    )
    fit_result_eff_energy_n = stats.bs_effmass(fit_result)

    plt.figure(figsize=(6, 5))
    plt.errorbar(
        efftime,
        np.average(eff_energy_n, axis=0),
        np.std(eff_energy_n, axis=0),
        elinewidth=1,
        capsize=4,
        color=_colors[0],
        fmt="s",
    )
    plt.xlim(0, 30)
    plt.ylim(0, 1)
    plt.plot(
        best_fit_n["x"][:-1],
        np.average(fit_result_eff_energy_n, axis=0),
        color=_colors[1],
    )
    plt.fill_between(
        best_fit_n["x"][:-1],
        np.average(fit_result_eff_energy_n, axis=0)
        - np.std(fit_result_eff_energy_n, axis=0),
        np.average(fit_result_eff_energy_n, axis=0)
        + np.std(fit_result_eff_energy_n, axis=0),
        label=f"chi-squared = {best_fit_n['redchisq']:.2f}",
        color=_colors[1],
        alpha=0.3,
        linewidth=0,
    )
    plt.legend()
    savefile = plotdir / Path(f"twopoint/bestfit_{title}_n_3exp.pdf")
    savefile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(savefile)
    plt.close()

    # ======================================================================
    # sigma fit
    eff_energy_s = stats.bs_effmass(best_fit_s["y"], time_axis=1, spacing=1)
    fit_result = np.array(
        [fitfunction(best_fit_s["x"], fitparam) for fitparam in best_fit_s["param"]]
    )
    fit_result_eff_energy_s = stats.bs_effmass(fit_result)

    plt.figure(figsize=(6, 5))
    plt.errorbar(
        efftime,
        np.average(eff_energy_s, axis=0),
        np.std(eff_energy_s, axis=0),
        elinewidth=1,
        capsize=4,
        color=_colors[0],
        fmt="s",
    )
    plt.xlim(0, 30)
    plt.ylim(0, 1)
    plt.plot(
        best_fit_s["x"][:-1],
        np.average(fit_result_eff_energy_s, axis=0),
        color=_colors[1],
    )
    plt.fill_between(
        best_fit_s["x"][:-1],
        np.average(fit_result_eff_energy_s, axis=0)
        - np.std(fit_result_eff_energy_s, axis=0),
        np.average(fit_result_eff_energy_s, axis=0)
        + np.std(fit_result_eff_energy_s, axis=0),
        label=f"chi-squared = {best_fit_s['redchisq']:.2f}",
        color=_colors[1],
        alpha=0.3,
        linewidth=0,
    )
    plt.legend()
    savefile = plotdir / Path(f"twopoint/bestfit_{title}_s_3exp.pdf")
    savefile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(savefile)
    plt.close()

    return


def read_2pt_fits_2exp(latticedir, plotdir, datadir, momenta, rel, tmin_choice):
    # momenta = ["p+0+0+0", "p+1+0+0", "p+1+1+0"]
    # tmin_choice = [5, 3, 3]
    for imom, mom in enumerate(momenta):
        print(f"\n{mom}")
        kappa_combs = [
            "kp121040kp121040",
            "kp121040kp120620",
        ]
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

        plot_2pt_2exp_fit(
            [fit_data_n, fit_data_s],
            tmin_choice[imom],
            datadir,
            plotdir,
            title=f"{mom}",
        )


def read_2pt_fits_3exp(latticedir, plotdir, datadir, momenta, rel, tmin_choice):
    # momenta = ["p+0+0+0", "p+1+0+0", "p+1+1+0"]
    # tmin_choice = [5, 3, 3]
    for imom, mom in enumerate(momenta):
        print(f"\n{mom}")
        kappa_combs = [
            "kp121040kp121040",
            "kp121040kp120620",
        ]
        datafile_n = datadir / Path(
            f"{kappa_combs[0]}_{mom}_{rel}_fitlist_2pt_3exp.pkl"
        )
        with open(datafile_n, "rb") as file_in:
            fit_data_n = pickle.load(file_in)

        datafile_s = datadir / Path(
            f"{kappa_combs[1]}_{mom}_{rel}_fitlist_2pt_3exp.pkl"
        )
        with open(datafile_s, "rb") as file_in:
            fit_data_s = pickle.load(file_in)

        plot_2pt_3exp_fit(
            [fit_data_n, fit_data_s],
            tmin_choice[imom],
            datadir,
            plotdir,
            title=f"{mom}",
        )


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

    momenta = ["p+0+0+0", "p+1+0+0", "p+1+1+0"]
    # tmin_choice = [5, 4, 4]
    # tmin_choice = [7, 7, 7]
    tmin_choice = [4, 4, 4]

    # ======================================================================
    # Read the two-point functions and fit a two-exponential function to it

    fit_2ptfn_2exp(latticedir, plotdir, datadir, rel="nr")
    # fit_2ptfn_3exp(latticedir, plotdir, datadir, rel="nr")

    print("\nFitting done")

    read_2pt_fits_2exp(latticedir, plotdir, datadir, momenta, "nr", tmin_choice)
    # read_2pt_fits_3exp(latticedir, plotdir, datadir, momenta, "nr", tmin_choice)


if __name__ == "__main__":
    main()
