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


def plot_ratio_fit_comp_paper(
    ratios,
    ratio_fit,
    delta_t,
    src_snk_times,
    redchisq,
    fit_param_boot,
    plotdir,
    plotparam,
    FH_matrix_element,
    FH_matrix_element_err,
    title="",
):
    time = np.arange(64)
    labels = [
        r"$t_{\mathrm{sep}}=10$",
        r"$t_{\mathrm{sep}}=13$",
        r"$t_{\mathrm{sep}}=16$",
    ]

    # f, axarr = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(11, 5))
    f, axarr = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(8, 5))
    f.subplots_adjust(wspace=0, bottom=0.2)
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

        axarr[icorr].set_title(labels[icorr])
        axarr[icorr].set_xlabel(r"$\tau-t_{\mathrm{sep}}/2$", labelpad=14, fontsize=18)
        axarr[icorr].set_ylabel(
            r"$R(\vec{p}\, ; t_{\mathrm{sep}}, \tau)$", labelpad=5, fontsize=18
        )
        axarr[icorr].label_outer()
        axarr[icorr].set_xlim(plot_time2[0] - 0.5, plot_time2[src_snk_times[-1]] + 0.5)

    axarr[3].axhline(
        FH_matrix_element,
        color=_colors[6],
    )
    axarr[3].fill_between(
        plot_time3,
        [FH_matrix_element - FH_matrix_element_err] * len(plot_time3),
        [FH_matrix_element + FH_matrix_element_err] * len(plot_time3),
        alpha=0.3,
        linewidth=0,
        color=_colors[6],
    )
    axarr[3].set_title(r"\textrm{FH}")
    # axarr[3].set_xticks([])

    savefile = plotdir / Path(f"{title}.pdf")
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


def main():
    plt.style.use("./mystyle.txt")
    plt.rc("text.latex", preamble=r"\usepackage{physics}")
    plt.rcParams.update({"figure.autolayout": False})

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
    operators_tex = ["$\gamma_4$"]
    operators = ["g3"]
    polarizations = ["UNPOL"]
    momenta = ["p+1+0+0"]
    delta_t_list = [5]
    tmin_choice = [7]

    # ======================================================================
    # Calculate the Q^2 values for each of the n2sig and sig2n transitions and momenta
    # Then save these to a file.
    # Qsquared_values_sig2n, Qsquared_values_n2sig = get_Qsquared_values(
    #     datadir, tmin_choice, tmin_choice
    # )

    # ======================================================================
    # plot the results of the three-point fn ratio fits
    normalisation = 0.863
    FH_matrix_element = 0.583 / normalisation
    FH_matrix_element_err = 0.036 / normalisation
    plot_3point_FH_comp_n2sig(
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
        tmin_choice[0],
        FH_matrix_element,
        FH_matrix_element_err,
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


def plot_3point_FH_comp_n2sig(
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
    FH_matrix_element,
    FH_matrix_element_err,
):
    """Loop over the operators and momenta"""

    src_snk_times = np.array([10, 13, 16])
    rel = "nr"
    print(momenta)
    for imom, mom in enumerate(momenta):
        print("here")
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

                for ir, reim in enumerate(["real"]):
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
                    # plot_ratio_fit(
                    #     full_ratio_list_reim[ir],
                    #     ratio_fit_boot,
                    #     delta_t_list[imom],
                    #     src_snk_times,
                    #     redchisq_ratio,
                    #     fit_param_ratio_boot,
                    #     plotdir,
                    #     [mom, operators_tex[iop], pol, reim],
                    #     title=f"{mom}/{pol}/ratio_fit_{reim}_{operator}_n2sig",
                    # )
                    print("here")
                    plot_ratio_fit_comp_paper(
                        full_ratio_list_reim[ir],
                        ratio_fit_boot,
                        delta_t_list[imom],
                        src_snk_times,
                        redchisq_ratio,
                        fit_param_ratio_boot,
                        plotdir,
                        [mom, operators_tex[iop], pol, reim],
                        FH_matrix_element,
                        FH_matrix_element_err,
                        title=f"{mom}/{pol}/ratio_comp_fit_{reim}_{operator}_{mom}_n2sig_paper",
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


if __name__ == "__main__":
    main()
