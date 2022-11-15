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

from gevpanalysis.util import read_config

_metadata = {"Author": "Mischa Batelaan", "Creator": __file__}

# _colors = [
#     (0, 0, 0),
#     (0.9, 0.6, 0),
#     (0.35, 0.7, 0.9),
#     (0, 0.6, 0.5),
#     (0.95, 0.9, 0.25),
#     (0, 0.45, 0.7),
#     (0.8, 0.4, 0),
#     (0.8, 0.6, 0.7),
# ]

_colors = [
    (0, 0, 0),
    (0.95, 0.9, 0.25),
    (0.35, 0.7, 0.9),
    (0.9, 0.6, 0),
    (0, 0.6, 0.5),
    (0, 0.45, 0.7),
    (0.8, 0.4, 0),
    (0.8, 0.6, 0.7),
]

# _fmts = ["s", "^", "o", "p", "x", "v", "P", ",", "*", "."]
_fmts = ["s", "p", "x", "^", "o", "v", "P", ",", "*", "."]


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
    FH_data,
    title="",
):
    time = np.arange(64)
    labels = [
        r"$t_{\mathrm{sep}}=10$",
        r"$t_{\mathrm{sep}}=13$",
        r"$t_{\mathrm{sep}}=16$",
    ]

    # f, axarr = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(11, 5))
    # f, axarr = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(8, 5))
    f, axarr = plt.subplots(1, 4, sharex=False, sharey=True, figsize=(8, 5))
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
        # axarr[icorr].set_xlim(plot_time2[0] - 0.5, plot_time2[src_snk_times[-1]] + 0.5)
        axarr[icorr].set_xlim(-10, 10)

    efftime = np.arange(63)
    axarr[3].errorbar(
        efftime[:20],
        np.average(FH_data["deltaE_eff"], axis=0)[:20],
        np.std(FH_data["deltaE_eff"], axis=0)[:20],
        capsize=4,
        elinewidth=1,
        color=_colors[3],
        fmt=_fmts[3],
        # label=labels[icorr],
    )
    axarr[3].plot(
        FH_data["ratio_t_range"],
        [np.average(FH_data["deltaE_fit"])] * len(FH_data["ratio_t_range"]),
        color=_colors[3],
    )
    axarr[3].fill_between(
        FH_data["ratio_t_range"],
        [np.average(FH_data["deltaE_fit"]) - np.std(FH_data["deltaE_fit"])]
        * len(FH_data["ratio_t_range"]),
        [np.average(FH_data["deltaE_fit"]) + np.std(FH_data["deltaE_fit"])]
        * len(FH_data["ratio_t_range"]),
        alpha=0.3,
        linewidth=0,
        color=_colors[7],
    )

    axarr[3].axhline(
        FH_data["FH_matrix_element"],
        color=_colors[6],
    )
    axarr[3].fill_between(
        plot_time3,
        [FH_data["FH_matrix_element"] - FH_data["FH_matrix_element_err"]]
        * len(plot_time3),
        [FH_data["FH_matrix_element"] + FH_data["FH_matrix_element_err"]]
        * len(plot_time3),
        alpha=0.3,
        linewidth=0,
        color=_colors[6],
    )
    axarr[3].set_title(r"\textrm{FH}")
    axarr[3].set_xlim(0, 20)
    axarr[3].set_ylim(0.4, 1.0)

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


def plot_ratio_fit_comp_paper_2(
    ratios,
    ratio_fit,
    delta_t,
    src_snk_times,
    redchisq,
    fit_param_boot,
    plotdir,
    plotparam,
    FH_data,
    title="",
):
    time = np.arange(64)
    labels = [
        r"$t_{\mathrm{sep}}=10$",
        r"$t_{\mathrm{sep}}=13$",
        r"$t_{\mathrm{sep}}=16$",
    ]

    f, axarr = plt.subplots(1, 1, figsize=(8, 5))
    # f, axarr = plt.subplots(1, 2, figsize=(8, 5), gridspec_kw={"width_ratios": [3, 1]})
    ydata0 = np.average(ratios[0], axis=0)
    yerror0 = np.std(ratios[0], axis=0)
    axarr.errorbar(
        time[1 : src_snk_times[0]],
        ydata0[1 : src_snk_times[0]],
        yerror0[1 : src_snk_times[0]],
        capsize=4,
        elinewidth=1,
        color=_colors[0],
        fmt=_fmts[0],
        label=labels[0],
    )

    ydata1 = np.average(ratios[1], axis=0)
    yerror1 = np.std(ratios[1], axis=0)
    axarr.errorbar(
        time[1 : src_snk_times[1]] + 0.07,
        ydata1[1 : src_snk_times[1]],
        yerror1[1 : src_snk_times[1]],
        capsize=4,
        elinewidth=1,
        color=_colors[1],
        fmt=_fmts[1],
        label=labels[1],
    )

    ydata2 = np.average(ratios[2], axis=0)
    yerror2 = np.std(ratios[2], axis=0)
    axarr.errorbar(
        time[1 : src_snk_times[2]] + 0.14,
        ydata2[1 : src_snk_times[2]],
        yerror2[1 : src_snk_times[2]],
        capsize=4,
        elinewidth=1,
        color=_colors[2],
        fmt=_fmts[2],
        label=labels[2],
    )

    # tau_values = np.arange(src_snk_times[icorr] + 1)[delta_t:-delta_t]
    # t_values = np.array([src_snk_times[icorr]] * len(tau_values))

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
    plot_x_values0 = np.arange(src_snk_times[0] + 1)[delta_t:-delta_t]
    axarr.plot(
        plot_x_values0,
        np.average(ratio_fit[:, step_indices[0] : step_indices[1]], axis=0),
        color=_colors[0],
    )
    plot_x_values1 = np.arange(src_snk_times[1] + 1)[delta_t:-delta_t]
    # print(ratio_fit[:, step_indices[1] : step_indices[2]])
    # print(np.average(ratio_fit[:, step_indices[1] : step_indices[1]], axis=0))
    axarr.plot(
        plot_x_values1,
        np.average(ratio_fit[:, step_indices[1] : step_indices[2]], axis=0),
        color=_colors[1],
    )
    plot_x_values2 = np.arange(src_snk_times[2] + 1)[delta_t:-delta_t]
    axarr.plot(
        plot_x_values2,
        np.average(ratio_fit[:, step_indices[2] : step_indices[3]], axis=0),
        color=_colors[2],
    )

    axarr.fill_between(
        plot_x_values0,
        np.average(ratio_fit[:, step_indices[0] : step_indices[1]], axis=0)
        - np.std(ratio_fit[:, step_indices[0] : step_indices[1]], axis=0),
        np.average(ratio_fit[:, step_indices[0] : step_indices[1]], axis=0)
        + np.std(ratio_fit[:, step_indices[0] : step_indices[1]], axis=0),
        alpha=0.3,
        linewidth=0,
        color=_colors[0],
    )
    axarr.fill_between(
        plot_x_values1,
        np.average(ratio_fit[:, step_indices[1] : step_indices[2]], axis=0)
        - np.std(ratio_fit[:, step_indices[1] : step_indices[2]], axis=0),
        np.average(ratio_fit[:, step_indices[1] : step_indices[2]], axis=0)
        + np.std(ratio_fit[:, step_indices[1] : step_indices[2]], axis=0),
        alpha=0.3,
        linewidth=0,
        color=_colors[1],
    )
    axarr.fill_between(
        plot_x_values2,
        np.average(ratio_fit[:, step_indices[2] : step_indices[3]], axis=0)
        - np.std(ratio_fit[:, step_indices[2] : step_indices[3]], axis=0),
        np.average(ratio_fit[:, step_indices[2] : step_indices[3]], axis=0)
        + np.std(ratio_fit[:, step_indices[2] : step_indices[3]], axis=0),
        alpha=0.3,
        linewidth=0,
        color=_colors[2],
    )
    #     axarr[icorr].axhline(
    #         np.average(fit_param_boot[:, 0]),
    #         color=_colors[5],
    #         # label=rf"fit = {err_brackets(np.average(fit_param_boot[:, 0]), np.std(fit_param_boot[:, 0]))}",
    #     )

    plot_time3 = np.array([0, 20])
    # axarr[icorr].fill_between(
    #     plot_time3,
    #     [np.average(fit_param_boot[:, 0]) - np.std(fit_param_boot[:, 0])]
    #     * len(plot_time3),
    #     [np.average(fit_param_boot[:, 0]) + np.std(fit_param_boot[:, 0])]
    #     * len(plot_time3),
    #     alpha=0.3,
    #     linewidth=0,
    #     color=_colors[5],
    # )

    # axarr[icorr].set_title(labels[icorr])
    # axarr[icorr].set_xlabel(r"$\tau-t_{\mathrm{sep}}/2$", labelpad=14, fontsize=18)
    # axarr[icorr].set_ylabel(
    #     r"$R(\vec{p}\, ; t_{\mathrm{sep}}, \tau)$", labelpad=5, fontsize=18
    # )
    # axarr[icorr].label_outer()
    # # axarr[icorr].set_xlim(plot_time2[0] - 0.5, plot_time2[src_snk_times[-1]] + 0.5)
    # axarr[icorr].set_xlim(-10, 10)

    efftime = np.arange(63)
    axarr.errorbar(
        efftime[:20] + 0.21,
        np.average(FH_data["deltaE_eff"], axis=0)[:20],
        np.std(FH_data["deltaE_eff"], axis=0)[:20],
        capsize=4,
        elinewidth=1,
        color=_colors[3],
        fmt=_fmts[3],
        label=r"$\textrm{FH}$",
    )
    # axarr.axhline(
    #     np.average(FH_data["deltaE_fit"]),
    #     color=_colors[7],
    # )
    # axarr.fill_between(
    #     plot_time3,
    #     [np.average(FH_data["deltaE_fit"]) - np.std(FH_data["deltaE_fit"])]
    #     * len(plot_time3),
    #     [np.average(FH_data["deltaE_fit"]) + np.std(FH_data["deltaE_fit"])]
    #     * len(plot_time3),
    #     alpha=0.3,
    #     linewidth=0,
    #     color=_colors[7],
    # )

    axarr.plot(
        FH_data["ratio_t_range"],
        [FH_data["FH_matrix_element"]] * len(FH_data["ratio_t_range"]),
        color=_colors[3],
    )
    axarr.fill_between(
        FH_data["ratio_t_range"],
        [FH_data["FH_matrix_element"] - FH_data["FH_matrix_element_err"]]
        * len(FH_data["ratio_t_range"]),
        [FH_data["FH_matrix_element"] + FH_data["FH_matrix_element_err"]]
        * len(FH_data["ratio_t_range"]),
        alpha=0.3,
        linewidth=0,
        color=_colors[3],
    )
    # axarr.set_title("")
    axarr.set_xlim(0, 18)
    axarr.set_ylim(0.3, 1.1)

    plt.legend(fontsize="x-small")
    savefile = plotdir / Path(f"{title}_one.pdf")
    savefile2 = plotdir / Path(f"{title}.png")
    savefile3 = plotdir / Path(f"{title}_small.png")
    savefile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(savefile)
    # plt.savefig(savefile2, dpi=500)
    # plt.savefig(savefile3, dpi=100)
    # plt.ylim(1.104, 1.181)
    # plt.savefig(savefile_ylim)
    # plt.show()
    plt.close()
    return


def plot_ratio_fit_comp_paper_3(
    ratios,
    ratio_fit,
    delta_t,
    src_snk_times,
    redchisq,
    fit_param_boot,
    plotdir,
    plotparam,
    FH_data,
    title="",
):
    time = np.arange(64)
    labels = [
        r"$t_{\mathrm{sep}}=10$",
        r"$t_{\mathrm{sep}}=13$",
        r"$t_{\mathrm{sep}}=16$",
    ]

    f, axarr = plt.subplots(
        1, 2, figsize=(7, 5), sharey=False, gridspec_kw={"width_ratios": [3, 1]}
    )
    f.subplots_adjust(wspace=0, bottom=0.15)

    # Plot the 3pt fn ratios
    offset = 0.08
    ydata0 = np.average(ratios[0], axis=0)
    yerror0 = np.std(ratios[0], axis=0)
    axarr[0].errorbar(
        time[1 : src_snk_times[0]],
        ydata0[1 : src_snk_times[0]],
        yerror0[1 : src_snk_times[0]],
        capsize=4,
        elinewidth=1,
        color=_colors[0],
        fmt=_fmts[0],
        label=labels[0],
    )
    ydata1 = np.average(ratios[1], axis=0)
    yerror1 = np.std(ratios[1], axis=0)
    axarr[0].errorbar(
        time[1 : src_snk_times[1]] + offset,
        ydata1[1 : src_snk_times[1]],
        yerror1[1 : src_snk_times[1]],
        capsize=4,
        elinewidth=1,
        color=_colors[1],
        fmt=_fmts[1],
        label=labels[1],
    )
    ydata2 = np.average(ratios[2], axis=0)
    yerror2 = np.std(ratios[2], axis=0)
    axarr[0].errorbar(
        time[1 : src_snk_times[2]] + offset * 2,
        ydata2[1 : src_snk_times[2]],
        yerror2[1 : src_snk_times[2]],
        capsize=4,
        elinewidth=1,
        color=_colors[2],
        fmt=_fmts[2],
        label=labels[2],
    )

    # set the indices which split the fit data into t10, t13, t16
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

    # Fit a constant to the ratios
    # t10_ratio_data = ratios[0][:, step_indices[0] : step_indices[1]]
    # t10_const_fit = np.average(t10_ratio_data, axis=1)
    # t13_ratio_data = ratios[1][:, step_indices[1] : step_indices[2]]
    # t13_const_fit = np.average(t13_ratio_data, axis=1)
    # t16_ratio_data = ratios[2][:, step_indices[2] : step_indices[3]]
    # t16_const_fit = np.average(t16_ratio_data, axis=1)
    t10_ratio_data = ratios[0][:, delta_t:-delta_t]
    t10_const_fit = np.average(t10_ratio_data, axis=1)
    t13_ratio_data = ratios[1][:, delta_t:-delta_t]
    fitparam_t13 = stats.fit_bootstrap(
        fitfunc.constant,
        [1],
        np.arange(len(t13_ratio_data[0])),
        t13_ratio_data,
        bounds=None,
        time=False,
        fullcov=False,
    )
    t13_const_fit = fitparam_t13["param"]
    print(f"{np.average(t13_const_fit)=}")

    t16_ratio_data = ratios[2][:, delta_t:-delta_t]
    print(f"\n{np.shape(t16_ratio_data)=}\n")
    print(f"\n{np.average(t16_ratio_data, axis=0)=}\n")
    fitparam_t16 = stats.fit_bootstrap(
        fitfunc.constant,
        np.array([1]),
        np.arange(len(t16_ratio_data[0])),
        t16_ratio_data,
        bounds=None,
        time=False,
    )
    t16_const_fit = fitparam_t16["param"]
    print(f"{np.average(t16_const_fit)=}")

    # plot the two-exp fit results
    plot_x_values0 = np.arange(src_snk_times[0] + 1)[delta_t:-delta_t]
    axarr[0].plot(
        plot_x_values0,
        np.average(ratio_fit[:, step_indices[0] : step_indices[1]], axis=0),
        color=_colors[0],
    )
    plot_x_values1 = np.arange(src_snk_times[1] + 1)[delta_t:-delta_t]
    axarr[0].plot(
        plot_x_values1,
        np.average(ratio_fit[:, step_indices[1] : step_indices[2]], axis=0),
        color=_colors[1],
    )
    plot_x_values2 = np.arange(src_snk_times[2] + 1)[delta_t:-delta_t]
    axarr[0].plot(
        plot_x_values2,
        np.average(ratio_fit[:, step_indices[2] : step_indices[3]], axis=0),
        color=_colors[2],
    )

    # Plot the two-exp fits to the ratio
    axarr[0].fill_between(
        plot_x_values0,
        np.average(ratio_fit[:, step_indices[0] : step_indices[1]], axis=0)
        - np.std(ratio_fit[:, step_indices[0] : step_indices[1]], axis=0),
        np.average(ratio_fit[:, step_indices[0] : step_indices[1]], axis=0)
        + np.std(ratio_fit[:, step_indices[0] : step_indices[1]], axis=0),
        alpha=0.3,
        linewidth=0,
        color=_colors[0],
    )
    axarr[0].fill_between(
        plot_x_values1,
        np.average(ratio_fit[:, step_indices[1] : step_indices[2]], axis=0)
        - np.std(ratio_fit[:, step_indices[1] : step_indices[2]], axis=0),
        np.average(ratio_fit[:, step_indices[1] : step_indices[2]], axis=0)
        + np.std(ratio_fit[:, step_indices[1] : step_indices[2]], axis=0),
        alpha=0.3,
        linewidth=0,
        color=_colors[1],
    )
    axarr[0].fill_between(
        plot_x_values2,
        np.average(ratio_fit[:, step_indices[2] : step_indices[3]], axis=0)
        - np.std(ratio_fit[:, step_indices[2] : step_indices[3]], axis=0),
        np.average(ratio_fit[:, step_indices[2] : step_indices[3]], axis=0)
        + np.std(ratio_fit[:, step_indices[2] : step_indices[3]], axis=0),
        alpha=0.3,
        linewidth=0,
        color=_colors[2],
    )

    # plot the Feynman-Hellmann data
    efftime = np.arange(63)
    axarr[0].errorbar(
        efftime[:20] + offset * 3,
        np.average(FH_data["deltaE_eff"], axis=0)[:20],
        np.std(FH_data["deltaE_eff"], axis=0)[:20],
        capsize=4,
        elinewidth=1,
        color=_colors[3],
        fmt=_fmts[3],
        label=r"$\textrm{FH}$",
    )
    # axarr.axhline(
    #     np.average(FH_data["deltaE_fit"]),
    #     color=_colors[7],
    # )
    # axarr.fill_between(
    #     plot_time3,
    #     [np.average(FH_data["deltaE_fit"]) - np.std(FH_data["deltaE_fit"])]
    #     * len(plot_time3),
    #     [np.average(FH_data["deltaE_fit"]) + np.std(FH_data["deltaE_fit"])]
    #     * len(plot_time3),
    #     alpha=0.3,
    #     linewidth=0,
    #     color=_colors[7],
    # )

    # plot the fit to the  Feynman-Hellmann data
    axarr[0].plot(
        FH_data["ratio_t_range"][:-2] + offset * 3,
        [np.average(FH_data["FH_matrix_element"])] * len(FH_data["ratio_t_range"][:-2]),
        color=_colors[3],
    )
    axarr[0].fill_between(
        FH_data["ratio_t_range"][:-2] + offset * 3,
        # [FH_data["FH_matrix_element"] - FH_data["FH_matrix_element_err"]]
        [
            np.average(FH_data["FH_matrix_element"])
            - np.std(FH_data["FH_matrix_element"])
        ]
        * len(FH_data["ratio_t_range"][:-2]),
        [
            np.average(FH_data["FH_matrix_element"])
            + np.std(FH_data["FH_matrix_element"])
        ]
        # [FH_data["FH_matrix_element"] + FH_data["FH_matrix_element_err"]]
        * len(FH_data["ratio_t_range"][:-2]),
        alpha=0.3,
        linewidth=0,
        color=_colors[3],
    )

    axarr[0].set_xlim(0, 16)
    axarr[0].set_ylim(0.2, 1.1)
    axarr[0].set_xlabel(r"$t$", labelpad=14, fontsize=18)
    axarr[0].set_ylabel(
        r"$R(\vec{q}\, ; t_{\mathrm{sep}}, \tau)$", labelpad=5, fontsize=18
    )
    axarr[0].label_outer()
    axarr[0].legend(fontsize="x-small")

    # Plot the fit results on the second subplot
    axarr[1].set_yticks([])
    axarr[1].errorbar(
        0,
        np.average(t10_const_fit),
        np.std(t10_const_fit),
        capsize=4,
        elinewidth=1,
        color=_colors[0],
        fmt=_fmts[0],
    )
    axarr[1].errorbar(
        1,
        np.average(t13_const_fit),
        np.std(t13_const_fit),
        capsize=4,
        elinewidth=1,
        color=_colors[1],
        fmt=_fmts[1],
    )
    axarr[1].errorbar(
        2,
        np.average(t16_const_fit),
        np.std(t16_const_fit),
        capsize=4,
        elinewidth=1,
        color=_colors[2],
        fmt=_fmts[2],
    )
    axarr[1].errorbar(
        3,
        np.average(fit_param_boot[:, 0]),
        np.std(fit_param_boot[:, 0]),
        capsize=4,
        elinewidth=1,
        color=_colors[4],
        fmt=_fmts[4],
    )
    axarr[1].errorbar(
        4,
        np.average(FH_data["FH_matrix_element"]),
        np.std(FH_data["FH_matrix_element"]),
        capsize=4,
        elinewidth=1,
        color=_colors[3],
        fmt=_fmts[3],
    )
    axarr[1].set_xticks(
        [0, 1, 2, 3, 4],
    )
    axarr[1].set_xticklabels(
        [labels[0], labels[1], labels[2], r"2-exp", r"FH"],
        fontsize="x-small",
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )
    axarr[1].tick_params(axis="x", which="minor", length=0)
    axarr[1].set_xlim(-0.8, 4.8)
    axarr[1].set_ylim(0.2, 1.1)

    savefile = plotdir / Path(f"{title}_two.pdf")
    savefile2 = plotdir / Path(f"{title}.png")
    savefile3 = plotdir / Path(f"{title}_small.png")
    savefile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(savefile)
    plt.savefig(savefile2, dpi=500)
    plt.savefig(savefile3, dpi=100)
    # plt.ylim(1.104, 1.181)
    # plt.savefig(savefile_ylim)
    # plt.show()
    plt.close()
    return


def plot_ratio_fit_comp_paper_4(
    ratios,
    ratio_fit,
    delta_t,
    src_snk_times,
    redchisq,
    fit_param_boot,
    plotdir,
    plotparam,
    FH_data,
    title="",
):
    time = np.arange(64)
    labels = [
        r"$t_{\mathrm{sep}}=10$",
        r"$t_{\mathrm{sep}}=13$",
        r"$t_{\mathrm{sep}}=16$",
    ]

    f, axarr = plt.subplots(
        1, 2, figsize=(7, 5), sharey=False, gridspec_kw={"width_ratios": [3, 1]}
    )
    f.subplots_adjust(wspace=0, bottom=0.15)

    # Plot the 3pt fn ratios
    ydata0 = np.average(ratios[0], axis=0)
    yerror0 = np.std(ratios[0], axis=0)
    plot_time0 = time - (src_snk_times[0]) / 2
    axarr[0].errorbar(
        plot_time0[1 : src_snk_times[0]],
        ydata0[1 : src_snk_times[0]],
        yerror0[1 : src_snk_times[0]],
        capsize=4,
        elinewidth=1,
        color=_colors[0],
        fmt=_fmts[0],
        label=labels[0],
    )
    ydata1 = np.average(ratios[1], axis=0)
    yerror1 = np.std(ratios[1], axis=0)
    plot_time1 = time - (src_snk_times[1]) / 2
    axarr[0].errorbar(
        plot_time1[1 : src_snk_times[1]],
        ydata1[1 : src_snk_times[1]],
        yerror1[1 : src_snk_times[1]],
        capsize=4,
        elinewidth=1,
        color=_colors[1],
        fmt=_fmts[1],
        label=labels[1],
    )
    ydata2 = np.average(ratios[2], axis=0)
    yerror2 = np.std(ratios[2], axis=0)
    plot_time2 = time - (src_snk_times[2]) / 2
    axarr[0].errorbar(
        plot_time2[1 : src_snk_times[2]],
        ydata2[1 : src_snk_times[2]],
        yerror2[1 : src_snk_times[2]],
        capsize=4,
        elinewidth=1,
        color=_colors[2],
        fmt=_fmts[2],
        label=labels[2],
    )

    # set the indices which split the fit data into t10, t13, t16
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

    # Fit a constant to the ratios
    t10_ratio_data = ratios[0][:, delta_t:-delta_t]
    t10_const_fit = np.average(t10_ratio_data, axis=1)

    t13_ratio_data = ratios[1][:, delta_t:-delta_t]
    fitparam_t13 = stats.fit_bootstrap(
        fitfunc.constant,
        [1],
        np.arange(len(t13_ratio_data[0])),
        t13_ratio_data,
        bounds=None,
        time=False,
        fullcov=False,
    )
    t13_const_fit = fitparam_t13["param"]
    print(f"{np.average(t13_const_fit)=}")

    t16_ratio_data = ratios[2][:, delta_t:-delta_t]
    print(f"\n{np.shape(t16_ratio_data)=}\n")
    print(f"\n{np.average(t16_ratio_data, axis=0)=}\n")
    fitparam_t16 = stats.fit_bootstrap(
        fitfunc.constant,
        np.array([1]),
        np.arange(len(t16_ratio_data[0])),
        t16_ratio_data,
        bounds=None,
        time=False,
    )
    t16_const_fit = fitparam_t16["param"]
    print(f"{np.average(t16_const_fit)=}")

    # plot the two-exp fit results
    plot_x_values0 = (
        np.arange(src_snk_times[0] + 1)[delta_t:-delta_t] - (src_snk_times[0]) / 2
    )
    axarr[0].plot(
        plot_x_values0,
        np.average(ratio_fit[:, step_indices[0] : step_indices[1]], axis=0),
        color=_colors[0],
    )

    plot_x_values1 = (
        np.arange(src_snk_times[1] + 1)[delta_t:-delta_t] - (src_snk_times[1]) / 2
    )
    axarr[0].plot(
        plot_x_values1,
        np.average(ratio_fit[:, step_indices[1] : step_indices[2]], axis=0),
        color=_colors[1],
    )
    plot_x_values2 = (
        np.arange(src_snk_times[2] + 1)[delta_t:-delta_t] - (src_snk_times[2]) / 2
    )
    axarr[0].plot(
        plot_x_values2,
        np.average(ratio_fit[:, step_indices[2] : step_indices[3]], axis=0),
        color=_colors[2],
    )

    # Plot the two-exp fits to the ratio
    axarr[0].fill_between(
        plot_x_values0,
        np.average(ratio_fit[:, step_indices[0] : step_indices[1]], axis=0)
        - np.std(ratio_fit[:, step_indices[0] : step_indices[1]], axis=0),
        np.average(ratio_fit[:, step_indices[0] : step_indices[1]], axis=0)
        + np.std(ratio_fit[:, step_indices[0] : step_indices[1]], axis=0),
        alpha=0.3,
        linewidth=0,
        color=_colors[0],
    )
    axarr[0].fill_between(
        plot_x_values1,
        np.average(ratio_fit[:, step_indices[1] : step_indices[2]], axis=0)
        - np.std(ratio_fit[:, step_indices[1] : step_indices[2]], axis=0),
        np.average(ratio_fit[:, step_indices[1] : step_indices[2]], axis=0)
        + np.std(ratio_fit[:, step_indices[1] : step_indices[2]], axis=0),
        alpha=0.3,
        linewidth=0,
        color=_colors[1],
    )
    axarr[0].fill_between(
        plot_x_values2,
        np.average(ratio_fit[:, step_indices[2] : step_indices[3]], axis=0)
        - np.std(ratio_fit[:, step_indices[2] : step_indices[3]], axis=0),
        np.average(ratio_fit[:, step_indices[2] : step_indices[3]], axis=0)
        + np.std(ratio_fit[:, step_indices[2] : step_indices[3]], axis=0),
        alpha=0.3,
        linewidth=0,
        color=_colors[2],
    )

    axarr[0].set_xlim(-8, 8)
    axarr[0].set_xlabel(r"$\tau - t_{\textrm{sep}}/2$", labelpad=14, fontsize=18)
    axarr[0].set_ylabel(
        r"$R(\vec{q}\, ; t_{\mathrm{sep}}, \tau)$", labelpad=5, fontsize=18
    )
    axarr[0].label_outer()
    axarr[0].legend(fontsize="x-small")

    # Plot the fit results on the second subplot
    axarr[1].set_yticks([])
    axarr[1].errorbar(
        0,
        np.average(t10_const_fit),
        np.std(t10_const_fit),
        capsize=4,
        elinewidth=1,
        color=_colors[0],
        fmt=_fmts[0],
    )
    axarr[1].errorbar(
        1,
        np.average(t13_const_fit),
        np.std(t13_const_fit),
        capsize=4,
        elinewidth=1,
        color=_colors[1],
        fmt=_fmts[1],
    )
    axarr[1].errorbar(
        2,
        np.average(t16_const_fit),
        np.std(t16_const_fit),
        capsize=4,
        elinewidth=1,
        color=_colors[2],
        fmt=_fmts[2],
    )
    axarr[1].errorbar(
        3,
        np.average(fit_param_boot[:, 0]),
        np.std(fit_param_boot[:, 0]),
        capsize=4,
        elinewidth=1,
        color=_colors[4],
        fmt=_fmts[4],
    )
    axarr[1].errorbar(
        4,
        np.average(FH_data["FH_matrix_element"]),
        np.std(FH_data["FH_matrix_element"]),
        capsize=4,
        elinewidth=1,
        color=_colors[3],
        fmt=_fmts[3],
    )
    axarr[1].set_xticks(
        [0, 1, 2, 3, 4],
    )
    axarr[1].set_xticklabels(
        [labels[0], labels[1], labels[2], r"2-exp", r"FH"],
        fontsize="x-small",
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )
    axarr[1].tick_params(axis="x", which="minor", length=0)
    axarr[1].set_xlim(-0.8, 4.8)

    axarr[0].set_ylim(0.63, 0.9)
    axarr[1].set_ylim(0.63, 0.9)

    savefile = plotdir / Path(f"{title}_three.pdf")
    savefile2 = plotdir / Path(f"{title}.png")
    savefile3 = plotdir / Path(f"{title}_small.png")
    savefile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(savefile)
    # plt.savefig(savefile2, dpi=500)
    # plt.savefig(savefile3, dpi=100)
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

    # Read data from the six_point.py gevp analysis script
    config = read_config("theta7")
    defaults = read_config("defaults")
    for key, value in defaults.items():
        config.setdefault(key, value)
    # time_choice_fh = 6
    # delta_t_fh = 4
    time_choice_fh = config["time_choice"]
    delta_t_fh = config["delta_t"]
    nucl_t_range = np.arange(config["tmin_nucl"], config["tmax_nucl"] + 1)
    sigma_t_range = np.arange(config["tmin_sigma"], config["tmax_sigma"] + 1)
    ratio_t_range = np.arange(config["tmin_ratio"], config["tmax_ratio"] + 1)

    with open(
        Path.home()
        / Path("Documents/PhD/analysis_results/six_point_fn_all/data/pickles/theta7/")
        / (f"lambda_dep_t{time_choice_fh}_dt{delta_t_fh}.pkl"),
        "rb",
    ) as file_in:
        data_fh = pickle.load(file_in)

    lambda_index = 6
    lambdas = data_fh[lambda_index]["lambdas"]
    gevp_correlators = data_fh[lambda_index]["order3_corrs"]
    gevp_ratio_fit = data_fh[lambda_index]["order3_fit"]
    ratio3 = np.abs(gevp_correlators[0] / gevp_correlators[1])
    print(f"{np.shape(gevp_correlators)=}")
    print(f"{np.shape(ratio3)=}")
    print(f"{lambdas=}")

    # Get the ratio at a second lambda value
    lambda_index2 = 8
    lambdas2 = data_fh[lambda_index2]["lambdas"]
    gevp_correlators2 = data_fh[lambda_index2]["order3_corrs"]
    gevp_ratio_fit2 = data_fh[lambda_index2]["order3_fit"]
    ratio32 = np.abs(gevp_correlators2[0] / gevp_correlators2[1])
    print(f"{lambdas2=}")

    deltaE_eff = stats.bs_effmass(ratio3) / (2 * lambdas)
    deltaE_fit = gevp_ratio_fit[:, 1] / (2 * lambdas)

    with open(
        Path.home()
        / Path("Documents/PhD/analysis_results/sig2n/data/form_factor_combination.pkl"),
        "rb",
    ) as file_in:
        [feynhell_points, threeptfn_points] = pickle.load(file_in)

    normalisation = 0.863
    FH_matrix_element = feynhell_points["ydata"][4, :] / normalisation

    double_ratio3 = ratio32 / ratio3
    deltaE_eff_double = stats.bs_effmass(double_ratio3) / (2 * (lambdas2 - lambdas))
    FH_data = {
        "deltaE_eff": deltaE_eff_double,
        "deltaE_fit": deltaE_fit,
        "ratio_t_range": ratio_t_range,
        "FH_matrix_element": FH_matrix_element,
        # "FH_matrix_element_err": FH_matrix_element_err,
    }

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
        FH_data,
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
    FH_data,
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

                    # # ======================================================================
                    # # Plot the results of the fit to the ratio
                    # # plot_ratio_fit(
                    # #     full_ratio_list_reim[ir],
                    # #     ratio_fit_boot,
                    # #     delta_t_list[imom],
                    # #     src_snk_times,
                    # #     redchisq_ratio,
                    # #     fit_param_ratio_boot,
                    # #     plotdir,
                    # #     [mom, operators_tex[iop], pol, reim],
                    # #     title=f"{mom}/{pol}/ratio_fit_{reim}_{operator}_n2sig",
                    # # )
                    # print("here")
                    # plot_ratio_fit_comp_paper(
                    #     full_ratio_list_reim[ir],
                    #     ratio_fit_boot,
                    #     delta_t_list[imom],
                    #     src_snk_times,
                    #     redchisq_ratio,
                    #     fit_param_ratio_boot,
                    #     plotdir,
                    #     [mom, operators_tex[iop], pol, reim],
                    #     FH_data,
                    #     title=f"{mom}/{pol}/ratio_comp_fit_{reim}_{operator}_{mom}_n2sig_paper",
                    # )

                    # # ========================================================================
                    # # --- Get the energies of the nucleon and sigma for the FH calculation ---
                    # (
                    #     nucl_fits,
                    #     sigma_fits,
                    #     nucl_energies,
                    #     sigma_energies,
                    # ) = get_energies()
                    # # --- Multiply matrix element with the energy factor ---
                    # energy_factor = np.sqrt(
                    #     2 * nucl_energies[4] / (nucl_energies[4] + nucl_energies[0])
                    # )
                    # FH_data["deltaE_eff"] = np.einsum(
                    #     "ij,i->ij", FH_data["deltaE_eff"], energy_factor
                    # )
                    # FH_data["FH_matrix_element"] = np.einsum(
                    #     "ij,i->ij", FH_data["FH_matrix_element"], energy_factor
                    # )

                    plot_ratio_fit_comp_paper_3(
                        full_ratio_list_reim[ir],
                        ratio_fit_boot,
                        delta_t_list[imom],
                        src_snk_times,
                        redchisq_ratio,
                        fit_param_ratio_boot,
                        plotdir,
                        [mom, operators_tex[iop], pol, reim],
                        FH_data,
                        title=f"{mom}/{pol}/ratio_comp_fit_{reim}_{operator}_{mom}_n2sig_paper",
                    )

                    plot_ratio_fit_comp_paper_4(
                        full_ratio_list_reim[ir],
                        ratio_fit_boot,
                        delta_t_list[imom],
                        src_snk_times,
                        redchisq_ratio,
                        fit_param_ratio_boot,
                        plotdir,
                        [mom, operators_tex[iop], pol, reim],
                        FH_data,
                        title=f"{mom}/{pol}/ratio_comp_fit_{reim}_{operator}_{mom}_n2sig_paper",
                    )

    return


def get_energies():
    """Get the energies of the nucleon and sigma for all the momenta"""
    resultsdir = Path.home() / Path("Documents/PhD/analysis_results")
    datadir_run6 = resultsdir / Path("six_point_fn_all/data/pickles/theta8/")
    datadir_run3 = resultsdir / Path("six_point_fn_all/data/pickles/theta3/")
    datadir_run4 = resultsdir / Path("six_point_fn_all/data/pickles/theta4/")
    datadir_run2 = resultsdir / Path("six_point_fn_all/data/pickles/theta5/")
    datadir_run5 = resultsdir / Path("six_point_fn_all/data/pickles/theta7/")
    datadir_run1 = resultsdir / Path("six_point_fn_all/data/pickles/qmax/")
    datadir_list = [
        datadir_run1,
        datadir_run2,
        datadir_run3,
        datadir_run4,
        datadir_run5,
        datadir_run6,
    ]
    nucl_fits = []
    sigma_fits = []
    nucl_energies = []
    sigma_energies = []
    for idir, datadir_ in enumerate(datadir_list):
        with open(datadir_ / "two_point_fits.pkl", "rb") as file_in:
            twopt_fit_data = pickle.load(file_in)
        nucl_fits.append(twopt_fit_data["chosen_nucl_fit"])
        sigma_fits.append(twopt_fit_data["chosen_sigma_fit"])
        nucl_energies.append(twopt_fit_data["chosen_nucl_fit"]["param"][:, 1])
        sigma_energies.append(twopt_fit_data["chosen_sigma_fit"]["param"][:, 1])
        print(np.average(nucl_energies[-1]))
        # print(np.average(sigma_energies[-1]))

    return nucl_fits, sigma_fits, nucl_energies, sigma_energies


if __name__ == "__main__":
    main()
