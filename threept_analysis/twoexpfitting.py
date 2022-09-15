import numpy as np
from pathlib import Path

import pickle
import csv
from plot_utils import save_plot
import matplotlib.pyplot as plt
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


class three_point:
    def __init__(self):
        self.npar = 2
        self.label = r"TwoexpRatio"
        # self.initpar=np.array([0.08,1.8e-6,0.8,8.4e-1]) #p+2+2+0
        # self.initpar=np.array([1.,1.15e-4,-1.4,1.4]) #p+0+0+0
        self.initpar = np.array([1.8e-4, 3.5e-4])  # p+1+0+0
        # self.bounds=([-np.inf,0,-np.inf,0],[np.inf,10,np.inf,10])
        self.q = [1.0, 1.0]
        # q[0] = A1/A0
        # q[1] = E_1 - E_0
        self.bounds = [(-1.0, 1.0), (-1.0, 3.0)]
        # print("Initialising Two exp fitter")

    def initparfnc(self, y, i=0):
        # self.initpar=np.array([1.8e-4,3.5e-4]) #p+1+0+0
        # self.initpar=np.array([1.,1.8e-4,-3.8e-4,3.5e-1]) #p+1+0+0
        pass

    def eval(self, x, p):
        """evaluate"""
        # return (np.exp(-x*p[0])+p[1]*np.exp(-x*(p[2]+p[3])))/(1+p[1]*np.exp(-x*p[2]))
        return (np.exp(-x * p[0]) + self.q[0] * np.exp(-x * (self.q[1] + p[1]))) / (
            1 + self.q[0] * np.exp(-x * self.q[1])
        )


def threeptBS(X, B00, B10, B01, B11):
    tau, t, A_E0i, A_E1i, A_E0f, A_E1f, E0i, E1i, E0f, E1f, m0, m1 = X
    return (
        -sqrt(A_E0i * A_E0f) * B00 * exp(-E0f * t) * exp(-(E0i - E0f) * tau)
        - sqrt(A_E1i * A_E0f) * B10 * exp(-E0f * t) * exp(-(E1i - E0f) * tau)
        - sqrt(A_E0i * A_E1f) * B01 * exp(-E1f * t) * exp(-(E0i - E1f) * tau)
        - sqrt(A_E1i * A_E1f) * B11 * exp(-E1f * t) * exp(-(E1i - E1f) * tau)
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


def plot_all_ratios(ratios, src_snk_times, plotdir, title=""):
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
        ydata = np.average(corr, axis=0)
        yerror = np.std(corr, axis=0)
        axarr[icorr].errorbar(
            plot_time2[: src_snk_times[icorr]],
            # ydata,
            # yerror,
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
            r"$R(\vec{p}\, ; t_{\mathrm{sep}}, \tau)$", labelpad=5, fontsize=18
        )
        axarr[icorr].label_outer()
        # axarr[icorr].set_xlim(-0.5, src_snk_times[icorr] + 0.5)
        axarr[icorr].set_xlim(
            plot_time2[0] - 0.5, plot_time2[src_snk_times[icorr]] + 0.5
        )

    f.suptitle(
        r"3-point function with $\hat{\mathcal{O}}=\gamma_4$, $\vec{q}\, =(0,0,0)$ for $t_{\mathrm{sep}}=10,13,16$"
    )
    savefile = plotdir / Path(f"{title}.pdf")
    savefile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(savefile)
    # plt.show()
    plt.close()
    return


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
    # # Read the two-point function and fit a two-exponential function to it
    # twoptfn_filename_sigma = latticedir / Path(
    #     "mass_spectrum/baryon_qcdsf/barspec/32x64/unpreconditioned_slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/barspec_nucleon_rel_500cfgs.pickle"
    # )
    # twoptfn_sigma = read_pickle(twoptfn_filename_sigma, nboot=500, nbin=1)
    # twoptfn_sigma_real = twoptfn_sigma[:, :, 0]

    # twoptfn_filename_neutron = latticedir / Path(
    #     "mass_spectrum/baryon_qcdsf/barspec/32x64/unpreconditioned_slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/barspec_nucleon_rel_500cfgs.pickle"
    # )
    # twoptfn_neutron = read_pickle(twoptfn_filename_neutron, nboot=500, nbin=1)
    # twoptfn_neutron_real = twoptfn_neutron[:, :, 0]

    # stats.bs_effmass(
    #     twoptfn_real,
    #     time_axis=1,
    #     plot=True,
    #     show=False,
    #     savefile=plotdir / Path("effmass_2pt_fn.pdf"),
    # )
    # ======================================================================
    # Read in the three point function data
    operators = [
        "g0",
        "g1",
        "g2",
        "g3",
        "g5",
        "g51",
        "g53",
        "g01",
        "g02",
        "g03",
        "g05",
        "g12",
        "g13",
        "g23",
        "g25",
        "gI",
    ]
    # polarizations = ["pol_3", "unpol"]
    polarizations = ["POL", "UNPOL"]
    momenta = ["p+0+0+0", "p+1+0+0", "p+1+1+0"]
    # momenta = ["p+1+0+0", "p+1+1+0"]
    src_snk_times = np.array([10, 13, 16])

    for imom, mom in enumerate(momenta):
        # ======================================================================
        # Read the two-point function and fit a two-exponential function to it
        twoptfn_filename_sigma = latticedir / Path(
            f"mass_spectrum/baryon_qcdsf/barspec/32x64/unpreconditioned_slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/barspec_nucleon_rel_500cfgs.pickle"
        )
        twoptfn_filename_neutron = latticedir / Path(
            f"mass_spectrum/baryon_qcdsf/barspec/32x64/unpreconditioned_slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/{mom}/barspec_nucleon_rel_500cfgs.pickle"
        )
        twoptfn_sigma = read_pickle(twoptfn_filename_sigma, nboot=500, nbin=1)
        twoptfn_neutron = read_pickle(twoptfn_filename_neutron, nboot=500, nbin=1)

        twoptfn_sigma_real = twoptfn_sigma[:, :, 0]
        twoptfn_neutron_real = twoptfn_neutron[:, :, 0]

        for iop, operator in enumerate(operators):
            for ipol, pol in enumerate(polarizations):
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
                # Plot the three-point functions
                plot_all_3ptfn(
                    np.array(
                        [
                            threeptfn_t10[:, :, 0],
                            threeptfn_t13[:, :, 0],
                            threeptfn_t16[:, :, 0],
                        ]
                    ),
                    src_snk_times,
                    plotdir,
                    title=f"{mom}/{pol}/threeptfns_real_{operator}",
                )
                plot_all_3ptfn(
                    np.array(
                        [
                            threeptfn_t10[:, :, 1],
                            threeptfn_t13[:, :, 1],
                            threeptfn_t16[:, :, 1],
                        ]
                    ),
                    src_snk_times,
                    plotdir,
                    title=f"{mom}/{pol}/threeptfns_imag_{operator}",
                )

                # ======================================================================
                # Construct the simple ratio of 3pt and 2pt functions
                ratio_unpol_t10 = np.einsum(
                    "ijk,i->ijk", threeptfn_t10, twoptfn_sigma_real[:, 10] ** (-1)
                )
                ratio_unpol_t13 = np.einsum(
                    "ijk,i->ijk", threeptfn_t13, twoptfn_sigma_real[:, 13] ** (-1)
                )
                ratio_unpol_t16 = np.einsum(
                    "ijk,i->ijk", threeptfn_t16, twoptfn_sigma_real[:, 16] ** (-1)
                )

                # ======================================================================
                # plot the real part of the 3pt fn ratio
                ratio_list_real = np.array(
                    [
                        ratio_unpol_t10[:, :, 0],
                        ratio_unpol_t13[:, :, 0],
                        ratio_unpol_t16[:, :, 0],
                    ]
                )
                plot_all_ratios(
                    ratio_list_real,
                    src_snk_times,
                    plotdir,
                    title=f"{mom}/{pol}/ratios_real_{operator}",
                )
                # ======================================================================
                # plot the imaginary part of the 3pt fn ratio
                ratio_list_imag = np.array(
                    [
                        ratio_unpol_t10[:, :, 1],
                        ratio_unpol_t13[:, :, 1],
                        ratio_unpol_t16[:, :, 1],
                    ]
                )
                plot_all_ratios(
                    ratio_list_imag,
                    src_snk_times,
                    plotdir,
                    title=f"{mom}/{pol}/ratios_imag_{operator}",
                )

                # ======================================================================
                # Construct the full ratio of 3pt and 2pt functions
                # ratio_unpol_t10 = np.einsum(
                #     "ijk,i->ijk", threeptfn_t10, twoptfn_sigma_real[:, 10] ** (-1)
                # )

                sqrt_factor_t10 = np.sqrt(
                    (twoptfn_sigma_real[:, :10] * twoptfn_neutron_real[:, 10 - 1 :: -1])
                    / (
                        twoptfn_neutron_real[:, :10]
                        * twoptfn_sigma_real[:, 10 - 1 :: -1]
                    )
                )
                prefactor_t10_full = np.einsum(
                    "ij,i->ij",
                    sqrt_factor_t10,
                    np.sqrt(twoptfn_sigma_real[:, 10] / twoptfn_neutron_real[:, 10])
                    / twoptfn_sigma_real[:, 10],
                )
                ratio_unpol_t10 = np.einsum(
                    "ijk,ij->ijk", threeptfn_t10[:, :10], prefactor_t10_full
                )
                print("sqrt factor")
                # t13
                sqrt_factor_t13 = np.sqrt(
                    (twoptfn_sigma_real[:, :13] * twoptfn_neutron_real[:, 13 - 1 :: -1])
                    / (
                        twoptfn_neutron_real[:, :13]
                        * twoptfn_sigma_real[:, 13 - 1 :: -1]
                    )
                )
                prefactor_t13_full = np.einsum(
                    "ij,i->ij",
                    sqrt_factor_t13,
                    np.sqrt(twoptfn_sigma_real[:, 13] / twoptfn_neutron_real[:, 13])
                    / twoptfn_sigma_real[:, 13],
                )
                ratio_unpol_t13 = np.einsum(
                    "ijk,ij->ijk", threeptfn_t13[:, :13], prefactor_t13_full
                )
                # t16
                sqrt_factor_t16 = np.sqrt(
                    (twoptfn_sigma_real[:, :16] * twoptfn_neutron_real[:, 16 - 1 :: -1])
                    / (
                        twoptfn_neutron_real[:, :16]
                        * twoptfn_sigma_real[:, 16 - 1 :: -1]
                    )
                )
                prefactor_t16_full = np.einsum(
                    "ij,i->ij",
                    sqrt_factor_t16,
                    np.sqrt(twoptfn_sigma_real[:, 16] / twoptfn_neutron_real[:, 16])
                    / twoptfn_sigma_real[:, 16],
                )
                ratio_unpol_t16 = np.einsum(
                    "ijk,ij->ijk", threeptfn_t16[:, :16], prefactor_t16_full
                )
                # ======================================================================
                # plot the real part of the 3pt fn ratio
                print(f"{np.shape(ratio_unpol_t10)=}")
                full_ratio_list_real = [
                    ratio_unpol_t10[:, :, 0],
                    ratio_unpol_t13[:, :, 0],
                    ratio_unpol_t16[:, :, 0],
                ]
                plot_all_ratios(
                    full_ratio_list_real,
                    src_snk_times,
                    plotdir,
                    title=f"{mom}/{pol}/full_ratios_real_{operator}",
                )

    exit()

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
    current_choice = 0
    pol_choice = 0
    mom_choice = 0

    evxptres_t10 = latticedir / Path(
        f"b5p50kp121040kp120620c2p6500-32x64_t10/d0/point/vector/{currents[current_choice]}/kp121040kp120620/p+0+0+0/{momenta[mom_choice]}/d_quark/{polarizations[pol_choice]}/dump/dump.res"
    )
    # evxptres_t10 = Path(
    #     "/home/mischa/Documents/AdelaideUniversity2019/PhD/2019/2expfitting/t10/point/p+0+0+0/q+0+0+0/g0/d/dump/evxpt.res"
    # )
    threepoint_t10 = evxptdata(evxptres_t10, numbers=[0], nboot=500, nbin=1)
    threepoint_t10_real = threepoint_t10[:, 0, :, 0]

    evxptres_t13 = latticedir / Path(
        f"b5p50kp121040kp120620c2p6500-32x64_t13/d0/point/vector/{currents[current_choice]}/kp121040kp120620/p+0+0+0/{momenta[mom_choice]}/d_quark/{polarizations[pol_choice]}/dump/dump.res"
    )
    # evxptres_t13 = Path(
    #     "/home/mischa/Documents/AdelaideUniversity2019/PhD/2019/2expfitting/t13/point/p+0+0+0/q+0+0+0/g0/d/dump/evxpt.res"
    # )
    threepoint_t13 = evxptdata(evxptres_t13, numbers=[0], nboot=500, nbin=1)
    threepoint_t13_real = threepoint_t13[:, 0, :, 0]

    evxptres_t16 = latticedir / Path(
        f"b5p50kp121040kp120620c2p6500-32x64_t16/d0/point/vector/{currents[current_choice]}/kp121040kp120620/p+0+0+0/{momenta[mom_choice]}/d_quark/{polarizations[pol_choice]}/dump/dump.res"
    )
    # evxptres_t16 = Path(
    #     "/home/mischa/Documents/AdelaideUniversity2019/PhD/2019/2expfitting/t16/point/p+0+0+0/q+0+0+0/g0/d/dump/evxpt.res"
    # )
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
    plot_all_3ptfn(threepoint_fns, src_snk_times, plotdir)

    # ======================================================================
    # Construct the simple ratio of 3pt and 2pt functions
    ratio_unpol_t10 = np.einsum(
        "ij,i->ij", threepoint_t10_real, twoptfn_real[:, 10] ** (-1)
    )
    ratio_unpol_t13 = np.einsum(
        "ij,i->ij", threepoint_t13_real, twoptfn_real[:, 13] ** (-1)
    )
    ratio_unpol_t16 = np.einsum(
        "ij,i->ij", threepoint_t16_real, twoptfn_real[:, 16] ** (-1)
    )

    print(f"{threepoint_t10_real=}")
    print(twoptfn_real[:, 10] ** (-1))

    # ratio_unpol_t10 = threepoint_t10_real / np.array([twoptfn_real[:, 10]] * 64).T
    # ratio_unpol_t13 = threepoint_t13_real / np.array([twoptfn_real[:, 13]] * 64).T
    # ratio_unpol_t16 = threepoint_t16_real / np.array([twoptfn_real[:, 16]] * 64).T

    ratio_unpol_t10 = threepoint_t10_real
    ratio_unpol_t13 = threepoint_t13_real
    ratio_unpol_t16 = threepoint_t16_real

    ratio_list = np.array([ratio_unpol_t10, ratio_unpol_t13, ratio_unpol_t16])
    src_snk_times = np.array([10, 13, 16])
    plot_all_ratios(ratio_list, src_snk_times, plotdir)

    # # ======================================================================
    # # Construct the more involved ratio of 3pt and 2pt functions
    # t10_sqrt = np.sqrt(twoptfn_real)
    # ratio_unpol_t10 = np.einsum(
    #     "ij,i->ij", threepoint_t10_real, twoptfn_real[:, 10] ** (-1)
    # )
    # ratio_unpol_t13 = np.einsum(
    #     "ij,i->ij", threepoint_t13_real, twoptfn_real[:, 13] ** (-1)
    # )
    # ratio_unpol_t16 = np.einsum(
    #     "ij,i->ij", threepoint_t16_real, twoptfn_real[:, 16] ** (-1)
    # )


if __name__ == "__main__":
    main()
