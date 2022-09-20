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


def fit_2ptfn(evxptdir, plotdir, datadir, rel="rel"):
    """Read the two-point function and fit a two-exponential function to it over a range of fit windows, then save the fit data to pickle files."""

    kappa_combs = [
        "kp121040kp121040",
        "kp121040kp120620",
        # "kp120620kp121040",
        # "kp120620kp120620",
    ]
    momenta = ["p+0+0+0", "p+1+0+0", "p+1+1+0"]
    twoexp_fitfunc = fitfunc.initffncs("Twoexp")
    # time_limits = [[1, 10], [15, 25]]
    time_limits = [[1, 3], [15, 19]]
    # time_limits = [[1, 1], [24, 24]]

    for ikappa, kappa in enumerate(kappa_combs):
        print(f"\n{kappa}")
        for imom, mom in enumerate(momenta):
            print(f"\n{mom}")
            twopointfn_filename = evxptdir / Path(
                f"mass_spectrum/baryon_qcdsf/barspec/32x64/unpreconditioned_slrc/{kappa}/sh_gij_p21_90-sh_gij_p21_90/{mom}/barspec_nucleon_{rel}_500cfgs.pickle"
            )
            twopoint_fn = read_pickle(twopointfn_filename, nboot=500, nbin=1)

            # Plot the effective mass of the two-point function
            twopoint_fn_real = twopoint_fn[:, :, 0]
            stats.bs_effmass(
                twopoint_fn_real,
                time_axis=1,
                plot=False,
                show=False,
                savefile=plotdir / Path(f"twopoint/{kappa}_{mom}_effmass_2pt_fn.pdf"),
            )
            fitlist_2pt = stats.fit_loop(
                twopoint_fn_real,
                twoexp_fitfunc,
                time_limits,
                plot=False,
                disp=True,
                time=False,
                weights_=True,
            )

            datafile = datadir / Path(f"{kappa}_{mom}_{rel}_fitlist_2pt.pkl")
            with open(datafile, "wb") as file_out:
                pickle.dump(fitlist_2pt, file_out)
    return


def main():
    plt.style.use("./mystyle.txt")
    plt.rc("text.latex", preamble=r"\usepackage{physics}")

    # --- directories ---
    evxptdir = Path.home() / Path(
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
    # Read the two-point functions and fit a two-exponential function to it
    fit_2ptfn(evxptdir, plotdir, datadir, rel="nr")
    # fit_2ptfn(evxptdir, plotdir, datadir)


if __name__ == "__main__":
    main()
