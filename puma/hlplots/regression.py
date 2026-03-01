"""Regression response plotting functions for high-level API.

Provides functions for evaluating jet regression performance:
- Response profiles (median and resolution vs a binning variable)
- Overall response distributions (pred/truth histogram)

Ported and generalized from DIAS plotting scripts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from puma import Histogram, HistogramPlot, VarVsVar, VarVsVarPlot

if TYPE_CHECKING:
    from puma.hlplots.tagger import Tagger


def get_mean_and_width(
    data: np.ndarray,
    method: str = "quantile_relative",
) -> tuple[float, float]:
    """Compute the central value and width of a distribution.

    Parameters
    ----------
    data : np.ndarray
        1D array of values (typically response = pred/truth).
    method : str, optional
        Method to use, by default "quantile_relative".
        - "mean_std": mean and standard deviation
        - "quantile": median and half the central 68.2% IQR
        - "quantile_relative": median and IQR/(2*median)

    Returns
    -------
    tuple[float, float]
        (central_value, width)

    Raises
    ------
    ValueError
        If method is not recognized.
    """
    valid_methods = ("mean_std", "quantile", "quantile_relative")
    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}, got '{method}'")

    if len(data) == 0:
        return 0.0, 0.0

    if method == "mean_std":
        return float(np.mean(data)), float(np.std(data))

    median = float(np.quantile(data, 0.5, method="linear"))
    plus_one_sigma = float(np.quantile(data, 0.841, method="linear"))
    minus_one_sigma = float(np.quantile(data, 0.159, method="linear"))

    if method == "quantile":
        return median, (plus_one_sigma - minus_one_sigma) / 2

    # quantile_relative
    if median == 0:
        return median, 0.0
    return median, (plus_one_sigma - minus_one_sigma) / (2 * median)


def bootstrap_uncertainties(
    data: np.ndarray,
    unc_func: Any,
    n_subsamples: int = 10,
    random_seed: int = 42,
) -> tuple[float, float]:
    """Estimate uncertainties on mean and width via bootstrap resampling.

    Splits data into ``n_subsamples`` interleaved subsets, computes ``unc_func``
    on each, and returns std/2 of the resulting distributions as error estimates.

    Parameters
    ----------
    data : np.ndarray
        1D array of values.
    unc_func : callable
        Function with signature ``(data) -> (mean, width)``.
    n_subsamples : int, optional
        Number of bootstrap subsamples, by default 10.
    random_seed : int, optional
        Random seed for reproducibility, by default 42.

    Returns
    -------
    tuple[float, float]
        (uncertainty_on_mean, uncertainty_on_width)
    """
    rng = np.random.RandomState(random_seed)
    shuffled = np.copy(data)
    rng.shuffle(shuffled)

    subsets = [shuffled[i::n_subsamples] for i in range(n_subsamples)]
    means = []
    sigmas = []
    for subset in subsets:
        mean, sigma = unc_func(subset)
        means.append(mean)
        sigmas.append(sigma)

    return float(np.std(means) / 2), float(np.std(sigmas) / 2)


def _compute_profile_data(
    taggers: list[Tagger],
    regression_target: str,
    truth_var: str,
    reco_var: str | None,
    x_bins: list[float] | np.ndarray,
    flavour_id: int | None,
    label_var: str,
    method: str,
    n_bootstrap: int,
) -> dict:
    """Compute profile statistics for all taggers and optional reco reference.

    Parameters
    ----------
    taggers : list[Tagger]
        List of Tagger objects with perf_vars and regression_targets loaded.
    regression_target : str
        Name of the regression target (key in tagger.regression_targets).
    truth_var : str
        Column name for truth values in perf_vars.
    reco_var : str or None
        Column name for reco values in perf_vars (for nominal calibration reference).
    x_bins : array-like
        Bin edges for the profile x-axis (in physical units, e.g. GeV).
    flavour_id : int or None
        If set, only use jets with label_var == flavour_id.
    label_var : str
        Name of the label variable in tagger.labels.
    method : str
        Method for get_mean_and_width.
    n_bootstrap : int
        Number of bootstrap subsamples.

    Returns
    -------
    dict
        Dictionary with keys: bin_centres, bin_widths, reco_data (or None),
        tagger_data (dict keyed by tagger label).
    """
    x_bins = np.asarray(x_bins, dtype=float)
    bin_centres = (x_bins[:-1] + x_bins[1:]) / 2
    bin_widths = x_bins[1:] - x_bins[:-1]

    # Build reco profile if reco_var is provided
    reco_data = None
    if reco_var is not None:
        # Use the first tagger to get truth and reco (they share the same input data)
        ref_tagger = taggers[0]
        truth = ref_tagger.perf_vars[truth_var]
        reco = ref_tagger.perf_vars[reco_var]

        # Apply flavour cut
        if flavour_id is not None:
            labels = ref_tagger.labels[label_var]
            flav_mask = labels == flavour_id
            truth = truth[flav_mask]
            reco = reco[flav_mask]

        reco_medians, reco_sigmas = [], []
        reco_medians_unc, reco_sigmas_unc = [], []

        for i in range(len(x_bins) - 1):
            bin_mask = (truth >= x_bins[i]) & (truth < x_bins[i + 1])
            valid = bin_mask & ~np.isnan(truth)
            response = reco[valid] / truth[valid]

            med, sig = get_mean_and_width(response, method=method)
            med_unc, sig_unc = bootstrap_uncertainties(
                response, get_mean_and_width, n_subsamples=n_bootstrap
            )

            reco_medians.append(med)
            reco_sigmas.append(sig)
            reco_medians_unc.append(med_unc)
            reco_sigmas_unc.append(sig_unc)

        reco_data = {
            "medians": reco_medians,
            "sigmas": reco_sigmas,
            "medians_unc": reco_medians_unc,
            "sigmas_unc": reco_sigmas_unc,
        }

    # Build tagger profiles
    tagger_data = {}
    for tagger in taggers:
        truth = tagger.perf_vars[truth_var]
        reg_col = tagger.regression_targets[regression_target]
        pred = tagger.perf_vars[reg_col]

        if flavour_id is not None:
            labels = tagger.labels[label_var]
            flav_mask = labels == flavour_id
            truth = truth[flav_mask]
            pred = pred[flav_mask]

        medians, sigmas = [], []
        medians_unc, sigmas_unc = [], []

        for i in range(len(x_bins) - 1):
            bin_mask = (truth >= x_bins[i]) & (truth < x_bins[i + 1])
            valid = bin_mask & ~np.isnan(truth)
            response = pred[valid] / truth[valid]

            med, sig = get_mean_and_width(response, method=method)
            med_unc, sig_unc = bootstrap_uncertainties(
                response, get_mean_and_width, n_subsamples=n_bootstrap
            )

            medians.append(med)
            sigmas.append(sig)
            medians_unc.append(med_unc)
            sigmas_unc.append(sig_unc)

        tagger_data[tagger.label] = {
            "medians": medians,
            "sigmas": sigmas,
            "medians_unc": medians_unc,
            "sigmas_unc": sigmas_unc,
            "colour": tagger.colour,
        }

    return {
        "bin_centres": bin_centres,
        "bin_widths": bin_widths,
        "reco_data": reco_data,
        "tagger_data": tagger_data,
    }


def plot_response_profile(
    taggers: list[Tagger],
    regression_target: str,
    truth_var: str,
    reco_var: str | None = None,
    x_bins: list[float] | np.ndarray | None = None,
    x_label: str | None = None,
    y_label_median: str | None = None,
    y_label_resolution: str | None = None,
    flavour_id: int | None = None,
    label_var: str = "HadronConeExclTruthLabelID",
    method: str = "quantile_relative",
    n_bootstrap: int = 10,
    atlas_first_tag: str | None = "Simulation Internal",
    atlas_second_tag: str | None = None,
    median_y_range: tuple[float, float] | None = None,
    resolution_y_range: tuple[float, float] | None = None,
    figsize: tuple[float, float] = (5.5, 4.5),
) -> tuple[VarVsVarPlot, VarVsVarPlot]:
    """Plot median response and relative resolution profiles vs a binning variable.

    Produces two plots: one for the median response and one for the relative
    resolution, both as a function of the truth variable binned by ``x_bins``.

    Parameters
    ----------
    taggers : list[Tagger]
        List of Tagger objects with ``regression_targets`` and ``perf_vars`` loaded.
    regression_target : str
        Key in ``tagger.regression_targets`` identifying the target variable.
    truth_var : str
        Column name for truth values in ``tagger.perf_vars``.
    reco_var : str or None, optional
        Column name for reco values (nominal calibration reference). If None,
        no reco reference is drawn, by default None.
    x_bins : list[float] or np.ndarray or None, optional
        Bin edges for the x-axis (in GeV). If None, uses 10 equal-width bins
        spanning the truth variable range, by default None.
    x_label : str or None, optional
        X-axis label, by default auto-generated from ``truth_var``.
    y_label_median : str or None, optional
        Y-axis label for median plot, by default auto-generated.
    y_label_resolution : str or None, optional
        Y-axis label for resolution plot, by default auto-generated.
    flavour_id : int or None, optional
        If set, restrict to jets with ``label_var == flavour_id``, by default None.
    label_var : str, optional
        Label variable name, by default "HadronConeExclTruthLabelID".
    method : str, optional
        Method for ``get_mean_and_width``, by default "quantile_relative".
    n_bootstrap : int, optional
        Number of bootstrap subsamples, by default 10.
    atlas_first_tag : str or None, optional
        ATLAS tag line 1, by default "Simulation Internal".
    atlas_second_tag : str or None, optional
        ATLAS tag line 2, by default None.
    median_y_range : tuple[float, float] or None, optional
        Y-axis range for the median plot, by default None.
    resolution_y_range : tuple[float, float] or None, optional
        Y-axis range for the resolution plot, by default None.
    figsize : tuple[float, float], optional
        Figure size, by default (5.5, 4.5).

    Returns
    -------
    tuple[VarVsVarPlot, VarVsVarPlot]
        (median_plot, resolution_plot)
    """
    if x_bins is None:
        ref_truth = taggers[0].perf_vars[truth_var]
        x_bins = np.linspace(float(np.min(ref_truth)), float(np.max(ref_truth)), 11)
    x_bins = np.asarray(x_bins, dtype=float)

    if x_label is None:
        x_label = f"Truth {regression_target} [GeV]"
    if y_label_median is None:
        y_label_median = "Median Response"
    if y_label_resolution is None:
        y_label_resolution = "Relative Resolution"

    profile = _compute_profile_data(
        taggers=taggers,
        regression_target=regression_target,
        truth_var=truth_var,
        reco_var=reco_var,
        x_bins=x_bins,
        flavour_id=flavour_id,
        label_var=label_var,
        method=method,
        n_bootstrap=n_bootstrap,
    )

    bin_centres = profile["bin_centres"]
    bin_widths = profile["bin_widths"]

    # --- Median response plot ---
    median_kwargs: dict[str, Any] = {
        "ylabel": y_label_median,
        "xlabel": x_label,
        "atlas_first_tag": atlas_first_tag,
        "atlas_second_tag": atlas_second_tag,
        "logy": False,
        "figsize": figsize,
        "n_ratio_panels": 1,
    }
    if median_y_range is not None:
        median_kwargs["ymin"] = median_y_range[0]
        median_kwargs["ymax"] = median_y_range[1]

    plot_median = VarVsVarPlot(**median_kwargs)

    if profile["reco_data"] is not None:
        reco = profile["reco_data"]
        reco_vvv = VarVsVar(
            bin_centres,
            reco["medians"],
            reco["medians_unc"],
            x_var_widths=bin_widths,
            plot_y_std=False,
            label="Nominal Calibration",
        )
        reco_vvv.linestyle = "dashed"
        plot_median.add(reco_vvv, reference=True)

    for label, data in profile["tagger_data"].items():
        vvv = VarVsVar(
            bin_centres,
            data["medians"],
            data["medians_unc"],
            x_var_widths=bin_widths,
            plot_y_std=False,
            label=label,
            colour=data["colour"],
        )
        # If no reco reference, make first tagger the reference
        is_ref = profile["reco_data"] is None and label == list(profile["tagger_data"])[0]
        plot_median.add(vvv, reference=is_ref)

    plot_median.axis_top.axhline(1, c="k", ls=":", alpha=0.5)
    plot_median.draw()

    # --- Resolution plot ---
    resolution_kwargs: dict[str, Any] = {
        "ylabel": y_label_resolution,
        "xlabel": x_label,
        "atlas_first_tag": atlas_first_tag,
        "atlas_second_tag": atlas_second_tag,
        "logy": False,
        "figsize": figsize,
        "n_ratio_panels": 1,
    }
    if resolution_y_range is not None:
        resolution_kwargs["ymin"] = resolution_y_range[0]
        resolution_kwargs["ymax"] = resolution_y_range[1]

    plot_resolution = VarVsVarPlot(**resolution_kwargs)

    if profile["reco_data"] is not None:
        reco = profile["reco_data"]
        reco_vvv = VarVsVar(
            bin_centres,
            reco["sigmas"],
            reco["sigmas_unc"],
            x_var_widths=bin_widths,
            plot_y_std=False,
            label="Nominal Calibration",
        )
        reco_vvv.linestyle = "dashed"
        plot_resolution.add(reco_vvv, reference=True)

    for label, data in profile["tagger_data"].items():
        vvv = VarVsVar(
            bin_centres,
            data["sigmas"],
            data["sigmas_unc"],
            x_var_widths=bin_widths,
            plot_y_std=False,
            label=label,
            colour=data["colour"],
        )
        is_ref = profile["reco_data"] is None and label == list(profile["tagger_data"])[0]
        plot_resolution.add(vvv, reference=is_ref)

    plot_resolution.draw()

    return plot_median, plot_resolution


def plot_response_distribution(
    taggers: list[Tagger],
    regression_target: str,
    truth_var: str,
    reco_var: str | None = None,
    bins: int = 80,
    bins_range: tuple[float, float] = (0.5, 1.5),
    flavour_id: int | None = None,
    label_var: str = "HadronConeExclTruthLabelID",
    atlas_first_tag: str | None = "Simulation Internal",
    atlas_second_tag: str | None = None,
    figsize: tuple[float, float] = (6, 5),
) -> HistogramPlot:
    """Plot overall response distribution (pred/truth) for each tagger.

    Parameters
    ----------
    taggers : list[Tagger]
        List of Tagger objects with ``regression_targets`` and ``perf_vars`` loaded.
    regression_target : str
        Key in ``tagger.regression_targets`` identifying the target variable.
    truth_var : str
        Column name for truth values in ``tagger.perf_vars``.
    reco_var : str or None, optional
        Column name for reco values (nominal calibration reference), by default None.
    bins : int, optional
        Number of histogram bins, by default 80.
    bins_range : tuple[float, float], optional
        Range for the histogram, by default (0.5, 1.5).
    flavour_id : int or None, optional
        If set, restrict to jets with ``label_var == flavour_id``, by default None.
    label_var : str, optional
        Label variable name, by default "HadronConeExclTruthLabelID".
    atlas_first_tag : str or None, optional
        ATLAS tag line 1, by default "Simulation Internal".
    atlas_second_tag : str or None, optional
        ATLAS tag line 2, by default None.
    figsize : tuple[float, float], optional
        Figure size, by default (6, 5).

    Returns
    -------
    HistogramPlot
        The response distribution plot.
    """
    plot = HistogramPlot(
        ylabel="a.u.",
        xlabel="Response (pred / truth)",
        atlas_first_tag=atlas_first_tag,
        atlas_second_tag=atlas_second_tag,
        figsize=figsize,
        n_ratio_panels=1,
    )

    # Common histogram kwargs
    hist_kwargs: dict[str, Any] = {
        "bins": bins,
        "bins_range": bins_range,
        "norm": True,
        "underoverflow": False,
    }

    # Add reco reference if provided
    if reco_var is not None:
        ref_tagger = taggers[0]
        truth = ref_tagger.perf_vars[truth_var]
        reco = ref_tagger.perf_vars[reco_var]

        if flavour_id is not None:
            labels = ref_tagger.labels[label_var]
            flav_mask = labels == flavour_id
            truth = truth[flav_mask]
            reco = reco[flav_mask]

        valid = ~np.isnan(truth) & (truth != 0)
        response = reco[valid] / truth[valid]
        hist = Histogram(response, label="Nominal Calibration", **hist_kwargs)
        hist.linestyle = "dashed"
        plot.add(hist, reference=True)

    # Add tagger predictions
    for tagger in taggers:
        truth = tagger.perf_vars[truth_var]
        reg_col = tagger.regression_targets[regression_target]
        pred = tagger.perf_vars[reg_col]

        if flavour_id is not None:
            labels = tagger.labels[label_var]
            flav_mask = labels == flavour_id
            truth = truth[flav_mask]
            pred = pred[flav_mask]

        valid = ~np.isnan(truth) & (truth != 0)
        response = pred[valid] / truth[valid]
        hist = Histogram(response, label=tagger.label, colour=tagger.colour, **hist_kwargs)
        is_ref = reco_var is None and tagger is taggers[0]
        plot.add(hist, reference=is_ref)

    plot.axis_top.axvline(1, c="k", ls=":", alpha=0.5)
    plot.draw()

    return plot
