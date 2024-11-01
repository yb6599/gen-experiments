import matplotlib.pyplot as plt
import numpy as np
import pysindy as ps

from . import config
from .plotting import compare_coefficient_plots, plot_pde_training_data
from .typing import ProbData
from .utils import (
    FullSINDyTrialData,
    SINDyTrialData,
    coeff_metrics,
    integration_metrics,
    make_model,
    unionize_coeff_matrices,
)

name = "pdes"
lookup_dict = vars(config)
metric_ordering = {
    "coeff_precision": "max",
    "coeff_f1": "max",
    "coeff_recall": "max",
    "coeff_mae": "min",
    "coeff_mse": "min",
    "mse_plot": "min",
    "mae_plot": "min",
}


def diffuse1D_periodic(t, u, dx, nx):
    u = np.reshape(u, nx)
    uxx = ps.differentiation.SpectralDerivative(d=2, axis=0)._differentiate(u, dx)
    return np.reshape(uxx, nx)


def burgers1D_periodic(t, u, dx, nx):
    u = np.reshape(u, nx)
    uxx = ps.differentiation.SpectralDerivative(d=2, axis=0)._differentiate(u, dx)
    ux = ps.differentiation.SpectralDerivative(d=1, axis=0)._differentiate(u, dx)
    return np.reshape((0.1 * uxx - u * ux), nx)


def ks_periodic(t, u, dx, nx):
    u = np.reshape(u, nx)
    ux = ps.differentiation.SpectralDerivative(d=1, axis=0)._differentiate(u, dx)
    uxx = ps.differentiation.SpectralDerivative(d=2, axis=0)._differentiate(u, dx)
    uxxxx = ps.differentiation.SpectralDerivative(d=4, axis=0)._differentiate(u, dx)
    return np.reshape(-uxx - uxxxx - u * ux, nx)


def kdv_periodic(t, u, dx, nx):
    u = np.reshape(u, nx)
    ux = ps.differentiation.SpectralDerivative(d=1, axis=0)._differentiate(u, dx)
    uxxx = ps.differentiation.SpectralDerivative(d=3, axis=0)._differentiate(u, dx)
    return np.reshape(6 * u * ux - uxxx, nx)


pde_setup = {
    "diffuse1D_periodic": {
        "rhsfunc": {"func": diffuse1D_periodic, "dimension": 1},
        "input_features": ["u"],
        "time_args": [0.1, 8],
        "coeff_true": [{"u_11": 1}],
        "spatial_grid": np.linspace(-8, 8, 64),
        "init_cond": np.exp(-((np.linspace(-8, 8, 64) + 2) ** 2) / 2),
    },
    "burgers1D_periodic": {
        "rhsfunc": {"func": burgers1D_periodic, "dimension": 1},
        "input_features": ["u"],
        "time_args": [0.1, 10],
        "coeff_true": [{"u_11": 0.1, "uu_1": -1}],
        "spatial_grid": np.linspace(-8, 8, 64),
        "init_cond": np.exp(-((np.linspace(-8, 8, 64) + 2) ** 2) / 2),
    },
    "ks_periodic": {
        "rhsfunc": {"func": ks_periodic, "dimension": 1},
        "input_features": ["u"],
        "time_args": [0.4, 100],
        "coeff_true": [
            {"u_11": -1, "u_1111": -1, "uu_1": -1},
        ],
        "spatial_grid": np.linspace(0, 100, 512),
        "init_cond": (np.cos(np.linspace(0, 100, 512))) * (
            1 + np.sin(np.linspace(0, 100, 512) - 0.5)
        ),
    },
    "kdv_periodic": {
        "rhsfunc": {"func": kdv_periodic, "dimension": 1},
        "input_features": ["u"],
        "time_args": [0.1, 8],
        "coeff_true": [{"uu_1": 6, "u_111": -1}],
        "spatial_grid": np.linspace(0, 60, 64),
        "init_cond": 0.5 * (1 / np.cosh(np.linspace(0, 60, 64)))**2,
    },
}


def run(
    data: ProbData,
    diff_params: dict,
    feat_params: dict,
    opt_params: dict,
    display: bool = True,
    return_all: bool = False,
) -> dict | tuple[dict, SINDyTrialData | FullSINDyTrialData]:
    dt = data.dt
    t_train = data.t_train
    x_train = data.x_train
    x_test = data.x_test
    x_dot_test = data.x_dot_test
    x_train_true = data.x_train_true
    coeff_true = data.coeff_true
    input_features = data.input_features
    model = make_model(input_features, dt, diff_params, feat_params, opt_params)

    model.fit(x_train, t=t_train)
    coeff_true, coefficients, feature_names = unionize_coeff_matrices(model, coeff_true)

    trial_data: SINDyTrialData = {
        "dt": dt,
        "coeff_true": coeff_true,
        "coeff_fit": coefficients,
        "feature_names": feature_names,
        "input_features": input_features,
        "t_train": t_train,
        "x_true": x_train_true,
        "x_train": x_train,
        "smooth_train": model.differentiation_method.smoothed_x_,
        "x_test": x_test,
        "x_dot_test": x_dot_test,
        "model": model,
        "t_end": data.t_end,
        "spatial_grid": data.spatial_grid,
    }
    if display:
        plot_pde_panel(trial_data)

    metrics = coeff_metrics(coefficients, coeff_true)
    metrics.update(integration_metrics(model, x_test, t_train, x_dot_test))
    if return_all:
        return (metrics, trial_data)
    return metrics


def plot_pde_panel(trial_data: FullSINDyTrialData):
    trial_data["model"].print()
    plot_pde_training_data(
        trial_data["x_train"],
        trial_data["x_true"],
        trial_data["smooth_train"],
        trial_data["t_end"],
        trial_data["spatial_grid"],
    )
    compare_coefficient_plots(
        trial_data["coeff_fit"],
        trial_data["coeff_true"],
        input_features=trial_data["input_features"],
        feature_names=trial_data["feature_names"],
    )
    plt.show()
