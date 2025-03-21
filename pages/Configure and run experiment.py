from dash import html, dcc, callback, Input, Output, State, callback_context, no_update, Patch, ALL, MATCH, clientside_callback, set_props
import dash
from dash.exceptions import PreventUpdate
from Graphs import Graphs
from Experiment import ExperimentMD
from FunctionParser import FunctionParser
import plotly.graph_objects as plotly 
import torch
import json 
import base64
import numpy as np
from PresetFuncs import AnisotropicQuadratic, Rastrigin, Rosenbrock, SimplexObjective, Booth, Ackley, CubicObjective, ExponentialObjective2D
from experiment_utils import (add_highlight, get_corresponding_value, setup_inits, create_compiled_metrics_dicts,
                               create_experiment_dict_min, get_objective_function, remove_highlights, clone_fig_shallow)
import os
import time
import re
import plotly.io as pio 
dash.register_page(__name__, path="/run-experiment")
graph = Graphs()
explanation_md = r"""
Here you can configure and run your own experiments with the mirror descent algorithm.
You can input your own custom function, 1D or 2D, or choose from the selection preset functions, which allow for some customisation in terms of curvature or where the optimum lies. 

"""

loading_md = r"""
There are several pre-configured experiments which you can load, each of which shows a scenario in which the corresponding mirror map is inherently suited to. Alternatively you have the option to save experiments after running,
and upload the configuration to run again.
"""

# default configurations to load in

default_config_path = os.path.join(os.path.dirname(__file__), '../experiments/base_experiment.json')
with open(default_config_path, 'r') as f:
    default_config = json.load(f)

euclidean_config_path = os.path.join(os.path.dirname(__file__), '../experiments/euclidean_experiment.json')
with open(euclidean_config_path, 'r') as f:
    euclidean_config = json.load(f)

mahalanobis_config_path = os.path.join(os.path.dirname(__file__), "../experiments/mahalanobis_experiment.json")
with open(mahalanobis_config_path, 'r') as f:
    mahalanobis_config = json.load(f)

kl_config_path = os.path.join(os.path.dirname(__file__), '../experiments/kl_experiment.json')
with open(kl_config_path, 'r') as f:
    kl_config = json.load(f)

itakura_config_path = os.path.join(os.path.dirname(__file__), '../experiments/itakura_experiment.json')
with open(itakura_config_path, 'r') as f:
    itakura_config = json.load(f)


config_dict = {
    "EUCLID": euclidean_config,
    "KL": kl_config,
    "MAHALANOBIS": mahalanobis_config,
    "ITAKURA-SAITO": itakura_config
}


def build_experiment_results_from_saved(saved_state, type):
    
    figs = saved_state.get("figures", {})
    graphs = []
    if type=="minimise":
    
        if "optim_fig" in figs:
            optim_fig = pio.from_json(figs["optim_fig"])
            graphs.append(optim_fig)
        if "optim_fig_3d" in figs:
            optim_fig3d = pio.from_json(figs["optim_fig_3d"])
            graphs.append(optim_fig3d)
        if "dual_optim_fig" in figs:
            dual_fig = pio.from_json(figs["dual_optim_fig"])
            graphs.append(dual_fig)
        if "gradient_fig" in figs:
            gradient_fig = pio.from_json(figs["gradient_fig"])
            graphs.append(gradient_fig)
        if "divergence_fig" in figs:
            divergence_fig = pio.from_json(figs["divergence_fig"])
            graphs.append(divergence_fig)
    elif type=="approximate":
        if "loss_fig" in figs:
            loss_fig = plotly.Figure(figs["loss_fig"])
            graphs.append(dcc.Graph(figure=loss_fig, id="optimisation-path-fig", config={'responsive': True}, className="graph"))
        if "gradient_fig" in figs:
            gradient_fig = plotly.Figure(figs["gradient_fig"])
            graphs.append(dcc.Graph(figure=gradient_fig, id="gradient-fig", config={'responsive': True}, className="graph"))
        if "divergence_fig" in figs:
            divergence_fig = plotly.Figure(figs["divergence_fig"])
            graphs.append(dcc.Graph(figure=divergence_fig, id="divergence-fig", config={'responsive': True}, className="graph"))
        if "results_fig" in figs:
            results_fig = plotly.Figure(figs["results_fig"])
            graphs.append(dcc.Graph(figure=results_fig, id="divergence-fig", config={'responsive': True}, className="graph"))
        
    return graphs




displaying_point_values_div = html.Div([
    dcc.Markdown(children="Figures to display: ", id="logging-markdown", className="logging-markdown"),
    html.Button("Trajectory (3D)", className="fig-button clicked", id="optim-3d-button"),
    html.Button("Trajectory (2D / 1D)", className="fig-button clicked", id="optim-contour-button"),
    html.Button("Bregman Divergence", className="fig-button clicked", id="div-button"),
    html.Button("Gradient Norm", className="fig-button clicked", id="grad-button"),
    html.Button("Trajectory (Dual Space)", className="fig-button clicked", id="dual-button"),

],id="hovered-values", className="log-container")



# CONSTRUCTOR FUNCTIONS 
# -------------------------------------------
def construct_mini_settings(idx, init_x=0.5, init_y=0.5, iterations=100, lr=0.01, p1=0.2, p2=0.3, p3=0.5, Q="2, 0, 0, 1", bregman="EUCLID"):
    # builds a number of input containers for an experiment configuration
    return html.Div([
        dcc.Markdown(f"**Algorithm parameters ({idx})**"),
        html.Div([
            html.Label("Initial value (X)"),
            dcc.Input(type="number", value=init_x, style={"marginBottom": "5px"},
                      className="input-values", id={"type": "initial-value-input", "index": idx}),
        ], className="input-row", id={"type": "init-row", "index": idx}),
        html.Div([
            html.Label("Initial value (Y)"),
            dcc.Input(type="number", value=init_y, style={"marginBottom": "5px"},
                      className="input-values", id={"type": "initial-value-input-2", "index": idx}),
        ], className="input-row", id={"type": "init-row-2", "index": idx}),
        html.Div([
            html.Label("Initial value (p1)"),
            dcc.Input(type="number", value=p1, style={"marginBottom": "5px"},
                      className="input-values", id={"type": "simplex-initial-value-input", "index": idx}),
        ], className="input-row hidden", id={"type": "simplex-init-row", "index": idx}),
        html.Div([
            html.Label("Initial value (p2)"),
            dcc.Input(type="number", value=p2, style={"marginBottom": "5px"},
                      className="input-values", id={"type": "simplex-initial-value-input-2", "index": idx}),
        ], className="input-row hidden", id={"type": "simplex-init-row-2", "index": idx}),
        html.Div([
            html.Label("Initial value (p3)"),
            dcc.Input(type="number", value=p3, style={"marginBottom": "5px"},
                      className="input-values", id={"type": "simplex-initial-value-input-3", "index": idx}),
        ], className="input-row hidden", id={"type": "simplex-init-row-3", "index": idx}),
        html.Div([
            html.Label("Iterations"),
            dcc.Input(type="number", value=iterations, style={"marginBottom": "5px"},
                      className="input-values", id={"type": "number-iterations-input", "index": idx}),
        ], className="input-row"),
        html.Div([
            html.Label("Learning Rate"),
            dcc.Input(type="number", value=lr, step=0.001, min=0, style={"marginBottom": "5px"},
                      className="input-values", id={"type": "lr-mini-input", "index": idx}),
        ], className="input-row"),
        html.Div([
            html.Label("Bregman"),
            dcc.Dropdown(
                options=[
                    {"label": "Euclidean", "value": "EUCLID"},
                    {"label": "KL", "value": "KL"},
                    {"label": "Mahalanobis", "value": "MAHALANOBIS"},
                    {"label": "Itakura-Saito", "value": "ITAKURA-SAITO"}
                ],
                value=bregman,
                id={"type": "bregman-mini-input", "index": idx},
                className="dropdown"
            )
        ], className="input-row"),
        html.Div([
            html.Label("Positive Definite Matrix"),
            dcc.Input(type="text", value=Q, step=0.001, min=0, style={"marginBottom": "5px"},
                      className="input-function", id={"type": "Q-input", "index": idx}),
        ], className="input-row hidden", id={"type": "Q-input-row", "index": idx})
    ], className="settings", id={"type": "minimise-settings", "index": idx})

# constructs metrics tables from the metrics dict returned from experiment.gather_metrics()
def construct_experiment_results(idx, metrics_dict):
    table_rows = []
    for key, value in metrics_dict.items():
        
        if isinstance(value, list):
            # skip arrays like step_sizes
            continue
        elif isinstance(value, float):
            if abs(value) > 1e6:
                display_value = f"{value:.3e}"
            else:
                display_value = f"{value:.5f}"
            row_value = str(display_value)
        else:
            row_value = str(value)
        table_rows.append(
            html.Tr(
                [
                    html.Td(key, className="metric-name"),
                    html.Td(row_value, className="metric-value")
                ],
                id={'type': 'metric-row', 'metric': key, 'table': idx},
                n_clicks=0,
                style={'cursor': 'pointer'}  # visually indicate that the row is clickable
            )
        )

    return html.Div([
        html.H4(f"Experiment {idx} results", className="experiment-header"),
        html.Table(
            className="metrics-table",
            children=[html.Tbody(table_rows)]
        )
    ],
    className="experiment-result animate-slide-in",
    id={"type": "experiment-result", "index": idx})




# builds the necessary number of configurations and each inputs value from a saved experiment
def build_minimise_config_from_saved(saved_state):
    experiments = saved_state.get("experiments", {})
    metrics = saved_state.get("metrics", {})
    metric_configs = []
    experiment_configs = []
    for i in range(len(experiments)):
        exp = experiments.get(f"experiment-{i+1}")
        config =  construct_mini_settings(i+1,exp.get("initial_value_x"),exp.get("initial_value_y"),exp.get("iterations"),exp.get("learning_rate"),exp.get("p1"),exp.get("p2"),exp.get("p3"),exp.get("Q"),exp.get("bregman"))
        if i == 0:
            experiment_configs.append(config)
        else:
            experiment_configs.append(html.Div(config, id=f"new-settings-{i+1}", className="option-columns-mlp"))
        
    for i in range(len(experiments)):
        metric = metrics.get(f"experiment-{i+1}-metrics", {})
        metric_configs.append(construct_experiment_results(i+1, metric))
    minimise_config = html.Div([html.Div([
    dcc.Markdown("**Objective function and algorithm parameters**"),
    html.Div([
        html.Label("Objective Function"),
        dcc.Input(type="text", value=saved_state["configuration"].get("function", ""), style={"marginBottom": "5px"}, className="input-function", id="function-mini-input"),
    ], className="input-row"),
    html.Div([
        html.Label("Function Presets"),
        dcc.Dropdown(
            options=[
                {"label": "Custom", "value": "CUSTOM"},
                {"label": "Anisotropic", "value": "ANISO"},
                {"label": "Cubic", "value": "CUBIC"},
                {"label": "3D Simplex", "value": "SIMPLEX"},
                {"label": "Itakura-based", "value": "ITAKURA"},
                {"label": "Rosenbrock", "value": "ROSENBROCK"},
                {"label": "Rastrigin", "value": "RASTRIGIN"},
                {"label": "Booth", "value": "BOOTH"},
                {"label": "Ackley", "value": "ACKLEY"},
                {"label": "Exponential", "value": "EXPONENTIAL"},
            ]   
        , id="preset-function-input",className="dropdown", value=saved_state["configuration"].get("function_preset", ""))
    ], className = "input-row"),
    html.Div([
        html.Label("Variable (a)"),
        dcc.Input(type="number", value=saved_state["configuration"].get("var_a"), style={"marginBottom": "5px"}, className="input-values", id="a-input")
    ], className="input-row hidden", id="a-input-row"),
    html.Div([
        html.Label("Variable (B)"),
        dcc.Input(type="number", value=saved_state["configuration"].get("var_b"), style={"marginBottom": "5px"}, className="input-values", id="b-input")
    ], className="input-row hidden", id="b-input-row"),
    html.Div([
        html.Label("Optimum (x)"),
        dcc.Input(type="number", value=saved_state["configuration"].get("opt_x"), style={"marginBottom": "5px"}, className="input-values", id="optim-x-input")
    ], className="input-row hidden", id="optim-x-input-row"),
    html.Div([
        html.Label("Optimum (y)"),
        dcc.Input(type="number", value=saved_state["configuration"].get("opt_y"), style={"marginBottom": "5px"}, className="input-values", id="optim-y-input")
    ], className="input-row hidden", id="optim-y-input-row"),
    html.Div([
        html.Label("Noise"),
        dcc.Input(type="number", value=saved_state["configuration"].get("noise"), max=50, min=0, step=0.1, style={"marginBottom": "5px"}, className="input-values", id="noise-input")
    ], className="input-row hidden", id="noise-input-row"),
    html.Div([
        html.Label("Target q1"),
        dcc.Input(type="number", value=saved_state["configuration"].get("q1"), max=1, min=0, step=0.01, style={"marginBottom": "5px"}, className="input-values", id="q1-input")
    ], className="input-row hidden", id="q1-input-row"),
    html.Div([
        html.Label("Target q2"),
        dcc.Input(type="number", value=saved_state["configuration"].get("q2"), max=1, min=0, step=0.01, style={"marginBottom": "5px"}, className="input-values", id="q2-input")
    ], className="input-row hidden", id="q2-input-row"),
    html.Div([
        html.Label("Target q3"),
        dcc.Input(type="number", value=saved_state["configuration"].get("q3"), max=1, min=0, step=0.01, style={"marginBottom": "5px"}, className="input-values", id="q3-input")
    ], className="input-row hidden", id="q3-input-row"),
    experiment_configs[0]]
    , className= "settings", id="inner-div")], id="minimise-config", className="option-columns-mlp")
    full_config = [minimise_config] + experiment_configs[1:]
    return full_config, metric_configs


# constructs the dictionary for an experiment configuration(s), for saving to a json
def construct_experiment_state(metrics_dict, experiment_config_dict, experiment_params, figures):
    experiment_state = {
            "configuration": {
                "experiment_type": "minimise",  
                "function": experiment_params[1], 
                "function_preset": experiment_params[0],
                "var_a": experiment_params[2],
                "var_b": experiment_params[3],
                "opt_x": experiment_params[7],
                "opt_y": experiment_params[8],
                "noise": experiment_params[9],
                "q1": experiment_params[4],
                "q2": experiment_params[5],
                "q3": experiment_params[6]     
            },
            "experiments": experiment_config_dict,
            "metrics": metrics_dict,
            "figures": {
                "optim_fig": pio.to_json(figures[0]),
                "optim_fig_3d": pio.to_json(figures[1]),
                "dual_optim_fig": pio.to_json(figures[2]),
                "gradient_fig": pio.to_json(figures[3]),
                "divergence_fig": pio.to_json(figures[4]),
            }
    }
    return experiment_state



# LAYOUT AND COMPONENTS 
# -----------------------------------

# constructs the figures from the default configuration to be shown when first loading the experiment page
new_experiment_results = build_experiment_results_from_saved(default_config, "minimise")

graphs_row1 = html.Div([
    html.Div([
        dcc.Loading(
            id="loading-optimisation-path-fig",
            type="default",
            color="#e8dac5",
            delay_show=1000,
            className="first-row",

            overlay_style={"visibility":"visible", "filter": "blur(2px)"},
            children=dcc.Graph(
                figure=new_experiment_results[0],
                id="optimisation-path-fig",
                config={'responsive': True},
                className="graph animate-slide-in",
            )
        )], id="optim-container", className="animate-slide-in"),
    html.Div([
        dcc.Loading(
            id="loading-optimisation-path-fig-3d",
            type="default",
            color="#e8dac5",
            className="first-row",
            delay_show=1000,
            overlay_style={"visibility":"visible", "filter": "blur(2px)"},
            children=dcc.Graph(
                figure=new_experiment_results[1],
                id="optimisation-path-fig-3d",
                config={'responsive': True},
                className="graph animate-slide-in",
            )
        )], id="optim-3d-container", className="animate-slide-in")], className="first-row")

graphs_row2 = html.Div([
    html.Div([
        dcc.Loading(
            id="loading-dual-fig",
            type="default",
            color="#e8dac5",
            delay_show=1000,
            className="second-row",

            overlay_style={"visibility":"visible", "filter": "blur(2px)"},
            children=dcc.Graph(
                figure=new_experiment_results[2],
                id="dual-fig",
                config={'responsive': True},
                className="graph animate-slide-in"
            )
    ),
    ], id="dual-container"),
    html.Div([
        dcc.Loading(
            id="loading-gradient-fig",
            type="default",
            color="#e8dac5",
            delay_show=1000,
            className="second-row",
            overlay_style={"visibility":"visible", "filter": "blur(2px)"},
            children=dcc.Graph(
                figure=new_experiment_results[3],
                id="gradient-fig",
                config={'responsive': True},
                className="graph animate-slide-in"
            ))
    ], id="grad-container"),
    html.Div([
        dcc.Loading(
            id="loading-divergence-fig",
            type="default",
            color="#e8dac5",
            delay_show=1000,
            className="second-row",

            overlay_style={"visibility":"visible", "filter": "blur(2px)"},
            children=[
                dcc.Graph(
                figure=new_experiment_results[4],
                id="divergence-fig",
                config={'responsive': True},
                className="graph animate-slide-in"
            )]
        )
    ], id="divergence-container")
    ], className="second-row")

experiment_figs = html.Div([displaying_point_values_div, graphs_row1, graphs_row2], id="experiment-output", className="experiment-graphs")


minimise_config = html.Div([html.Div([
    dcc.Markdown("**Objective function and algorithm parameters**"),
    html.Div([
        html.Label("Function Presets"),
        dcc.Dropdown(
            options=[
                {"label": "Custom", "value": "CUSTOM"},
                {"label": "Anisotropic", "value": "ANISO"},
                {"label": "Cubic", "value": "CUBIC"},
                {"label": "3D Simplex", "value": "SIMPLEX"},
                {"label": "Itakura-based", "value": "ITAKURA"},
                {"label": "Rosenbrock", "value": "ROSENBROCK"},
                {"label": "Rastrigin", "value": "RASTRIGIN"},
                {"label": "Booth", "value": "BOOTH"},
                {"label": "Ackley", "value": "ACKLEY"},
                {"label": "Exponential", "value": "EXPONENTIAL"},
            ]    
        , id="preset-function-input",className="dropdown", value="CUSTOM")
    ], className = "input-row"),
    html.Div([
        html.Label("Objective Function"),
        dcc.Input(type="text", value="X**2 + Y**2", style={"marginBottom": "5px"}, className="input-function", id="function-mini-input"),
    ], className="input-row"),
    html.Div([
        html.Label("Variable (a)"),
        dcc.Input(type="number", value=10, style={"marginBottom": "5px"}, className="input-values", id="a-input")
    ], className="input-row hidden", id="a-input-row"),
    html.Div([
        html.Label("Variable (B)"),
        dcc.Input(type="number", value=2, style={"marginBottom": "5px"}, className="input-values", id="b-input")
    ], className="input-row hidden", id="b-input-row"),
    html.Div([
        html.Label("Optimum (x)"),
        dcc.Input(type="number", value=10, style={"marginBottom": "5px"}, className="input-values", id="optim-x-input")
    ], className="input-row hidden", id="optim-x-input-row"),
    html.Div([
        html.Label("Optimum (y)"),
        dcc.Input(type="number", value=10, style={"marginBottom": "5px"}, className="input-values", id="optim-y-input")
    ], className="input-row hidden", id="optim-y-input-row"),
    html.Div([
        html.Label("Noise"),
        dcc.Input(type="number", value=0.0, max=50, min=0, step=0.1, style={"marginBottom": "5px"}, className="input-values", id="noise-input")
    ], className="input-row hidden", id="noise-input-row"),
    html.Div([
        html.Label("Target q1"),
        dcc.Input(type="number", value=0.1, max=1, min=0, step=0.01, style={"marginBottom": "5px"}, className="input-values", id="q1-input")
    ], className="input-row hidden", id="q1-input-row"),
    html.Div([
        html.Label("Target q2"),
        dcc.Input(type="number", value=0.7, max=1, min=0, step=0.01, style={"marginBottom": "5px"}, className="input-values", id="q2-input")
    ], className="input-row hidden", id="q2-input-row"),
    html.Div([
        html.Label("Target q3"),
        dcc.Input(type="number", value=0.2, max=1, min=0, step=0.01, style={"marginBottom": "5px"}, className="input-values", id="q3-input")
    ], className="input-row hidden", id="q3-input-row"),
    construct_mini_settings(1)
], className= "settings", id="inner-div")], id="minimise-config", className="option-columns-mlp")

# configuration panel buttons
minimise_run_button = html.Button("Run Experiment", className="run-button", n_clicks=0, id="run-button-minimise", disabled=False)
minimise_add_button = html.Button("+", className="add-button", n_clicks=1, id="add-button-minimise", title="add a configuration")
minimise_save_button = html.Button("Save", className="save-button", id="save-button-minimise", disabled=True, title="Cannot save until experiment has ran", n_clicks=0)
minimise_remove_button = html.Button("-", className="add-button", id="remove-button-minimise", disabled=True, title="Must be at least one configuration", n_clicks=0)
run_button_container = html.Div([minimise_run_button, minimise_add_button, minimise_remove_button, minimise_save_button], id="run-button-container")

# metrics panel
experiment_results = html.Div([
        html.H3("Metrics", id="metrics-header"),
        
        html.Div([], className="bottom-container", id="metrics-bottom")
    ], id="experiment-metrics", className="experiment-metrics hidden")

experiment_settings_type_store = dcc.Store(id="experiment-settings-type", data="minimise")

# configuration panel 
config_options = html.Div([
    dcc.Markdown("#### Configuration", className="markdown-config", id="config-title"),
    minimise_config,
    run_button_container,
    html.Div([
    ], id="configuration-bottom", className="bottom-container"),
    
], className="configuration-options", id="config-options")

# dcc.store components to act as global variables for callback logic 
# stores for the current number of experiment configurations 
num_experiments_min_store = dcc.Store(id="num-experiments-min", data=1)

# stores for the last run configuration 
last_run_config_min_store = dcc.Store(id="last-min-config", data=None)

# boolean flags which are used to initiate the loading of a minimisation/approximation experiment from json
load_min_bool = dcc.Store(id="load-min", data=False)

# download component that gets triggered when a save button is clicked 
run_config_download = dcc.Download(id="save-config")

# stores the current number save button clicks 
mini_save_clicks_store = dcc.Store(id="mini-save-clicks", data=0)

# stores the current number of approx/mini button clicks 
global_min_clicks_store = dcc.Store(id="min-clicks", data=0)
selected_metrics_store = dcc.Store(id="selected-metrics", data=[])

# stores the currently inputted Q matrices as tensors for use in experiment callbacks
Q_store = dcc.Store(id="Q-store", data=None)

# stores used for experiment callback chain
current_experiment = dcc.Store(id="current-experiment", data=0)
metrics = dcc.Store(id="metrics", data=[])
experiment_parameters = dcc.Store(id="experiment-params", data=None)
experiment_dict = dcc.Store(id="experiment-dict", data=None)

# binary store indicating whether to disable the contour figure, set to true by experiment callbacks
disable_contour_store = dcc.Store(id="disable-3d", data=False)

# dim store 
dim_store = dcc.Store(id="dim-store", data=1)
default_config_store = dcc.Store(id="default-experiment-config", data=default_config)
layout = html.Div([
    html.Div([
        html.Div([
            html.H3("Running an Experiment"),
            dcc.Markdown(explanation_md, className="padding-markdown")
        ], className="experiment-desc"),
        html.Div([
            html.H3("Loading Experiments"),
            dcc.Markdown(loading_md, className="padding-markdown"),
            html.Div([
                html.Div([
                    dcc.Dropdown(
                    options = [
                        {"label": "Euclidean", "value": "EUCLID"},
                        {"label": "KL", "value": "KL"},
                        {"label": "Mahalanobis", "value": "MAHALANOBIS"},
                        {"label": "Itakura-Saito", "value": "ITAKURA-SAITO"}
                    ]    
                , id="preset-dropdown",className="dropdown", value="EUCLID"),
                html.Button("Load", className="load-button", id="preset-load")
                ], className="preset-load"),
                html.Div([
                    dcc.Upload([
                        html.Div([
                            dcc.Markdown("**Drag and Drop** or **Select File**", id="drag-drop-md")

                        ], className="uploading-box")
                    ], className="upload", id="upload-config", multiple=False, accept=".json"),
                    html.Button("Load", className="load-button", id="upload-load")
                ], className="upload-load")               
            ], className="loading-div")
        ], className="load-experiments")
    ], className="experiment-page-toprow")
    ,
    html.Div([
        config_options,
        experiment_figs,
        experiment_results
    ], className= "experiment-div"),
    experiment_settings_type_store,
    num_experiments_min_store,
    global_min_clicks_store,
    last_run_config_min_store,
    run_config_download,
    mini_save_clicks_store,
    load_min_bool,
    Q_store,
    dim_store,
    selected_metrics_store,
    default_config_store,
    current_experiment,
    metrics,
    experiment_parameters,
    experiment_dict,
    disable_contour_store,
    dcc.Interval(
        id='experiment-interval',
        interval=500,  # in milliseconds (1 second interval, adjust as needed)
        n_intervals=0,
        disabled=True  # Initially disabled
    ),
    
    
], style={"padding": "5px 20px 20px 20px"})


# CALLBACKS 
# ----------------------------------------

# callback highlights corresponding points on gradient, divergence and dual space traj graphs when hovering over a point
# on the primal space traj graphs
@callback(
    Output('optimisation-path-fig', 'figure', allow_duplicate=True),
    Output('dual-fig', 'figure', allow_duplicate=True),
    Output('gradient-fig', 'figure', allow_duplicate=True),
    Output('divergence-fig', 'figure', allow_duplicate=True),
    Input('optimisation-path-fig', 'hoverData'),
    Input('optimisation-path-fig-3d', 'hoverData'),
    State('optimisation-path-fig', 'figure'),
    State('optimisation-path-fig-3d', 'figure'),
    State('dual-fig', 'figure'),
    State('gradient-fig', 'figure'),
    State('divergence-fig', 'figure'),
    State("preset-function-input", "value"),
    prevent_initial_call=True
)
def sync_hover_others(hover2d, hover3d, fig2d, fig3d, dual_fig, grad_fig, div_fig, preset_value):
    # determine which hover event is triggered (contour or 3d optim traj)
    ctx = callback_context
    active_hover = None
    source_fig = None

    if ctx.triggered:
        triggered_id = ctx.triggered[0]['prop_id']
        if triggered_id.startswith('optimisation-path-fig-3d'):
            active_hover = hover3d
            source_fig = fig3d
        elif triggered_id.startswith('optimisation-path-fig'):
            active_hover = hover2d
            source_fig = fig2d

    if active_hover is None:
        active_hover = hover3d if hover3d is not None else hover2d
        source_fig = fig3d if hover3d is not None else fig2d

    # clone the figures 
    new_optim = clone_fig_shallow(fig2d)
    new_dual  = clone_fig_shallow(dual_fig)
    new_grad  = clone_fig_shallow(grad_fig)
    new_div   = clone_fig_shallow(div_fig)

    new_dual = remove_highlights(new_dual)
    new_grad = remove_highlights(new_grad)
    new_div  = remove_highlights(new_div)


    # lots of try/except blocks here as this callback was very error prone when developing
    if active_hover is None:
        return new_optim, new_dual, new_grad, new_div

    try:
        point = active_hover['points'][0]
    except (KeyError, IndexError):
        return new_optim, new_dual, new_grad, new_div

    # get the point index being hovered over
    pt_index = point.get('pointIndex') or point.get('pointNumber')
    if pt_index is None:
        return new_optim, new_dual, new_grad, new_div

    curve_idx = point.get('curveNumber', 0)
    try:
        hovered_trace = source_fig['data'][curve_idx]
    except (KeyError, IndexError):
        return new_optim, new_dual, new_grad, new_div

    trace_name = hovered_trace.get('name', '')
    # ignore objective function traces e.g. contour or surface plots
    if "objective" in trace_name.lower():
        return new_optim, new_dual, new_grad, new_div

    
    m = re.search(r'\d+', trace_name)
    if not m:
        return new_optim, new_dual, new_grad, new_div
    exp_num = int(m.group(0))

    optim_x = hovered_trace.get('x', [None])[pt_index]
    optim_y = hovered_trace.get('y', [None])[pt_index]

    # use helper function to get corresponding values for gradient and divergence figs fig
    _, grad_val = get_corresponding_value(grad_fig, exp_num, pt_index, return_hovertext=False)
    _, div_val  = get_corresponding_value(div_fig, exp_num, pt_index, return_hovertext=False)

    if div_val is not None:
        # divergence can vary drastically depending on the configuration so selectively apply different levels of precision
        if div_val < 0.00001:
            div_string = f"Divergence: {div_val:.4e}<br>"
        else:
            div_string = f"Divergence: {div_val:.5f}<br>"
    else:
        div_string = ""

    # the hovertext needs to be different for the simplex figure
    if preset_value == "SIMPLEX":
        if 'hovertext' in hovered_trace and isinstance(hovered_trace['hovertext'], list):
            optim_info = hovered_trace['hovertext'][pt_index]
        else:
            optim_info = f"p1: {optim_x:.2f}, p2: {optim_y:.2f}, p3: {1 - optim_x - optim_y:.2f}"
        dual_info = get_corresponding_value(dual_fig, exp_num, pt_index, return_hovertext=True)
        custom_hovertemplate = (
            f"Primal: ({optim_info})<br>" +
            (f"Dual: ({dual_info})<br>" if dual_info is not None else "") +
            (f"Gradient: {grad_val:.2f}<br>" if grad_val is not None else "") +
            div_string +
            f"Iteration: {pt_index}" +
            "<extra></extra>"
        )
    else:
        dual_x, dual_y = get_corresponding_value(dual_fig, exp_num, pt_index, return_hovertext=False)
        custom_hovertemplate = (
            f"Primal: ({optim_x:.2f}, {optim_y:.2f})<br>" +
            (f"Dual: ({dual_x:.2f}, {dual_y:.2f})<br>" if dual_x is not None and dual_y is not None else "") +
            (f"Gradient: {grad_val:.2f}<br>" if grad_val is not None else "") +
            div_string +
            f"Iteration: {pt_index}" +
            "<extra></extra>"
        )

    # update hovered trace hover text with info from th eother figures
    new_optim['data'][curve_idx]['hovertemplate'] = custom_hovertemplate

    # add highlight markers to other figures
    new_dual = add_highlight(new_dual, "Dual space coords", exp_num, pt_index)
    new_grad = add_highlight(new_grad, "Gradient norm", exp_num, pt_index)
    new_div  = add_highlight(new_div, "Divergence", exp_num, pt_index)

    return new_optim, new_dual, new_grad, new_div


@callback(
    Output("a-input-row", "className"),
    Output("b-input-row", "className"),
    Output("optim-x-input-row", "className"),
    Output("optim-y-input-row", "className"),
    Output("noise-input-row", "className"),
    Output("q1-input-row", "className"),
    Output("q2-input-row", "className"),
    Output("q3-input-row", "className"),
    Output("function-mini-input", "value"),
    Input("preset-function-input", "value"),
    Input("config-options", "children"),
    State("function-mini-input", "value"),
    prevent_initial_call=True
)
def add_preset_variable_inputs(preset_function, children, obj_string):
    if preset_function == "ANISO":
        return ["input-row"]*5 + ["input-row hidden"]*3 + ["a*(x - optx)**2 + b*(y-opty)**2"]
    elif preset_function == "SIMPLEX":
        return ["input-row hidden"]*4 + ["input-row"] + ["input-row"]*3 + ["sum(q * log(q / p))"]
    elif preset_function == "ROSENBROCK":
        return ["input-row"]*2 + ["input-row hidden"]*2 + ["input-row"] + ["input-row hidden"]*3 + ["(a - x)**2 + b*(y - x**2)**2"]
    elif preset_function == "RASTRIGIN":
        return ["input-row hidden"]*4 + ["input-row"] + ["input-row hidden"]*3 + ["20 + (x**2 - 10*cos(2*pi*x)) + (y**2 - 10*cos(2*pi*y))"]
    elif preset_function == "BOOTH":
        return ["input-row hidden"]*4 + ["input-row"] + ["input-row hidden"]*3 + ["(x + 2*y - 7)**2 + (2*x + y -5)**2"]
    elif preset_function == "ACKLEY":
        return ["input-row hidden"]*4 + ["input-row"] + ["input-row hidden"]*3 + ["-20*e(-0.2*sqrt(0.5*(x**2 + y**2))) - e(0.5*(cos(2*pi*x) + cos(2*pi*y))) + e + 20"]
    elif preset_function == "CUSTOM":
        if obj_string != "X**2 + Y**2":
            func = obj_string
        else: 
            func = "X**2 + Y**2"
        return ["input-row hidden"]*5 + ["input-row hidden"]*3 + [func]
    elif preset_function == "EXPONENTIAL":
        return ["input-row hidden"]*2 + ["input-row"]*3 + ["input-row hidden"]*3 + ["e*(x-optx) - (x - optx) + e(y - opty) - (y - opty)"]
    elif preset_function == "CUBIC":
        return ["input-row hidden"]*2 + ["input-row"]*3 + ["input-row hidden"]*3 + ["1/3 * (|x - optx|**3 + |y - opty|**3)"]
    elif preset_function == "ITAKURA":
        return ["input-row"] + ["input-row hidden"]*3 + ["input-row"] + ["input-row hidden"]*3 + ["a/(x*y) - log(a/(x*y)) - 1"]
    else:
        return no_update

# shows/hides initial value inputs for simplex/others
# simplex needs 3 p inputs while all of the others just need x, y inputs
@callback(
    Output({"type": "init-row", "index": ALL}, "className"),
    Output({"type": "init-row-2", "index": ALL}, "className"),
    Output({"type": "simplex-init-row", "index": ALL}, "className"),
    Output({"type": "simplex-init-row-2", "index": ALL}, "className"),
    Output({"type": "simplex-init-row-3", "index": ALL}, "className"),
    State("num-experiments-min", "data"),
    Input("preset-function-input", "value")
)
def update_init_rows(num_experiments, function_preset):
    if function_preset == "SIMPLEX":
        return ["input-row hidden"]*num_experiments, ["input-row hidden"]*num_experiments, ["input-row"]*num_experiments, ["input-row"]*num_experiments, ["input-row"]*num_experiments
    else:
        return ["input-row"]*num_experiments, ["input-row"]*num_experiments, ["input-row hidden"]*num_experiments, ["input-row hidden"]*num_experiments, ["input-row hidden"]*num_experiments



@callback(
    Output({"type": "initial-value-input-2", "index" : ALL}, "disabled"),
    Input("function-mini-input", "value"),
    State("num-experiments-min", "data")
)
def update_initial_value_input(func, num_experiments):
    
    if ("Y" in func.upper()) and ("X" in func.upper()):
        return [False]*num_experiments
    else: 
        return [True]*num_experiments


# callback to add/remove configurations for a minimisation experiment
@callback(
    Output("config-options", "children", allow_duplicate=True),
    Output("num-experiments-min", "data", allow_duplicate=True),
    Input("add-button-minimise", "n_clicks"),
    Input("remove-button-minimise", "n_clicks"),
    State("config-options", "children"),
    State("num-experiments-min", "data"),
    State("experiment-settings-type", "data"),
    prevent_initial_call=True
)
def update_configuration_mini(n_clicks_add, n_clicks_remove, current_children, num_experiments, experiment_type):
    ctx = callback_context
    if not ctx.triggered:
        return no_update

    triggered_prop = ctx.triggered[0]["prop_id"]

    # adding a configuration
    if "add-button-minimise" in triggered_prop:
        if (not n_clicks_add) or (n_clicks_add <= num_experiments) or (experiment_type != "minimise"):
            return no_update
        num_experiments += 1
        print(f"add min ran - {num_experiments}")
        
        new_settings = html.Div(
            [construct_mini_settings(num_experiments)],
            id=f"new-settings-{num_experiments}", className="option-columns-mlp animate-slide-in"
        )
        updated_children = Patch()
        updated_children.insert(-2, new_settings)
        return updated_children, num_experiments

    # removing a configuration
    elif "remove-button-minimise" in triggered_prop:
        if (not n_clicks_remove) or (n_clicks_remove == num_experiments) or (experiment_type != "minimise"):
            return no_update
        num_experiments -= 1
        print(f"remove min ran - {num_experiments}")
        
        updated_children = Patch()

        del updated_children[-3]
        return updated_children, num_experiments

    return no_update

# disables/enables the add configuration button dependent on whether num experiments < 5 or not
@callback(
    Output("add-button-minimise", "disabled"),
    Input("num-experiments-min", "data")
)
def disable_add(num_experiments):
    if num_experiments < 5:
        return False
    else:
        return True

@callback(
    Output("remove-button-minimise", "disabled"),
    Input("num-experiments-min", "data")
) 
def disable_enable_remove_button_minimise(num_experiments):
    if num_experiments > 1:
        return False
    else: 
        return True
    

@callback(
    Output("dim-store", "data"),
    Input("preset-function-input", "value"),
    Input({"type": "initial-value-input-2", "index": ALL}, "disabled")
)
def update_dim_store(preset_function, second_input_bool):
    if preset_function == "CUSTOM":
        if second_input_bool[0] == True:
            return 1
        else:
            return 2
    elif preset_function == "SIMPLEX":
        return 3 

# callback to disabled function input if preset is not custom
@callback(
    Output("function-mini-input", "disabled"),
    Input("preset-function-input", "value")
)
def disable_function_input(function_preset):
    if function_preset != "CUSTOM":
        return True
    else:
        return False

# disables mahalanobis for 1d custom functions, causes error with the mirror map attempting to matmul with a 0d scalar (Q in the 1d case)
@callback(
    Output({"type": "bregman-mini-input", "index": ALL}, "options"),
    Input("dim-store", "data"),
    State("num-experiments-min", "data")
)
def manage_bregman_options(current_dim, num_experiments):
    base_options =[
            {"label": "Euclidean", "value": "EUCLID"},
            {"label": "KL", "value": "KL"},
            {"label": "Mahalanobis", "value": "MAHALANOBIS"},
            {"label": "Itakura-Saito", "value": "ITAKURA-SAITO"},
        ]
    if current_dim == 1:
        for opt in base_options:
            if opt["value"] == "MAHALANOBIS":
                
                opt["disabled"] = True
    return [base_options]*num_experiments



# callback to show/hide input field for positive definite matrix Q when mahalanobis distance is selected
# also sets default value of Q to correct dims depending on the problem selected - 3x3 for simplex or 2x2 for 2D functions
@callback(
    Output({"type": "Q-input-row", "index": ALL}, "className"),
    Output({"type": "Q-input", "index": ALL}, "value"),
    Input({"type": "bregman-mini-input", "index": ALL}, "value"),
    Input("preset-function-input", "value")
)
def show_Q_input(bregman_fields, function_preset):
    new_q_input_classes = []
    q_default_values = []
    for i in range(len(bregman_fields)):
        if bregman_fields[i] == "MAHALANOBIS":
            new_q_input_classes.append("input-row")
        else:
            new_q_input_classes.append("input-row hidden")
        if function_preset == "SIMPLEX": 
            q_default_values.append("3, 0, 0, 0, 3, 0, 0, 0, 3")
        else:
            q_default_values.append("2, 0, 0, 1")


    return new_q_input_classes, q_default_values

# callback to check if the input for the above is positive definite, shows invalid input with red border if not
@callback(
    Output({"type": "Q-input", "index": ALL}, "className"),
    Output("Q-store", "data"),
    Input({"type": "Q-input", "index": ALL}, "value"),
    State({"type": "Q-input-row", "index": ALL}, "className"),
    Input("preset-function-input", "value")
)
def check_positive_definite(input_values, total_input_components, preset_function):
    classnames = []
    qs = []
    if preset_function == "SIMPLEX":
        dim = 3
    else:
        dim = 2
    for input_str in input_values:
        try:
            numbers = list(map(float, input_str.split(',')))

            required_count = dim * dim
            if len(numbers) != required_count:
                classnames.append("input-function invalid")
                qs.append([0, 0])
            else:
                matrix = np.array(numbers).reshape((dim, dim))
                # ensuring positivity in eigenvalues
                if np.all(np.linalg.eigvals(matrix) > 0):
                    classnames.append("input-function")
                    Q = torch.tensor(matrix, dtype=torch.float64)
                    Q_inv = torch.linalg.inv(Q)
                    qs.append([Q.tolist(), Q_inv.tolist()])
                else:
                    classnames.append("input-function invalid")
                    qs.append([0, 0])

        except Exception:
            classnames.append("input-function invalid")
            qs.append([0, 0])

    return classnames, qs

@callback(
    Output("save-button-minimise", "disabled", allow_duplicate=True),
    Input("function-mini-input", "value"),
    Input({"type": "initial-value-input", "index": ALL}, "value"),
    Input({"type": "initial-value-input-2", "index": ALL}, "value"),
    Input({"type": "number-iterations-input", "index": ALL}, "value"),
    Input({"type": "lr-mini-input", "index": ALL}, "value"),
    Input({"type": "bregman-mini-input", "index": ALL}, "value"),
    Input({"type": "Q-input", "index": ALL}, "value"),
    Input("num-experiments-min", "data"),
    Input({"type": "initial-value-input-2", "index": ALL}, "disabled"),
    State("preset-function-input", "value"),
    State({"type": "simplex-initial-value-input", "index": ALL}, "value"),
    State({"type": "simplex-initial-value-input-2", "index": ALL}, "value"),
    State({"type": "simplex-initial-value-input-3", "index": ALL}, "value"),
    State("last-min-config", "data"),
    prevent_initial_call=True
)
def listen_then_disable_save_min(objective_string, init_x, init_y, iter, lr, bregman, q_strings,
                                  num_experiments, second_input_bool, preset_function, p1s, p2s, p3s, last_config):
    # this callback listens for changes in any input paramaters for the minimisation variant.
    # if configuration has changed since the last run, then disable the save button to prevent users saving a run that has results for a different configuration

    experiments_dict = create_experiment_dict_min(num_experiments, init_x, init_y, iter, lr, bregman, second_input_bool, q_strings, p1s, p2s, p3s)
    current_config = {
            "configuration": {
                "experiment_type": "minimise",  
                "function": objective_string, 
                "function_preset": preset_function     
            }}
    current_config["experiments"] = experiments_dict


    if last_config == None: 
        return no_update
    
    if (current_config["configuration"] != last_config["configuration"]) or (current_config["experiments"] != last_config["experiments"]):
        return True
    else:
        return False



# replaces "drag and drop..." text with the filename once one has been uploaded 
@callback(
        Output("drag-drop-md", "children"),
        Input("upload-config", "filename")
)
def update_upload_prompt(filename):
    if filename: 
        return filename 
    return "**Drag and Drop** or **Select File**"

@callback(
    Output("save-config", "data"),
    Output("save-button-minimise", "n_clicks"),
    Input("save-button-minimise", "n_clicks"),
    State("last-min-config", "data"),
    prevent_initial_call=True
)
def download_minimise_experiment(n_clicks_min, experiment_data_min):
    # triggers a download upon save button being clicked
    ctx = callback_context
    if not ctx.triggered:
        return no_update

    triggered_prop = ctx.triggered[0]["prop_id"]
    # Check if the triggered prop_id exactly matches "save-button-minimise.n_clicks"
    if "save-button-minimise" in triggered_prop:
        if n_clicks_min == 0 or not experiment_data_min:
            return no_update
        
        json_str = json.dumps(experiment_data_min, indent=2)

        return dict(content=json_str, filename="experiment.json"), 0
    




# updates which figures are currently being displayed as per the users choice by applying/removing the "true-hidden" class
@callback(
    Output("optim-3d-button", "className"),
    Output("optim-contour-button", "className"),
    Output("div-button", "className"),
    Output("grad-button", "className"),
    Output("dual-button", "className"),
    Output("optim-3d-container", "className"),
    Output("optim-container", "className"),
    Output("divergence-container", "className"),
    Output("grad-container", "className"),
    Output("dual-container", "className"),
    Input("optim-3d-button", "n_clicks"),
    Input("optim-contour-button", "n_clicks"),
    Input("div-button", "n_clicks"),
    Input("grad-button", "n_clicks"),
    Input("dual-button", "n_clicks"),
    State("optim-3d-button", "className"),
    State("optim-contour-button", "className"),
    State("div-button", "className"),
    State("grad-button", "className"),
    State("dual-button", "className"),
    State("optim-3d-container", "className"),
    State("optim-container", "className"),
    State("divergence-container", "className"),
    State("grad-container", "className"),
    State("dual-container", "className"),
    State("dim-store", "data")
)
def update_displayed_figures(o3d_clicks, o_clicks, div_clicks, grad_clicks, dual_clicks,
                             o3d_btn_class, o_btn_class, div_btn_class, grad_btn_class, dual_btn_class,
                             o3d_cont_class, o_cont_class, div_cont_class, grad_cont_class, dual_cont_class, dim_store):
    ctx = callback_context
    if not ctx.triggered:
        return no_update

    # safely converts a value to a string to work with
    def safe_class(val):
        if val is no_update or val is None:
            return "animate-slide-in"
        return val

    btn_classes = [no_update] * 5
    cont_classes = [no_update] * 5

    
    triggered_prop = ctx.triggered[0]["prop_id"]

    # optim-3d figure
    if triggered_prop == "optim-3d-button.n_clicks":
        if o3d_btn_class == "fig-button clicked" or dim_store == 1:
            btn_classes[0] = "fig-button"
            cont_classes[0] = "animate-slide-in true-hidden"
        else:
            btn_classes[0] = "fig-button clicked"
            cont_classes[0] = "animate-slide-in"
    else:
        btn_classes[0] = o3d_btn_class
        cont_classes[0] = o3d_cont_class

    # optim contour figure
    if triggered_prop == "optim-contour-button.n_clicks":
        if o_btn_class == "fig-button clicked":
            btn_classes[1] = "fig-button"
            cont_classes[1] = "animate-slide-in true-hidden"
        else:
            btn_classes[1] = "fig-button clicked"
            cont_classes[1] = "animate-slide-in"
    else:
        btn_classes[1] = o_btn_class
        cont_classes[1] = o_cont_class

    # divergence figure
    if triggered_prop == "div-button.n_clicks":
        if div_btn_class == "fig-button clicked":
            btn_classes[2] = "fig-button"
            cont_classes[2] = "animate-slide-in true-hidden"
        else:
            btn_classes[2] = "fig-button clicked"
            cont_classes[2] = "animate-slide-in"
    else:
        btn_classes[2] = div_btn_class
        cont_classes[2] = div_cont_class

    # gradient figure
    if triggered_prop == "grad-button.n_clicks":
        if grad_btn_class == "fig-button clicked":
            btn_classes[3] = "fig-button"
            cont_classes[3] = "animate-slide-in true-hidden"
        else:
            btn_classes[3] = "fig-button clicked"
            cont_classes[3] = "animate-slide-in"
    else:
        btn_classes[3] = grad_btn_class
        cont_classes[3] = grad_cont_class

    # dual space figure
    if triggered_prop == "dual-button.n_clicks":
        if dual_btn_class == "fig-button clicked":
            btn_classes[4] = "fig-button"
            cont_classes[4] = "animate-slide-in true-hidden"
        else:
            btn_classes[4] = "fig-button clicked"
            cont_classes[4] = "animate-slide-in"
    else:
        btn_classes[4] = dual_btn_class
        cont_classes[4] = dual_cont_class

    # if any of the figures are on their own in their respective row, adds/removes a "solo" class which decreases the width 
    # so figures arent stretched to fill the row
    first_row_indexes = [0, 1]
    for i in first_row_indexes:
        cont_classes[i] = safe_class(cont_classes[i]).replace(" solo", "")

    # determine visible containers in first row
    first_row_visibles = [1 if "true-hidden" not in safe_class(cont_classes[i]).split() else 0 for i in first_row_indexes]
    if sum(first_row_visibles) == 1:
        for i in first_row_indexes:
            if "true-hidden" not in safe_class(cont_classes[i]).split():
                cont_classes[i] = safe_class(cont_classes[i]) + " solo"

    # same for second row
    second_row_indexes = [2, 3, 4]
    for i in second_row_indexes:
        cont_classes[i] = safe_class(cont_classes[i]).replace(" solo", "")
    second_row_visibles = [1 if "true-hidden" not in safe_class(cont_classes[i]).split() else 0 for i in second_row_indexes]
    if sum(second_row_visibles) == 1:
        for i in second_row_indexes:
            if "true-hidden" not in safe_class(cont_classes[i]).split():
                cont_classes[i] = safe_class(cont_classes[i]) + " solo"

    return (btn_classes[0],
            btn_classes[1],
            btn_classes[2],
            btn_classes[3],
            btn_classes[4],
            cont_classes[0],
            cont_classes[1],
            cont_classes[2],
            cont_classes[3],
            cont_classes[4])

# disables 3d graph if function is 1d
@callback(
    Output("optim-3d-button", "className", allow_duplicate=True),
    Output("optim-3d-button", "disabled", allow_duplicate=True),
    Output("optim-3d-container", "className", allow_duplicate=True),
    Output("optim-container", "className", allow_duplicate=True),
    Input("disable-3d", "data"),
    prevent_initial_call=True
)
def disable_3d_fig(disable):
    ctx = callback_context
    if not ctx.triggered:
        return no_update
    if disable:
        return "fig-button", True, "animate-slide-in true-hidden", "animate-slide-in solo"
    else:
        return "fig-button clicked", False, "animate-slide-in", "animate-slide-in"



# clientside callback to make the figures themselves invisible when the containers are changing size due to added/removed figures
# from the above callback
# necessary as the figures take about half a second longer to change size than the containers, so it looks jittery and messy without this
clientside_callback(
    """
    function(optim3dContClass, optimContClass, divContClass, gradContClass, dualContClass) {
        // all figure ids
        var graphIds = [
            "optimisation-path-fig",
            "optimisation-path-fig-3d",
            "divergence-fig",
            "gradient-fig",
            "dual-fig"
        ];
        graphIds.forEach(function(id) {
            var graphElem = document.getElementById(id);
            if (graphElem) {
                // add "transitioning" class to hide the figure
                graphElem.classList.add("transitioning");
                // remove the 'transitioning' class after 600ms so that the plot fades in
                setTimeout(function() {
                    graphElem.classList.remove("transitioning");
                }, 800);
            }
        });
        
        
    }
    """,
    Input("optim-3d-container", "className"),
    Input("optim-container", "className"),
    Input("divergence-container", "className"),
    Input("grad-container", "className"),
    Input("dual-container", "className")
)

# rebuilds the graphs from the saved experiment json
@callback(
    Output("config-options", "children", allow_duplicate=True),
    Output("optimisation-path-fig", "figure", allow_duplicate=True),
    Output("optimisation-path-fig-3d", "figure", allow_duplicate=True),
    Output("dual-fig", "figure", allow_duplicate=True),
    Output("gradient-fig", "figure", allow_duplicate=True),
    Output("divergence-fig", "figure", allow_duplicate=True),
    Output("experiment-metrics", "children", allow_duplicate=True),
    Output("num-experiments-min", "data", allow_duplicate=True),
    Output("add-button-minimise", "n_clicks", allow_duplicate=True),
    Output("experiment-settings-type", "data", allow_duplicate=True),
    Output("minimise-config", "className", allow_duplicate=True),
    Output("experiment-metrics", "className", allow_duplicate=True),
    Input("upload-load", "n_clicks"),
    Input("preset-load", "n_clicks"),
    State("upload-config", "contents"),
    State("upload-config", "filename"),
    State("config-options", "children"),
    State("experiment-metrics", "children"),
    State("preset-dropdown", "value"),
    prevent_initial_call=True
)
def load_experiment(load_clicks, preset_load_clicks, contents, filename, current_children, current_children_metrics, preset_dropdown):
    
    triggered = callback_context.triggered
    if not triggered:
        raise PreventUpdate
    
    triggered_id = triggered[0]["prop_id"]
    if (triggered_id != "upload-load.n_clicks" and triggered_id != "preset-load.n_clicks") or (load_clicks == 0 and preset_load_clicks == 0):
        raise PreventUpdate
    

    
    print("Loading Experiment...")
    if triggered_id == "preset-load.n_clicks":
        saved_experiment = config_dict[preset_dropdown]
    
    else:

        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        saved_experiment = json.loads(decoded.decode("utf-8"))

    num_experiments = len(saved_experiment.get("experiments", {}))
    if saved_experiment["configuration"].get("experiment_type") == "minimise":
        print("?")
        minimise_config_new, metrics = build_minimise_config_from_saved(saved_experiment)
        print("minimise config len ", len(minimise_config_new))
        new_children = current_children[:1] + minimise_config_new + current_children[-2:]

        new_experiment_results = build_experiment_results_from_saved(saved_experiment, "minimise")
        new_metrics = [current_children_metrics[0]] + metrics + [current_children_metrics[-1]]

        return new_children, new_experiment_results[0], new_experiment_results[1], new_experiment_results[2], new_experiment_results[3], new_experiment_results[4],new_metrics, num_experiments, num_experiments, "minimise", "option-columns-mlp", "experiment-metrics"


# callback initialises a run of experiment configurations. This callback initialises all experiment parameters and runs the first experiment.
# after completion it enables the experiment-interval which calls run_next_experiment
@callback(
    Output("experiment-interval", "disabled", allow_duplicate=True),
    Output("current-experiment", "data", allow_duplicate=True),
    Output("metrics", "data", allow_duplicate=True),
    Output("experiment-params", "data", allow_duplicate=True),
    Output("experiment-dict", "data", allow_duplicate=True),
    Output("optimisation-path-fig", "figure", allow_duplicate=True),
    Output("optimisation-path-fig-3d", "figure", allow_duplicate=True),
    Output("dual-fig", "figure", allow_duplicate=True),
    Output("divergence-fig", "figure", allow_duplicate=True),
    Output("gradient-fig", "figure", allow_duplicate=True),
    Output("experiment-metrics", "children", allow_duplicate=True),
    Output("experiment-metrics", "className", allow_duplicate=True),
    Output("run-button-minimise", "n_clicks"),
    Output("run-button-minimise", "disabled", allow_duplicate=True), 
    Output("disable-3d", "data"),
    Input("run-button-minimise", "n_clicks"),
    State("function-mini-input", "value"),
    State({"type": "initial-value-input", "index": ALL}, "value"),
    State({"type": "initial-value-input-2", "index": ALL}, "value"),
    State({"type": "number-iterations-input", "index": ALL}, "value"),
    State({"type": "lr-mini-input", "index": ALL}, "value"),
    State({"type": "bregman-mini-input", "index": ALL}, "value"),
    State("num-experiments-min", "data"),
    State({"type": "initial-value-input-2", "index": ALL}, "disabled"),
    State("Q-store", "data"),
    State({"type": "simplex-initial-value-input", "index": ALL}, "value"),
    State({"type": "simplex-initial-value-input-2", "index": ALL}, "value"),
    State({"type": "simplex-initial-value-input-3", "index": ALL}, "value"),
    State("q1-input", "value"),
    State("q2-input", "value"),
    State("q3-input", "value"),
    State("a-input", "value"),
    State("b-input", "value"),
    State("optim-x-input", "value"),
    State("optim-y-input", "value"),
    State("noise-input", "value"),
    State("preset-function-input", "value"),
    State("experiment-metrics", "children"),
    State({"type": "Q-input", "index": ALL}, "value"),
    State("optimisation-path-fig-3d", "figure"),
    running=[Output("run-button-minimise", "disabled"), True, True],
    prevent_initial_call=True,
    allow_duplicate=True,
    suppress_callback_exceptions=True
)
def initialise_experiment_run(n_clicks, objective_string, init_x, init_y, iter,
                            lr, bregman, num_experiments, second_input_bool, q_store,
                            p1s, p2s, p3s, q1, q2, q3, a, b, optx, opty, noise_std,
                            preset_function, current_metrics, q_strings, current3dfig):
    print("init experiment ran")
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    # every time the configuration panel is reloaded, the run-button is also reloaded causing this callback to trigger even if button has not been pressed
    # hence checking for n_clicks to see if callback has definitely been triggered by the user
    if n_clicks != 0:
        inits, dim = setup_inits(preset_function, second_input_bool, init_x, init_y, p1s, p2s, p3s)
        objective = get_objective_function(preset_function, objective_string, a, b, q1, q2, q3, optx, opty, noise_std=noise_std)

        # get the optimum its coords from the objective class if its a preset function, if not then set these to None
        if preset_function != "CUSTOM":
            optimum = objective(objective.optimum)
            optimum_coords = objective.optimum

        else:
            optimum, optimum_coords = None, None

        # initialising the experiment class
        experiment = ExperimentMD(objective, bregman=bregman[0], Q=torch.tensor(q_store[0][0], dtype=torch.float64),
                                   Q_inv=torch.tensor(q_store[0][1], dtype=torch.float64), x_star=optimum_coords, f_star=optimum, dim=dim)
        
        # clear trajectories from the graph object for a new set of experiments
        graph.trajectories = []
        
        # running the experiment and constructing figures and metrics
        experiment.run_experiment_minimise(inits[0], iter[0], float(lr[0]))
        metrics = experiment.gather_metrics()
        metrics_div = construct_experiment_results(1, metrics)
        optimisation_path_fig = graph.create_optimisation_path_graph(experiment.minimisation_guesses, experiment.objective, dim)

        # for the 1d custom function case, disable the 3d graph
        if dim != 1:
            disable_3d = False
            optimisation_path_fig3d = graph.create_optimisation_path_3d_graph(experiment.minimisation_guesses, experiment.objective, dim)
        else:
            disable_3d = True
            optimisation_path_fig3d = current3dfig

        gradient_fig = graph.create_gradient_norm_graph(experiment.gradient_logs)
        divergence_fig = graph.create_divergence_graph(experiment.avg_divergence_logs)
        dual_fig = graph.create_dual_space_trajectory_graph(experiment.optimiser.logs["dual"],experiment.objective, dim)

        # create a dictionary of the experiment parameters to pass to run_next_experiment
        experiments_dict = create_experiment_dict_min(num_experiments, init_x, init_y, iter, lr, bregman, second_input_bool, q_strings, p1s, p2s, p3s)
        experiment_params = {
            "inits": inits,
            "dim": dim,
            "iter": iter,
            "optimum": optimum,
            "optimum_coords": optimum_coords,
            "objective_params": [preset_function, objective_string, a, b, q1, q2, q3, optx, opty,noise_std],
            "bregman": bregman,
            "learning_rate": lr
        }
        
        # get rid of any old metrics tables and replace with the first table of metrics from this first experiment configuration
        new_metrics_children = [current_metrics[0]] + [metrics_div] + [current_metrics[-1]]

        # enable the interval to call run_next_experiment
        interval_disabled = False
        next_experiment = 2
        return (interval_disabled, next_experiment, [metrics], experiment_params, experiments_dict, optimisation_path_fig, optimisation_path_fig3d, dual_fig, divergence_fig, gradient_fig, new_metrics_children, "experiment-metrics", 0, no_update, disable_3d)
    else:
        return [no_update]*13 + [False] + [no_update]




@callback(
    Output("experiment-interval", "disabled", allow_duplicate=True),
    Output("current-experiment", "data", allow_duplicate=True),
    Output("metrics", "data", allow_duplicate=True),
    Output("optimisation-path-fig", "figure"),
    Output("optimisation-path-fig-3d", "figure"),
    Output("dual-fig", "figure", allow_duplicate=True),
    Output("divergence-fig", "figure", allow_duplicate=True),
    Output("gradient-fig", "figure", allow_duplicate=True),
    Output("experiment-metrics", "children", allow_duplicate=True),
    Output("experiment-metrics", "className", allow_duplicate=True),
    Output("last-min-config", "data", allow_duplicate=True),
    Output("save-button-minimise", "disabled"),
    Output("run-button-minimise", "disabled"),
    Input("experiment-interval", "n_intervals"),
    State("current-experiment", "data"),
    State("experiment-params", "data"),
    State("metrics", "data"),
    State("num-experiments-min", "data"),
    State("experiment-dict", "data"),
    State("optimisation-path-fig", "figure"),
    State("optimisation-path-fig-3d", "figure"),
    State("dual-fig", "figure"),
    State("divergence-fig", "figure"),
    State("gradient-fig", "figure"),
    State("Q-store", "data"),
    State("experiment-metrics", "children"),
    State("experiment-interval", "disabled"),
    # disable the experiment-interval while the callback is running, otherwise it can cause an infinite loop if callback
    #  is triggered again before the previous experiemnt has fi nished
    running=[(Output("experiment-interval", "disabled"), True, False), (Output("run-button-minimise", "disabled"), True, True)], 
    suppress_callback_exceptions=True,
    prevent_initial_call=True,
)
def run_next_experiment(triggered, current_experiment, params, metrics,  num_experiments, experiment_dict,
                        optimisation_path_fig, optimisation_path_fig_3d, dual_fig, divergence_fig, gradient_fig,
                        q_store, current_metrics_div, interval_disabled):
    print("run next experiment ran")
    # checks if the interval has actually triggered
    if triggered == 0:
        return [no_update]*13
    
    if current_experiment > num_experiments:
        # disable the interval now that the final experiment has ran
        interval_disabled = True
        # creates multiple experiment dicts as well as final figures for each experiment completed in this looping callback
        metrics_dict = create_compiled_metrics_dicts(num_experiments, metrics)
        figures = [optimisation_path_fig, optimisation_path_fig_3d, dual_fig, divergence_fig, gradient_fig]
        experiment_state = construct_experiment_state(metrics_dict, experiment_dict, params["objective_params"], figures)
        print("All experiments complete and configurations ready for saving")
        return (interval_disabled, no_update, None, no_update, no_update, no_update, no_update, no_update, no_update, no_update, experiment_state, False, False)
      
    else:
        time1 = time.time()
        # gather all params from dict
        inits, dim, iter = params["inits"], params["dim"], params["iter"]
        optimum, optimum_coords = params["optimum"], params["optimum_coords"]

        # optimum gets converted back to a scalar rather than a tensor when stored in a dcc.Store component so must convert back
        if optimum is not None: 
            optimum = torch.tensor(optimum, dtype=torch.float64)
        bregman, lr = params["bregman"], params["learning_rate"]
        obj_params = params["objective_params"]
        objective = get_objective_function(obj_params[0], obj_params[1], obj_params[2], obj_params[3], obj_params[4], obj_params[5], obj_params[6] ,obj_params[7], obj_params[8], noise_std=obj_params[9]) 

        # run next experiment with corresponding params
        exp_idx = current_experiment-1
        experiment = ExperimentMD(objective, bregman=bregman[exp_idx], Q=torch.tensor(q_store[exp_idx][0], dtype=torch.float64),
                                  Q_inv = torch.tensor(q_store[exp_idx][1], dtype=torch.float64), x_star=optimum_coords, f_star=optimum, dim=dim)
        experiment.run_experiment_minimise(inits[exp_idx], iter[exp_idx], float(lr[exp_idx]))
        g_time1  =time.time()

        # generate figures
        optimisation_path_fig, optimisation_path_fig_3d, gradient_fig, divergence_fig, dual_fig = graph.update_all_graphs_min(experiment.minimisation_guesses, experiment.gradient_logs,
                                                                                                     experiment.avg_divergence_logs, experiment.optimiser.logs["dual"],
                                                                                                     experiment.objective, current_experiment, dim=dim)
        if dim == 1:
            optimisation_path_fig_3d = no_update
            # dont update the 3d figure for 1d objectives
        g_time2 = time.time()
        print("time taken to construct graphs: ", g_time2 - g_time1)

        # calculate metrics and construct the table div
        new_metrics = experiment.gather_metrics()
        new_metrics_div = construct_experiment_results(current_experiment, new_metrics)
        metrics.append(new_metrics)
        
        # insert the new metrics div after all of the previous experiment results
        metrics_div = Patch()
        metrics_div.insert(-1, new_metrics_div)
        time2 =time.time()
        print(f"experiment {current_experiment} ran in :", time2-time1)
        current_experiment += 1
        print(f"Next experiment number:  ({current_experiment})")
        return no_update, current_experiment, metrics, optimisation_path_fig, optimisation_path_fig_3d, dual_fig, divergence_fig, gradient_fig, metrics_div, "experiment-metrics", no_update, no_update, no_update
    

#callback loads a base experiment when the page initialises
@callback(
    Output("config-options", "children", allow_duplicate=True),
    Output("experiment-metrics", "children", allow_duplicate=True),
    Output("num-experiments-min", "data", allow_duplicate=True),
    Output("experiment-metrics", "className", allow_duplicate=True),
    Input("default-experiment-config", "data"),
    State("config-options", "children"),
    State("experiment-metrics", "children"),
    prevent_initial_call='initial_duplicate',
    suppress_callback_exceptions=True
)
def load_default_experiment(saved_experiment, config_children, metrics_children):
    if saved_experiment is None:
         raise PreventUpdate
    print("test")
    minimise_config_new, metrics = build_minimise_config_from_saved(saved_experiment)
    new_children = config_children[:1] + minimise_config_new + config_children[-2:]
    new_metrics = [metrics_children[0]] + metrics + [metrics_children[-1]]
    num_experiments = len(saved_experiment.get("experiments", {}))
    
    return new_children, new_metrics, num_experiments, "experiment-metrics"


