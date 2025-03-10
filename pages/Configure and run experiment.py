from dash import html, dcc, callback, Input, Output, State, callback_context, no_update, Patch, ALL
import dash
from Graphs import Graphs
from Experiment import ExperimentMD
from FunctionParser import FunctionParser
import plotly.graph_objects as plotly 
import torch
import json 
import base64
import numpy as np
from PresetFuncs import AnisotropicQuadratic, Rastrigin, Rosenbrock, SimplexObjective, Booth, Ackley

dash.register_page(__name__, path="/run-experiment")

explanation_md = r"""
Here you can configure and run your own experiments with the mirror descent algorithm.
There are two different types of experiments available: 
* Minimising an objective function using mirror descent.
* Approximating a function with a simple regression model, using mirror descent to minimise the loss over training. 
"""

loading_md = r"""
There are several pre-configured experiments which you can load. Alternatively you have the option to save experiments after running,
and upload the configuration to run again.
"""

# constructor functions 
def construct_md_settings(idx, batch_size=500, bregman="EUCLID", loss="MSE", lr=0.01):
    return html.Div([
    dcc.Markdown(f"**Mirror Descent Options ({idx})**"),
    html.Div([
        html.Label("Batch Size"),
        dcc.Input(type="number", value=batch_size, style={"marginBottom": "5px"}, className="input-values", id={"type": "batch-size-input", "index": idx}),
    ], className="input-row"),
    html.Div([
        html.Label("Bregman"),
        dcc.Dropdown(
            options=[
                {"label": "Euclidean", "value": "EUCLID"},
                {"label": "KL", "value": "KL"},
                {"label": "Mahalanobis", "value": "MAHALANOBIS"},
                {"label": "Itakura-Saito", "value": "ITAKURA-SAITO"}
            ]    
        , id={"type": "bregman-input", "index": idx},className="bregman-loss-input", value=bregman)
    ], className = "input-row"),
    html.Div([
        html.Label("Loss"),
        dcc.Dropdown(
            options = [
                {"label": "MSE", "value": "MSE"},
                {"label": "MAE", "value": "MAE"},
                {"label": "Huber", "value": "Huber"}
            ]
        , id={"type": "loss-input", "index": idx}, className="bregman-loss-input", value=loss)
    ], className = "input-row"),
    html.Div([
        html.Label("Learning Rate"),
        dcc.Input(type="number", value=lr, step=0.001, min=0, style={"marginBottom": "5px"}, className="input-values", id={"type": "lr-input", "index": idx}),
    ], className="input-row")
    ], className="model-settings")


def construct_model_settings(idx, layers=2, neurons=10, epochs=2000):
    
    return html.Div([
    dcc.Markdown(f"**Model options ({idx})**"),
    html.Div([
        html.Label("Layers"),
        dcc.Input(type="number", value=layers, style={"marginBottom": "5px"}, className="input-values", id={"type": "layers-input", "index": idx})
    ], className="input-row"),
    html.Div([
        html.Label("Neurons"),
        dcc.Input(type="number", value=neurons, style={"marginBottom": "5px"}, className="input-values", id={"type": "neuron-input", "index": idx})
    ], className="input-row"),
    html.Div([
        html.Label("Epochs"),
        dcc.Input(type="number", value=epochs, style={"marginBottom": "5px"}, className="input-values", id={"type": "epoch-input", "index": idx})
    ], className="input-row"),
], className="model-settings-left", id=f"model-settings-{idx}")

def construct_mini_settings(idx, init_x=0.5, init_y=0.5, iterations=100, lr=0.01, p1=0.2, p2=0.3, p3=0.5, Q="2, 0, 0, 1", bregman="EUCLID"):
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
        ], className="input-row", id={"type": "Q-input-row", "index": idx})
    ], className="settings", id={"type": "minimise-settings", "index": idx})


function_md_settings = html.Div([
    dcc.Markdown("**Function / Data values**"),
    html.Div([
        html.Label("Function"),
        dcc.Input(type="text", value="X**2 + 3*X", style={"marginBottom": "5px"}, className="input-function", id="function-input")
    ], className="input-row"),
    html.Div([
        html.Label("Input range"),
        dcc.Input(type="number", value=-5, style={"marginBottom": "5px"}, className="input-values", id="data-lbound-input"),
        dcc.Input(type="number", value=5, style={"marginBottom": "5px"}, className="input-values", id="data-ubound-input"),
    ], className="input-row"),
    html.Div([
        html.Label("Samples"),
        dcc.Input(type="number", value=500, style={"marginBottom": "5px"}, className="input-values", id="num-samples-input"),
    ], className="input-row"),
    construct_md_settings(1)
    
], className="model-settings", id="function-md-settings")



minimise_config = html.Div([html.Div([
    dcc.Markdown("**Objective function and algorithm parameters**"),
    html.Div([
        html.Label("Function Presets"),
        dcc.Dropdown(
            options=[
                {"label": "Custom", "value": "CUSTOM"},
                {"label": "Anisotropic", "value": "ANISO"},
                {"label": "3D Simplex", "value": "SIMPLEX"},
                {"label": "Rosenbrock", "value": "ROSENBROCK"},
                {"label": "Rastrigin", "value": "RASTRIGIN"},
                {"label": "Booth", "value": "BOOTH"},
                {"label": "Ackley", "value": "ACKLEY"},
                
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
        dcc.Input(type="number", value=0.0, max=1, min=0, step=0.01, style={"marginBottom": "5px"}, className="input-values", id="noise-input")
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

approx_config = html.Div([
                    construct_model_settings(1), function_md_settings],
                id="approx-config", className="option-columns-mlp hidden")


metrics_data = [
    {"name": "Objective Value", "value": 0.2345},
    {"name": "Gradient Norm", "value": 0.000123},
    {"name": "Bregman Divergence", "value": 1.234e-5},
]

def construct_experiment_results(idx, metrics_dict):
       # function converts the metrics_dict into a table
    table_rows = []
    for key, value in metrics_dict.items():
        
        if isinstance(value, list):
            # skip arrays like step_sizes
            continue
        elif isinstance(value, float):
            display_value = f"{value:.5f}"
            row_value = str(display_value)
        else:
            row_value = str(value)
        table_rows.append(
            html.Tr([
                html.Td(key, className="metric-name"),
                html.Td(row_value, className="metric-value")
            ])
        )

    return html.Div([
        html.H4(f"Experiment {idx} results", className="experiment-header"),
        html.Table(
            className="metrics-table",
            children=[html.Tbody(table_rows)]
        )
    ],
    className="experiment-result",
    id={"type": "experiment-result", "index": idx})




# placeholder variables to be updated via callback
minimise_run_button = html.Button("Run Experiment", className="run-button", n_clicks=0, id="run-button-minimise")
minimise_add_button = html.Button("+", className="add-button", n_clicks=1, id="add-button-minimise", title="add a configuration")
minimise_save_button = html.Button("Save", className="save-button", id="save-button-minimise", disabled=True, title="Cannot save until experiment has ran", n_clicks=0)
minimise_remove_button = html.Button("-", className="add-button", id="remove-button-minimise", disabled=True, title="Must be at least one configuration", n_clicks=0)

approximate_run_button = html.Button("Run Experiment", className="run-button", n_clicks=0, id="run-button-approximate")
approximate_add_button = html.Button("+", className="add-button", n_clicks=1, id="add-button-approximate", title="add a configuration")
approximate_save_button = html.Button("Save", className="save-button", id="save-button-approximate", disabled=True,  title="Cannot save until experiment has ran", n_clicks=0)
approximate_remove_button = html.Button("-", className="add-button", id="remove-button-approximate", disabled=True, title="Must be at least one configuration", n_clicks=0)


run_button_container = html.Div([minimise_run_button, minimise_add_button, minimise_remove_button, approximate_run_button , minimise_save_button, approximate_add_button, approximate_remove_button, approximate_save_button], id="run-button-container")
experiment_figs = html.Div([], id="experiment-output", className="experiment-graphs")
experiment_results = html.Div([
    html.H3("Metrics", id="metrics-header"),
], id="experiment-metrics", className="experiment-metrics")
experiment_settings_type_store = dcc.Store(id="experiment-settings-type", data="minimise")

config_options = html.Div([
    dcc.Markdown("#### Configuration", className="markdown-config", id="config-title"),
    html.Div([
        html.Button("Approximate", className="config-button", id="approx-button", n_clicks=0),
        html.Button("Minimise",   className="config-button-clicked", id="min-button",   n_clicks=0)
    ], id="top-div-config"),
    minimise_config,
    approx_config,
    run_button_container
], className="configuration-options", id="config-options")

# dcc.store components to act as global variables for callback logic 
# stores for the current number of experiment configurations 
num_experiments_min_store = dcc.Store(id="num-experiments-min", data=1)
num_experiments_approx_store = dcc.Store(id="num-experiments-approx", data=1)

# stores for the last run configuration 
last_run_config_min_store = dcc.Store(id="last-min-config", data=None)
last_run_config_approx_store = dcc.Store(id="last-approx-config", data=None)

# boolean flags which are used to initiate the loading of a minimisation/approximation experiment from json
load_min_bool = dcc.Store(id="load-min", data=False)
load_approx_bool = dcc.Store(id="load-approx", data=False)

# download component that gets triggered when a save button is clicked 
run_config_download = dcc.Download(id="save-config")

# stores the current number save button clicks 
mini_save_clicks_store = dcc.Store(id="mini-save-clicks", data=0)
approx_save_clicks_store = dcc.Store(id="approx-save-clicks", data=0)

# stores the current number of approx/mini button clicks 
global_approx_clicks_store = dcc.Store(id="approx-clicks", data=0)
global_min_clicks_store = dcc.Store(id="min-clicks", data=0)

# stores the currently inputted Q matrices as tensors for use in experiment callbacks
Q_store = dcc.Store(id="Q-store", data=None)

# dim store 
dim_store = dcc.Store(id="dim-store", data=1)

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
    num_experiments_approx_store,
    global_approx_clicks_store,
    global_min_clicks_store,
    last_run_config_approx_store,
    last_run_config_min_store,
    run_config_download,
    mini_save_clicks_store,
    approx_save_clicks_store,
    load_min_bool,
    load_approx_bool,
    Q_store,
    dim_store
    
], style={"padding": "5px 20px 20px 20px"})



# callback to hide/show necessary/optional inputs for each function preset
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
    prevent_initial_call=True
)
def add_preset_variable_inputs(preset_function, children):
    if preset_function == "ANISO":
        return ["input-row"]*5 + ["input-row hidden"]*3 + ["a*(x - optx)**2 + b*(y-opty)"]
    elif preset_function == "SIMPLEX":
        return ["input-row hidden"]*4 + ["input-row"] + ["input-row"]*3 + ["sum(q * log(q / p))"]
    elif preset_function == "ROSENBROCK":
        return ["input-row"]*2 + ["input-row hidden"]*2 + ["input-row"] + ["input-row hidden"]*3 + ["(a - x)**2 + b*(y - x**2)**2"]
    elif preset_function == "RASTRIGIN":
        return ["input-row hidden"]*4 + ["input-row"] + ["input-row hidden"]*3 + ["20 + (x**2 - 10*cos(2*pi*x)) + (y**2 - 10*cos(2*pi*y))"]
    elif preset_function == "BOOTH":
        return ["input-row hidden"]*4 + ["input-row"] + ["input-row hidden"]*3 + ["(x + 2*y - 7)**2 + (2*x + y -5)**2"]
    elif preset_function == "ACKLEY":
        return ["input-row hidden"]*4 + ["input-row"] + ["input-row hidden"] + ["-20*e(-0.2*sqrt(0.5*(x**2 + y**2))) - e(0.5*(cos(2*pi*x) + cos(2*pi*y))) + e + 20"]
    elif preset_function == "CUSTOM":
        return ["input-row hidden"]*5 + ["input-row hidden"]*3 + ["X**2 + Y**2"]
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


# callback to add/remove configurations for approximation experiment
@callback(
    Output("config-options", "children", allow_duplicate=True),
    Output("num-experiments-approx", "data", allow_duplicate=True),
    Input("add-button-approximate", "n_clicks"),
    Input("remove-button-approximate", "n_clicks"),
    State("config-options", "children"),
    State("num-experiments-approx", "data"),
    State("experiment-settings-type", "data"),
    prevent_initial_call=True
)
def update_configuration_approx(n_clicks_add, n_clicks_remove, current_children, num_experiments, experiment_type):
    ctx = callback_context
    if not ctx.triggered:
        return no_update

    triggered_prop = ctx.triggered[0]["prop_id"]

    # adding a configuration
    if "add-button-approximate" in triggered_prop:
        if (not n_clicks_add) or (n_clicks_add <= num_experiments) or (experiment_type != "approximate"):
            return no_update
        num_experiments += 1
        print(f"add approx ran - {num_experiments}")
        
        new_settings = html.Div(
            [construct_model_settings(num_experiments), construct_md_settings(num_experiments)],
            id=f"new-settings-{num_experiments}", className="option-columns-mlp"
        )
        updated_children = Patch()
        updated_children.insert(-1, new_settings)
        return updated_children, num_experiments

    # removing a configuration
    elif "remove-button-approximate" in triggered_prop:
        if (not n_clicks_remove) or (n_clicks_remove == num_experiments) or (experiment_type != "approximate"):
            return no_update
        num_experiments -= 1
        print(f"remove approx ran - {num_experiments}")
        
        updated_children = Patch()
    
        del updated_children[-2]
        return updated_children, num_experiments

    return no_update

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
            id=f"new-settings-{num_experiments}", className="option-columns-mlp"
        )
        updated_children = Patch()
        updated_children.insert(-1, new_settings)
        return updated_children, num_experiments

    # removing a configuration
    elif "remove-button-minimise" in triggered_prop:
        if (not n_clicks_remove) or (n_clicks_remove == num_experiments) or (experiment_type != "minimise"):
            return no_update
        num_experiments -= 1
        print(f"remove min ran - {num_experiments}")
        
        updated_children = Patch()
    
        del updated_children[-2]
        return updated_children, num_experiments

    return no_update

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
    Output("remove-button-approximate", "disabled"),
    Input("num-experiments-approx", "data")
) 
def disable_enable_remove_button_approximate(num_experiments):
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
    base_options = [
        {"label": "Euclidean", "value": "EUCLID"},
        {"label": "KL", "value": "KL"},
        {"label": "Mahalanobis", "value": "MAHALANOBIS"},
        {"label": "Itakura-Saito", "value": "ITAKURA-SAITO"}
    ]
    if current_dim == 1:
        for opt in base_options:
            if opt["value"] == "MAHALANOBIS":
                
                opt["disabled"] = True
    return [base_options]*num_experiments


# assumes batch mirror descent, and automatically sets the batch size to be equal to the number of samples
# if user changes the batch size for mirror descent it shouldn't change back unless the sample number is changed
@callback(
    Output("batch-size-input", "value"),
    Input("num-samples-input", "value")
)
def update_batch_size(num_samples):
    return num_samples

# callback to add input field for positive definite matrix Q when mahalanobis distance is selected
@callback(
    Output({"type": "Q-input-row", "index": ALL}, "className"),
    Input({"type": "bregman-mini-input", "index": ALL}, "value")
)
def show_Q_input(bregman_fields):
    new_q_input_classes = []
    for i in range(len(bregman_fields)):
        if bregman_fields[i] == "MAHALANOBIS":
            new_q_input_classes.append("input-row")
        else:
            new_q_input_classes.append("input-row hidden")

    return new_q_input_classes

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
            print("matrix nums: ", numbers)

            required_count = dim * dim
            if len(numbers) != required_count:
                classnames.append("input-function invalid")
                qs.append([None, None])
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
                    qs.append([None, None])

        except Exception:
            classnames.append("input-function invalid")
            qs.append([None, None])

    return classnames, qs


    
         
    

# callback triggers when either the minimise or approximate button is clicked and updates experiment_type store
# also makes sure to remove any extra configurations that have been added for the previous type of experiment
@callback(
    Output("config-options", "children", allow_duplicate=True),
    Output("approx-button", "className"),
    Output("min-button", "className"),
    Output("experiment-settings-type", "data"),
    Output("approx-clicks", "data"),
    Output("min-clicks", "data"),
    Output("num-experiments-min", "data", allow_duplicate=True),
    Output("num-experiments-approx", "data", allow_duplicate=True),
    Output("minimise-config", "className"),
    Output("approx-config", "className"),
    Input("approx-button", "n_clicks"),
    Input("min-button", "n_clicks"),
    State("config-options", "children"),
    State("approx-clicks", "data"),
    State("min-clicks", "data"),
    prevent_initial_call=True
   
)
def update_experiment_settings(approx_clicks, min_clicks, current_children, g_approx_clicks, g_min_clicks):
    # this callback sometimes gets triggered when adding an experiment, as the min or approx buttons are reloaded and 
    # this counts as a condition for the callback
    # to combat this, im keeping track of the actual clicks with a dcc.Store element so I can check if the button has actually been clicked
    print("update_experiment_settings ran")
    context = callback_context
    
    new_children_mini = current_children[:2] + [minimise_config, approx_config] + [current_children[-1]]
    new_children_approx = current_children[:2] + [minimise_config, approx_config] + [current_children[-1]]
    if not context.triggered:
    
        return new_children_mini, "config-button", "config-button-clicked", "minimise", g_approx_clicks, g_min_clicks, 1, 1, "option-columns-mlp", "option-columns-mlp hidden"
    else:
        # find which button was clicked then update accordingly
        button_triggered = context.triggered[0]["prop_id"].split(".")[0]
        if button_triggered == "min-button":
            if min_clicks == g_min_clicks:
                return no_update
            g_min_clicks = min_clicks
            return new_children_mini, "config-button", "config-button-clicked", "minimise",g_approx_clicks, g_min_clicks, 1, 1, "option-columns-mlp", "option-columns-mlp hidden"
        elif button_triggered == "approx-button": 
            if approx_clicks == g_approx_clicks:
                return no_update
            
            g_approx_clicks = approx_clicks
        
            return new_children_approx, "config-button-clicked", "config-button", "approximate", g_approx_clicks, g_min_clicks, 1, 1, "option-columns-mlp hidden", "option-columns-mlp"
        else:
            if approx_clicks == g_approx_clicks:
               
                return no_update
       
            g_min_clicks = min_clicks
            return new_children_mini, "config-button", "config-button-clicked", "minimise", g_approx_clicks, g_min_clicks, 1, 1, "option-columns-mlp", "option-columns-mlp hidden"


# updates which run button (approx or minimise) to display based on the experiment type store
@callback(
    Output("run-button-minimise", "className"),
    Output("add-button-minimise", "className"),
    Output("save-button-minimise", "className"),
    Output("remove-button-minimise", "className"),
    Output("run-button-approximate", "className"),
    Output("add-button-approximate", "className"),
    Output("save-button-approximate", "className"),
    Output("remove-button-approximate", "className"),
    Input("experiment-settings-type", "data"),
)
def update_run_button(experiment_type):
    if experiment_type == "minimise":
        return "run-button", "add-button", "save-button", "add-button", "run-button hidden", "add-button hidden", "save-button hidden", "add-button hidden"
    elif experiment_type == "approximate":
        return "run-button hidden", "add-button hidden", "save-button hidden", "add-button hidden", "run-button", "add-button", "save-button", "add-button"
    else: 
        return ValueError("Unrecognised experiment type")

# run an approximation experiment, taking in single or multiple experiment configurations
# overlays the graphs to directly compare results immediately
@callback(
    Output("experiment-output", "children", allow_duplicate=True),
    Output("run-button-approximate", "n_clicks"),
    Output("save-button-approximate", "disabled"),
    Output("last-approx-config", "data"),
    Input("run-button-approximate", "n_clicks"),
    State({"type": "layers-input", "index": ALL}, "value"),
    State({"type": "neuron-input", "index": ALL}, "value"),
    State({"type": "epoch-input", "index": ALL}, "value"),
    State("function-input", "value"),
    State("data-lbound-input", "value"),
    State("data-ubound-input", "value"),
    State("num-samples-input", "value"),
    State({"type": "batch-size-input", "index": ALL}, "value"),
    State({"type": "bregman-input", "index": ALL}, "value"),
    State({"type": "loss-input", "index": ALL}, "value"),
    State({"type": "lr-input", "index": ALL}, "value"),
    State("num-experiments-approx", "data"),
    State("Q-store", "data"),
    prevent_initial_call=True,
    allow_duplicate=True,
    suppress_callback_exceptions=True,
    debug=False
)
def run_experiment_mlp(n_clicks, layers, neurons, epochs, objective_string, range_min, range_max,
                        n_samples, batch_size, bregman, loss, lr, num_experiments, q_store):
    if n_clicks != 0:       
        # parse from string the function to approximate
        parser = FunctionParser(objective_string)
        parser.test_function()
        fta = parser.string_to_lambda()
        # test the function on a value of 1
        print(fta(torch.tensor(1)))

        # instantiate the experiment object
        experiment = ExperimentMD(fta, bregman=bregman[0], Q=q_store[0][0], Q_inv=q_store[0][1])
        print("Experiment instantiated")
        experiment.criterion = experiment.losses[loss[0]]

        # insantiate the graph object
        graph = Graphs()

        
        # run the first experiment, generate and return figures
        experiment.run_experiment_mlp(range_min, range_max, n_samples, batch_size[0], layers[0],
                                       neurons[0], epochs[0], float(lr[0]))

        loss_fig = graph.create_loss_curve(experiment.loss_logs)
        gradient_fig = graph.create_gradient_norm_graph(experiment.gradient_logs)
        divergence_fig = graph.create_divergence_graph(experiment.divergence_logs)
        results_fig = graph.create_function_approximation_plot(experiment.prediction_data)

        for i in range(1, num_experiments):
            # clear the experiment logs for the next experiment
            experiment.clear() 
            # update bregman and loss
            experiment.bregman, experiment.criterion, experiment.Q, experiment.Q_inv = bregman[i], experiment.losses[loss[i]], q_store[i][0], q_store[i][1]
            # run the new experiment
            experiment.run_experiment_mlp(range_min, range_max, n_samples, batch_size[i], layers[i],
                                       neurons[i], epochs[i], float(lr[i]))
            # update graphs
            loss_fig, gradient_fig, divergence_fig, results_fig = graph.update_all_graphs_approx(experiment.loss_logs, experiment.gradient_logs,
                                                                                                 experiment.divergence_logs, experiment.prediction_data, i+1)
        
        experiments_dict = create_experiment_dict_approx(num_experiments, layers, neurons, epochs, batch_size, lr, bregman, loss)
        experiment_state = {
            "configuration": {
                "experiment_type": "approximate",  
                "function": objective_string,
                "range_min": range_min,
                "range_max": range_max,
                "num_samples": n_samples  
            },
            "experiments": experiments_dict,
            "results": {
                "loss_logs": experiment.loss_logs,
                "gradient_logs": experiment.gradient_logs,
                "divergence_logs": experiment.avg_divergence_logs,
                "prediction_logs": experiment.prediction_data
            },
            "figures": {
                "loss_fig": loss_fig.to_plotly_json(),
                "gradient_fig": gradient_fig.to_plotly_json(),
                "divergence_fig": divergence_fig.to_plotly_json(),
                "results_fig": results_fig.to_plotly_json()
            }
        }

        
        return [dcc.Graph(figure=loss_fig, id="loss-curve", config={'responsive': True},className="graph"),
                dcc.Graph(figure=gradient_fig, id="gradient-fig",config={'responsive': True}, className="graph"),
                dcc.Graph(figure=divergence_fig, id="divergence-fig",config={'responsive': True}, className="graph"),
                dcc.Graph(figure=results_fig, id="results_fig",config={'responsive': True}, className="graph")],0, False, experiment_state
                    
    else: 
        return no_update

# function takes in the current preset type (custom/rosenbrock/simplex etc..)
# returns the correct objective function with correct initialisation parameters
def get_objective_function(preset_value, objective_string, a, b, q1, q2, q3, optx, opty, noise_std=0.0):
    if preset_value == "CUSTOM":
        # use the function parser for custom functions to generate the lambda expression
        parser = FunctionParser(objective_string)
        return parser.string_to_lambda()  
    else:
        presets = {
            "ANISO": lambda: AnisotropicQuadratic(a=float(a),
                                                  b=float(b),
                                                  optimum=torch.tensor([optx, opty]),
                                                  noise_std=noise_std),
            "SIMPLEX": lambda: SimplexObjective(weights=torch.tensor([q1, q2, q3]),
                                                noise_std=noise_std),
            "ROSENBROCK": lambda: Rosenbrock(a=float(a),
                                             b=float(b),
                                             noise_std=noise_std),
            "RASTRIGIN": lambda: Rastrigin(noise_std=noise_std),
            "BOOTH": lambda: Booth(noise_std=noise_std),
            "ACKLEY": lambda: Ackley(noise_std=noise_std)
        }
        return presets[preset_value]()
    
# sets up the initial points for different function types and determines the dimension
def setup_inits(preset_function, second_input_bool, init_x, init_y, p1s, p2s, p3s):
    if preset_function == "CUSTOM":
            if second_input_bool[0]==False:
                inits = [[float(x), float(y)] for x, y in zip(init_x, init_y)]
                print("?")
                dim = 2
                test = [1, 2]
            else: 
                inits = [float(x) for x in init_x]
                dim = 1
                test = 1
    elif preset_function == "SIMPLEX":
        inits = [[float(p1), float(p2), float(p3)] for p1, p2, p3 in zip(p1s, p2s, p3s)]
        dim = 3
    else:
        inits = [[float(x), float(y)] for x, y in zip(init_x, init_y)]
        dim = 2
    return inits, dim
        
# run a minimisation experiment, taking in single or multiple experiment configurations
# overlays the graphs to directly compare results immediately
@callback(
    Output("experiment-output", "children"),
    Output("run-button-minimise", "n_clicks"),
    Output("save-button-minimise", "disabled"), 
    Output("last-min-config", "data"),
    Output("experiment-metrics", "children"),
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
    prevent_initial_call=True,
    allow_duplicate=True,
    suppress_callback_exceptions=True
    
)
def run_experiment_minimise(n_clicks, objective_string, init_x, init_y, iter,
                            lr, bregman, num_experiments, second_input_bool, q_store,
                            p1s, p2s, p3s, q1, q2, q3, a, b, optx, opty, noise_std,
                            preset_function, current_metrics, q_strings):
    if n_clicks != 0:
        # parse objective function from string 
        print(second_input_bool)
        print(f"inputted initials {init_x} {init_y}")
        
        # defines the inits variable for 3 cases:  either 1d/2d custom functions, simplex functions, rest of preset functions (all 2d)
        inits, dim = setup_inits(preset_function, second_input_bool, init_x, init_y, p1s, p2s, p3s)
        print(inits)
        print(init_y)
        
        objective = get_objective_function(preset_function, objective_string, a, b, q1, q2, q3, optx, opty, noise_std=noise_std)
    
        print(q_store)

        # minimum/optimum always known for preset functions, not for user input functions
        if preset_function != "CUSTOM":
            optimum = objective(objective.optimum)
            optimum_coords = objective.optimum

        else:
            optimum, optimum_coords = None, None

        # instantiate experiment object
        experiment = ExperimentMD(objective, bregman=bregman[0], Q=torch.tensor(q_store[0][0], dtype=torch.float64),
                                   Q_inv=torch.tensor(q_store[0][1], dtype=torch.float64), x_star=optimum_coords, f_star=optimum, dim=dim)
        print("Experiment instantiated")
        print(experiment.Q)
        experiment_metrics = []
        # run experiment, generate and return figures
        print(experiment.objective)
        experiment.run_experiment_minimise(inits[0], iter[0], float(lr[0]))
        experiment_metrics.append(experiment.gather_metrics())
        print("experiment complete")

        # instantiate the graph class
        graph = Graphs()

        optimisation_path_fig = graph.create_optimisation_path_graph(experiment.minimisation_guesses, experiment.objective, dim)
        gradient_fig = graph.create_gradient_norm_graph(experiment.gradient_logs)
        divergence_fig = graph.create_divergence_graph(experiment.avg_divergence_logs)
        dual_fig = graph.create_dual_space_trajectory_graph(experiment.optimiser.logs["dual"],experiment.objective, dim)

        for i in range(1, num_experiments):
            experiment.clear()
            experiment.bregman, experiment.Q, experiment.Q_inv = bregman[i],  torch.tensor(q_store[i][0], dtype=torch.float64), torch.tensor(q_store[i][1], dtype=torch.float64)
            experiment.run_experiment_minimise(inits[i], iter[i], float(lr[i]))
            experiment_metrics.append(experiment.gather_metrics())
            optimisation_path_fig, gradient_fig, divergence_fig, dual_fig = graph.update_all_graphs_min(experiment.minimisation_guesses, experiment.gradient_logs,
                                                                                                        experiment.avg_divergence_logs, experiment.optimiser.logs["dual"],
                                                                                                        experiment.objective, i+1, dim)

        
        new_metrics = []
        for i in range(num_experiments):
            new_metrics.append(construct_experiment_results(i+1, experiment_metrics[i]))

        metrics = [current_metrics[0]] + new_metrics
        # store the run configuration for saving 
        experiments_dict = create_experiment_dict_min(num_experiments, init_x, init_y, iter, lr, bregman, second_input_bool, q_strings, p1s, p2s, p3s)
        metrics_dict = create_compiled_metrics_dicts(num_experiments, experiment_metrics)
        experiment_state = {
            "configuration": {
                "experiment_type": "minimise",  
                "function": objective_string, 
                "function_preset": preset_function,
                "var_a": a,
                "var_b": b,
                "opt_x": optx,
                "opt_y": opty,
                "noise": noise_std,
                "q1": q1,
                "q2": q2,
                "q3": q3     
            },
            "experiments": experiments_dict,
            "metrics": metrics_dict,
            "figures": {
                "optim_fig": optimisation_path_fig.to_plotly_json(),
                "dual_optim_fig": dual_fig.to_plotly_json(),
                "gradient_fig": gradient_fig.to_plotly_json(),
                "divergence_fig": divergence_fig.to_plotly_json(),
            }
}

        return [dcc.Graph(figure=optimisation_path_fig, id="optimisation-path-fig", config={'responsive': True},className="graph"),
                dcc.Graph(figure=dual_fig, id="dual-fig",config={'responsive': True}, className="graph"),
                dcc.Graph(figure=gradient_fig, id="gradient-fig",config={'responsive': True}, className="graph"),
                dcc.Graph(figure=divergence_fig, id="divergence-fig",config={'responsive': True}, className="graph")], 0, False, experiment_state, metrics
    
    else:
        return no_update


# function that creates a dictionary of the different experiment configurations used for a minimisation run 
def create_experiment_dict_min(num_experiments, init_x, init_y, iter, lr, bregman, second_input_bool, qs, p1s, p2s, p3s):
    experiments_dict = {}
    for i in range(num_experiments):
        experiments_dict[f"experiment-{i+1}"] = {
            "initial_value_x": init_x[i],
            "initial_value_y": init_y[i],
            "iterations": iter[i],
            "learning_rate": lr[i],
            "bregman": bregman[i],
            "p1": p1s[i],
            "p2": p2s[i],
            "p3": p3s[i],
            "Q": qs[i] 
    }
    return experiments_dict

def create_compiled_metrics_dicts(num_experiments, metric_dicts):
    metric_dict_compiled = {}
    for i in range(num_experiments):
        metric_dict_compiled[f"experiment-{i+1}-metrics"] = metric_dicts[i]
    return metric_dict_compiled
            
def create_experiment_dict_approx(num_experiments, layers, neurons, epochs, batch_size, lr, bregman, loss):
    experiments_dict = {}
    for i in range(num_experiments):
        experiments_dict[f"experiment-{i+1}"] = {
            "layers": layers[i],
            "neurons": neurons[i],
            "epochs": epochs[i],
            "learning_rate": lr[i],
            "batch_size": batch_size[i],
            "bregman": bregman[i],
            "loss": loss[i]
    }
    return experiments_dict

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
    print(current_config["experiments"])
    print(last_config["experiments"])
    if (current_config["configuration"] != last_config["configuration"]) or (current_config["experiments"] != last_config["experiments"]):
        return True
    else:
        return False


# this callback listens for changes in any input paramaters for the approximation variant.
# if configuration has changed since the last run, then disable the save button to prevent users saving a run that has results for a different configuration
@callback(
    Output("save-button-approximate", "disabled", allow_duplicate=True),
    Input({"type": "layers-input", "index": ALL}, "value"),
    Input({"type": "neuron-input", "index": ALL}, "value"),
    Input({"type": "epoch-input", "index": ALL}, "value"),
    Input("function-input", "value"),
    Input("data-lbound-input", "value"),
    Input("data-ubound-input", "value"),
    Input("num-samples-input", "value"),
    Input({"type": "batch-size-input", "index": ALL}, "value"),
    Input({"type": "bregman-input", "index": ALL}, "value"),
    Input({"type": "loss-input", "index": ALL}, "value"),
    Input({"type": "lr-input", "index": ALL}, "value"),
    Input("num-experiments-approx", "data"),
    State("last-approx-config", "data"),
    prevent_initial_call=True
)
def listen_then_disable_save_approx(layers, neurons, epochs, objective_string, range_min, range_max,
                        n_samples, batch_size, bregman, loss, lr, num_experiments, last_config):
    # if num_experiments > len(layers):
    #     return no_update
    experiments_dict = create_experiment_dict_approx(num_experiments, layers, neurons, epochs, batch_size, lr, bregman, loss)
    current_config = {
            "configuration": {
                "experiment_type": "approximate",  
                "function": objective_string,
                "range_min": range_min,
                "range_max": range_max,
                "num_samples": n_samples      
            }}
    current_config["experiments"] = experiments_dict

    if last_config == None: 
        return no_update
    print(current_config["experiments"])
    print(last_config["experiments"])
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
    Output("save-button-approximate", "n_clicks"),
    Input("save-button-minimise", "n_clicks"),
    Input("save-button-approximate", "n_clicks"),
    State("last-min-config", "data"),
    State("last-approx-config", "data"),
    prevent_initial_call=True
)
def download_minimise_experiment(n_clicks_min, n_clicks_approx, experiment_data_min, experiment_data_approx):
    # triggers a download upon save button being clicked
    # have to store n_clicks in a global dcc.store in order to compare to the save buttons value
    # as this callback gets triggered when the user adds a configuration due to the save button reloading 
    ctx = callback_context
    if not ctx.triggered:
        return no_update

    triggered_prop = ctx.triggered[0]["prop_id"]
    # Check if the triggered prop_id exactly matches "save-button-minimise.n_clicks"
    print("save button n_clicks = ", n_clicks_min)
    if "save-button-minimise" in triggered_prop:
        if n_clicks_min == 0 or not experiment_data_min:
            return no_update
        
        json_str = json.dumps(experiment_data_min, indent=2)

        return dict(content=json_str, filename="experiment.json"), 0, 0
    
    elif "save-button-approximate" in triggered_prop:
        if n_clicks_approx == 0 or not experiment_data_approx:
            return no_update
        
        json_str = json.dumps(experiment_data_approx, indent=2)

        return dict(content=json_str, filename="experiment.json"), 0, 0
    
    else: 
        return no_update


# builds a matching minimise-configuration from the saved json
def build_minimise_config_from_saved(saved_state):
    experiments = saved_state.get("experiments", {})
    metrics = saved_state.get("metrics", {})
    metric_configs = []
    experiment_configs = []
    for i in range(len(experiments)):
        exp = experiments.get(f"experiment-{i+1}")
        experiment_configs.append(construct_mini_settings(i+1,exp.get("initial_value_x"),
                                                        exp.get("initial_value_y"),
                                                        exp.get("iterations"),
                                                        exp.get("learning_rate"),
                                                        exp.get("p1"),
                                                        exp.get("p2"),
                                                        exp.get("p3"),
                                                        exp.get("Q"),
                                                        exp.get("bregman"),))
        
    for i in range(len(experiments)):
        metric = metrics.get(f"experiment-{i+1}-metrics", {})
        metric_configs.append(construct_experiment_results(i+1, metric))
    print("length of experiment configs ", len(experiment_configs))
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
                {"label": "3D Simplex", "value": "SIMPLEX"},
                {"label": "Rosenbrock", "value": "ROSENBROCK"},
                {"label": "Rastrigin", "value": "RASTRIGIN"},
                {"label": "Booth", "value": "BOOTH"},
                {"label": "Ackley", "value": "ACKLEY"},
                
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
        dcc.Input(type="number", value=saved_state["configuration"].get("noise"), max=1, min=0, step=0.01, style={"marginBottom": "5px"}, className="input-values", id="noise-input")
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
    ], className="input-row hidden", id="q3-input-row")] + experiment_configs
    , className= "settings", id="inner-div")], id="minimise-config", className="option-columns-mlp")

    return minimise_config, metric_configs

def build_approximate_config_from_saved(saved_state):
    experiments = saved_state.get("experiments", {})

    experiment_configs = [] 
    num_experiments = len(experiments)
    for i in range(num_experiments):
        print(i+1)
        exp = experiments.get(f"experiment-{i+1}")
        experiment_configs.append([construct_model_settings(i+1, exp.get("layers"),
                                                            exp.get("neurons"),
                                                            exp.get("epochs")),
                                    construct_md_settings(i+1,exp.get("batch_size"),
                                                            exp.get("bregman"),
                                                            exp.get("loss"),
                                                            exp.get("learning_rate"))])
    settings_divs = []
    for i in range(1, num_experiments):
        settings_divs.append(
            html.Div([
                experiment_configs[i][0], experiment_configs[i][1]
            ],id=f"new-settings-{i+1}", className="option-columns-mlp"
        )
        )
    function_md_settings = html.Div([
        dcc.Markdown("**Function / Data values**"),
        html.Div([
            html.Label("Function"),
            dcc.Input(type="text", value=saved_state["configuration"].get("function", ""), style={"marginBottom": "5px"}, className="input-function", id="function-input")
        ], className="input-row"),
        html.Div([
            html.Label("Input range"),
            dcc.Input(type="number", value=saved_state["configuration"].get("range_min"), style={"marginBottom": "5px"}, className="input-values", id="data-lbound-input"),
            dcc.Input(type="number", value=saved_state["configuration"].get("range_max"), style={"marginBottom": "5px"}, className="input-values", id="data-ubound-input"),
        ], className="input-row"),
        html.Div([
            html.Label("Samples"),
            dcc.Input(type="number", value=saved_state["configuration"].get("num_samples"), style={"marginBottom": "5px"}, className="input-values", id="num-samples-input"),
        ], className="input-row"),
        experiment_configs[0][1]
    
    ], className="settings", id="function-md-settings")

    if len(experiment_configs) > 1: 
        approx_config = [html.Div([experiment_configs[0][0], function_md_settings], id="approx-config", className="option-columns-mlp")] +[settings_divs[i] for i in range(len(settings_divs))]
    else: 
        approx_config = html.Div([
            experiment_configs[0][0], function_md_settings],
        id="approx-config", className="option-columns-mlp")

    return approx_config

# rebuilds the graphs from the saved experiment json
def build_experiment_results_from_saved(saved_state, type):
    
    figs = saved_state.get("figures", {})
    graphs = []
    if type=="minimise":
    
        if "optim_fig" in figs:
            optim_fig = plotly.Figure(figs["optim_fig"])
            graphs.append(dcc.Graph(figure=optim_fig, id="optimisation-path-fig", config={'responsive': True}, className="graph"))
        if "dual_optim_fig" in figs:
            dual_fig = plotly.Figure(figs["dual_optim_fig"])
            graphs.append(dcc.Graph(figure=dual_fig, id="divergence-fig", config={'responsive': True}, className="graph"))
    
        if "gradient_fig" in figs:
            gradient_fig = plotly.Figure(figs["gradient_fig"])
            graphs.append(dcc.Graph(figure=gradient_fig, id="gradient-fig", config={'responsive': True}, className="graph"))
        if "divergence_fig" in figs:
            divergence_fig = plotly.Figure(figs["divergence_fig"])
            graphs.append(dcc.Graph(figure=divergence_fig, id="divergence-fig", config={'responsive': True}, className="graph"))
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

@callback(
    Output("config-options", "children", allow_duplicate=True),
    Output("experiment-output", "children", allow_duplicate=True),
    Output("experiment-metrics", "children", allow_duplicate=True),
    Output("num-experiments-min", "data", allow_duplicate=True),
    Output("num-experiments-approx", "data", allow_duplicate=True),
    Output("add-button-minimise", "n_clicks", allow_duplicate=True),
    Output("approx-button", "className", allow_duplicate=True),
    Output("min-button", "className", allow_duplicate=True),
    Output("experiment-settings-type", "data", allow_duplicate=True),
    Output("minimise-config", "className", allow_duplicate=True),
    Output("approx-config", "className", allow_duplicate=True),
    Input("upload-load", "n_clicks"),
    State("upload-config", "contents"),
    State("upload-config", "filename"),
    State("config-options", "children"),
    State("experiment-metrics", "children"),
    prevent_initial_call=True
)
def load_experiment(load_clicks, contents, filename, current_children, current_children_metrics):
    
    triggered = callback_context.triggered
    if not triggered:
        return no_update
    
    triggered_id = triggered[0]["prop_id"]
    if triggered_id != "upload-load.n_clicks" or load_clicks == 0:
        return no_update, no_update
    
    print("load experiment ran")
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    saved_experiment = json.loads(decoded.decode("utf-8"))

    num_experiments = len(saved_experiment.get("experiments", {}))
    if saved_experiment["configuration"].get("experiment_type") == "minimise":

        minimise_config_new, metrics = build_minimise_config_from_saved(saved_experiment)
        print("minimise config len ", len(minimise_config_new))
        new_children = current_children[:2] + [minimise_config_new, approx_config] + [current_children[-1]]

        new_experiment_results = build_experiment_results_from_saved(saved_experiment, "minimise")
        new_metrics = [current_children_metrics[0]] + metrics

        return new_children, new_experiment_results, new_metrics, num_experiments, no_update, num_experiments, "config-button", "config-button-clicked", "minimise", "option-columns-mlp", "option-columns-mlp hidden"
    
    elif saved_experiment["configuration"].get("experiment_type") == "approximate":
        approx_config_new = build_approximate_config_from_saved(saved_experiment)
        if num_experiments == 1:
            new_children = current_children[:2] + [minimise_config] + [approx_config_new] + [current_children[-1]]
        else:
            new_children = current_children[:2] + [minimise_config] + [approx_config_new[i] for i in range(len(approx_config_new))] + [current_children[-1]]

        new_experiment_results = build_experiment_results_from_saved(saved_experiment, "approximate")

        return new_children, new_experiment_results, no_update, num_experiments, num_experiments, "config-button-clicked", "config-button", "approximate", "option-columns-mlp hidden", "option-columns-mlp"







    
    
