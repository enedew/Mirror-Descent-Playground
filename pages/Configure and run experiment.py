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
import os


dash.register_page(__name__, path="/run-experiment")
graph = Graphs()
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

default_config_path = os.path.join(os.path.dirname(__file__), '../base_experiment.json')
with open(default_config_path, 'r') as f:
    default_config = json.load(f)

import plotly.io as pio 
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

new_experiment_results = build_experiment_results_from_saved(default_config, "minimise")

graphs_row1 = html.Div([
    html.Div([
        dcc.Loading(
            id="loading-optimisation-path-fig",
            type="default",
            color="#e8dac5",
            delay_show=500,
            className="first-row",

            overlay_style={"visibility":"visible", "filter": "blur(2px)"},
            children=dcc.Graph(
                figure=new_experiment_results[0],
                id="optimisation-path-fig",
                config={'responsive': True},
                className="graph animate-slide-in",
            )
        )], id="optim-container"),
    html.Div([
        dcc.Loading(
            id="loading-optimisation-path-fig-3d",
            type="default",
            color="#e8dac5",
            className="first-row",
            delay_show=500,
            overlay_style={"visibility":"visible", "filter": "blur(2px)"},
            children=dcc.Graph(
                figure=new_experiment_results[1],
                id="optimisation-path-fig-3d",
                config={'responsive': True},
                className="graph animate-slide-in",
            )
        )], id="optim-3d-container")], className="first-row")

graphs_row2 = html.Div([
    dcc.Loading(
        id="loading-dual-fig",
        type="default",
        color="#e8dac5",
        delay_show=500,
        className="second-row",

        overlay_style={"visibility":"visible", "filter": "blur(2px)"},
        children=dcc.Graph(
            figure=new_experiment_results[2],
            id="dual-fig",
            config={'responsive': True},
            className="graph animate-slide-in"
        )
    ),
    dcc.Loading(
        id="loading-gradient-fig",
        type="default",
        color="#e8dac5",
        delay_show=500,
        className="second-row",
        overlay_style={"visibility":"visible", "filter": "blur(2px)"},
        children=dcc.Graph(
            figure=new_experiment_results[4],
            id="gradient-fig",
            config={'responsive': True},
            className="graph animate-slide-in"
        )
    ),
    dcc.Loading(
        id="loading-divergence-fig",
        type="default",
        color="#e8dac5",
        delay_show=500,
        className="second-row",

        overlay_style={"visibility":"visible", "filter": "blur(2px)"},
        children=[
            dcc.Graph(
            figure=new_experiment_results[3],
            id="divergence-fig",
            config={'responsive': True},
            className="graph animate-slide-in"
        )]
    )
    ], className="second-row")

displaying_point_values_div = html.Div([
    dcc.Markdown(children="Figures to display: ", id="logging-markdown", className="logging-markdown"),
    html.Button("Trajectory (3D)", className="fig-button clicked", id="optim-3d-button"),
    html.Button("Trajectory (Contour)", className="fig-button clicked", id="optim-contour-button"),
    html.Button("Bregman Divergence", className="fig-button clicked", id="div-button"),
    html.Button("Gradient Norm", className="fig-button clicked", id="grad-button"),
    html.Button("Trajectory (Dual Space)", className="fig-button clicked", id="dual-button"),

],id="hovered-values", className="log-container")

experiment_figs = html.Div([displaying_point_values_div, graphs_row1, graphs_row2], id="experiment-output", className="experiment-graphs")


# constructor functions 


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
                    {"label": "Itakura-Saito", "value": "ITAKURA-SAITO"},
                    {"label": "Power-3", "value": "POWER3"},
                    {"label": "Exponential", "value": "EXPONENTIAL"}
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


import re

def construct_experiment_results(idx, metrics_dict):
       # function converts the metrics_dict into a table
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


# callback determines which metrics have been selected for highlighting
@callback(
    Output("selected-metrics", "data"),
    Input({'type': 'metric-row', 'metric': ALL, 'table': ALL}, 'n_clicks'),
    State("selected-metrics", "data"),
    prevent_initial_call=True,
)
def update_selected_metrics(n_clicks_list, selected_metrics):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    
    if all(n == 0 for n in n_clicks_list):
        raise PreventUpdate
    
    triggered_id = json.loads(ctx.triggered[0]["prop_id"].split('.')[0])
    metric = triggered_id['metric']

    
    if metric in selected_metrics:
        selected_metrics.remove(metric)
    else:
        selected_metrics.append(metric)
    return selected_metrics


@callback(
    Output({'type': 'metric-row', 'metric': ALL, 'table': ALL}, 'style'),
    Input("selected-metrics", "data"),
    State({'type': 'metric-row', 'metric': ALL, 'table': ALL}, 'id')
)
def update_metric_row_style(selectedMetrics, ids):
    
    if selectedMetrics is None:
        selectedMetrics = []
    
    styles = []
    
    for idObj in ids:
        
        base_style = {"cursor": "pointer"}
        
        if idObj.get("metric") in selectedMetrics:
            base_style["backgroundColor"] = "#a2bab2"
        styles.append(base_style)
    
    return styles

# placeholder variables to be updated via callback
minimise_run_button = html.Button("Run Experiment", className="run-button", n_clicks=0, id="run-button-minimise")
minimise_add_button = html.Button("+", className="add-button", n_clicks=1, id="add-button-minimise", title="add a configuration")
minimise_save_button = html.Button("Save", className="save-button", id="save-button-minimise", disabled=True, title="Cannot save until experiment has ran", n_clicks=0)
minimise_remove_button = html.Button("-", className="add-button", id="remove-button-minimise", disabled=True, title="Must be at least one configuration", n_clicks=0)



run_button_container = html.Div([minimise_run_button, minimise_add_button, minimise_remove_button, minimise_save_button], id="run-button-container")

experiment_results = html.Div([
        html.H3("Metrics", id="metrics-header"),
        dcc.Loading(
            id="loading-metrics",
            type="default",
            color="#e8dac5",
            delay_show=500,
            children=None),
        html.Div([], className="bottom-container", id="metrics-bottom")
    ], id="experiment-metrics", className="experiment-metrics hidden")

experiment_settings_type_store = dcc.Store(id="experiment-settings-type", data="minimise")

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

# Stores used for experiment callback chain
current_experiment = dcc.Store(id="current-experiment", data=0)
metrics = dcc.Store(id="metrics", data=[])
trigger_next = dcc.Store(id="trigger-next", data=None)
experiment_parameters = dcc.Store(id="experiment-params", data=None)
experiment_dict = dcc.Store(id="experiment-dict", data=None)

experiment_running = dcc.Store(id="experiment-running", data=False)

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
    trigger_next,
    experiment_parameters,
    experiment_dict,
    experiment_running,
    dcc.Interval(
        id='experiment-interval',
        interval=1500,  # in milliseconds (1 second interval, adjust as needed)
        n_intervals=0,
        disabled=True  # Initially disabled
    ),
    dcc.Interval(
        id='logging-interval',
        interval=2000,  # in milliseconds (1 second interval, adjust as needed)
        n_intervals=0,
        disabled=False  # Initially disabled
    )
    
], style={"padding": "5px 20px 20px 20px"})




# callback highlights corresponding points on gradient, divergence and dual space traj graphs when hovering over a point
# on the primal space traj graphs
@callback(
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
    prevent_initial_call=True
)
def sync_hover_others(hover2d, hover3d, fig2d, fig3d, dual_fig, grad_fig, div_fig):
    
    from dash import callback_context

    # use callback_context to see which input triggered the callback
    ctx = callback_context
    active_hover = None
    source_fig = None

    if ctx.triggered:
        triggered_id = ctx.triggered[0]['prop_id']
        # if the 3d figure triggered the callback, use its hoverData
        if triggered_id.startswith('optimisation-path-fig-3d'):
            active_hover = hover3d
            source_fig = fig3d
        # if the 2d figure triggered the callback, use its hoverData
        elif triggered_id.startswith('optimisation-path-fig'):
            active_hover = hover2d
            source_fig = fig2d

    # if none triggered (or if the triggered value is empty), fallback to prefer 3d hoverData if available
    if active_hover is None:
        active_hover = hover3d if hover3d is not None else hover2d
        source_fig = fig3d if hover3d is not None else fig2d

    # clone only the figures we need to update (dual, gradient, divergence)
    def clone_fig(fig):
        return {
            'data': [trace.copy() for trace in fig.get('data', [])],
            'layout': fig.get('layout', {}).copy()
        }
    new_dual = clone_fig(dual_fig)
    new_grad = clone_fig(grad_fig)
    new_div = clone_fig(div_fig)

    # remove any existing highlight traces (with name exactly 'Highlight')
    def remove_highlights(fig):
        fig['data'] = [trace for trace in fig.get('data', []) if trace.get('name') != 'Highlight']
        return fig
    new_dual = remove_highlights(new_dual)
    new_grad = remove_highlights(new_grad)
    new_div = remove_highlights(new_div)

    if active_hover is None:
        return new_dual, new_grad, new_div

    try:
        point = active_hover['points'][0]
    except (KeyError, IndexError):
        return new_dual, new_grad, new_div

    # get the hovered point index, using pointIndex (2d) or falling back to pointNumber (3d)
    pt_index = point.get('pointIndex') or point.get('pointNumber')
    if pt_index is None:
        return new_dual, new_grad, new_div

    # get the curve index; if not provided, default to 0
    curve_idx = point.get('curveNumber', 0)
    try:
        hovered_trace = source_fig['data'][curve_idx]
    except (KeyError, IndexError):
        return new_dual, new_grad, new_div

    trace_name = hovered_trace.get('name', '')
    # ignore static traces such as the objective function trace
    if "objective" in trace_name.lower():
        return new_dual, new_grad, new_div

    # extract the experiment number from the trace name (e.g. "(1)" or "1")
    m = re.search(r'\d+', trace_name)
    if not m:
        return new_dual, new_grad, new_div
    exp_num = int(m.group(0))

    # helper: add a highlight marker to each trace in a figure that corresponds to the experiment number
    def add_highlight(fig):
        highlights = []
        for tr in fig.get('data', []):
            # skip contour traces
            if tr.get('type') == 'contour':
                continue
            candidate_name = tr.get('name', '')
            try:
                candidate_num = int(''.join(filter(str.isdigit, candidate_name)))
            except Exception:
                candidate_num = None
            # also accept a trace named "dual trajectory" for experiment 1
            if (exp_num == 1 and candidate_name.lower() == "dual trajectory") or \
               (candidate_num is not None and candidate_num == exp_num):
                xs = tr.get('x', [])
                ys = tr.get('y', [])
                if pt_index < len(xs) and pt_index < len(ys):
                    highlights.append({
                        'x': [xs[pt_index]],
                        'y': [ys[pt_index]],
                        'mode': 'markers',
                        'marker': {'size': 4, 'color': '#322634'},
                        'name': 'Highlight'
                    })
        fig['data'].extend(highlights)
        return fig

    new_dual = add_highlight(new_dual)
    new_grad = add_highlight(new_grad)
    new_div = add_highlight(new_div)

    return new_dual, new_grad, new_div

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
        return ["input-row hidden"]*4 + ["input-row"] + ["input-row hidden"]*3 + ["-20*e(-0.2*sqrt(0.5*(x**2 + y**2))) - e(0.5*(cos(2*pi*x) + cos(2*pi*y))) + e + 20"]
    elif preset_function == "CUSTOM":
        return ["input-row hidden"]*5 + ["input-row hidden"]*3 + ["X**2 + Y**2"]
    elif preset_function == "EXPONENTIAL":
        return ["input-row hidden"]*2 + ["input-row"]*3 + ["input-row hidden"]*3 + ["e*(x-optx) - (x - optx) + e(y - opty) - (y - opty)"]
    elif preset_function == "CUBIC":
        return ["input-row hidden"]*2 + ["input-row"]*3 + ["input-row hidden"]*3 + ["1/3 * (|x - optx|**3 + |y - opty|**3)"]
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
            {"label": "Power-3", "value": "POWER3"},
            {"label": "Exponential", "value": "EXPONENTIAL"}
        ]
    if current_dim == 1:
        for opt in base_options:
            if opt["value"] == "MAHALANOBIS":
                
                opt["disabled"] = True
    return [base_options]*num_experiments



# callback to add input field for positive definite matrix Q when mahalanobis distance is selected
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
            "ACKLEY": lambda: Ackley(noise_std=noise_std),
            "CUBIC": lambda: ExponentialObjective2D(optimum = torch.tensor([optx, opty]), noise_std=noise_std),
            "EXPONENTIAL": lambda: CubicObjective(optimum = torch.tensor([optx, opty]), noise_std=noise_std)
        }
        return presets[preset_value]()
    
# sets up the initial points for different function types and determines the dimension
def setup_inits(preset_function, second_input_bool, init_x, init_y, p1s, p2s, p3s):
    if preset_function == "CUSTOM":
            if second_input_bool[0]==False:
                inits = [[float(x), float(y)] for x, y in zip(init_x, init_y)]
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
        

import time


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
    # have to store n_clicks in a global dcc.store in order to compare to the save buttons value
    # as this callback gets triggered when the user adds a configuration due to the save button reloading 
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
    


# builds a matching minimise-configuration from the saved json
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
    ], className="input-row hidden", id="q3-input-row"),
    experiment_configs[0]]
    , className= "settings", id="inner-div")], id="minimise-config", className="option-columns-mlp")
    full_config = [minimise_config] + experiment_configs[1:]
    return full_config, metric_configs


# updates which figures are currently being displayed as per the users choice
@callback(
    Output("optim-3d-button", "className"),
    Output("optim-contour-button", "className"),
    Output("div-button", "className"),
    Output("grad-button", "className"),
    Output("dual-button", "className"),
    Output("optim-3d-container", "className"),
    Output("optimisation-path-fig", "className"),
    Output("divergence-fig", "className"),
    Output("gradient-fig", "className"),
    Output("dual-fig", "className"),
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
)
def update_displayed_figures(o3d_button_clicks, o_button_clicks, div_button_clicks, grad_button_clicks, dual_button_clicks,
                             o3d_button_class, o_button_class, div_button_class, grad_button_class, dual_button_class):
    ctx = callback_context
    if not ctx.triggered:
        return no_update

    triggered_prop = ctx.triggered[0]["prop_id"]
    print(triggered_prop)
    if triggered_prop == "optim-3d-button.n_clicks":
        if o3d_button_class == "fig-button clicked":
            return ["fig-button"] + [no_update]*4 + ["true-hidden"] + [no_update]*4
        else:
            return ["fig-button clicked"] + [no_update]*4 + [""]+ [no_update]*4
    else:
        return no_update




# rebuilds the graphs from the saved experiment json
@callback(
    Output("config-options", "children", allow_duplicate=True),
    Output("optimisation-path-fig", "figure", allow_duplicate=True),
    Output("dual-fig", "figure", allow_duplicate=True),
    Output("gradient-fig", "figure", allow_duplicate=True),
    Output("divergence-fig", "figure", allow_duplicate=True),
    Output("loading-metrics", "children", allow_duplicate=True),
    Output("num-experiments-min", "data", allow_duplicate=True),
    Output("add-button-minimise", "n_clicks", allow_duplicate=True),
    Output("experiment-settings-type", "data", allow_duplicate=True),
    Output("minimise-config", "className", allow_duplicate=True),
    Output("experiment-metrics", "className", allow_duplicate=True),
    Input("upload-load", "n_clicks"),
    State("upload-config", "contents"),
    State("upload-config", "filename"),
    State("config-options", "children"),
    State("loading-metrics", "children"),
    prevent_initial_call=True
)
def load_experiment(load_clicks, contents, filename, current_children, current_children_metrics):
    
    triggered = callback_context.triggered
    if not triggered:
        return no_update
    
    triggered_id = triggered[0]["prop_id"]
    if triggered_id != "upload-load.n_clicks" or load_clicks == 0:
        return no_update, no_update
    
    print("Loading Experiment...")
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    saved_experiment = json.loads(decoded.decode("utf-8"))

    num_experiments = len(saved_experiment.get("experiments", {}))
    if saved_experiment["configuration"].get("experiment_type") == "minimise":

        minimise_config_new, metrics = build_minimise_config_from_saved(saved_experiment)
        print("minimise config len ", len(minimise_config_new))
        new_children = current_children[:1] + minimise_config_new + [current_children[-1]]

        new_experiment_results = build_experiment_results_from_saved(saved_experiment, "minimise")
        new_metrics = metrics

        return new_children, new_experiment_results[0], new_experiment_results[1], new_experiment_results[2], new_experiment_results[3],new_metrics, num_experiments, num_experiments, "minimise", "option-columns-mlp", "experiment-metrics"


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
    Output("loading-metrics", "children", allow_duplicate=True),
    Output("experiment-metrics", "className", allow_duplicate=True),
    Output("run-button-minimise", "n_clicks"),
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
    State("loading-metrics", "children"),
    State({"type": "Q-input", "index": ALL}, "value"),
    prevent_initial_call=True,
    allow_duplicate=True,
    suppress_callback_exceptions=True
)
def initialise_experiment_run(n_clicks, objective_string, init_x, init_y, iter,
                            lr, bregman, num_experiments, second_input_bool, q_store,
                            p1s, p2s, p3s, q1, q2, q3, a, b, optx, opty, noise_std,
                            preset_function, current_metrics, q_strings):
    print("init experiment ran")
    if n_clicks != 0:
        inits, dim = setup_inits(preset_function, second_input_bool, init_x, init_y, p1s, p2s, p3s)
        objective = get_objective_function(preset_function, objective_string, a, b, q1, q2, q3, optx, opty, noise_std=noise_std)
        if preset_function != "CUSTOM":
            optimum = objective(objective.optimum)
            optimum_coords = objective.optimum

        else:
            optimum, optimum_coords = None, None

        experiment = ExperimentMD(objective, bregman=bregman[0], Q=torch.tensor(q_store[0][0], dtype=torch.float64),
                                   Q_inv=torch.tensor(q_store[0][1], dtype=torch.float64), x_star=optimum_coords, f_star=optimum, dim=dim)
        
        experiment.run_experiment_minimise(inits[0], iter[0], float(lr[0]))
        metrics = experiment.gather_metrics()
        metrics_div = construct_experiment_results(1, metrics)
        optimisation_path_fig = graph.create_optimisation_path_graph(experiment.minimisation_guesses, experiment.objective, dim)
        optimisation_path_fig3d = graph.create_optimisation_path_3d_graph(experiment.minimisation_guesses, experiment.objective, dim)

        gradient_fig = graph.create_gradient_norm_graph(experiment.gradient_logs)
        divergence_fig = graph.create_divergence_graph(experiment.avg_divergence_logs)
        dual_fig = graph.create_dual_space_trajectory_graph(experiment.optimiser.logs["dual"],experiment.objective, dim)

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
        
    
        interval_disabled = False
        next_experiment = 2
        return (interval_disabled, next_experiment, [metrics], experiment_params, experiments_dict, optimisation_path_fig, optimisation_path_fig3d, dual_fig, divergence_fig, gradient_fig, metrics_div, "experiment-metrics", 0)
    else:
        raise PreventUpdate

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

current_exp_number = 1
@callback(
    Input("current-experiment", "data"),
)
def update_expnumber(current_experiment):
    print("do I ever run")
    current_exp_number = current_experiment

@callback(
    Output("experiment-interval", "disabled", allow_duplicate=True),
    Output("current-experiment", "data", allow_duplicate=True),
    Output("metrics", "data", allow_duplicate=True),
    Output("optimisation-path-fig", "figure"),
    Output("optimisation-path-fig-3d", "figure"),
    Output("dual-fig", "figure", allow_duplicate=True),
    Output("divergence-fig", "figure", allow_duplicate=True),
    Output("gradient-fig", "figure", allow_duplicate=True),
    Output("loading-metrics", "children", allow_duplicate=True),
    Output("experiment-metrics", "className", allow_duplicate=True),
    Output("last-min-config", "data", allow_duplicate=True),
    Output("save-button-minimise", "disabled"),
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
    State("loading-metrics", "children"),
    State("experiment-interval", "disabled"),
    suppress_callback_exceptions=True,
    prevent_initial_call=True,
)
def run_next_experiment(triggered, current_experiment, params, metrics,  num_experiments, experiment_dict,
                        optimisation_path_fig, optimisation_path_fig_3d, dual_fig, divergence_fig, gradient_fig,
                        q_store, current_metrics_div, interval_disabled):
    print("run next experiment ran")
    print(triggered)
    print(interval_disabled)
    if triggered == 0:
        print("check123")
        return [no_update]*10
    
    if current_experiment > num_experiments:
        print("check1")
        interval_disabled = True
        metrics_dict = create_compiled_metrics_dicts(num_experiments, metrics)
        figures = [optimisation_path_fig, optimisation_path_fig_3d, dual_fig, divergence_fig, gradient_fig]
        experiment_state = construct_experiment_state(metrics_dict, experiment_dict, params["objective_params"], figures)
        print("check2")
        return (interval_disabled, no_update, None, no_update, no_update, no_update, no_update, no_update, no_update, no_update, experiment_state, False)
      
    else:
        time1 = time.time()
        inits, dim, iter = params["inits"], params["dim"], params["iter"]
        optimum, optimum_coords = params["optimum"], params["optimum_coords"]
        bregman, lr = params["bregman"], params["learning_rate"]
        obj_params = params["objective_params"]
        objective = get_objective_function(*obj_params)
        exp_idx = current_experiment-1
        experiment = ExperimentMD(objective, bregman=bregman[exp_idx], Q=torch.tensor(q_store[exp_idx][0], dtype=torch.float64),
                                  Q_inv = torch.tensor(q_store[exp_idx][1], dtype=torch.float64), x_star=optimum_coords, f_star=optimum, dim=dim)
        experiment.run_experiment_minimise(inits[exp_idx], iter[exp_idx], float(lr[exp_idx]))
        optimisation_path_fig, optimisation_path_fig_3d, gradient_fig, divergence_fig, dual_fig = graph.update_all_graphs_min(experiment.minimisation_guesses, experiment.gradient_logs,
                                                                                                     experiment.avg_divergence_logs, experiment.optimiser.logs["dual"],
                                                                                                     experiment.objective, current_experiment, dim=dim)
        new_metrics = experiment.gather_metrics()
        new_metrics_div = construct_experiment_results(current_experiment, new_metrics)
        metrics.append(new_metrics)
        
        if isinstance(current_metrics_div, dict):
            new_metrics_divs = [current_metrics_div] + [new_metrics_div]
        else:
            print(type(current_metrics_div))
            new_metrics_divs = current_metrics_div[0:] + [new_metrics_div]
        time2 =time.time()
        print(f"experiment {current_experiment} ran in :", time2-time1)
        current_experiment += 1
        print(f"Next experiment number:  ({current_experiment})")
        return no_update, current_experiment, metrics, optimisation_path_fig, optimisation_path_fig_3d, dual_fig, divergence_fig, gradient_fig, new_metrics_divs, "experiment-metrics", no_update, no_update
    
#
#callback loads a base experiment when the page initialises
@callback(
    Output("config-options", "children", allow_duplicate=True),
    Output("loading-metrics", "children", allow_duplicate=True),
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
    new_metrics = metrics
    num_experiments = len(saved_experiment.get("experiments", {}))
    
    return new_children, new_metrics, num_experiments, "experiment-metrics"


# @callback(
#     Output("run-button-minimise", "n_clicks"),
#     Output("metrics", "data", allow_duplicate=True),
#     Output("optimisation-path-fig", "figure"),
#     Output("optimisation-path-fig-3d", "figure"),
#     Output("dual-fig", "figure", allow_duplicate=True),
#     Output("divergence-fig", "figure", allow_duplicate=True),
#     Output("gradient-fig", "figure", allow_duplicate=True),
#     Output("loading-metrics", "children", allow_duplicate=True),
#     Output("experiment-metrics", "className", allow_duplicate=True),
#     Output("last-min-config", "data", allow_duplicate=True),
#     Output("save-button-minimise", "disabled"),
#     Input("run-button-minimise", "n_clicks"),
#     State("function-mini-input", "value"),
#     State({"type": "initial-value-input", "index": ALL}, "value"),
#     State({"type": "initial-value-input-2", "index": ALL}, "value"),
#     State({"type": "number-iterations-input", "index": ALL}, "value"),
#     State({"type": "lr-mini-input", "index": ALL}, "value"),
#     State({"type": "bregman-mini-input", "index": ALL}, "value"),
#     State("num-experiments-min", "data"),
#     State({"type": "initial-value-input-2", "index": ALL}, "disabled"),
#     State("Q-store", "data"),
#     State({"type": "simplex-initial-value-input", "index": ALL}, "value"),
#     State({"type": "simplex-initial-value-input-2", "index": ALL}, "value"),
#     State({"type": "simplex-initial-value-input-3", "index": ALL}, "value"),
#     State("q1-input", "value"),
#     State("q2-input", "value"),
#     State("q3-input", "value"),
#     State("a-input", "value"),
#     State("b-input", "value"),
#     State("optim-x-input", "value"),
#     State("optim-y-input", "value"),
#     State("noise-input", "value"),
#     State("preset-function-input", "value"),
#     State("loading-metrics", "children"),
#     State({"type": "Q-input", "index": ALL}, "value"),
#     prevent_initial_call=True,
#     allow_duplicate=True,
#     suppress_callback_exceptions=True,
# )
# def combined_experiment_run(n_clicks, objective_string, init_x, init_y, iter, lr, bregman, num_experiments,
#                             second_input_bool, q_store, p1s, p2s, p3s, q1, q2, q3, a, b, optx, opty,
#                             noise_std, preset_function, loading_metrics, q_strings):
#     print("Combined experiment run triggered")
#     if n_clicks == 0:
#         # If the run button hasn’t been clicked, do nothing.
#         return [no_update] * 12

#     # Initialize experiment parameters and starting values.
#     inits, dim = setup_inits(preset_function, second_input_bool, init_x, init_y, p1s, p2s, p3s)
#     objective = get_objective_function(preset_function, objective_string, a, b, q1, q2, q3, optx, opty, noise_std=noise_std)
#     if preset_function != "CUSTOM":
#         optimum = objective(objective.optimum)
#         optimum_coords = objective.optimum
#     else:
#         optimum, optimum_coords = None, None

#     # Create a dictionary of all experiment configurations.
#     experiments_dict = create_experiment_dict_min(num_experiments, init_x, init_y, iter, lr, bregman,
#                                                    second_input_bool, q_strings, p1s, p2s, p3s)
#     experiment_params = {
#         "inits": inits,
#         "dim": dim,
#         "iter": iter,
#         "optimum": optimum,
#         "optimum_coords": optimum_coords,
#         "objective_params": [preset_function, objective_string, a, b, q1, q2, q3, optx, opty, noise_std],
#         "bregman": bregman,
#         "learning_rate": lr
#     }
    
#     all_metrics = []      # list to collect metrics from each experiment
#     metrics_divs = []     # list to collect HTML metrics divs for display
    
#     # --- Run the first experiment ---
#     print("Running experiment 1")
#     experiment = ExperimentMD(objective,
#                               bregman=bregman[0],
#                               Q=torch.tensor(q_store[0][0], dtype=torch.float64),
#                               Q_inv=torch.tensor(q_store[0][1], dtype=torch.float64),
#                               x_star=optimum_coords,
#                               f_star=optimum,
#                               dim=dim)
#     experiment.run_experiment_minimise(inits[0], iter[0], float(lr[0]))
#     metrics_exp = experiment.gather_metrics()
#     all_metrics.append(metrics_exp)
#     metrics_divs.append(construct_experiment_results(1, metrics_exp))
    
#     # Create initial figures based on the first experiment.
#     optimisation_path_fig = graph.create_optimisation_path_graph(experiment.minimisation_guesses, experiment.objective, dim)
#     # For 3D (if applicable) – this uses your custom 3d graph creation.
#     optimisation_path_fig_3d = graph.create_optimisation_path_3d_graph(experiment.minimisation_guesses, experiment.objective, dim)
#     gradient_fig = graph.create_gradient_norm_graph(experiment.gradient_logs)
#     divergence_fig = graph.create_divergence_graph(experiment.avg_divergence_logs)
#     dual_fig = graph.create_dual_space_trajectory_graph(experiment.optimiser.logs["dual"], experiment.objective, dim)

#     # --- Run experiments 2 to num_experiments in a for loop ---
#     for exp in range(1, num_experiments):
#         print(f"Running experiment {exp+1}")
#         # Use the corresponding index from the parameter lists.
#         experiment = ExperimentMD(objective,
#                                   bregman=bregman[exp],
#                                   Q=torch.tensor(q_store[exp][0], dtype=torch.float64),
#                                   Q_inv=torch.tensor(q_store[exp][1], dtype=torch.float64),
#                                   x_star=optimum_coords,
#                                   f_star=optimum,
#                                   dim=dim)
#         experiment.run_experiment_minimise(inits[exp], iter[exp], float(lr[exp]))
#         # Update all graphs with the new experiment’s data.
#         optimisation_path_fig, optimisation_path_fig_3d, gradient_fig, divergence_fig, dual_fig = \
#             graph.update_all_graphs_min(experiment.minimisation_guesses,
#                                         experiment.gradient_logs,
#                                         experiment.avg_divergence_logs,
#                                         experiment.optimiser.logs["dual"],
#                                         experiment.objective,
#                                         exp+1,
#                                         dim=dim)
#         # Gather and store the metrics.
#         metrics_exp = experiment.gather_metrics()
#         all_metrics.append(metrics_exp)
#         metrics_div = construct_experiment_results(exp+1, metrics_exp)
#         metrics_divs.append(metrics_div)

#     # Compile the metrics into a dictionary for saving.
#     metrics_dict = create_compiled_metrics_dicts(num_experiments, all_metrics)
#     # Arrange final figures into a list (order can be adjusted as needed).
#     final_figures = [optimisation_path_fig, optimisation_path_fig_3d, dual_fig, divergence_fig, gradient_fig]
#     # Create the saved experiment state using your helper.
#     experiment_state = construct_experiment_state(metrics_dict, experiments_dict, experiment_params["objective_params"], final_figures)

   
#     current_experiment_out = no_update

   
#     return (0,
#             all_metrics,
#             optimisation_path_fig,
#             optimisation_path_fig_3d,
#             dual_fig,
#             divergence_fig,
#             gradient_fig,
#             metrics_divs,
#             "experiment-metrics",
#             experiment_state,
#             False)