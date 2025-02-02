from dash import html, dcc, callback, Input, Output, State, callback_context, no_update
import dash
from Experiment import ExperimentMD
from FunctionParser import FunctionParser
import torch

dash.register_page(__name__, path="/run-experiment")

explanation_md = r"""
### Running an experiment 
Here you can configure and run your own experiments with the mirror descent algorithm.
There are two different types of experiments available: 
* Minimising an objective function using mirror descent.
* Approximating a function with a simple regression model, using mirror descent to minimise the loss over training. 
"""

model_settings = html.Div([
    dcc.Markdown("**Model options**"),
    html.Div([
        html.Label("Layers"),
        dcc.Input(type="number", value=2, style={"marginBottom": "5px"}, className="input-values", id="layers-input")
    ], className="input-row"),
    html.Div([
        html.Label("Neurons"),
        dcc.Input(type="number", value=10, style={"marginBottom": "5px"}, className="input-values", id="neuron-input")
    ], className="input-row"),
    html.Div([
        html.Label("Epochs"),
        dcc.Input(type="number", value=2000, style={"marginBottom": "5px"}, className="input-values", id="epoch-input")
    ], className="input-row"),
], className="settings")

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
    dcc.Markdown("**Mirror Descent Options**"),
    html.Div([
        html.Label("Bregman"),
        dcc.Dropdown(
            options = [
                {"label": "Euclidean", "value": "EUCLID"},
                {"label": "KL", "value": "KL"}
            ]    
        , id="bregman-input",className="dropdown", value="EUCLID")
    ], className = "input-row"),
    html.Div([
        html.Label("Loss"),
        dcc.Dropdown(
            options = [
                {"label": "MSE", "value": "MSE"},
                {"label": "MAE", "value": "MAE"},
                {"label": "Huber", "value": "Huber"}
            ]
        , id="loss-input", className="dropdown", value="MSE")
    ], className = "input-row"),
    html.Div([
        html.Label("Learning Rate"),
        dcc.Input(type="number", value=0.01, step=0.001, min=0, style={"marginBottom": "5px"}, className="input-values", id="lr-input"),
    ], className="input-row"),
    
], className="settings")

minimise_config = html.Div([
    dcc.Markdown("**Objective function and algorithm parameters**"),
    html.Div([
        html.Label("Objective Function"),
        dcc.Input(type="text", value="X**2 + 3*X", style={"marginBottom": "5px"}, className="input-function", id="function-mini-input"),
    ], className="input-row"),
    html.Div([
        html.Label("Initial value"),
        dcc.Input(type="number", value=0.5, style={"marginBottom": "5px"}, className="input-values", id="initial-value-input"),
    ], className="input-row"),
    html.Div([
        html.Label("Iterations"),
        dcc.Input(type="number", value=100, style={"marginBottom": "5px"}, className="input-values", id="number-iterations-input"),
    ], className="input-row"),
    html.Div([
        html.Label("Learning Rate"),
        dcc.Input(type="number", value=0.01, step=0.001, style={"marginBottom": "5px"}, className="input-values", id="lr-mini-input"),
    ], className="input-row"),
    html.Div([
        html.Label("Bregman"),
        dcc.Dropdown(
            options = [
                {"label": "Euclidean", "value": "EUCLID"},
                {"label": "KL", "value": "KL"}
            ]    
        , id="bregman-mini-input",className="dropdown", value="EUCLID")
    ], className = "input-row"),
], className= "settings", id="minimise-config")

approximate_config = html.Div([
        model_settings,
        function_md_settings
    ], className="option-columns-mlp", id="approximate-config")

# placeholder variable to be updated with either approximate_config or minimise_config
experiment_settings_container = html.Div(id="experiment-settings-container", className="option-columns-mlp")

config_options = html.Div([
    dcc.Markdown("#### Configuration", className="markdown-config"),
    html.Div([
        html.Button("Approximate", className="config-button", id="approx-button", n_clicks=0),
        html.Button("Minimise",   className="config-button", id="min-button",   n_clicks=0)
    ], id="top-div-config"),
    experiment_settings_container,
    html.Button("Run Experiment", className="run-button", n_clicks=0, id="run-button")
], className="configuration-options")

experiment_results = html.Div([
    dcc.Markdown("### results placeholder", className="markdown")
], id="experiment-output", className="experiment-graphs")

experiment_settings_type_store = dcc.Store(id="experiment-settings-type", data="minimise")
experiment_run_type_store = dcc.Store(id="experiment-run-type", data="minimise")

layout = html.Div([
    dcc.Markdown(explanation_md, className="markdown", id="config-info"),
    html.Div([
        config_options,
        experiment_results
    ], className= "experiment-div"),
    experiment_settings_type_store,
    experiment_run_type_store
], style={"padding": "20px"})



# callback triggers when either the minimise or approximate button is clicked 
@callback(
    Output("experiment-settings-container", "children"),
    Output("approx-button", "className"),
    Output("min-button", "className"),
    Output("experiment-settings-type", "data"),
    Input("approx-button", "n_clicks"),
    Input("min-button", "n_clicks")
)
def update_experiment_settings(approx_clicks, min_clicks):
    context = callback_context
    if not context.triggered:
        return minimise_config, "config-button", "config-button-clicked", "minimise"
    else:
        # find which button was clicked then update accordingly
        button_triggered = context.triggered[0]["prop_id"].split(".")[0]
        if button_triggered == "min-button":
            return minimise_config, "config-button", "config-button-clicked", "minimise"
        elif button_triggered == "approx-button": 
            return [model_settings, function_md_settings], "config-button-clicked", "config-button", "approximate"
        else:
            return minimise_config, "config-button", "config-button-clicked", "minimise"





@callback(
    Output("experiment-run-type", "data"),
    Input("run-button", "n_clicks"),
    State("experiment-settings-type", "data"),
    suppress_callback_exceptions=True,
    prevent_initial_call=True 
)
def choose_and_run(n_clicks, experiment_type):
    if experiment_type == "approximate":
        return "approximate"
    elif experiment_type == "minimise":
        return "minimise"



def on_errors(err):
    print("temporary fix for callback error")

@callback(
    Output("experiment-output", "children", allow_duplicate=True),
    Input("experiment-run-type", "data"),
    State("layers-input", "value"),
    State("neuron-input", "value"),
    State("epoch-input", "value"),
    State("function-input", "value"),
    State("data-lbound-input", "value"),
    State("data-ubound-input", "value"),
    State("num-samples-input", "value"),
    State("bregman-input", "value"),
    State("loss-input", "value"),
    State("lr-input", "value"),
    prevent_initial_call=True,
    allow_duplicate=True,
    suppress_callback_exceptions=True,
    debug=False
)
def run_experiment_mlp(experiment_type, layers, neurons, epochs, function, range_min, range_max, n_samples, bregman, loss, learning_rate):
    if experiment_type == "approximate":       
        # parse from string the function to approximate
        parser = FunctionParser(function)
        parser.test_function()
        fta = parser.string_to_lambda()
        # test function
        print(fta(torch.tensor(1)))

        # instantiate the experiment object
        experiment = ExperimentMD(fta, bregman=bregman)
        print("Experiment instantiated")
        experiment.criterion = experiment.losses[loss]
        
        # run the experiment, generate and return figures
        experiment.run_experiment_mlp(range_min, range_max, n_samples, layers, neurons, epochs, float(learning_rate))

        loss_fig = experiment.create_loss_curve()
        gradient_fig = experiment.create_gradient_norm_graph()
        divergence_fig = experiment.create_divergence_graph()
        results_fig = experiment.create_function_approximation_plot()

        return [dcc.Graph(figure=loss_fig, id="loss-curve", config={'responsive': True},className="graph"),
                dcc.Graph(figure=gradient_fig, id="gradient-fig",config={'responsive': True}, className="graph"),
                dcc.Graph(figure=divergence_fig, id="divergence-fig",config={'responsive': True}, className="graph"),
                dcc.Graph(figure=results_fig, id="results_fig",config={'responsive': True}, className="graph")]
    else:
        return no_update

@callback(
    Output("experiment-output", "children"),
    Input("experiment-run-type", "data"),
    State("function-mini-input", "value"),
    State("initial-value-input", "value"),
    State("number-iterations-input", "value"),
    State("lr-mini-input", "value"),
    State("bregman-mini-input", "value"),
    prevent_initial_call=True,
    allow_duplicate=True,
    suppress_callback_exceptions=True
    
)
def run_experiment_minimise(experiment_type, objective_string, init, iter, lr, bregman):
    if experiment_type == "minimise": 
        # parse objective function from string 
        parser = FunctionParser(objective_string)
        parser.test_function()
        objective = parser.string_to_lambda()

        # instantiate experiment object
        experiment = ExperimentMD(objective, bregman=bregman)
        print("Experiment instantiated")

        # run experiment, generate and return figures
        print(experiment.objective)
        experiment.run_experiment_minimise(init, iter, lr)
        print("experiment complete")
        optimisation_path_fig = experiment.create_optimisation_path_graph()
        gradient_fig = experiment.create_gradient_norm_graph()
        divergence_fig = experiment.create_divergence_graph()

        return [dcc.Graph(figure=optimisation_path_fig, id="optimisation-path-fig", config={'responsive': True},className="graph"),
                dcc.Graph(figure=gradient_fig, id="gradient-fig",config={'responsive': True}, className="graph"),
                dcc.Graph(figure=divergence_fig, id="divergence-fig",config={'responsive': True}, className="graph")]












    
    
