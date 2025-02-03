from dash import html, dcc, callback, Input, Output, State, callback_context, no_update, Patch
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

# constructor functions 
def construct_md_settings(idx):
    return html.Div([
    dcc.Markdown(f"**Mirror Descent Options ({idx})**"),
    html.Div([
        html.Label("Batch Size"),
        dcc.Input(type="number", value=500, style={"marginBottom": "5px"}, className="input-values", id={"type": "batch-size-input", "index": idx}),
    ], className="input-row"),
    html.Div([
        html.Label("Bregman"),
        dcc.Dropdown(
            options = [
                {"label": "Euclidean", "value": "EUCLID"},
                {"label": "KL", "value": "KL"}
            ]    
        , id={"type": "bregman-input", "index": idx},className="bregman-loss-input", value="EUCLID")
    ], className = "input-row"),
    html.Div([
        html.Label("Loss"),
        dcc.Dropdown(
            options = [
                {"label": "MSE", "value": "MSE"},
                {"label": "MAE", "value": "MAE"},
                {"label": "Huber", "value": "Huber"}
            ]
        , id={"type": "loss-input", "index": idx}, className="bregman-loss-input", value="MSE")
    ], className = "input-row"),
    html.Div([
        html.Label("Learning Rate"),
        dcc.Input(type="number", value=0.01, step=0.001, min=0, style={"marginBottom": "5px"}, className="input-values", id={"type": "lr-input", "index": idx}),
    ], className="input-row")
    ], className="settings")


def construct_model_settings(idx):
    
    return html.Div([
                dcc.Markdown(f"**Model options ({idx})**"),
                html.Div([
                    html.Label("Layers"),
                    dcc.Input(type="number", value=2, style={"marginBottom": "5px"}, className="input-values", id={"type": "layers-input", "index": idx})
                ], className="input-row"),
                html.Div([
                    html.Label("Neurons"),
                    dcc.Input(type="number", value=10, style={"marginBottom": "5px"}, className="input-values", id={"type": "neuron-input", "index": idx})
                ], className="input-row"),
                html.Div([
                    html.Label("Epochs"),
                    dcc.Input(type="number", value=2000, style={"marginBottom": "5px"}, className="input-values", id={"type": "epoch-input", "index": idx})
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
    construct_md_settings(1)
    
], className="settings", id="function-md-settings")

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

# placeholder variables to be updated via callback
experiment_settings_container = html.Div(id="experiment-settings-container", className="option-columns-mlp")
run_button_container = html.Div(id="run-button-container")
experiment_results = html.Div([], id="experiment-output", className="experiment-graphs")
experiment_settings_type_store = dcc.Store(id="experiment-settings-type", data="minimise")

config_options = html.Div([
    dcc.Markdown("#### Configuration", className="markdown-config"),
    html.Div([
        html.Button("Approximate", className="config-button", id="approx-button", n_clicks=0),
        html.Button("Minimise",   className="config-button", id="min-button",   n_clicks=0)
    ], id="top-div-config"),
    experiment_settings_container,
    run_button_container
], className="configuration-options", id="config-options")

num_experiments_store = dcc.Store(id="num-experiments", data=1)

layout = html.Div([
    dcc.Markdown(explanation_md, className="markdown", id="config-info"),
    html.Div([
        config_options,
        experiment_results
    ], className= "experiment-div"),
    experiment_settings_type_store,
    num_experiments_store
    
], style={"padding": "20px"})


minimise_run_button = html.Button("Run Experiment", className="run-button", n_clicks=0, id="run-button-minimise")
approximate_run_button = html.Button("Run Experiment", className="run-button", n_clicks=0, id="run-button-approximate")
minimise_add_button = html.Button("+", className="add-button", n_clicks=0, id="add-button-minimise")
approximate_add_button = html.Button("+", className="add-button", n_clicks=0, id="add-button-approximate")
@callback(
        Output("config-options", "children"),
        Output("num-experiments", "data"),
        Input("add-button-approximate", "n_clicks"),
        State("config-options", "children"),
        State("num-experiments", "data"),
        prevent_initial_call=True
)
def add_configuration(n_clicks, current_children, num_experiments):
    # Only add a configuration if n_clicks is at least 1.
    if not n_clicks:
        return current_children
    num_experiments += 1
    # Construct the new settings container.
    new_settings = html.Div(
        [construct_model_settings(num_experiments), construct_md_settings(num_experiments)],
        id=f"new-settings-{num_experiments}", className="option-columns-mlp"
    )

    # Assuming that run_button_container is always the last child,
    # insert new_settings right before it.
    updated_children = current_children[:-1] + [new_settings] + [current_children[-1]]
    return updated_children, num_experiments





@callback(
    Output("batch-size-input", "value"),
    Input("num-samples-input", "value")
)
def update_batch_size(num_samples):
    # assumes batch mirror descent, and automatically sets the batch size to be equal to the number of samples
    # if user changes the batch size for mirror descent it shouldn't change back unless the sample number is changed
    return num_samples

# callback triggers when either the minimise or approximate button is clicked and updates experiment_type store
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
            return [construct_model_settings(1), function_md_settings], "config-button-clicked", "config-button", "approximate"
        else:
            return minimise_config, "config-button", "config-button-clicked", "minimise"


@callback(
    Output("run-button-container", "children"),
    Input("experiment-settings-type", "data"),
)
def update_run_button(experiment_type):
    if experiment_type == "minimise":
        return [minimise_run_button, minimise_add_button]
    elif experiment_type == "approximate":
        return [approximate_run_button, approximate_add_button]
    else: 
        return ValueError("Unrecognised experiment type")



def on_errors(err):
    print("temporary fix for callback error")

@callback(
    Output("experiment-output", "children", allow_duplicate=True),
    Output("run-button-approximate", "n_clicks"),
    Input("run-button-approximate", "n_clicks"),
    State("layers-input", "value"),
    State("neuron-input", "value"),
    State("epoch-input", "value"),
    State("function-input", "value"),
    State("data-lbound-input", "value"),
    State("data-ubound-input", "value"),
    State("num-samples-input", "value"),
    State("batch-size-input", "value"),
    State("bregman-input", "value"),
    State("loss-input", "value"),
    State("lr-input", "value"),
    prevent_initial_call=True,
    allow_duplicate=True,
    suppress_callback_exceptions=True,
    debug=False
)
def run_experiment_mlp(n_clicks, layers, neurons, epochs, function, range_min, range_max, n_samples, batch_size, bregman, loss, learning_rate):
    if n_clicks != 0:       
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
        experiment.run_experiment_mlp(range_min, range_max, n_samples, batch_size, layers, neurons, epochs, float(learning_rate))

        loss_fig = experiment.create_loss_curve()
        gradient_fig = experiment.create_gradient_norm_graph()
        divergence_fig = experiment.create_divergence_graph()
        results_fig = experiment.create_function_approximation_plot()
        n_clicks = 0 
        return [dcc.Graph(figure=loss_fig, id="loss-curve", config={'responsive': True},className="graph"),
                dcc.Graph(figure=gradient_fig, id="gradient-fig",config={'responsive': True}, className="graph"),
                dcc.Graph(figure=divergence_fig, id="divergence-fig",config={'responsive': True}, className="graph"),
                dcc.Graph(figure=results_fig, id="results_fig",config={'responsive': True}, className="graph")], 0
    else: 
        return no_update

        
        

@callback(
    Output("experiment-output", "children"),
    Output("run-button-minimise", "n_clicks"),
    Input("run-button-minimise", "n_clicks"),
    State("function-mini-input", "value"),
    State("initial-value-input", "value"),
    State("number-iterations-input", "value"),
    State("lr-mini-input", "value"),
    State("bregman-mini-input", "value"),
    prevent_initial_call=True,
    allow_duplicate=True,
    suppress_callback_exceptions=True
    
)
def run_experiment_minimise(n_clicks, objective_string, init, iter, lr, bregman):
    if n_clicks != 0:
        # parse objective function from string 
        parser = FunctionParser(objective_string)
        parser.test_function()
        objective = parser.string_to_lambda()

        # instantiate experiment object
        experiment = ExperimentMD(objective, bregman=bregman)
        print("Experiment instantiated")

        # run experiment, generate and return figures
        print(experiment.objective)
        experiment.run_experiment_minimise(init, iter, float(lr))
        print("experiment complete")
        optimisation_path_fig = experiment.create_optimisation_path_graph()
        gradient_fig = experiment.create_gradient_norm_graph()
        divergence_fig = experiment.create_divergence_graph()

        return [dcc.Graph(figure=optimisation_path_fig, id="optimisation-path-fig", config={'responsive': True},className="graph"),
                dcc.Graph(figure=gradient_fig, id="gradient-fig",config={'responsive': True}, className="graph"),
                dcc.Graph(figure=divergence_fig, id="divergence-fig",config={'responsive': True}, className="graph")], 0 
    else:
        return no_update











    
    
