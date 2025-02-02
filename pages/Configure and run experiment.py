from dash import html, dcc, callback, Input, Output, State
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

approximation_settings = html.Div([
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
        dcc.Input(type="number", value=0.01, step=0.01, min=0, style={"marginBottom": "5px"}, className="input-values", id="lr-input"),
    ], className="input-row"),
    
], className="settings")

config_options = html.Div([



    dcc.Markdown("#### Configuration", className="markdown-config"),
    
    
    html.Div([
        html.Button("Approximate", className="config-button", id="approx-button", n_clicks=0),
        html.Button("Minimise",   className="config-button", id="min-button",   n_clicks=0)
    ], id="top-div-config"),
    
    
    html.Div([
        approximation_settings,
        function_md_settings
    ], className="option-columns"),

    html.Button("Run Experiment", className="run-button", n_clicks=0, id="run-button")

], className="configuration-options")

experiment_results = html.Div([
    dcc.Markdown("### results placeholder", className="markdown")
], id="experiment-output", className="experiment-graphs")



layout = html.Div([
    dcc.Markdown(explanation_md, className="markdown", id="config-info"),
    html.Div([
        config_options,
        experiment_results
    ], className= "experiment-div")
    
    
    
], style={"padding": "20px"})



@callback(
    Output("experiment-output", "children"),
    Input("run-button", "n_clicks"),
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
    prevent_initial_call=True
)
def run_experiment_mlp(n_clicks, layers, neurons, epochs, function, range_min, range_max, n_samples, bregman, loss, learning_rate):
    # parse from string the function to approximate
    parser = FunctionParser(function)
    parser.test_function()
    fta = parser.string_to_lambda()
    print(fta(1))
    # instantiate the experiment object
    
    experiment = ExperimentMD(fta, bregman=bregman)
    print("experiment class - complete")
    experiment.criterion = experiment.losses[loss]
    
    experiment.run_experiment(range_min, range_max, n_samples, layers, neurons, epochs, float(learning_rate))

    loss_fig = experiment.create_loss_curve()
    gradient_fig = experiment.create_gradient_norm_graph()
    divergence_fig = experiment.create_divergence_graph()
    results_fig = experiment.create_function_approximation_plot()

    return [dcc.Graph(figure=loss_fig, id="loss-curve", config={'responsive': True},className="graph"),
            dcc.Graph(figure=gradient_fig, id="gradient-fig",config={'responsive': True}, className="graph"),
            dcc.Graph(figure=divergence_fig, id="divergence-fig",config={'responsive': True}, className="graph"),
            dcc.Graph(figure=results_fig, id="results_fig",config={'responsive': True}, className="graph")]


    
    
