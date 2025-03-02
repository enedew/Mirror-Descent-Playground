from dash import html, dcc, callback, Input, Output, State, callback_context, no_update, Patch, ALL
import dash
from Graphs import Graphs
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

def construct_mini_settings(idx):
    return html.Div([
    dcc.Markdown(f"**Algorithm parameters ({idx})**"),
    html.Div([
        html.Label("Initial value (X)"),
        dcc.Input(type="number", value=0.5, style={"marginBottom": "5px"}, className="input-values", id={"type": "initial-value-input", "index" : idx}),
    ], className="input-row"),
    html.Div([
        html.Label("Initial value (Y)"),
        dcc.Input(type="number", value=0.5, style={"marginBottom": "5px"}, className="input-values", id={"type": "initial-value-input-2", "index" : idx}),
    ], className="input-row"),
    html.Div([
        html.Label("Iterations"),
        dcc.Input(type="number", value=100, style={"marginBottom": "5px"}, className="input-values", id={"type": "number-iterations-input", "index" : idx}),
    ], className="input-row"),
    html.Div([
        html.Label("Learning Rate"),
        dcc.Input(type="number", value=0.01, step=0.001, style={"marginBottom": "5px"}, className="input-values", id={"type": "lr-mini-input", "index" : idx}),
    ], className="input-row"),
    html.Div([
        html.Label("Bregman"),
        dcc.Dropdown(
            options = [
                {"label": "Euclidean", "value": "EUCLID"},
                {"label": "KL", "value": "KL"},
                {"label": "Mahalanobis", "value": "MAHALANOBIS"},
                {"label": "Itakura-Saito", "value": "ITAKURA-SAITO"}
            ]    
        , id={"type": "bregman-mini-input", "index" : idx},className="dropdown", value="EUCLID")
    ], className = "input-row")
    ], className="settings", id={"type": "minimise-settings", "index" : idx})


function_md_settings = html.Div([
    dcc.Markdown("**Function / Data values**"),
    html.Div([
        html.Label("Function"),
        dcc.Input(type="text", value="X**2 + Y**2", style={"marginBottom": "5px"}, className="input-function", id="function-input")
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

minimise_config = html.Div([html.Div([
    dcc.Markdown("**Objective function and algorithm parameters**"),
    html.Div([
        html.Label("Objective Function"),
        dcc.Input(type="text", value="X**2 + Y**2", style={"marginBottom": "5px"}, className="input-function", id="function-mini-input"),
    ], className="input-row"),
    html.Div([
        html.Label("Variables"),
        dcc.Input(type="number", value=1, step=1, min=1, max=2, style={"marginBottom": "5px"}, className="input-values", id="num-variables-input"),
    ], className="input-row"),
    construct_mini_settings(1)
], className= "settings", id="minimise-config")], id="experiment-settings-container", className="option-columns-mlp")

approx_config = html.Div([
                    construct_model_settings(1), function_md_settings],
                id="experiment-settings-container", className="option-columns-mlp")


# placeholder variables to be updated via callback
experiment_settings_container = minimise_config
run_button_container = html.Div(id="run-button-container")
experiment_results = html.Div([], id="experiment-output", className="experiment-graphs")
experiment_settings_type_store = dcc.Store(id="experiment-settings-type", data="minimise")

config_options = html.Div([
    dcc.Markdown("#### Configuration", className="markdown-config"),
    html.Div([
        html.Button("Approximate", className="config-button", id="approx-button", n_clicks=0),
        html.Button("Minimise",   className="config-button-clicked", id="min-button",   n_clicks=0)
    ], id="top-div-config"),
    experiment_settings_container,
    run_button_container
], className="configuration-options", id="config-options")

num_experiments_min_store = dcc.Store(id="num-experiments-min", data=1)
num_experiments_approx_store = dcc.Store(id="num-experiments-approx", data=1)

global_approx_clicks_store = dcc.Store(id="approx-clicks", data=0)
global_min_clicks_store = dcc.Store(id="min-clicks", data=0)


layout = html.Div([
    dcc.Markdown(explanation_md, className="markdown", id="config-info"),
    html.Div([
        config_options,
        experiment_results
    ], className= "experiment-div"),
    experiment_settings_type_store,
    num_experiments_min_store,
    num_experiments_approx_store,
    global_approx_clicks_store,
    global_min_clicks_store
    
], style={"padding": "20px"})


minimise_run_button = html.Button("Run Experiment", className="run-button", n_clicks=0, id="run-button-minimise")
approximate_run_button = html.Button("Run Experiment", className="run-button", n_clicks=0, id="run-button-approximate")
minimise_add_button = html.Button("+", className="add-button", n_clicks=1, id="add-button-minimise")
approximate_add_button = html.Button("+", className="add-button", n_clicks=1, id="add-button-approximate")

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


# Callback to add another experiment configuration when running an approximation experiment
@callback(
        Output("config-options", "children"),
        Output("num-experiments-approx", "data"),
        Input("add-button-approximate", "n_clicks"),
        State("config-options", "children"),
        State("num-experiments-approx", "data"),
        State("experiment-settings-type", "data"),
        prevent_initial_call=True
)
def add_configuration_approx(n_clicks, current_children, num_experiments, experiment_type):
    # if n_clicks is equal to num_experiments, this has been falsely called due to a resetting of the button,
    # not an actual update to n_clicks

    if (not n_clicks) or (n_clicks == num_experiments) or (experiment_type != "approximate"):
        print("hello")
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

# callback to add another configuration when running a minimisation experiment
@callback(
        Output("config-options", "children", allow_duplicate=True),
        Output("num-experiments-min", "data", allow_duplicate=True),
        Input("add-button-minimise", "n_clicks"),
        State("config-options", "children"),
        State("num-experiments-min", "data"),
        State("experiment-settings-type", "data"),
        prevent_initial_call=True
)
def add_configuration_mini(n_clicks, current_children, num_experiments, experiment_type):
    if (not n_clicks) or (n_clicks == num_experiments) or (experiment_type != "minimise"):
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

# assumes batch mirror descent, and automatically sets the batch size to be equal to the number of samples
# if user changes the batch size for mirror descent it shouldn't change back unless the sample number is changed
@callback(
    Output("batch-size-input", "value"),
    Input("num-samples-input", "value")
)
def update_batch_size(num_samples):
    return num_samples


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
    
    new_children_mini = current_children[:2] + [minimise_config] + [current_children[-1]]
    new_children_approx = current_children[:2] + [approx_config] + [current_children[-1]]
    if not context.triggered:
        
        return new_children_mini, "config-button", "config-button-clicked", "minimise", g_approx_clicks, g_min_clicks, 1, 1
    else:
        # find which button was clicked then update accordingly
        button_triggered = context.triggered[0]["prop_id"].split(".")[0]
        if button_triggered == "min-button":
            if min_clicks == g_min_clicks:
                return no_update
            g_min_clicks = min_clicks
            return new_children_mini, "config-button", "config-button-clicked", "minimise",g_approx_clicks, g_min_clicks, 1, 1
        elif button_triggered == "approx-button": 
            if approx_clicks == g_approx_clicks:
                return no_update
            
            g_approx_clicks = approx_clicks
            return new_children_approx, "config-button-clicked", "config-button", "approximate", g_approx_clicks, g_min_clicks, 1, 1
        else:
            if approx_clicks == g_approx_clicks:
                return no_update
            g_min_clicks = min_clicks
            return new_children_mini, "config-button", "config-button-clicked", "minimise", g_approx_clicks, g_min_clicks, 1, 1


# updates which run button (approx or minimise) to display based on the experiment type store
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

# run an approximation experiment, taking in single or multiple experiment configurations
# overlays the graphs to directly compare results immediately
@callback(
    Output("experiment-output", "children", allow_duplicate=True),
    Output("run-button-approximate", "n_clicks"),
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
    prevent_initial_call=True,
    allow_duplicate=True,
    suppress_callback_exceptions=True,
    debug=False
)
def run_experiment_mlp(n_clicks, layers, neurons, epochs, function, range_min, range_max,
                        n_samples, batch_size, bregman, loss, learning_rate, num_experiments):
    if n_clicks != 0:       
        # parse from string the function to approximate
        parser = FunctionParser(function)
        parser.test_function()
        fta = parser.string_to_lambda()
        # test the function on a value of 1
        print(fta(torch.tensor(1)))

        # instantiate the experiment object
        experiment = ExperimentMD(fta, bregman=bregman[0])
        print("Experiment instantiated")
        experiment.criterion = experiment.losses[loss[0]]

        # insantiate the graph object
        graph = Graphs()

        
        # run the first experiment, generate and return figures
        experiment.run_experiment_mlp(range_min, range_max, n_samples, batch_size[0], layers[0],
                                       neurons[0], epochs[0], float(learning_rate[0]))

        loss_fig = graph.create_loss_curve(experiment.loss_logs)
        gradient_fig = graph.create_gradient_norm_graph(experiment.gradient_logs)
        divergence_fig = graph.create_divergence_graph(experiment.divergence_logs)
        results_fig = graph.create_function_approximation_plot(experiment.prediction_data)

        for i in range(1, num_experiments):
            # clear the experiment logs for the next experiment
            experiment.clear() 
            # update bregman and loss
            experiment.bregman, experiment.criterion = bregman[i], experiment.losses[loss[i]]
            # run the new experiment
            experiment.run_experiment_mlp(range_min, range_max, n_samples, batch_size[i], layers[i],
                                       neurons[i], epochs[i], float(learning_rate[i]))
            # update graphs
            loss_fig, gradient_fig, divergence_fig, results_fig = graph.update_all_graphs_approx(experiment.loss_logs, experiment.gradient_logs,
                                                                                                 experiment.divergence_logs, experiment.prediction_data, i+1)




        n_clicks = 0 
        return [dcc.Graph(figure=loss_fig, id="loss-curve", config={'responsive': True},className="graph"),
                dcc.Graph(figure=gradient_fig, id="gradient-fig",config={'responsive': True}, className="graph"),
                dcc.Graph(figure=divergence_fig, id="divergence-fig",config={'responsive': True}, className="graph"),
                dcc.Graph(figure=results_fig, id="results_fig",config={'responsive': True}, className="graph")], 0
    else: 
        return no_update

        
        
# run a minimisation experiment, taking in single or multiple experiment configurations
# overlays the graphs to directly compare results immediately
@callback(
    Output("experiment-output", "children"),
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
    prevent_initial_call=True,
    allow_duplicate=True,
    suppress_callback_exceptions=True
    
)
def run_experiment_minimise(n_clicks, objective_string, init_x, init_y, iter, lr, bregman, num_experiments, second_input_bool):
    if n_clicks != 0:
        # parse objective function from string 
        print(second_input_bool)
        if second_input_bool[0]==False:
            inits = [[float(x), float(y)] for x, y in zip(init_x, init_y)]
            print("?")
            dim = 2
            test = [1, 2]
        else: 
            inits = [float(x) for x in init_x]
            dim = 1
            test = 1
        print(inits)
        print(init_y)
        parser = FunctionParser(objective_string)
        
        objective = parser.string_to_lambda()

        # instantiate experiment object
        experiment = ExperimentMD(objective, bregman=bregman[0])
        print("Experiment instantiated")

        # run experiment, generate and return figures
        print(experiment.objective)
        experiment.run_experiment_minimise(inits[0], iter[0], float(lr[0]))
        print("experiment complete")

        # instantiate the graph class
        graph = Graphs()

        optimisation_path_fig = graph.create_optimisation_path_graph(experiment.minimisation_guesses, experiment.objective, dim)
        gradient_fig = graph.create_gradient_norm_graph(experiment.gradient_logs)
        divergence_fig = graph.create_divergence_graph(experiment.divergence_logs)

        for i in range(1, num_experiments):
            experiment.clear()
            experiment.bregman = bregman[i]
            experiment.run_experiment_minimise(inits[i], iter[i], float(lr[i]))
            optimisation_path_fig, gradient_fig, divergence_fig = graph.update_all_graphs_min(experiment.minimisation_guesses, experiment.gradient_logs,experiment.divergence_logs, experiment.objective, i+1, dim)

        n_clicks = 0
        return [dcc.Graph(figure=optimisation_path_fig, id="optimisation-path-fig", config={'responsive': True},className="graph"),
                dcc.Graph(figure=gradient_fig, id="gradient-fig",config={'responsive': True}, className="graph"),
                dcc.Graph(figure=divergence_fig, id="divergence-fig",config={'responsive': True}, className="graph")], 0 
    
    else:
        return no_update











    
    
