from dash import html, dcc, callback, Input, Output, State, callback_context, no_update, Patch, ALL
import dash
from Graphs import Graphs
from Experiment import ExperimentMD
from FunctionParser import FunctionParser
import plotly.graph_objects as plotly 
import torch
import json 
import base64

dash.register_page(__name__, path="/run-experiment")

explanation_md = r"""
### Running an experiment 
Here you can configure and run your own experiments with the mirror descent algorithm.
There are two different types of experiments available: 
* Minimising an objective function using mirror descent.
* Approximating a function with a simple regression model, using mirror descent to minimise the loss over training. 
"""

loading_md = r"""
### Loading experiments
There are several pre-configured experiments which you can load. Alternatively you have the option to save experiments after running,
and upload the configuration to run again.
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

def construct_mini_settings(idx, init_x=0.5, init_y=0.5, iterations=100, lr=0.01, bregman="EUCLID"):
    return html.Div([
        dcc.Markdown(f"**Algorithm parameters ({idx})**"),
        html.Div([
            html.Label("Initial value (X)"),
            dcc.Input(type="number", value=init_x, style={"marginBottom": "5px"},
                      className="input-values", id={"type": "initial-value-input", "index": idx}),
        ], className="input-row"),
        html.Div([
            html.Label("Initial value (Y)"),
            dcc.Input(type="number", value=init_y, style={"marginBottom": "5px"},
                      className="input-values", id={"type": "initial-value-input-2", "index": idx}),
        ], className="input-row"),
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
        ], className="input-row")
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
], className= "settings", id="inner-div")], id="minimise-config", className="option-columns-mlp")

approx_config = html.Div([
                    construct_model_settings(1), function_md_settings],
                id="approx-config", className="option-columns-mlp hidden")


# placeholder variables to be updated via callback

run_button_container = html.Div(id="run-button-container")
experiment_results = html.Div([], id="experiment-output", className="experiment-graphs")
experiment_settings_type_store = dcc.Store(id="experiment-settings-type", data="minimise")

config_options = html.Div([
    dcc.Markdown("#### Configuration", className="markdown-config"),
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

# download component that gets triggered when a save button is clicked 
run_config_download = dcc.Download(id="save-config")

# stores the current number save button clicks 
mini_save_clicks_store = dcc.Store(id="mini-save-clicks", data=0)
approx_save_clicks_store = dcc.Store(id="approx-save-clicks", data=0)

# stores the current number of approx/mini button clicks 
global_approx_clicks_store = dcc.Store(id="approx-clicks", data=0)
global_min_clicks_store = dcc.Store(id="min-clicks", data=0)


layout = html.Div([
    html.Div([
        html.Div([
            dcc.Markdown(explanation_md, className="padding-markdown")
        ], className="experiment-desc"),
        html.Div([
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
    approx_save_clicks_store
    
], style={"padding": "5px 20px 20px 20px"})


minimise_run_button = html.Button("Run Experiment", className="run-button", n_clicks=0, id="run-button-minimise")
approximate_run_button = html.Button("Run Experiment", className="run-button", n_clicks=0, id="run-button-approximate")
minimise_add_button = html.Button("+", className="add-button", n_clicks=1, id="add-button-minimise", title="add a configuration")
approximate_add_button = html.Button("+", className="add-button", n_clicks=1, id="add-button-approximate", title="add a configuration")
minimise_save_button = html.Button("Save", className="save-button", id="save-button-minimise", disabled=True, title="Cannot save until experiment has ran", n_clicks=0)
approximate_save_button = html.Button("Save", className="save-button", id="save-button-approximate", disabled=True,  title="Cannot save until experiment has ran", n_clicks=0)



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
    Output("run-button-container", "children"),
    Input("experiment-settings-type", "data"),
)
def update_run_button(experiment_type):
    if experiment_type == "minimise":
        return [minimise_run_button, minimise_add_button, minimise_save_button]
    elif experiment_type == "approximate":
        return [approximate_run_button, approximate_add_button, approximate_save_button]
    else: 
        return ValueError("Unrecognised experiment type")

# run an approximation experiment, taking in single or multiple experiment configurations
# overlays the graphs to directly compare results immediately
@callback(
    Output("experiment-output", "children", allow_duplicate=True),
    Output("run-button-approximate", "n_clicks"),
    Output("save-button-approximate", "disabled"),
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
                dcc.Graph(figure=results_fig, id="results_fig",config={'responsive': True}, className="graph")],0, False
                    
    else: 
        return no_update

# function that creates a dictionary of the different experiment configurations used for a minimisation run 
def create_experiment_dict(num_experiments, init_x, init_y, iter, lr, bregman, second_input_bool):
    experiments_dict = {}
    for i in range(num_experiments):
        experiments_dict[f"experiment-{i+1}"] = {
            "initial_value_x": init_x[i],
            "initial_value_y": init_y[i] if not second_input_bool[0] else None,
            "iterations": iter[i],
            "learning_rate": lr[i],
            "bregman": bregman[i]
    }
    return experiments_dict
        
# run a minimisation experiment, taking in single or multiple experiment configurations
# overlays the graphs to directly compare results immediately
@callback(
    Output("experiment-output", "children"),
    Output("run-button-minimise", "n_clicks"),
    Output("save-button-minimise", "disabled"), 
    Output("last-min-config", "data"),
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
        print(f"inputted initials {init_x} {init_y}")
        
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

        # store the run configuration for saving 
        experiments_dict = create_experiment_dict(num_experiments, init_x, init_y, iter, lr, bregman, second_input_bool)

        experiment_state = {
            "configuration": {
                "experiment_type": "minimise",  
                "function": objective_string,      
            },
            "experiments": experiments_dict,
            "results": {
                "mini_guess_logs": experiment.minimisation_guesses,
                "gradient_logs": experiment.gradient_logs,
                "divergence_logs": experiment.divergence_logs,
            },
            "figures": {
                "optim_fig": optimisation_path_fig.to_plotly_json(),
                "gradient_fig": gradient_fig.to_plotly_json(),
                "divergence_fig": divergence_fig.to_plotly_json(),
            }
}

        return [dcc.Graph(figure=optimisation_path_fig, id="optimisation-path-fig", config={'responsive': True},className="graph"),
                dcc.Graph(figure=gradient_fig, id="gradient-fig",config={'responsive': True}, className="graph"),
                dcc.Graph(figure=divergence_fig, id="divergence-fig",config={'responsive': True}, className="graph")], 0, False, experiment_state
    
    else:
        return no_update

@callback(
    Output("save-button-minimise", "disabled", allow_duplicate=True),
    Input("function-mini-input", "value"),
    Input({"type": "initial-value-input", "index": ALL}, "value"),
    Input({"type": "initial-value-input-2", "index": ALL}, "value"),
    Input({"type": "number-iterations-input", "index": ALL}, "value"),
    Input({"type": "lr-mini-input", "index": ALL}, "value"),
    Input({"type": "bregman-mini-input", "index": ALL}, "value"),
    Input("num-experiments-min", "data"),
    Input({"type": "initial-value-input-2", "index": ALL}, "disabled"),
    State("last-min-config", "data"),
    prevent_initial_call=True
)
def listen_then_disable_save(objective_string, init_x, init_y, iter, lr, bregman, num_experiments, second_input_bool, last_config):
    # this callback listens for changes in any input paramaters for the minimisation variant.
    # if configuration has changed since the last run, then disable the save button to prevent users saving a run that has results for a different configuration

    experiments_dict = create_experiment_dict(num_experiments, init_x, init_y, iter, lr, bregman, second_input_bool)
    current_config = {
            "configuration": {
                "experiment_type": "minimise",  
                "function": objective_string,      
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
    Input("save-button-minimise", "n_clicks"),
    State("last-min-config", "data"),
    prevent_initial_call=True
)
def download_minimise_experiment(n_clicks, experiment_data):
    # triggers a download upon save button being clicked
    # have to store n_clicks in a global dcc.store in order to compare to the save buttons value
    # as this callback gets triggered when the user adds a configuration due to the save button reloading 
    triggered = callback_context.triggered
    if not triggered:
        return no_update
    # Check if the triggered prop_id exactly matches "save-button-minimise.n_clicks"
    print("save button n_clicks = ", n_clicks)
    triggered_id = triggered[0]["prop_id"]
    if triggered_id != "save-button-minimise.n_clicks" or n_clicks == 0 or not experiment_data:
        return no_update
    
    json_str = json.dumps(experiment_data, indent=2)

    return dict(content=json_str, filename="experiment.json"), 0
    

# builds a matching minimise-configuration from the saved json
def build_minimise_config_from_saved(saved_state):
    experiments = saved_state.get("experiments", {})

    experiment_configs = []
    for i in range(len(experiments)):
        exp = experiments.get(f"experiment-{i+1}")
        experiment_configs.append(construct_mini_settings(i+1,exp.get("initial_value_x"),
                                                        exp.get("initial_value_y"),
                                                        exp.get("iterations"),
                                                        exp.get("learning_rate"),
                                                        exp.get("bregman")))
    print("length of experiment configs ", len(experiment_configs))
    minimise_config = html.Div([html.Div([
    dcc.Markdown("**Objective function and algorithm parameters**"),
    html.Div([
        html.Label("Objective Function"),
        dcc.Input(type="text", value=saved_state["configuration"].get("function", ""), style={"marginBottom": "5px"}, className="input-function", id="function-mini-input"),
    ], className="input-row"),
    html.Div([
        html.Label("Variables"),
        dcc.Input(type="number", value=1, step=1, min=1, max=2, style={"marginBottom": "5px"}, className="input-values", id="num-variables-input"),
    ], className="input-row")] + experiment_configs
    , className= "settings", id="inner-div")], id="minimise-config", className="option-columns-mlp")

    return minimise_config

# rebuilds the graphs from the saved experiment json
def build_experiment_results_from_saved(saved_state):
    
    figs = saved_state.get("figures", {})
    graphs = []
    if "optim_fig" in figs:
        optim_fig = plotly.Figure(figs["optim_fig"])
        graphs.append(dcc.Graph(figure=optim_fig, id="optimisation-path-fig", config={'responsive': True}, className="graph"))
    if "gradient_fig" in figs:
        gradient_fig = plotly.Figure(figs["gradient_fig"])
        graphs.append(dcc.Graph(figure=gradient_fig, id="gradient-fig", config={'responsive': True}, className="graph"))
    if "divergence_fig" in figs:
        divergence_fig = plotly.Figure(figs["divergence_fig"])
        graphs.append(dcc.Graph(figure=divergence_fig, id="divergence-fig", config={'responsive': True}, className="graph"))
    
    return graphs

@callback(
    Output("config-options", "children", allow_duplicate=True),
    Output("experiment-output", "children", allow_duplicate=True),
    Output("num-experiments-min", "data", allow_duplicate=True),
    Output("add-button-minimise", "n_clicks", allow_duplicate=True),
    Input("upload-load", "n_clicks"),
    State("upload-config", "contents"),
    State("upload-config", "filename"),
    State("config-options", "children"),
    prevent_initial_call=True
)
def load_experiment(n_clicks, contents, filename, current_children):
    
    triggered = callback_context.triggered
    if not triggered:
        return no_update
    # Check if the triggered prop_id exactly matches "save-button-minimise.n_clicks"
    print("save button n_clicks = ", n_clicks)
    triggered_id = triggered[0]["prop_id"]
    if triggered_id != "upload-load.n_clicks" or n_clicks == 0 or not contents:
        return no_update
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    saved_experiment = json.loads(decoded.decode("utf-8"))

    num_experiments = len(saved_experiment.get("experiments", {}))

    minimise_config = build_minimise_config_from_saved(saved_experiment)
    print("minimise config len ", len(minimise_config))
    new_children = current_children[:2] + [minimise_config, approx_config] + [current_children[-1]]

    new_experiment_results = build_experiment_results_from_saved(saved_experiment)
    return new_children, new_experiment_results, num_experiments, num_experiments
    





    
    
