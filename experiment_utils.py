from dash import html
from FunctionParser import FunctionParser
from PresetFuncs import AnisotropicQuadratic, SimplexObjective, CubicObjective, Rosenbrock, Rastrigin, Booth, Ackley, ExponentialObjective2D
import torch

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
    className="experiment-result",
    id={"type": "experiment-result", "index": idx})

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
    className="experiment-result",
    id={"type": "experiment-result", "index": idx})


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


# adds a highlight on figures by matching the point index
def add_highlight(fig, label_prefix, exp_num, pt_index):
        highlights = []
        for tr in fig.get('data', []):
            if tr.get('type') == 'contour':
                continue
            candidate_name = tr.get('name', '')
            try:
                candidate_num = int(''.join(filter(str.isdigit, candidate_name)))
            except Exception:
                candidate_num = None
            if (exp_num == 1 and candidate_name.lower() == "dual trajectory") or \
               (candidate_num is not None and candidate_num == exp_num):
                xs = tr.get('x', [])
                ys = tr.get('y', [])
                if pt_index < len(xs) and pt_index < len(ys):
                    highlights.append({
                        'x': [xs[pt_index]],
                        'y': [ys[pt_index]],
                        'mode': 'markers',
                        'marker': {'size': 8, 'color': '#322634'},
                        'name': 'Highlight',
                        'hovertemplate': f"{label_prefix}: ({xs[pt_index]:.2f}, {ys[pt_index]:.2f})<extra></extra>",
                    })
        fig['data'].extend(highlights)
        return fig

# removes any highlight traces when user isnt hovering the optim trajs
def remove_highlights(fig):
    fig['data'] = [trace for trace in fig.get('data', []) if trace.get('name') != 'Highlight']
    return fig

# makes a shallow copy of the figure
def clone_fig_shallow(fig):
    new_fig = fig.copy()
    new_fig['data'] = fig.get('data', [])[:]
    new_fig['layout'] = fig.get('layout', {}).copy()
    return new_fig

# finds x,y for non-simplex problems, for simplex problems, reads the already generated hovertext from the Graphs.py function
# to get p1, p2, p3
def get_corresponding_value(fig, exp_num, pt_index, return_hovertext=False, ):
    for tr in fig.get('data', []):
        if tr.get('type') == 'contour':
            continue
        candidate_name = tr.get('name', '')
        try:
            candidate_num = int(''.join(filter(str.isdigit, candidate_name)))
        except Exception:
            candidate_num = None
        if (exp_num == 1 and candidate_name.lower() == "dual trajectory") or \
        (candidate_num is not None and candidate_num == exp_num):
            if return_hovertext and 'hovertext' in tr and isinstance(tr['hovertext'], list):
                return tr['hovertext'][pt_index]
            else:
                xs = tr.get('x', [])
                ys = tr.get('y', [])
                if pt_index < len(xs) and pt_index < len(ys):
                    return xs[pt_index], ys[pt_index]
    return (None, None) if not return_hovertext else None


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
            