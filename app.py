import dash
from dash import dcc, html, Input, Output, callback_context, no_update, callback, State, Patch, ALL, MATCH, set_props

import plotly.express as px
from Graphs import Graphs
from Experiment import ExperimentMD
from dash.long_callback import DiskcacheLongCallbackManager
import torch
import diskcache
import time
from experiment_utils import setup_inits, get_objective_function, create_compiled_metrics_dicts, create_experiment_dict_min, construct_experiment_results
import plotly.io as pio 
from dash.long_callback import DiskcacheManager
cache = diskcache.Cache("./cache")
background_callback_manager = DiskcacheManager(cache)
app = dash.Dash(__name__, use_pages=True, suppress_callback_exceptions=True)

app.index_string = """
<!DOCTYPE html>
<html>
<head>
    <title>Mirror Descent & Bregman Divergences</title>
    <!-- Load MathJax 3 from a CDN -->
    {%css%}
</head>
<body>
    <div id="react-entry-point">
        {%app_entry%}
    </div>
    <footer>
        {%config%}
        {%scripts%}
        {%renderer%}
    </footer>
</body>
</html>
"""

app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    html.H1("Mirror Descent Optimisation Toolkit",
            className="headers"),
    html.Div([
        html.Div(
            dcc.Link(f"{page['name']}", href=page["relative_path"]), className="navlinks"
        ) for page in dash.page_registry.values()
    ], className="navbar", id="navbar"),
    dash.page_container
])

@callback(
    Output("navbar", "children"), 
    Input("url", "pathname")
)
def update_navbar(pathname):
    links = [] 
    for page in dash.page_registry.values():
        active = (page)
        active = (page['relative_path'] == pathname)
        classname = "navlinks-active" if active else "navlinks"
        links.append(
            dcc.Link(page['name'], href=page['relative_path'], className=classname)
        )
    return links
# @app.callback(
#     Output("last-min-config", "data"),
#     Input("run-button-minimise", "n_clicks"),
#     State("function-mini-input", "value"),
#     State("aggregated-inputs", "data"),  # aggregated pattern-matched inputs (including q_strings)
#     State("Q-store", "data"),
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
#     background=True,
#     running=[
#         (Output("run-button-minimise", "disabled"), True, False),
#         (Output("save-button-minimise", "disabled"), True, False)
#     ],
#     cancel=[],  # no cancel inputs in this example
#     manager=background_callback_manager,
#     prevent_initial_call=True,
#     allow_duplicate=True,
#     suppress_callback_exceptions=True,
# )
# def run_experiment_minimise_bg(n_clicks, objective_string, agg_inputs, q_store,
#                                q1, q2, q3, a, b, optx, opty, noise_std,
#                                preset_function, loading_metrics):
#     # only proceed if run-button n_clicks is nonzero
#     if not n_clicks:
#         return no_update

#     # unpack aggregated inputs
#     init_x    = agg_inputs.get("init_vals")
#     init_y    = agg_inputs.get("init_vals2")
#     iter_list = agg_inputs.get("iters")
#     lr        = agg_inputs.get("lrs")
#     bregman   = agg_inputs.get("bregmans")
#     # pattern-matched inputs for Q (if needed), simplex parameters and q_strings
#     q_inputs  = agg_inputs.get("q_inputs")
#     p1s       = agg_inputs.get("p1s")
#     p2s       = agg_inputs.get("p2s")
#     p3s       = agg_inputs.get("p3s")
#     q_strings = agg_inputs.get("q_strings")
    
#     # parse objective function and set up initial values
#     print(f"inputted initials {init_x} {init_y}")
#     inits, dim = setup_inits(preset_function, None, init_x, init_y, p1s, p2s, p3s)
#     objective = get_objective_function(preset_function, objective_string, a, b, q1, q2, q3, optx, opty, noise_std=noise_std)
    
#     if preset_function != "CUSTOM":
#         optimum = objective(objective.optimum)
#         optimum_coords = objective.optimum
#     else:
#         optimum, optimum_coords = None, None

#     # instantiate experiment object
#     experiment = ExperimentMD(objective, bregman=bregman[0],
#                                 Q=torch.tensor(q_store[0][0], dtype=torch.float64),
#                                 Q_inv=torch.tensor(q_store[0][1], dtype=torch.float64),
#                                 x_star=optimum_coords, f_star=optimum, dim=dim)
#     print("experiment instantiated")
#     print(experiment.Q)
#     experiment_metrics = []
    
#     # run first experiment and gather metrics
#     print(experiment.objective)
#     experiment.run_experiment_minimise(inits[0], iter_list[0], float(lr[0]))
#     experiment_metrics.append(construct_experiment_results(1, experiment.gather_metrics()))
#     print("experiment complete")
    
#     # instantiate graph class and create figures;
#     # force any generator outputs to materialize by wrapping with list()
#     graph = Graphs()
#     optimisation_path_fig = graph.create_optimisation_path_graph(list(experiment.minimisation_guesses), 
#                                                                 experiment.objective, dim)
#     gradient_fig = graph.create_gradient_norm_graph(list(experiment.gradient_logs))
#     divergence_fig = graph.create_divergence_graph(list(experiment.avg_divergence_logs))
#     dual_fig = graph.create_dual_space_trajectory_graph(list(experiment.optimiser.logs["dual"]), 
#                                                         experiment.objective, dim)
    
#     # update iterative outputs via set_props
#     set_props("optimisation-path-fig", {"figure": optimisation_path_fig})
#     set_props("dual-fig", {"figure": dual_fig})
#     set_props("divergence-fig", {"figure": divergence_fig})
#     set_props("gradient-fig", {"figure": gradient_fig})
#     set_props("loading-metrics", {"children": experiment_metrics})
    
#     # run subsequent experiments and update iteratively
#     for i in range(1, len(iter_list)):
#         experiment.clear()
#         experiment.bregman, experiment.Q, experiment.Q_inv = bregman[i], \
#             torch.tensor(q_store[i][0], dtype=torch.float64), \
#             torch.tensor(q_store[i][1], dtype=torch.float64)
#         experiment.run_experiment_minimise(inits[i], iter_list[i], float(lr[i]))
#         experiment_metrics.append(construct_experiment_results(i+1, experiment.gather_metrics()))
#         figs = graph.update_all_graphs_min(
#             experiment.minimisation_guesses, list(experiment.gradient_logs),
#             list(experiment.avg_divergence_logs), list(experiment.optimiser.logs["dual"]),
#             experiment.objective, i+1, dim
#         )
#         optimisation_path_fig, gradient_fig, divergence_fig, dual_fig = tuple(figs)
#         set_props("optimisation-path-fig", {"figure": optimisation_path_fig})
#         set_props("dual-fig", {"figure": dual_fig})
#         set_props("divergence-fig", {"figure": divergence_fig})
#         set_props("gradient-fig", {"figure": gradient_fig})
#         set_props("loading-metrics", {"children": experiment_metrics})
#         time.sleep(0.1)  # allow UI to update between iterations
    
#     # store the run configuration for saving
#     experiments_dict = create_experiment_dict_min(len(iter_list), init_x, init_y, iter_list, lr, bregman, None, q_strings, p1s, p2s, p3s)
#     metrics_dict = create_compiled_metrics_dicts(len(iter_list), experiment_metrics)
#     experiment_state = {
#         "configuration": {
#             "experiment_type": "minimise",
#             "function": objective_string,
#             "function_preset": preset_function,
#             "var_a": a,
#             "var_b": b,
#             "opt_x": optx,
#             "opt_y": opty,
#             "noise": noise_std,
#             "q1": q1,
#             "q2": q2,
#             "q3": q3
#         },
#         "experiments": experiments_dict,
#         "metrics": metrics_dict,
#         "figures": {
#             "optim_fig": pio.to_json(optimisation_path_fig),
#             "dual_optim_fig": pio.to_json(dual_fig),
#             "gradient_fig": pio.to_json(gradient_fig),
#             "divergence_fig": pio.to_json(divergence_fig),
#         }
#     }
#     # reset run button and enable save button via set_props
#     set_props("run-button-minimise", {"n_clicks": 0})
#     set_props("save-button-minimise", {"disabled": False})
    
#     return experiment_state

if __name__ == "__main__":
    app.run_server(debug=True)