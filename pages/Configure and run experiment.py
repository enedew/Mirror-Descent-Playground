from dash import html, dcc
import dash

dash.register_page(__name__)

layout = html.Div([
    html.H2("Mirror Descent Experiments"),
    html.P("Use the controls below to run a PyTorch-based mirror descent experiment."),

    # Example slider for learning rate
    html.Label("Learning Rate"),
    dcc.Slider(id="learning-rate-slider", min=0.001, max=0.1, step=0.001, value=0.01),

    html.Br(),

    # Button to run the experiment
    html.Button("Run Experiment", id="run-experiment-button", n_clicks=0),

    html.Br(), html.Br(),

    # Graph to display results
    dcc.Graph(id="experiment-results-graph"),

    # Button to go back to the info page
    html.Br(),
    html.Button("Back to Info", id="go-back-home-button", n_clicks=0),
], style={"padding": "20px"})