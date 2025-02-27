import dash
from dash import dcc, html, Input, Output, callback_context, no_update, callback
import plotly.express as px

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

if __name__ == "__main__":
    app.run_server(debug=True)