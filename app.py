import dash
from dash import dcc, html, Input, Output, callback_context, no_update
import plotly.express as px


app = dash.Dash(__name__, use_pages=True)

app.index_string = """
<!DOCTYPE html>
<html>
<head>
    <title>Mirror Descent & Bregman Divergences</title>
    <!-- Load MathJax 3 from a CDN -->
    <script
      type="text/javascript"
      async
      src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"
    ></script>
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
    html.H1("Mirror Descent Optimisation Toolkit",
            className="headers"),
    html.Div([
        html.Div(
            dcc.Link(f"{page['name']}", href=page["relative_path"]), className="navlinks"
        ) for page in dash.page_registry.values()
    ], className="navbar"),
    dash.page_container
])


if __name__ == "__main__":
    app.run_server(debug=True)