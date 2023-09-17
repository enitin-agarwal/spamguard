import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from url_scanner_dashboard import classify_url  # Import your classification code

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("URL Spam Classifier"),

    dcc.Input(id="url-input", type="text", placeholder="Enter URL"),

    html.Div(id="classification-result"),

    dcc.Graph(id="probability-chart")
])

@app.callback(
    [Output("classification-result", "children"), Output("probability-chart", "figure")],
    [Input("url-input", "value")]
)
def update_display(url):
    if url:
        # Call your classification function to get the result and probabilities
        classification, probabilities = classify_url(url)

        # Update the display based on the classification result and probabilities
        result_display = f"Classification Result: {classification}"
        # Create a bar chart or any visualization to display probabilities
        probability_chart_data = {
    'data': [
        {
            'x': ['Legitimate', 'Spam'],
            'y': probabilities,
            'type': 'bar',
        }
    ],
    'layout': {
        'title': 'Classification Probabilities',
        'xaxis': {'title': 'Class'},
        'yaxis': {'title': 'Probability'},
    }
}
        print(result_display)

        return classification,probability_chart_data

    return "", {}

if __name__ == "__main__":
    app.run_server(debug=True)
