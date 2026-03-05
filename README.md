
# Mirror Descent Playground

An interactive web application built with Dash for visualizing and interacting with Mirror Descent and different Bregman divergences and their corresponding mirror maps. The web app is currently live at https://www.mirror-descent-playground.com

## Prerequisites

Ensure you have Python installed. The application relies on several libraries, including Dash, Plotly, PyTorch, and NumPy.

## Installation

Clone the repository and install the necessary dependencies listed in the requirements file. 

```bash
pip install -r requirements.txt
```

## Usage 
To start the application server, execute the following command in the root directory:
```bash
python app.py
```
Then navigate to `http://127.0.0.1:8050/` in your web browser to use the application.

## Project Code Structure 
* `app.py`: Main entry point for the dash application.
* `Experiment.py`: Experiment engine.
* `FunctionParser.py`: Class for parsing textual inputs for objective functions e.g. "x * y^2".
* `MirrorDescent.py`: Custom PyTorch Mirror Descent optimiser.
* `PresetFuncs.py`: Contains several customisable preset objective functions.
* `Graphs.py`: Class containing all methods for creating the experiment page's Plotly figures.
* `experiment_utils.py`: Several utility functions for the experiment functionality.


