# Embeddings Visualizer

`Embeddings Visualizer` is a Python package that provides tools for visualizing text embeddings generated from the OpenAI API. The package uses Streamlit for creating interactive visualization dashboards, and can be executed within a local Python environment or deployed to a web server.

## Initialization

Initialize the package using the `init` command:

`ev init`

This will guide you through the process of setting up your configuration. Please ensure you have your OpenAI API key available.

## Usage

Once initialized, you can start the Streamlit application using the `start-app` command:

`ev start-app`

This will start the Streamlit application where you can upload your dataset and interactively visualize the embeddings.

To open the notebook, use the `open-notebook` command:

`ev open-notebook`

This will open the `embeddings.ipynb` Jupyter notebook in your browser, where you can interactively experiment with generating and visualizing embeddings.

## Requirements

Python 3.9 or later is required to use this package. It also depends on several Python libraries, including Streamlit, Typer, numpy, pandas, OpenAI, python-dotenv, scikit-learn, plotly, matplotlib and langchain. These dependencies are automatically installed when you install the `Embeddings Visualizer` package.

For more information, please refer to the `pyproject.toml` file in the root directory of the project.
