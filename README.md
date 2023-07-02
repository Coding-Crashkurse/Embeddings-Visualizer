# Text Embeddings Visualization with Plotly

This project is a web application built with [Streamlit](https://streamlit.io/). It allows users to upload a CSV file and visualize text embeddings in a 3D space using [Plotly](https://plotly.com/).

The application leverages the power of AI through [OpenAI's text-embedding-ada-002 model](https://www.openai.com/) to transform natural language text into high-dimensional vectors, or "embeddings". It uses [Principal Component Analysis (PCA)](https://en.wikipedia.org/wiki/Principal_component_analysis) to reduce the dimensionality of these vectors for visualization purposes.

## Features

- Upload CSV files.
- Select the columns of interest.
- Apply PCA to reduce the dimensionality of the text embeddings.
- Visualize the embeddings in a 3D plot using Plotly.
- Inspect the data in a table view.

## Installation

1. Clone the repository or download the files.
2. Install the required libraries:
   ```
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

## Usage

1. Start the application. A web page should open in your default web browser.
2. Use the file uploader to upload a CSV file. The file should have at least two columns: one for categories and one for the text to be embedded.
3. Use the dropdowns to select the columns that contain the categories and the text.
4. Click the button to generate the embeddings and visualize them in a 3D plot.

Please note: The quality and the number of text embeddings directly affect the time it takes to generate the plot.

## Requirements

This project requires Python 3.9 or later and the following Python libraries installed:

- [streamlit](https://streamlit.io/)
- [pandas](https://pandas.pydata.org/)
- [openai](https://www.openai.com/)
- [plotly](https://plotly.com/)
- [numpy](https://numpy.org/)
- [python-dotenv](https://pypi.org/project/python-dotenv/)
- [scikit-learn](https://scikit-learn.org/stable/)

You will also need to have the OpenAI API key. If you do not have one, you can request it from the [OpenAI website](https://beta.openai.com/). Once you have the key, create a `.env` file in the root directory of this project with the following content:

```
OPENAI_API_KEY=your_openai_api_key
```
