import os
import shutil
import subprocess
from pathlib import Path
from enum import Enum

import typer

app = typer.Typer()


class Provider(str, Enum):
    OPENAI = "openai"
    AZURE = "azure"


def is_valid_provider(value: str) -> str:
    if value not in Provider:
        raise typer.BadParameter("Provider should be either 'openai' or 'azure'.")
    return value


@app.command()
def init(provider: Provider):
    """
    Initialize the application:
    - Create a .env file with API settings
    - Copy the app.py, embeddings.ipynb, and questions.csv files to the current directory

    The 'provider' argument should be either 'openai' or 'azure'.
    """
    # Part 1: Create .env
    if Path(".env").exists():
        typer.echo("Warning: .env file already exists.")
    else:
        with open(".env", "w") as f:
            if provider == Provider.OPENAI:
                f.write("OPENAI_API_TYPE=openai\n")
                f.write("OPENAI_API_KEY=\n")
            elif provider == Provider.AZURE:
                f.write("OPENAI_API_TYPE=azure\n")
                f.write("OPENAI_API_BASE=https://<your-endpoint>.openai.azure.com/\n")
                f.write("OPENAI_API_KEY=<your AzureOpenAI key>\n")
                f.write("OPENAI_API_VERSION=2023-03-15-preview\n")
        typer.echo(f".env file with {provider.value.upper()} API settings created.")

    # Part 2: Copy files
    current_dir = Path.cwd()
    for filename in ["app.py", "embeddings.ipynb", "questions.csv"]:
        source_file = Path(__file__).parent / filename
        target_file = current_dir / filename
        if not source_file.exists():
            typer.echo(f"Error: Could not find source file at {source_file}.")
            raise typer.Exit(code=1)
        if target_file.exists():
            typer.echo(f"Warning: {target_file} already exists.")
        else:
            shutil.copy2(source_file, target_file)
            typer.echo(f"Copied {filename} to {current_dir}.")


@app.command()
def start_app():
    """
    Start the app
    """
    subprocess.call(["streamlit", "run", "app.py"])


@app.command()
def open_notebook():
    """
    Open the embeddings.ipynb with Jupyter
    """
    subprocess.call(["jupyter", "notebook", "embeddings.ipynb"])


if __name__ == "__main__":
    app()
