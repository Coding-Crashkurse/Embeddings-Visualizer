import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import find_dotenv, load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from openai.embeddings_utils import get_embeddings
from sklearn.decomposition import PCA

load_dotenv(find_dotenv())

st.set_page_config(layout="wide")

st.title("Text Embeddings Visualization")

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file:
    delimiter = st.text_input("Enter the delimiter", ";")
    drop_na = st.checkbox("Drop rows with N/A values")
    drop_duplicates = st.checkbox("Drop duplicate rows")
    df = pd.read_csv(uploaded_file, delimiter=delimiter)

    if drop_na:
        df = df.dropna()

    if drop_duplicates:
        df = df.drop_duplicates()

    columns = df.columns.tolist()

    category_column = st.selectbox("Choose a column for categories:", columns)
    answer_column = st.selectbox("Choose a column for answers:", columns)

    df_filtered = df[[category_column, answer_column]].dropna()
    st.write(f"The DataFrame has {df_filtered.shape[0]} rows after filtering.")

    if st.button("Compute embeddings and plot"):
        categories = sorted(df_filtered[category_column].unique())

        api_type = os.getenv("OPENAI_API_TYPE", "openai")
        if api_type == "openai":
            matrix = get_embeddings(
                df_filtered[answer_column].to_list(), engine="text-embedding-ada-002"
            )
        elif api_type == "azure":
            deployment_name = os.getenv("OPENAI_AZURE_DEPLOYMENT_NAME")
            embeddings = OpenAIEmbeddings(deployment=deployment_name)
            matrix = embeddings.embed_query(df_filtered[answer_column].to_list())

        pca = PCA(n_components=3)
        vis_dims = pca.fit_transform(matrix)
        df_filtered["embed_vis"] = vis_dims.tolist()

        cmap = px.colors.qualitative.Plotly
        fig = go.Figure()
        for i, cat in enumerate(categories):
            sub_matrix = np.array(
                df_filtered[df_filtered[category_column] == cat]["embed_vis"].to_list()
            )
            x = sub_matrix[:, 0]
            y = sub_matrix[:, 1]
            z = sub_matrix[:, 2]
            answers = df_filtered[df_filtered[category_column] == cat][
                answer_column
            ].tolist()
            color = cmap[i % len(cmap)]
            fig.add_trace(
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode="markers",
                    marker=dict(color=color, size=6, opacity=0.8),
                    hovertemplate="%{text}",
                    hoverlabel=dict(font_size=16),
                    text=answers,
                    name=cat,
                )
            )

        fig.update_layout(
            title="PCA of Text Embeddings Grouped by Categories",
            height=800,
            width=1200,
            scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"),
        )

        col1, col2 = st.columns(2)
        col1.plotly_chart(fig, use_container_width=True)
        col2.dataframe(df_filtered.head(10))
