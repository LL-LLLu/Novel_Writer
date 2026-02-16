import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
import re
from collections import Counter

st.set_page_config(
    page_title="Novel Writer Dashboard",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Novel Writer Data Dashboard")
st.markdown("Analyze your training dataset and model performance")

# Sidebar
st.sidebar.header("Configuration")
dataset_path = st.sidebar.text_input(
    "Dataset Path",
    value="data/processed/train.jsonl"
)

# Load dataset
@st.cache_data
def load_dataset(path: str):
    if not Path(path).exists():
        return None

    entries = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            entries.append(json.loads(line))

    df = pd.DataFrame(entries)
    df['length'] = df['output'].apply(len)
    return df

df = load_dataset(dataset_path)

if df is None:
    st.error(f"Dataset not found: {dataset_path}")
    st.stop()

# Overview
st.header("Dataset Overview")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Entries", len(df))

with col2:
    st.metric("Avg Chunk Length", f"{df['length'].mean():.0f}")

with col3:
    st.metric("Total Characters", f"{df['length'].sum():,}")

# Chunk length distribution
st.header("Chunk Length Distribution")

fig_length = px.histogram(
    df,
    x='length',
    nbins=50,
    title="Distribution of Chunk Lengths",
    labels={'length': 'Characters'}
)
st.plotly_chart(fig_length, use_container_width=True)

# Dialogue analysis
st.header("Dialogue Analysis")

df['dialogue_ratio'] = df['output'].apply(
    lambda x: x.count('"') / len(x) if len(x) > 0 else 0
)
df['dialogue_lines'] = df['output'].apply(
    lambda x: len(re.findall(r'"[^"]+"', x))
)

col1, col2 = st.columns(2)

with col1:
    fig_dialogue = px.box(
        df,
        y='dialogue_ratio',
        title="Dialogue Ratio Distribution"
    )
    st.plotly_chart(fig_dialogue, use_container_width=True)

with col2:
    fig_dialogue_lines = px.histogram(
        df,
        x='dialogue_lines',
        nbins=30,
        title="Number of Dialogue Lines per Chunk"
    )
    st.plotly_chart(fig_dialogue_lines, use_container_width=True)

# Vocabulary variety
st.header("Vocabulary Analysis")

df['unique_words'] = df['output'].apply(
    lambda x: len(set(x.lower().split())) / len(x.split()) if x.split() else 0
)

fig_variety = px.histogram(
    df,
    x='unique_words',
    nbins=50,
    title="Vocabulary Variety (Unique/Total Words)",
    labels={'unique_words': 'Variety Score'}
)
st.plotly_chart(fig_variety, use_container_width=True)

# Sample inspection
st.header("Sample Data")

st.subheader("Quality Distribution")
df['quality_score'] = (
    df['length'].clip(0, 4000) / 4000 * 0.3 +
    df['dialogue_ratio'].clip(0, 0.1) / 0.1 * 0.4 +
    df['unique_words'] * 0.3
)

fig_quality = px.histogram(
    df,
    x='quality_score',
    nbins=30,
    title="Quality Score Distribution"
)
st.plotly_chart(fig_quality, use_container_width=True)

st.subheader("Top/Bottom Samples")

col1, col2 = st.columns(2)

with col1:
    st.write("### Highest Quality Chunks")
    top_samples = df.nlargest(5, 'quality_score')
    for i, row in top_samples.iterrows():
        st.text_area(
            f"Score: {row['quality_score']:.2f}",
            row['output'][:500] + "...",
            height=150,
            key=f"top_{i}"
        )

with col2:
    st.write("### Lowest Quality Chunks")
    bottom_samples = df.nsmallest(5, 'quality_score')
    for i, row in bottom_samples.iterrows():
        st.text_area(
            f"Score: {row['quality_score']:.2f}",
            row['output'][:500] + "...",
            height=150,
            key=f"bottom_{i}"
        )

# Download
st.header("Export Analysis")

st.download_button(
    label="Download Analysis CSV",
    data=df.to_csv(index=False),
    file_name="dataset_analysis.csv",
    mime="text/csv"
)
