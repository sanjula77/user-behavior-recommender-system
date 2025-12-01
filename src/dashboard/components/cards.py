import streamlit as st

def metric_card(title: str, value: str, delta: str = None, help_text: str = None):
    """
    Displays a metric card with a title, value, and optional delta.
    """
    st.metric(label=title, value=value, delta=delta, help=help_text)

def kpi_grid(metrics: list, cols: int = 4):
    """
    Displays a grid of metric cards.
    metrics: List of dicts with keys: title, value, delta, help
    """
    columns = st.columns(cols)
    for i, metric in enumerate(metrics):
        with columns[i % cols]:
            metric_card(
                title=metric.get("title"),
                value=metric.get("value"),
                delta=metric.get("delta"),
                help_text=metric.get("help")
            )
