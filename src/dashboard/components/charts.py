import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from src.dashboard.config import COLORS

def plot_trend_line(df: pd.DataFrame, x: str, y: str, title: str, color: str = None):
    """
    Creates a line chart for trends.
    """
    if df.empty or x not in df.columns or y not in df.columns:
        # Return empty figure with error message
        fig = go.Figure()
        fig.add_annotation(
            text=f"Data not available: Missing {x} or {y} column",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    fig = px.line(df, x=x, y=y, title=title, markers=True)
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color=COLORS["secondary"] if pd.isna(color) else "white",
        xaxis_showgrid=False,
        yaxis_showgrid=True,
        yaxis_gridcolor="rgba(255,255,255,0.1)",
        hovermode="x unified"
    )
    fig.update_traces(line_color=COLORS["primary"], line_width=3)
    return fig

def plot_segment_distribution(df: pd.DataFrame, names: str, values: str, title: str):
    """
    Creates a donut chart for segment distribution.
    """
    if df.empty or names not in df.columns or values not in df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Data not available: Missing {names} or {values} column",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    fig = px.pie(df, names=names, values=values, title=title, hole=0.4)
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig

def plot_multi_line(df: pd.DataFrame, x: str, y: str, color: str, title: str):
    """
    Creates a multi-line chart for comparing segments.
    """
    if df.empty or x not in df.columns or y not in df.columns or color not in df.columns:
        fig = go.Figure()
        missing = [col for col in [x, y, color] if col not in df.columns]
        fig.add_annotation(
            text=f"Data not available: Missing {', '.join(missing)} column(s)",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    fig = px.line(df, x=x, y=y, color=color, title=title, markers=True)
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis_showgrid=False,
        yaxis_showgrid=True,
        yaxis_gridcolor="rgba(255,255,255,0.1)",
        hovermode="x unified"
    )
    return fig

def plot_bar_chart(df: pd.DataFrame, x: str, y: str, title: str, color: str = None):
    """
    Creates a bar chart.
    """
    if df.empty or x not in df.columns or y not in df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Data not available: Missing {x} or {y} column",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    fig = px.bar(df, x=x, y=y, title=title, color=color)
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis_showgrid=False,
        yaxis_showgrid=True,
        yaxis_gridcolor="rgba(255,255,255,0.1)",
    )
    if not color:
        fig.update_traces(marker_color=COLORS["primary"])
    return fig
