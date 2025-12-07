import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dash import Dash, dcc, html, Input, Output
import pickle

# -------------------------------------------------------------------
# 1. CONFIG: Provide your centers DataFrame and feature-name mapping
# -------------------------------------------------------------------

# EXAMPLE ONLY: replace this with loading your actual centers DataFrame
# centers = pd.read_pickle("cluster_centers.pkl")
# or however you're storing it

# Dummy example centers (4 clusters, 5 features)
# REMOVE/REPLACE THIS BLOCK IN YOUR REAL CODE
np.random.seed(0)
centers = pd.DataFrame(
    np.random.rand(4, 5),
    index=[0, 1, 2, 3],
    columns=[
        "num__age",
        "num__income",
        "cat__gender_F",
        "cat__gender_M",
        "behav__num_products",
    ],
)

with open("rejected_centers.pkl", "rb") as f:
    centers = pickle.load(f)

# Mapping from raw feature names (from ColumnTransformer) to readable labels.
# 👉 EDIT THIS DICT FOR YOUR OWN FEATURES.
FEATURE_NAME_MAP = {
    "num_trf__AMT_INCOME_TOTAL": "Total Income",
    "num_trf__AMT_CREDIT_x": "Credit Amount",
    "num_trf__EXT_SOURCE_2": "External Source 2",
    "num_trf__DAYS_BIRTH": "Days Since Birth",
    "num_trf__DAYS_EMPLOYED": "Days Employed",
    "num_trf__CNT_FAM_MEMBERS": "Family Members Count",
    "num_trf__AMT_REQ_CREDIT_BUREAU_HOUR": "Credit Bureau Requests (Hour)",
    "num_trf__APARTMENTS_AVG": "Average Apartments"
    # "raw_name": "Readable Name",
}

TRACE_NAME_PREFIX = "Cluster"

# -------------------------------------------------------------------
# 2. Helper: Make radar figure (no fig.show; just return Figure)
# -------------------------------------------------------------------

def make_radar_figure(
    centers: pd.DataFrame,
    selected_features,
    selected_clusters,
    normalize: bool = True,
    feature_name_map=None,
    trace_name_prefix: str = "Cluster",
    chart_title: str = "Cluster Profiles (Radar)",
):
    """
    Create a radar plotly figure for the selected clusters & features.
    """

    if feature_name_map is None:
        feature_name_map = {}

    if not isinstance(centers, pd.DataFrame):
        raise ValueError("centers must be a pandas DataFrame with cluster centers as rows.")

    # If nothing selected, return an empty figure
    if not selected_features or not selected_clusters:
        return go.Figure()

    # Ensure we only use valid clusters/features
    selected_clusters = [cid for cid in selected_clusters if cid in centers.index]
    selected_features = [f for f in selected_features if f in centers.columns]

    if len(selected_clusters) == 0 or len(selected_features) == 0:
        return go.Figure()

    sub = centers.loc[selected_clusters, selected_features]

    # Normalize per feature
    if normalize:
        vmin = sub.min(axis=0)
        vmax = sub.max(axis=0)
        sub = (sub - vmin) / (vmax - vmin + 1e-8)

    # Friendly labels for axes
    theta_raw = selected_features
    theta_display = [feature_name_map.get(f, f) for f in theta_raw]

    fig = go.Figure()

    for cid in selected_clusters:
        vals = sub.loc[cid].values.astype(float)

        fig.add_trace(
            go.Scatterpolar(
                r=vals,
                theta=theta_display,
                fill="toself",
                name=f"{trace_name_prefix} {cid}",
                hovertemplate=(
                    "<b>%{theta}</b><br>"
                    "value: %{r:.2f}"
                    f"<extra>{trace_name_prefix} {cid}</extra>"
                ),
            )
        )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1] if normalize else None,
                tickfont=dict(size=9),
            ),
            angularaxis=dict(
                tickfont=dict(size=10),
            ),
        ),
        showlegend=True,
        title=chart_title,
        legend=dict(orientation="h", y=-0.2),
        margin=dict(l=40, r=40, t=80, b=40),
    )

    return fig


# -------------------------------------------------------------------
# 3. Extra chart: Bar chart for one selected cluster
# -------------------------------------------------------------------

def make_cluster_bar_chart(
    centers: pd.DataFrame,
    selected_features,
    selected_cluster,
    normalize: bool = True,
    feature_name_map=None,
    chart_title: str = "Feature Values for Selected Cluster",
):
    if feature_name_map is None:
        feature_name_map = {}

    if (selected_cluster is None) or (selected_cluster not in centers.index):
        return go.Figure()

    if not selected_features:
        return go.Figure()

    selected_features = [f for f in selected_features if f in centers.columns]
    if len(selected_features) == 0:
        return go.Figure()

    sub = centers.loc[:, selected_features]

    if normalize:
        vmin = sub.min(axis=0)
        vmax = sub.max(axis=0)
        sub = (sub - vmin) / (vmax - vmin + 1e-8)

    vals = sub.loc[selected_cluster].values.astype(float)
    labels = [feature_name_map.get(f, f) for f in selected_features]

    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=vals,
                text=[f"{v:.2f}" for v in vals],
                textposition="auto",
            )
        ]
    )

    fig.update_layout(
        title=f"{chart_title} - {TRACE_NAME_PREFIX} {selected_cluster}",
        xaxis_title="Feature",
        yaxis_title="Value (normalized)" if normalize else "Value",
        margin=dict(l=40, r=40, t=60, b=100),
    )

    return fig


# -------------------------------------------------------------------
# 4. Dash App Layout
# -------------------------------------------------------------------

app = Dash(__name__)
app.title = "Cluster Radar Dashboard"

available_clusters = centers.index.tolist()
available_features = centers.columns.tolist()

app.layout = html.Div(
    [
        html.H2("Cluster Profiling Dashboard", style={"textAlign": "center"}),

        html.Div(
            [
                # Cluster selection
                html.Div(
                    [
                        html.Label("Select Cluster(s):"),
                        dcc.Dropdown(
                            id="cluster-dropdown",
                            options=[
                                {"label": f"{TRACE_NAME_PREFIX} {cid}", "value": cid}
                                for cid in available_clusters
                            ],
                            value=available_clusters,  # default: all clusters
                            multi=True,
                            clearable=False,
                        ),
                    ],
                    style={"width": "30%", "display": "inline-block", "padding": "0 10px"},
                ),

                # Feature selection
                html.Div(
                    [
                        html.Label("Select Features:"),
                        dcc.Dropdown(
                            id="feature-dropdown",
                            options=[
                                {
                                    "label": FEATURE_NAME_MAP.get(col, col),
                                    "value": col,
                                }
                                for col in available_features
                            ],
                            value=available_features,  # default: all features
                            multi=True,
                        ),
                    ],
                    style={"width": "50%", "display": "inline-block", "padding": "0 10px"},
                ),

                # Normalize checkbox
                html.Div(
                    [
                        html.Label("Options:"),
                        dcc.Checklist(
                            id="normalize-checklist",
                            options=[{"label": "Normalize to [0, 1]", "value": "normalize"}],
                            value=["normalize"],  # default: normalize
                        ),
                    ],
                    style={"width": "20%", "display": "inline-block", "padding": "0 10px"},
                ),
            ],
            style={"padding": "10px 0"},
        ),

        html.Div(
            [
                dcc.Graph(id="radar-graph", style={"height": "550px"}),
            ]
        ),

        html.Hr(),

        html.Div(
            [
                html.Label("Pick a cluster for detailed view:"),
                dcc.Dropdown(
                    id="single-cluster-dropdown",
                    options=[
                        {"label": f"{TRACE_NAME_PREFIX} {cid}", "value": cid}
                        for cid in available_clusters
                    ],
                    value=available_clusters[0] if available_clusters else None,
                    clearable=False,
                    style={"width": "30%"},
                ),
                dcc.Graph(id="cluster-bar-chart", style={"height": "450px"}),
            ],
            style={"padding": "10px 0"},
        ),
    ],
    style={"maxWidth": "1200px", "margin": "0 auto"},
)

# -------------------------------------------------------------------
# 5. Callbacks
# -------------------------------------------------------------------

@app.callback(
    Output("radar-graph", "figure"),
    [
        Input("cluster-dropdown", "value"),
        Input("feature-dropdown", "value"),
        Input("normalize-checklist", "value"),
    ],
)
def update_radar(selected_clusters, selected_features, normalize_values):
    # Dash sends single value if not multi; force list
    if isinstance(selected_clusters, (int, str)):
        selected_clusters = [selected_clusters]
    normalize = "normalize" in (normalize_values or [])

    fig = make_radar_figure(
        centers=centers,
        selected_features=selected_features,
        selected_clusters=selected_clusters,
        normalize=normalize,
        feature_name_map=FEATURE_NAME_MAP,
        trace_name_prefix=TRACE_NAME_PREFIX,
        chart_title="Cluster Profiles (Radar)",
    )
    return fig


@app.callback(
    Output("cluster-bar-chart", "figure"),
    [
        Input("single-cluster-dropdown", "value"),
        Input("feature-dropdown", "value"),
        Input("normalize-checklist", "value"),
    ],
)
def update_bar(selected_cluster, selected_features, normalize_values):
    normalize = "normalize" in (normalize_values or [])
    fig = make_cluster_bar_chart(
        centers=centers,
        selected_features=selected_features,
        selected_cluster=selected_cluster,
        normalize=normalize,
        feature_name_map=FEATURE_NAME_MAP,
        chart_title="Feature Values for Selected Cluster",
    )
    return fig


# -------------------------------------------------------------------
# 6. Run app
# -------------------------------------------------------------------

if __name__ == "__main__":
    # In your real code, make sure `centers` and `FEATURE_NAME_MAP` are defined before this.
    app.run(debug=True)
