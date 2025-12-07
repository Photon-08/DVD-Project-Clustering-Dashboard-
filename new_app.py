import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dash import Dash, dcc, html, Input, Output

# -------------------------------------------------------------------
# 1. Load your centers DataFrame here
# -------------------------------------------------------------------
# TODO: Replace this dummy example with your real centers DataFrame.
# Example:
# centers = pd.read_pickle("cluster_centers.pkl")

with open("rejected_centers.pkl", "rb") as f:
    centers = pd.read_pickle(f)

# -------------------------------------------------------------------
# 2. Feature maps (EDIT THESE FOR YOUR DATA)
# -------------------------------------------------------------------

# Features to appear on the RADAR (spider) chart
FEATURE_NAME_MAP = {
    "num_trf__AMT_INCOME_TOTAL": "Total Income",
    "num_trf__AMT_CREDIT_x": "Credit Amount",
    "num_trf__EXT_SOURCE_2": "External Source 2",
    "num_trf__DAYS_BIRTH": "Days Since Birth",
    "num_trf__DAYS_EMPLOYED": "Days Employed",
    "num_trf__CNT_FAM_MEMBERS": "Family Members Count",
    "num_trf__AMT_REQ_CREDIT_BUREAU_HOUR": "Credit Bureau Requests (Hour)",
    "num_trf__APARTMENTS_AVG": "Average Apartments",
}

# Features to appear in the **detail chart** (outside spider)
# 👉 You control this subset entirely by editing this dict.
DETAIL_FEATURE_MAP = {
    "num_trf__AMT_INCOME_TOTAL": "Total Income",
    "num_trf__AMT_CREDIT_x": "Credit Amount",
    "num_trf__EXT_SOURCE_2": "External Source 2",
    "num_trf__DAYS_BIRTH": "Days Since Birth",
    "num_trf__DAYS_EMPLOYED": "Days Employed",
    "num_trf__CNT_FAM_MEMBERS": "Family Members Count",
    "num_trf__AMT_REQ_CREDIT_BUREAU_HOUR": "Credit Bureau Requests (Hour)",
    "num_trf__APARTMENTS_AVG": "Average Apartments",
}

TRACE_NAME_PREFIX = "Cluster"

# Intersect with actual columns in centers to avoid errors
RADAR_FEATURES = [f for f in FEATURE_NAME_MAP.keys() if f in centers.columns]
DETAIL_FEATURES = [f for f in DETAIL_FEATURE_MAP.keys() if f in centers.columns]

# -------------------------------------------------------------------
# 3. Helpers to build figures
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
    if feature_name_map is None:
        feature_name_map = {}

    if not isinstance(centers, pd.DataFrame):
        raise ValueError("centers must be a pandas DataFrame with cluster centers as rows.")

    if not selected_features or not selected_clusters:
        return go.Figure()

    # Keep only valid clusters/features
    selected_clusters = [cid for cid in selected_clusters if cid in centers.index]
    selected_features = [f for f in selected_features if f in centers.columns]

    if len(selected_clusters) == 0 or len(selected_features) == 0:
        return go.Figure()

    sub = centers.loc[selected_clusters, selected_features]

    if normalize:
        vmin = sub.min(axis=0)
        vmax = sub.max(axis=0)
        sub = (sub - vmin) / (vmax - vmin + 1e-8)

    theta_display = [feature_name_map.get(f, f) for f in selected_features]

    fig = go.Figure()

    for cid in selected_clusters:
        vals = sub.loc[cid].values.astype(float)

        fig.add_trace(
            go.Scatterpolar(
                r=vals,
                theta=theta_display,
                fill="toself",
                name=f"{trace_name_prefix} {cid}",
                hovertemplate="<b>%{theta}</b><br>value: %{r:.2f}"
                              f"<extra>{trace_name_prefix} {cid}</extra>",
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


def make_cluster_detail_chart(
    centers: pd.DataFrame,
    selected_cluster,
    normalize: bool = True,
    detail_features=None,
    feature_name_map=None,
    chart_title: str = "Selected Features for Cluster",
):
    if feature_name_map is None:
        feature_name_map = {}
    if detail_features is None:
        detail_features = []

    if (selected_cluster is None) or (selected_cluster not in centers.index):
        return go.Figure()

    detail_features = [f for f in detail_features if f in centers.columns]
    if len(detail_features) == 0:
        return go.Figure()

    sub = centers.loc[:, detail_features]

    if normalize:
        vmin = sub.min(axis=0)
        vmax = sub.max(axis=0)
        sub = (sub - vmin) / (vmax - vmin + 1e-8)

    vals = sub.loc[selected_cluster].values.astype(float)
    labels = [feature_name_map.get(f, f) for f in detail_features]

    # For now we use a single bar chart – all these features are numeric
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
# 4. Dash app
# -------------------------------------------------------------------

app = Dash(__name__)
app.title = "Cluster Radar Dashboard"

available_clusters = centers.index.tolist()

app.layout = html.Div(
    [
        html.H2("Cluster Profiling Dashboard", style={"textAlign": "center"}),

        html.Div(
            [
                # Multi-cluster selection for radar
                html.Div(
                    [
                        html.Label("Select Cluster(s):"),
                        dcc.Dropdown(
                            id="cluster-dropdown",
                            options=[
                                {"label": f"{TRACE_NAME_PREFIX} {cid}", "value": cid}
                                for cid in available_clusters
                            ],
                            value=available_clusters,  # default: all
                            multi=True,
                            clearable=False,
                        ),
                    ],
                    style={"width": "30%", "display": "inline-block", "padding": "0 10px"},
                ),

                # Radar features (only from FEATURE_NAME_MAP)
                html.Div(
                    [
                        html.Label("Radar Features:"),
                        dcc.Dropdown(
                            id="radar-feature-dropdown",
                            options=[
                                {
                                    "label": FEATURE_NAME_MAP.get(col, col),
                                    "value": col,
                                }
                                for col in RADAR_FEATURES
                            ],
                            value=RADAR_FEATURES,  # default: all mapped radar features
                            multi=True,
                        ),
                    ],
                    style={"width": "50%", "display": "inline-block", "padding": "0 10px"},
                ),

                # Normalize toggle
                html.Div(
                    [
                        html.Label("Options:"),
                        dcc.Checklist(
                            id="normalize-checklist",
                            options=[{"label": "Normalize to [0, 1]", "value": "normalize"}],
                            value=["normalize"],
                        ),
                    ],
                    style={"width": "20%", "display": "inline-block", "padding": "0 10px"},
                ),
            ],
            style={"padding": "10px 0"},
        ),

        dcc.Graph(id="radar-graph", style={"height": "550px"}),

        html.Hr(),

        # Detail view for a single cluster, using DETAIL_FEATURE_MAP only
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
                dcc.Graph(id="cluster-detail-chart", style={"height": "450px"}),
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
        Input("radar-feature-dropdown", "value"),
        Input("normalize-checklist", "value"),
    ],
)
def update_radar(selected_clusters, selected_features, normalize_values):
    if isinstance(selected_clusters, (int, str)):
        selected_clusters = [selected_clusters]
    normalize = "normalize" in (normalize_values or [])

    return make_radar_figure(
        centers=centers,
        selected_features=selected_features,
        selected_clusters=selected_clusters,
        normalize=normalize,
        feature_name_map=FEATURE_NAME_MAP,
        trace_name_prefix=TRACE_NAME_PREFIX,
        chart_title="Cluster Profiles (Radar)",
    )


@app.callback(
    Output("cluster-detail-chart", "figure"),
    [
        Input("single-cluster-dropdown", "value"),
        Input("normalize-checklist", "value"),
    ],
)
def update_detail(selected_cluster, normalize_values):
    normalize = "normalize" in (normalize_values or [])

    return make_cluster_detail_chart(
        centers=centers,
        selected_cluster=selected_cluster,
        normalize=normalize,
        detail_features=DETAIL_FEATURES,          # <- only these features used
        feature_name_map=DETAIL_FEATURE_MAP,
        chart_title="Selected Features for Cluster",
    )

# -------------------------------------------------------------------
# 6. Run
# -------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
