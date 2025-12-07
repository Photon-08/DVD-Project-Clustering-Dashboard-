import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pickle

from dash import Dash, dcc, html, Input, Output

# -------------------------------------------------------------------
# 1. CONFIG: your cluster centers + feature mappings
# -------------------------------------------------------------------

# TODO: Replace this with your real centers DataFrame
# Example:
# centers = pd.read_pickle("cluster_centers.pkl")
#   - rows = clusters
#   - columns = transformed feature names

# DUMMY EXAMPLE (remove in real code)
np.random.seed(0)
centers = pd.DataFrame(
    np.random.rand(10, 8),
    index=list(range(10)),
    columns=[
        "num_trf__AMT_INCOME_TOTAL",
        "num_trf__AMT_CREDIT_x",
        "num_trf__EXT_SOURCE_2",
        "num_trf__DAYS_BIRTH",
        "num_trf__DAYS_EMPLOYED",
        "num_trf__CNT_FAM_MEMBERS",
        "num_trf__AMT_REQ_CREDIT_BUREAU_HOUR",
        "num_trf__APARTMENTS_AVG",
    ],
)
centers = pickle.load(open("rejected_centers.pkl", "rb"))   
# === FEATURES USED ON THE RADAR (SPIDER) CHART ===
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

# This is the *only* set of features that will appear in the radar.
RADAR_FEATURES = [f for f in FEATURE_NAME_MAP.keys() if f in centers.columns]

# === ADDITIONAL FEATURES & CHART TYPES (OUTSIDE RADAR) ===
# You can keep the same features or include others; one chart per feature.
# chart can be: "bar" (cluster vs value). Extend if you want more types.
ADDITIONAL_FEATURES = {
    
    "num_trf__APARTMENTS_AVG": {
        "label": "Average Apartments",
        "chart": "bar",
    },

    "num_trf__LANDAREA_AVG": {
        "label": "Average Land Area",
        "chart": "bar",
    },
    "num_trf__AMT_REQ_CREDIT_BUREAU_QRT": {
        "label": "Credit Bureau Requests (Quarter) Per Quarter",
        "chart": "bar",
    },
    "num_trf__DAYS_LAST_DUE_1ST_VERSION": {
        "label": "Days Since Last Due First Version",
        "chart": "bar",
    }
}

TRACE_NAME_PREFIX = "Cluster"

# -------------------------------------------------------------------
# 2. Helper: create radar chart (no fig.show here)
# -------------------------------------------------------------------

def make_radar_figure(
    centers: pd.DataFrame,
    selected_clusters,
    normalize: bool = True,
    radar_features=None,
    feature_name_map=None,
    trace_name_prefix: str = "Cluster",
    chart_title: str = "Cluster Profiles (Radar)",
):
    if radar_features is None:
        radar_features = []
    if feature_name_map is None:
        feature_name_map = {}

    if not isinstance(centers, pd.DataFrame):
        raise ValueError("centers must be a pandas DataFrame with cluster centers as rows.")

    # If nothing selected: return empty fig
    if not selected_clusters:
        return go.Figure()

    # Keep only valid clusters/features
    selected_clusters = [cid for cid in selected_clusters if cid in centers.index]
    radar_features = [f for f in radar_features if f in centers.columns]

    if len(selected_clusters) == 0 or len(radar_features) == 0:
        return go.Figure()

    sub = centers.loc[selected_clusters, radar_features]

    # Normalize per feature
    if normalize:
        vmin = sub.min(axis=0)
        vmax = sub.max(axis=0)
        sub = (sub - vmin) / (vmax - vmin + 1e-8)

    theta_display = [feature_name_map.get(f, f) for f in radar_features]

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
# 3. Helpers: additional charts (one per feature)
# -------------------------------------------------------------------

def make_additional_chart(
    centers: pd.DataFrame,
    feature_name: str,
    selected_clusters,
    normalize: bool = True,
    label: str = None,
    chart_type: str = "bar",
):
    """
    For each feature, we create a chart comparing selected clusters.
    Currently supports chart_type="bar".
    """
    if feature_name not in centers.columns:
        return go.Figure()

    if not selected_clusters:
        return go.Figure()

    selected_clusters = [cid for cid in selected_clusters if cid in centers.index]
    if len(selected_clusters) == 0:
        return go.Figure()

    series = centers.loc[selected_clusters, feature_name]

    if normalize:
        vmin = centers[feature_name].min()
        vmax = centers[feature_name].max()
        series = (series - vmin) / (vmax - vmin + 1e-8)

    label = label or feature_name
    x_labels = [f"{TRACE_NAME_PREFIX} {cid}" for cid in selected_clusters]
    y_vals = series.values.astype(float)

    if chart_type == "bar":
        fig = go.Figure(
            data=[
                go.Bar(
                    x=x_labels,
                    y=y_vals,
                    text=[f"{v:.2f}" for v in y_vals],
                    textposition="auto",
                )
            ]
        )
        fig.update_layout(
            title=f"{label} by Cluster",
            xaxis_title="Cluster",
            yaxis_title="Value (normalized)" if normalize else "Value",
            margin=dict(l=40, r=40, t=60, b=40),
        )
        return fig

    # Fallback empty fig for unknown types
    return go.Figure()


# -------------------------------------------------------------------
# 4. Dash App Layout
# -------------------------------------------------------------------

app = Dash(__name__)
app.title = "Cluster Profiling Dashboard - Rejected Applications"

available_clusters = centers.index.tolist()

app.layout = html.Div(
    [
        html.H2(
            "Cluster Profiling Dashboard - Rejected Applications",
            style={"textAlign": "center", "marginBottom": "20px"}
        ),

        # ====== CONTROL PANEL ======
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Select Cluster(s):", style={"fontWeight": "bold"}),
                        dcc.Dropdown(
                            id="cluster-dropdown",
                            options=[
                                {"label": f"{TRACE_NAME_PREFIX} {cid}", "value": cid}
                                for cid in available_clusters
                            ],
                            value=available_clusters[:3] if available_clusters else [],
                            multi=True,
                            clearable=False,
                        ),
                    ],
                    style={"width": "40%", "display": "inline-block", "padding": "10px"},
                ),

                html.Div(
                    [
                        html.Label("Options:", style={"fontWeight": "bold"}),
                        dcc.Checklist(
                            id="normalize-checklist",
                            options=[{"label": " Normalize to [0, 1]", "value": "normalize"}],
                            value=["normalize"],
                        ),
                    ],
                    style={"width": "30%", "display": "inline-block", "padding": "10px"},
                ),
            ],
            style={
                "display": "flex",
                "justifyContent": "center",
                "marginBottom": "15px",
                "backgroundColor": "#F7F7F7",
                "padding": "10px",
                "borderRadius": "8px",
                "border": "1px solid #DDD"
            },
        ),

        # ====== DASHBOARD GRID ======
        html.Div(
            [
                # LEFT PANEL → Radar chart
                html.Div(
                    [
                        html.Div(
                            dcc.Graph(id="radar-graph", style={"height": "550px"}),
                            style={
                                "backgroundColor": "white",
                                "padding": "15px",
                                "borderRadius": "10px",
                                "boxShadow": "0 2px 6px rgba(0,0,0,0.15)"
                            },
                        )
                    ],
                    style={"width": "48%", "display": "inline-block", "verticalAlign": "top"},
                ),

                # RIGHT PANEL → Additional charts grid
                html.Div(
                    id="additional-charts",
                    style={
                        "width": "48%",
                        "display": "inline-block",
                        "verticalAlign": "top",
                        "paddingLeft": "10px"
                    },
                ),
            ],
            style={"display": "flex", "justifyContent": "space-between"},
        ),
    ],
    style={"maxWidth": "1400px", "margin": "0 auto"},
)


# -------------------------------------------------------------------
# 5. Callbacks
# -------------------------------------------------------------------

@app.callback(
    Output("radar-graph", "figure"),
    [
        Input("cluster-dropdown", "value"),
        Input("normalize-checklist", "value"),
    ],
)
def update_radar(selected_clusters, normalize_values):
    if isinstance(selected_clusters, (int, str)):
        selected_clusters = [selected_clusters]
    normalize = "normalize" in (normalize_values or [])

    fig = make_radar_figure(
        centers=centers,
        selected_clusters=selected_clusters,
        normalize=normalize,
        radar_features=RADAR_FEATURES,
        feature_name_map=FEATURE_NAME_MAP,
        trace_name_prefix=TRACE_NAME_PREFIX,
        chart_title="Cluster Profiles (Radar)",
    )
    return fig


@app.callback(
    Output("additional-charts", "children"),
    [
        Input("cluster-dropdown", "value"),
        Input("normalize-checklist", "value"),
    ],
)
def update_additional_charts(selected_clusters, normalize_values):
    if isinstance(selected_clusters, (int, str)):
        selected_clusters = [selected_clusters]
    normalize = "normalize" in (normalize_values or [])

    children = []

    # For each additional feature, create its own chart
    for feature_name, cfg in ADDITIONAL_FEATURES.items():
        if feature_name not in centers.columns:
            continue

        label = cfg.get("label", feature_name)
        chart_type = cfg.get("chart", "bar")

        fig = make_additional_chart(
            centers=centers,
            feature_name=feature_name,
            selected_clusters=selected_clusters,
            normalize=normalize,
            label=label,
            chart_type=chart_type,
        )

        children.append(
            html.Div(
                [
                    dcc.Graph(
                        id=f"graph-{feature_name}",
                        figure=fig,
                        style={"height": "400px"},
                    )
                ],
                style={"marginBottom": "30px"},
            )
        )

    return children


# -------------------------------------------------------------------
# 6. Run app
# -------------------------------------------------------------------

if __name__ == "__main__":
    #app.run_server(debug=True)
    app.run(debug=True)