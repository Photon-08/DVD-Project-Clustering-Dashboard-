import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pickle

from dash import Dash, dcc, html, Input, Output

# -------------------------------------------------------------------
# 1. LOAD CLUSTER CENTERS
# -------------------------------------------------------------------

# Make sure these files exist in the same folder:
#   accepted_centers.pkl
#   rejected_centers.pkl
np.random.seed(0)

accepted_centers = pickle.load(open("accepted_centers.pkl", "rb"))
rejected_centers = pickle.load(open("rejected_centers.pkl", "rb"))

# -------------------------------------------------------------------
# 2. FEATURE CONFIG (COMMON TO BOTH)
# -------------------------------------------------------------------

FEATURE_NAME_MAP = {
    "num_trf__AMT_INCOME_TOTAL": "Total Income",
    "num_trf__AMT_CREDIT_x": "Credit Amount",

    "num_trf__DAYS_BIRTH": "Age (Days)",
    "num_trf__DAYS_EMPLOYED": "Days Employed",

    # Family Features
    "num_trf__CNT_FAM_MEMBERS": "Family Members Count",
    "num_trf__CNT_CHILDREN": "Children Count",

    # Demographics
    "cat_trf__CODE_GENDER_M": "Male",
    "cat_trf__CODE_GENDER_F": "Female",
    "num_trf__APARTMENTS_AVG": "Average Apartment Size",

    # Credit Behavior
    "num_trf__OBS_30_CNT_SOCIAL_CIRCLE": "Peers with 30+ Day Payment Delay",
    "num_trf__OBS_60_CNT_SOCIAL_CIRCLE": "Peers with 60+ Day Payment Delay",
    "num_trf__AMT_REQ_CREDIT_BUREAU_MON": "Credit Bureau Requests (Month)",
    "num_trf__EXT_SOURCE_2": "External Source 2",
}

# Radar features per dataset
RADAR_FEATURES_ACCEPTED = [
    f for f in FEATURE_NAME_MAP.keys() if f in accepted_centers.columns
]
RADAR_FEATURES_REJECTED = [
    f for f in FEATURE_NAME_MAP.keys() if f in rejected_centers.columns
]

# Additional chart configuration
ADDITIONAL_FEATURES = {
    "num_trf__AMT_GOODS_PRICE_x": {
        "label": "Price of Financed Goods",
        "chart": "bar",
    },
    "num_trf__AMT_DOWN_PAYMENT": {
        "label": "Down Payment",
        "chart": "bar",
    },
    "ORGANIZATION_TYPE_GROUP": {
        "label": "Organization Type",
        "chart": "grouped_bar",
        "features": [
            "cat_trf__ORGANIZATION_TYPE_Agriculture",
            "cat_trf__ORGANIZATION_TYPE_Government",
            "cat_trf__ORGANIZATION_TYPE_Self-employed",
            "cat_trf__ORGANIZATION_TYPE_Military",
        ],
        "feature_labels": ["Agri", "Government", "Self-employed", "Military"],
    },
}

TRACE_NAME_PREFIX = "Profile"

# -------------------------------------------------------------------
# 3. SHARED HELPER FUNCTIONS
# -------------------------------------------------------------------

def make_radar_figure(
    centers: pd.DataFrame,
    selected_clusters,
    normalize: bool = True,
    radar_features=None,
    feature_name_map=None,
    trace_name_prefix: str = "Profile",
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


def make_grouped_bar_chart(
    centers: pd.DataFrame,
    features: list,
    feature_labels: list,
    selected_clusters,
    normalize: bool = True,
    label: str = None,
):
    """
    Creates a grouped bar chart for a set of features.
    """
    if not selected_clusters:
        return go.Figure()

    selected_clusters = [cid for cid in selected_clusters if cid in centers.index]
    if len(selected_clusters) == 0:
        return go.Figure()

    valid_features = [f for f in features if f in centers.columns]
    if not valid_features:
        return go.Figure()

    sub_df = centers.loc[selected_clusters, valid_features]

    if normalize:
        vmin = centers[valid_features].min()
        vmax = centers[valid_features].max()
        sub_df = (sub_df - vmin) / (vmax - vmin + 1e-8)

    fig = go.Figure()
    cluster_labels = [f"{TRACE_NAME_PREFIX} {cid}" for cid in selected_clusters]

    for i, feature in enumerate(valid_features):
        feature_label = feature_labels[i] if i < len(feature_labels) else feature
        fig.add_trace(
            go.Bar(
                x=cluster_labels,
                y=sub_df[feature],
                name=feature_label,
                text=[f"{v:.2f}" for v in sub_df[feature]],
                textposition="auto",
            )
        )

    fig.update_layout(
        barmode="group",
        title=f"{label} by Cluster",
        xaxis_title="Cluster",
        yaxis_title="Value (normalized)" if normalize else "Value",
        margin=dict(l=40, r=40, t=60, b=40),
        legend_title_text="Organization Type",
    )
    return fig


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
# 4. LAYOUT BUILDERS FOR EACH TAB
# -------------------------------------------------------------------

def build_tab_layout(
    tab_id_prefix: str,
    title_text: str,
    centers: pd.DataFrame,
):
    """
    tab_id_prefix: string prefix to keep component IDs unique
    title_text: header title for this dashboard
    centers: corresponding centers DataFrame
    """
    available_clusters = centers.index.tolist()

    # Figure out radar features based on which dataset this is
    if tab_id_prefix == "accepted":
        radar_features = RADAR_FEATURES_ACCEPTED
    else:
        radar_features = RADAR_FEATURES_REJECTED

    return html.Div(
        [
            # Sub-title inside tab
            html.Div(
                [
                    html.H3(
                        title_text,
                        style={
                            "textAlign": "center",
                            "marginBottom": "10px",
                            "fontWeight": "600",
                        },
                    ),
                    html.P(
                        "Explore segment-wise behaviors and compare profiles across clusters.",
                        style={
                            "textAlign": "center",
                            "marginTop": "0px",
                            "color": "#555",
                            "fontSize": "13px",
                        },
                    ),
                ],
                style={"marginBottom": "10px"},
            ),

            # ====== CONTROL PANEL ======
            html.Div(
                [
                    html.Div(
                        [
                            html.Label(
                                "Select Cluster(s):",
                                style={"fontWeight": "bold", "fontSize": "13px"},
                            ),
                            dcc.Dropdown(
                                id=f"cluster-dropdown-{tab_id_prefix}",
                                options=[
                                    {
                                        "label": f"{TRACE_NAME_PREFIX} {cid}",
                                        "value": cid,
                                    }
                                    for cid in available_clusters
                                ],
                                value=available_clusters[:3] if available_clusters else [],
                                multi=True,
                                clearable=False,
                            ),
                        ],
                        style={
                            "width": "40%",
                            "display": "inline-block",
                            "padding": "10px",
                        },
                    ),
                    html.Div(
                        [
                            html.Label(
                                "Options:",
                                style={"fontWeight": "bold", "fontSize": "13px"},
                            ),
                            dcc.Checklist(
                                id=f"normalize-checklist-{tab_id_prefix}",
                                options=[
                                    {
                                        "label": " Normalize to [0, 1]",
                                        "value": "normalize",
                                    }
                                ],
                                value=["normalize"],
                                style={"fontSize": "12px"},
                            ),
                        ],
                        style={
                            "width": "30%",
                            "display": "inline-block",
                            "padding": "10px",
                        },
                    ),
                ],
                style={
                    "display": "flex",
                    "justifyContent": "center",
                    "marginBottom": "15px",
                    "backgroundColor": "#F7F7F7",
                    "padding": "10px",
                    "borderRadius": "8px",
                    "border": "1px solid #DDD",
                },
            ),

            # ====== DASHBOARD GRID ======
            html.Div(
                [
                    # LEFT PANEL → Radar chart
                    html.Div(
                        [
                            html.Div(
                                dcc.Graph(
                                    id=f"radar-graph-{tab_id_prefix}",
                                    style={"height": "550px"},
                                ),
                                style={
                                    "backgroundColor": "white",
                                    "padding": "15px",
                                    "borderRadius": "10px",
                                    "boxShadow": "0 2px 6px rgba(0,0,0,0.12)",
                                },
                            )
                        ],
                        style={
                            "width": "48%",
                            "display": "inline-block",
                            "verticalAlign": "top",
                        },
                    ),
                    # RIGHT PANEL → Additional charts grid
                    html.Div(
                        id=f"additional-charts-{tab_id_prefix}",
                        style={
                            "width": "48%",
                            "display": "inline-block",
                            "verticalAlign": "top",
                            "paddingLeft": "10px",
                        },
                    ),
                ],
                style={"display": "flex", "justifyContent": "space-between"},
            ),
        ]
    )


# -------------------------------------------------------------------
# 5. DASH APP LAYOUT (UNIFIED)
# -------------------------------------------------------------------

app = Dash(__name__)
app.title = "Loan Application Cluster Profiling"

app.layout = html.Div(
    [
        html.H1(
            "Loan Application Cluster Profiling Studio",
            style={
                "textAlign": "center",
                "marginBottom": "5px",
                "marginTop": "15px",
            },
        ),
        html.P(
            "Interactive cluster comparison for accepted vs rejected applications.",
            style={
                "textAlign": "center",
                "marginBottom": "25px",
                "color": "#666",
            },
        ),
        dcc.Tabs(
            id="dataset-tabs",
            value="accepted",
            children=[
                dcc.Tab(
                    label="Accepted Applications",
                    value="accepted",
                    children=build_tab_layout(
                        tab_id_prefix="accepted",
                        title_text="Cluster Profiling - Accepted Applications",
                        centers=accepted_centers,
                    ),
                ),
                dcc.Tab(
                    label="Rejected Applications",
                    value="rejected",
                    children=build_tab_layout(
                        tab_id_prefix="rejected",
                        title_text="Cluster Profiling - Rejected Applications",
                        centers=rejected_centers,
                    ),
                ),
            ],
            style={"fontSize": "14px"},
        ),
        html.Div(
            "Tip: Use the same cluster IDs in both tabs to mentally compare accepted vs rejected profiles.",
            style={
                "marginTop": "15px",
                "fontSize": "11px",
                "color": "#777",
                "textAlign": "center",
            },
        ),
    ],
    style={"maxWidth": "1400px", "margin": "0 auto", "fontFamily": "Arial, sans-serif"},
)


# -------------------------------------------------------------------
# 6. CALLBACKS (SEPARATE FOR ACCEPTED & REJECTED)
# -------------------------------------------------------------------

@app.callback(
    Output("radar-graph-accepted", "figure"),
    [
        Input("cluster-dropdown-accepted", "value"),
        Input("normalize-checklist-accepted", "value"),
    ],
)
def update_radar_accepted(selected_clusters, normalize_values):
    if isinstance(selected_clusters, (int, str)):
        selected_clusters = [selected_clusters]
    normalize = "normalize" in (normalize_values or [])

    fig = make_radar_figure(
        centers=accepted_centers,
        selected_clusters=selected_clusters,
        normalize=normalize,
        radar_features=RADAR_FEATURES_ACCEPTED,
        feature_name_map=FEATURE_NAME_MAP,
        trace_name_prefix=TRACE_NAME_PREFIX,
        chart_title="Cluster Profiles (Radar) - Accepted",
    )
    return fig


@app.callback(
    Output("additional-charts-accepted", "children"),
    [
        Input("cluster-dropdown-accepted", "value"),
        Input("normalize-checklist-accepted", "value"),
    ],
)
def update_additional_charts_accepted(selected_clusters, normalize_values):
    if isinstance(selected_clusters, (int, str)):
        selected_clusters = [selected_clusters]
    normalize = "normalize" in (normalize_values or [])

    children = []
    for feature_name, cfg in ADDITIONAL_FEATURES.items():
        label = cfg.get("label", feature_name)
        chart_type = cfg.get("chart", "bar")

        if chart_type == "grouped_bar":
            features = cfg.get("features", [])
            feature_labels = cfg.get("feature_labels", [])
            if not all(f in accepted_centers.columns for f in features):
                continue
            fig = make_grouped_bar_chart(
                centers=accepted_centers,
                features=features,
                feature_labels=feature_labels,
                selected_clusters=selected_clusters,
                normalize=normalize,
                label=label,
            )
        else:
            if feature_name not in accepted_centers.columns:
                continue
            fig = make_additional_chart(
                centers=accepted_centers,
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
                        id=f"graph-accepted-{feature_name}",
                        figure=fig,
                        style={"height": "400px"},
                    )
                ],
                style={"marginBottom": "30px"},
            )
        )

    return children


@app.callback(
    Output("radar-graph-rejected", "figure"),
    [
        Input("cluster-dropdown-rejected", "value"),
        Input("normalize-checklist-rejected", "value"),
    ],
)
def update_radar_rejected(selected_clusters, normalize_values):
    if isinstance(selected_clusters, (int, str)):
        selected_clusters = [selected_clusters]
    normalize = "normalize" in (normalize_values or [])

    fig = make_radar_figure(
        centers=rejected_centers,
        selected_clusters=selected_clusters,
        normalize=normalize,
        radar_features=RADAR_FEATURES_REJECTED,
        feature_name_map=FEATURE_NAME_MAP,
        trace_name_prefix=TRACE_NAME_PREFIX,
        chart_title="Cluster Profiles (Radar) - Rejected",
    )
    return fig


@app.callback(
    Output("additional-charts-rejected", "children"),
    [
        Input("cluster-dropdown-rejected", "value"),
        Input("normalize-checklist-rejected", "value"),
    ],
)
def update_additional_charts_rejected(selected_clusters, normalize_values):
    if isinstance(selected_clusters, (int, str)):
        selected_clusters = [selected_clusters]
    normalize = "normalize" in (normalize_values or [])

    children = []
    for feature_name, cfg in ADDITIONAL_FEATURES.items():
        label = cfg.get("label", feature_name)
        chart_type = cfg.get("chart", "bar")

        if chart_type == "grouped_bar":
            features = cfg.get("features", [])
            feature_labels = cfg.get("feature_labels", [])
            if not all(f in rejected_centers.columns for f in features):
                continue
            fig = make_grouped_bar_chart(
                centers=rejected_centers,
                features=features,
                feature_labels=feature_labels,
                selected_clusters=selected_clusters,
                normalize=normalize,
                label=label,
            )
        else:
            if feature_name not in rejected_centers.columns:
                continue
            fig = make_additional_chart(
                centers=rejected_centers,
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
                        id=f"graph-rejected-{feature_name}",
                        figure=fig,
                        style={"height": "400px"},
                    )
                ],
                style={"marginBottom": "30px"},
            )
        )

    return children


# -------------------------------------------------------------------
# 7. RUN APP
# -------------------------------------------------------------------

if __name__ == "__main__":
    # Single unified dashboard
    app.run(debug=True, port=8050)
