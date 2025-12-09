import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pickle

import base64
from datetime import datetime

from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

# =============================================================================
# 1. LOAD CLUSTER CENTERS
# =============================================================================

np.random.seed(0)

accepted_centers = pickle.load(open("accepted_centers.pkl", "rb"))
rejected_centers = pickle.load(open("rejected_centers.pkl", "rb"))

# =============================================================================
# 2. FEATURE CONFIG (COMMON TO BOTH)
# =============================================================================

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

RADAR_FEATURES_ACCEPTED = [
    f for f in FEATURE_NAME_MAP.keys() if f in accepted_centers.columns
]
RADAR_FEATURES_REJECTED = [
    f for f in FEATURE_NAME_MAP.keys() if f in rejected_centers.columns
]

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

# =============================================================================
# 3. SHARED HELPER FUNCTIONS FOR CLUSTER DASHBOARD
# =============================================================================

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

    if not selected_clusters:
        return go.Figure()

    selected_clusters = [cid for cid in selected_clusters if cid in centers.index]
    radar_features = [f for f in radar_features if f in centers.columns]

    if len(selected_clusters) == 0 or len(radar_features) == 0:
        return go.Figure()

    sub = centers.loc[selected_clusters, radar_features]

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

    return go.Figure()


def build_tab_layout(
    tab_id_prefix: str,
    title_text: str,
    centers: pd.DataFrame,
):
    available_clusters = centers.index.tolist()

    if tab_id_prefix == "accepted":
        radar_features = RADAR_FEATURES_ACCEPTED
    else:
        radar_features = RADAR_FEATURES_REJECTED

    return html.Div(
        [
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
            html.Div(
                [
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

# =============================================================================
# 4. STORYBOARD HELPERS (YOUR STORY SCRIPT)
# =============================================================================

def hero_section():
    return dbc.Container(
        [
            html.H1(
                "From Rejected Applications to New Profit Pools",
                className="display-5 fw-bold mb-2"
            ),
            html.P(
                "A storytelling dashboard for the Bank Marketing Team – using "
                "cluster profiles of rejected applicants to design new loan products.",
                className="lead text-muted"
            ),
            dbc.Badge(
                "Unsupervised Learning · Customer Profiling · Product Strategy",
                color="primary",
                pill=True,
                className="mt-2"
            ),
        ],
        className="py-4",
        fluid=True,
    )


def story_timeline():
    steps = [
        {
            "title": "Context",
            "body": (
                "Risk has already reduced NPAs using credit-scoring models. "
                "Now Marketing looks at the rejected pool and asks: "
                "‘Is there hidden opportunity here?’"
            ),
        },
        {
            "title": "Idea",
            "body": (
                "Instead of treating all rejected customers as the same, "
                "we cluster them using unsupervised learning to reveal "
                "distinct behavioural profiles."
            ),
        },
        {
            "title": "Insight",
            "body": (
                "Some profiles are structurally risky and should stay rejected. "
                "Others, like the ‘Financially Stretched Households’, have "
                "high loan appetite but are manageable with controlled exposure."
            ),
        },
        {
            "title": "Action",
            "body": (
                "For each profile, we design targeted loan propositions, "
                "approval policies, and communication themes – converting "
                "a flat ‘Reject’ list into a pipeline of future profit pools."
            ),
        },
    ]

    items = []
    for i, step in enumerate(steps, start=1):
        items.append(
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            [
                                html.Div(
                                    str(i).zfill(2),
                                    className="rounded-circle border border-primary "
                                              "px-2 py-1 small fw-bold text-primary"
                                ),
                            ],
                            className="d-flex justify-content-center"
                        ),
                        width=1,
                        className="d-flex align-items-start"
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H5(step["title"], className="card-title mb-1"),
                                    html.P(step["body"], className="card-text small text-muted"),
                                ]
                            ),
                            className="mb-3 shadow-sm border-0"
                        ),
                        width=11,
                    ),
                ],
                className="g-2",
            )
        )

    return html.Div(items, className="mt-3")


def persona_card():
    return dbc.Card(
        [
            dbc.CardHeader("Example Persona – Profile 1", className="fw-bold"),
            dbc.CardBody(
                [
                    html.H5("“Financially Stretched Households”", className="card-title"),
                    html.P(
                        "Cluster characteristics (illustrative): many dependents, "
                        "lower average income, higher reliance on short-term credit, "
                        "and more females as primary applicants.",
                        className="small"
                    ),
                    html.H6("Marketing Opportunity", className="mt-2 mb-1"),
                    html.Ul(
                        [
                            html.Li("Volume-heavy segment with strong credit appetite."),
                            html.Li("Responds well to small-ticket, flexible products."),
                            html.Li("Can be profitable if exposure and limits are controlled."),
                        ],
                        className="small"
                    ),
                    html.H6("Strategic Product Ideas", className="mt-2 mb-1"),
                    html.Ul(
                        [
                            html.Li("Family Support Micro-Loans (₹10k–₹50k, small EMIs)."),
                            html.Li("Salary / Income Booster loans for short-term gaps."),
                            html.Li("Women-centric credit programs with lower initial limits."),
                            html.Li("Risk-controlled BNPL / small-ticket EMI cards."),
                        ],
                        className="small"
                    ),
                ]
            ),
        ],
        className="h-100 shadow-sm border-0"
    )


def recommendations_card():
    return dbc.Card(
        [
            dbc.CardHeader("Profile-Wise Product Planning", className="fw-bold"),
            dbc.CardBody(
                [
                    html.P(
                        "For each cluster, we link data patterns to product actions.",
                        className="small mb-2"
                    ),
                    html.Ul(
                        [
                            html.Li("Size & potential of the segment."),
                            html.Li("Risk posture – who remains ‘hard reject’."),
                            html.Li("New loan / credit products to pilot."),
                            html.Li("Cross-sell or graduation paths to safer products."),
                        ],
                        className="small"
                    ),
                    html.P(
                        "The dashboard becomes a living ‘playbook’ for Marketing, "
                        "rather than a one-time PPT.",
                        className="small mt-2 text-muted"
                    ),
                ]
            )
        ],
        className="h-100 shadow-sm border-0"
    )


def upload_section():
    return dbc.Card(
        [
            dbc.CardHeader(
                [
                    html.Span("Dashboard Snapshot", className="fw-bold"),
                    html.Span(
                        " – Drop a screenshot from your analytics app.",
                        className="text-muted small"
                    ),
                ]
            ),
            dbc.CardBody(
                [
                    dcc.Upload(
                        id="upload-image",
                        children=html.Div(
                            [
                                html.I(className="bi bi-cloud-arrow-up me-2"),
                                "Drag & drop or ",
                                html.Span("click to upload", className="text-primary fw-semibold"),
                                html.Br(),
                                html.Span(
                                    "Use a PNG/JPG screenshot of your cluster visualization.",
                                    className="text-muted small"
                                ),
                            ]
                        ),
                        style={
                            "width": "100%",
                            "borderStyle": "dashed",
                            "borderWidth": "1px",
                            "borderRadius": "10px",
                            "textAlign": "center",
                            "padding": "25px",
                            "marginBottom": "15px",
                        },
                        multiple=False,
                    ),
                    html.Div(
                        id="output-image",
                        className="mt-2",
                    ),
                    html.Div(
                        id="upload-meta",
                        className="mt-1 text-muted small fst-italic",
                    )
                ]
            ),
        ],
        className="shadow-sm border-0"
    )

# =============================================================================
# 5. APP INITIALISATION
# =============================================================================

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True,
    title="Cluster Storyboard & Profiling Studio"
)

server = app.server

# =============================================================================
# 6. LAYOUT – MAIN TABS: STORYBOARD + CLUSTER DASHBOARD
# =============================================================================

def storyboard_tab():
    return dbc.Container(
        [
            hero_section(),
            html.Hr(),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H4("The Story at a Glance", className="mb-2"),
                            story_timeline(),
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            upload_section(),
                        ],
                        md=6,
                    ),
                ],
                className="mt-3 g-4",
            ),
            html.Hr(className="my-4"),
            dbc.Row(
                [
                    dbc.Col(persona_card(), md=6, className="mb-3"),
                    dbc.Col(recommendations_card(), md=6, className="mb-3"),
                ],
                className="g-4",
            ),
            html.Footer(
                [
                    html.Small(
                        [
                            "Cluster Storyboard · Built for internal demo | ",
                            html.Span(id="footer-time"),
                        ],
                        className="text-muted",
                    )
                ],
                className="mt-4 mb-2",
            ),
        ],
        fluid=True,
        className="pb-4",
    )


def cluster_dashboard_tab():
    return html.Div(
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


app.layout = dbc.Container(
    [
        html.Br(),
        dcc.Tabs(
            id="main-tabs",
            value="story",
            children=[
                dcc.Tab(label="Storyboard", value="story", children=storyboard_tab()),
                dcc.Tab(label="Cluster Dashboard", value="dashboard", children=cluster_dashboard_tab()),
            ],
        ),
    ],
    fluid=True,
)

# =============================================================================
# 7. CALLBACKS – STORYBOARD
# =============================================================================

def parse_contents(contents, filename):
    if contents is None:
        return None

    return html.Div(
        [
            html.Img(
                src=contents,
                style={
                    "width": "100%",
                    "marginTop": "5px",
                    "borderRadius": "8px",
                    "boxShadow": "0 0.25rem 0.75rem rgba(0,0,0,0.15)",
                },
            ),
            html.Div(
                filename,
                className="text-center small text-muted mt-2",
            ),
        ]
    )


@app.callback(
    Output("output-image", "children"),
    Output("upload-meta", "children"),
    Input("upload-image", "contents"),
    State("upload-image", "filename"),
)
def update_output(contents, filename):
    if contents is None:
        return (
            html.Div(
                "No screenshot uploaded yet.",
                className="text-muted small text-center",
            ),
            "",
        )

    img_block = parse_contents(contents, filename)
    meta = f"Screenshot loaded: {filename} · Last updated at {datetime.now().strftime('%H:%M:%S')}"
    return img_block, meta


@app.callback(
    Output("footer-time", "children"),
    Input("output-image", "children")
)
def update_footer(_):
    return f"Last refreshed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

# =============================================================================
# 8. CALLBACKS – CLUSTER DASHBOARD
# =============================================================================

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

# =============================================================================
# 9. RUN
# =============================================================================

if __name__ == "__main__":
    app.run(debug=True, port=8057)
