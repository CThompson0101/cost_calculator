import streamlit as st
import pulp as pl
import pandas as pd

# Optional chart library for a more "product" look
try:
    import altair as alt
    _ALTAIR_OK = True
except Exception:
    _ALTAIR_OK = False

from io import BytesIO
import zipfile

# ------------------------------------------------------------
# Streamlit config MUST be first Streamlit command
# ------------------------------------------------------------
st.set_page_config(
    layout="wide",
    page_title="Portfolio Optimization Tool",
    page_icon="📊",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------
# "IOPs-style" UI polish (CSS)
# ------------------------------------------------------------
def inject_css():
    st.markdown(
        """
        <style>
          /* Global spacing */
          .block-container { padding-top: 1.1rem; padding-bottom: 2.2rem; }

          /* Hide Streamlit chrome (optional – looks more like a product) */
          #MainMenu { visibility: hidden; }
          footer { visibility: hidden; }

          /* Keep header alive so sidebar toggle can exist */
          header, header[data-testid="stHeader"]{
            visibility: visible !important;
            background: transparent !important;
            box-shadow: none !important;
          }

          button[kind="secondary"],
          button[data-testid="baseButton-secondary"],
          button[data-testid="baseButton-tertiary"]{
            color: #000 !important;
          }

          button[kind="secondary"] *,
          button[data-testid="baseButton-secondary"] *,
          button[data-testid="baseButton-tertiary"] *{
            color: #000 !important;
          }

          /* Hide the top decoration strip */
          div[data-testid="stDecoration"]{ display: none !important; }

          /*
            IMPORTANT:
            Don't do: div[data-testid="stToolbar"]{ display: none; }
            That can remove the sidebar reopen button in some Streamlit versions.
            Instead hide only the toolbar action area (Deploy / menu widgets).
          */
          div[data-testid="stToolbarActions"]{ display: none !important; }
          div[data-testid="stStatusWidget"]{ display: none !important; }

          /* Force the sidebar toggle to be visible (covers multiple Streamlit versions) */
          button[data-testid="stSidebarCollapseButton"],
          button[data-testid="stSidebarCollapsedControl"],
          div[data-testid="stSidebarCollapseButton"],
          div[data-testid="stSidebarCollapsedControl"],
          [data-testid="collapsedControl"],
          button[title*="sidebar" i],
          button[aria-label*="sidebar" i]{
            display: flex !important;
            visibility: visible !important;
            position: fixed !important;
            top: 0.75rem;
            left: 0.75rem;
            z-index: 999999;
          }

          /* Sidebar styling (dark, product-like) */
          section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0B0F19 0%, #111827 100%);
            border-right: 1px solid rgba(255,255,255,0.06);
          }
          section[data-testid="stSidebar"] * { color: #FFFFFF !important; }
          section[data-testid="stSidebar"] p { opacity: 0.9; }
          section[data-testid="stSidebar"] a { color: #86BC25 !important; }
          section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] small { opacity: 0.85; }

          /* Inputs in sidebar */
          section[data-testid="stSidebar"] input,
          section[data-testid="stSidebar"] textarea,
          section[data-testid="stSidebar"] .stSlider {
            color: #0B0F19 !important;
          }

          /* Header */
          .app-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 16px 18px;
            border-radius: 16px;
            background: linear-gradient(135deg, #0B0F19 0%, #111827 55%, #0B0F19 100%);
            border: 1px solid rgba(255,255,255,0.08);
            margin-bottom: 14px;
          }
          .app-header-left { display: flex; align-items: center; gap: 12px; }
          .brand-dot {
            width: 12px; height: 12px; border-radius: 999px;
            background: #86BC25;
            box-shadow: 0 0 0 5px rgba(134,188,37,0.15);
          }
          .app-title { font-size: 18px; font-weight: 700; color: #F9FAFB; line-height: 1.2; }
          .app-subtitle { font-size: 12px; color: rgba(249,250,251,0.78); margin-top: 2px; }
          .header-chip {
            font-size: 12px; color: rgba(249,250,251,0.82);
            padding: 6px 10px; border-radius: 999px;
            border: 1px solid rgba(255,255,255,0.10);
            background: rgba(255,255,255,0.06);
            white-space: nowrap;
          }

          /* Metric cards */
          div[data-testid="metric-container"] {
            background: #FFFFFF;
            border: 1px solid #E7EAF0;
            border-radius: 16px;
            padding: 14px 14px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.04);
          }
          div[data-testid="stMetricValue"] { font-size: 1.9rem; }
          div[data-testid="stMetricLabel"] { font-size: 0.95rem; opacity: 0.9; }

          /* Tabs */
          button[data-baseweb="tab"] {
            border-radius: 12px;
            padding: 10px 14px;
          }
          button[aria-selected="true"][data-baseweb="tab"] {
            background: rgba(134,188,37,0.14);
          }

          /* Primary buttons */
          .stButton button[kind="primary"] {
            border-radius: 14px !important;
            box-shadow: 0 1px 2px rgba(0,0,0,0.04);
          }

          /* Soft section containers using markdown wrapper */
          .section-card {
            background: #FFFFFF;
            border: 1px solid #E7EAF0;
            border-radius: 16px;
            padding: 14px 16px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.03);
            margin: 8px 0 14px 0;
          }

          /* Dataframe border rounding */
          div[data-testid="stDataFrame"] {
            border: 1px solid #E7EAF0;
            border-radius: 16px;
            overflow: hidden;
          }

          /* Subtle captions */
          .muted { color: rgba(11,15,25,0.68); font-size: 0.92rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

inject_css()

# ------------------------------------------------------------
# 6-period horizon: N to N+5
# ------------------------------------------------------------
T = [1, 2, 3, 4, 5, 6]
Lbl = {t: ("N" if t == 1 else f"N+{t-1}") for t in T}
LBL_ORDER = [Lbl[t] for t in T]
T_MAX = max(T)

# Default values (increased to allow all enablers)
default_budget = {1: 2.0, 2: 2.0, 3: 6.5, 4: 6.0, 5: 6.0, 6: 5.5}
default_fte_cap = {1: 10, 2: 10, 3: 9, 4: 8, 5: 8, 6: 7}

# Default use cases
default_use_cases = ["UC1", "UC2", "UC3", "UC4"]
default_uc_name = {
    "UC1": "Vision Quality Inspection",
    "UC2": "Operator Assist (Vision + Guided Work)",
    "UC3": "Predictive Maintenance",
    "UC4": "Yard / Inventory Vision",
}
default_benefit = {
    "UC1": {1: 1.8, 2: 2.0, 3: 2.0, 4: 2.1, 5: 2.1, 6: 2.2},
    "UC2": {1: 0.8, 2: 1.1, 3: 1.2, 4: 1.2, 5: 1.3, 6: 1.3},
    "UC3": {1: 0.5, 2: 0.8, 3: 1.0, 4: 1.1, 5: 1.1, 6: 1.2},
    "UC4": {1: 0.4, 2: 0.6, 3: 0.8, 4: 0.9, 5: 0.9, 6: 1.0},
}

# Default enablers
default_enablers = ["E1", "E2", "E3", "E4", "E5", "E6"]
default_en_name = {
    "E1": "Industrial Camera Kit",
    "E2": "NVIDIA Jetson Edge Nodes",
    "E3": "Edge Runtime + Fleet Mgmt",
    "E4": "Plant Data Hub (basic storage/ingest)",
    "E5": "MLOps (registry/monitoring)",
    "E6": "OT Integration (MES/SCADA/CMMS connectors)",
}
default_cost = {
    "E1": {1: 1.5, 2: 1.5, 3: 1.4, 4: 1.4, 5: 1.3, 6: 1.3},
    "E2": {1: 1.2, 2: 1.2, 3: 1.1, 4: 1.1, 5: 1.0, 6: 1.0},
    "E3": {1: 0.8, 2: 0.8, 3: 0.8, 4: 0.7, 5: 0.7, 6: 0.7},
    "E4": {1: 1.4, 2: 1.4, 3: 1.3, 4: 1.3, 5: 1.2, 6: 1.2},
    "E5": {1: 1.0, 2: 1.0, 3: 0.9, 4: 0.9, 5: 0.9, 6: 0.8},
    "E6": {1: 1.2, 2: 1.2, 3: 1.1, 4: 1.1, 5: 1.0, 6: 1.0},
}
default_fte = {
    "E1": {1: 4.0, 2: 4.0, 3: 4.0, 4: 4.0, 5: 4.0, 6: 4.0},
    "E2": {1: 2.5, 2: 2.5, 3: 2.5, 4: 2.5, 5: 2.5, 6: 2.5},
    "E3": {1: 2.5, 2: 2.5, 3: 2.5, 4: 2.5, 5: 2.5, 6: 2.5},
    "E4": {1: 4.0, 2: 4.0, 3: 4.0, 4: 4.0, 5: 4.0, 6: 4.0},
    "E5": {1: 2.5, 2: 2.5, 3: 2.5, 4: 2.5, 5: 2.5, 6: 2.5},
    "E6": {1: 3.0, 2: 3.0, 3: 3.0, 4: 3.0, 5: 3.0, 6: 3.0},
}

# Default requirements (matrix)
default_req = {
    "UC1": {"E1": 1, "E2": 1, "E3": 1, "E4": 0, "E5": 0, "E6": 0},
    "UC2": {"E1": 1, "E2": 1, "E3": 1, "E4": 0, "E5": 0, "E6": 0},
    "UC3": {"E1": 0, "E2": 0, "E3": 0, "E4": 1, "E5": 1, "E6": 1},
    "UC4": {"E1": 1, "E2": 1, "E3": 0, "E4": 1, "E5": 0, "E6": 0},
}

# Default dependencies
default_prereq_ready = {
    "E3": ["E2"],
    "E5": ["E4"],
    "E6": ["E4"],
}

# ------------------------------------------------------------
# Vision alignment scores (0..1) — used ONLY as a tie-breaker priority
# (does NOT change the reported Objective, which remains Benefit - Cost)
# ------------------------------------------------------------
default_vision = {
    "UC1": 0.95,
    "UC2": 0.85,
    "UC3": 0.55,
    "UC4": 0.75,
}

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def df_to_period_dict(df, id_col, value_cols):
    result = {}
    for _, row in df.iterrows():
        id_val = row[id_col]
        result[id_val] = {}
        for t, label in Lbl.items():
            if label in value_cols:
                result[id_val][t] = row[label]
    return result

def req_dict_to_df(req_dict, use_cases, enablers, uc_name_map):
    rows = []
    for uc in use_cases:
        row = {"Use Case": uc, "Name": uc_name_map.get(uc, uc)}
        for e in enablers:
            row[e] = int(req_dict.get(uc, {}).get(e, 0))
        rows.append(row)
    return pd.DataFrame(rows)

def df_to_req_dict(df, use_cases, enablers):
    req = {}
    for _, row in df.iterrows():
        uc = row["Use Case"]
        req[uc] = {}
        for e in enablers:
            req[uc][e] = int(row[e])
    return req

def dep_df_to_dict(df):
    prereq = {}
    for _, row in df.iterrows():
        child = row["Dependent"]
        parent = row["Prerequisite"]
        prereq.setdefault(child, [])
        if parent not in prereq[child]:
            prereq[child].append(parent)
    return prereq

def dict_to_dep_df(prereq_dict):
    rows = []
    for child, parents in prereq_dict.items():
        for p in parents:
            rows.append({"Dependent": child, "Prerequisite": p})
    return pd.DataFrame(rows)

# Vision helpers
def vision_dict_to_df(vision_dict, use_cases, uc_name_map):
    return pd.DataFrame([
        {"Use Case": uc, "Name": uc_name_map.get(uc, uc), "Vision Score (0-1)": float(vision_dict.get(uc, 0.0))}
        for uc in use_cases
    ])

def df_to_vision_dict(df):
    vision = {}
    for _, row in df.iterrows():
        vision[row["Use Case"]] = float(row["Vision Score (0-1)"])
    return vision

def period_label_to_t(label: str) -> int:
    return next(k for k, v in Lbl.items() if v == label)

def _blank_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = [""] * len(out)
    return out

def build_download_zip(files: dict[str, bytes]) -> bytes:
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buf.getvalue()

# ------------------------------------------------------------
# Initialize session state (and reset helper)
# ------------------------------------------------------------
def init_session_state(force_reset: bool = False):
    if force_reset:
        keys = [
            "df_cons", "df_ben", "df_cost", "df_fte", "df_req", "df_dep", "df_vision",
            "prev_enablers", "prev_use_cases_for_vision",
            "solver_result", "current_data",
        ]
        for k in keys:
            if k in st.session_state:
                del st.session_state[k]

    if "df_cons" not in st.session_state:
        st.session_state.df_cons = pd.DataFrame([
            {"Period": Lbl[t], "Budget ($M)": default_budget[t], "FTE Cap": default_fte_cap[t]}
            for t in T
        ])

    if "df_ben" not in st.session_state:
        rows = []
        for uc in default_use_cases:
            row = {"Use Case": uc, "Name": default_uc_name[uc]}
            row.update({Lbl[t]: default_benefit[uc][t] for t in T})
            rows.append(row)
        st.session_state.df_ben = pd.DataFrame(rows)

    if "df_cost" not in st.session_state:
        rows = []
        for e in default_enablers:
            row = {"Enabler": e, "Name": default_en_name[e]}
            row.update({Lbl[t]: default_cost[e][t] for t in T})
            rows.append(row)
        st.session_state.df_cost = pd.DataFrame(rows)

    if "df_fte" not in st.session_state:
        rows = []
        for e in default_enablers:
            row = {"Enabler": e, "Name": default_en_name[e]}
            row.update({Lbl[t]: default_fte[e][t] for t in T})
            rows.append(row)
        st.session_state.df_fte = pd.DataFrame(rows)

    if "df_req" not in st.session_state:
        st.session_state.df_req = req_dict_to_df(default_req, default_use_cases, default_enablers, default_uc_name)

    if "df_dep" not in st.session_state:
        st.session_state.df_dep = dict_to_dep_df(default_prereq_ready)

    if "df_vision" not in st.session_state:
        st.session_state.df_vision = vision_dict_to_df(default_vision, default_use_cases, default_uc_name)

init_session_state()

# ------------------------------------------------------------
# Extract current data
# ------------------------------------------------------------
def get_current_data():
    budget, fte_cap = {}, {}
    for _, row in st.session_state.df_cons.iterrows():
        label = row["Period"]
        t = next(k for k, v in Lbl.items() if v == label)
        budget[t] = float(row["Budget ($M)"])
        fte_cap[t] = float(row["FTE Cap"])

    use_cases = st.session_state.df_ben["Use Case"].tolist()
    uc_name_map = dict(zip(st.session_state.df_ben["Use Case"], st.session_state.df_ben["Name"]))
    enablers = st.session_state.df_cost["Enabler"].tolist()
    en_name_map = dict(zip(st.session_state.df_cost["Enabler"], st.session_state.df_cost["Name"]))

    benefit = df_to_period_dict(st.session_state.df_ben, "Use Case", [Lbl[t] for t in T])
    cost = df_to_period_dict(st.session_state.df_cost, "Enabler", [Lbl[t] for t in T])
    fte = df_to_period_dict(st.session_state.df_fte, "Enabler", [Lbl[t] for t in T])

    req = df_to_req_dict(st.session_state.df_req, use_cases, enablers)
    prereq_ready = dep_df_to_dict(st.session_state.df_dep)

    vision = df_to_vision_dict(st.session_state.df_vision)

    return {
        "budget": budget,
        "fte_cap": fte_cap,
        "benefit": benefit,
        "cost": cost,
        "fte": fte,
        "req": req,
        "prereq_ready": prereq_ready,
        "vision": vision,  # used as tie-breaker only
        "use_cases": use_cases,
        "enablers": enablers,
        "uc_name_map": uc_name_map,
        "en_name_map": en_name_map,
    }

# ------------------------------------------------------------
# Solver: primary objective = Benefit - Cost
# Vision is a SMALL tie-breaker: + vision_priority * vision_points
# ------------------------------------------------------------
def solve_model(data, roi_periods, vision_priority):
    use_cases = data["use_cases"]
    enablers = data["enablers"]
    T_local = T

    m = pl.LpProblem("Roadmap_MaxROI", pl.LpMaximize)

    startE = pl.LpVariable.dicts("startE", (enablers, T_local), cat="Binary")
    availE = pl.LpVariable.dicts("availE", (enablers, T_local), cat="Binary")
    startUC = pl.LpVariable.dicts("startUC", (use_cases, T_local), cat="Binary")
    activeUC = pl.LpVariable.dicts("activeUC", (use_cases, T_local), cat="Binary")

    # Start at most once
    for e in enablers:
        m += pl.lpSum(startE[e][t] for t in T_local) <= 1
    for uc in use_cases:
        m += pl.lpSum(startUC[uc][t] for t in T_local) <= 1

    # Availability/active propagation
    for e in enablers:
        for t in T_local:
            m += availE[e][t] == pl.lpSum(startE[e][k] for k in T_local if k <= t)
    for uc in use_cases:
        for t in T_local:
            m += activeUC[uc][t] == pl.lpSum(startUC[uc][k] for k in T_local if k <= t)

    # Tech dependencies
    for child, prereqs in data["prereq_ready"].items():
        if child not in enablers:
            continue
        for t in T_local:
            for p in prereqs:
                if p not in enablers:
                    continue
                m += startE[child][t] <= availE[p][t]

    # Use case requirements
    for uc in use_cases:
        required = [e for e in enablers if data["req"].get(uc, {}).get(e, 0) == 1]
        for t in T_local:
            for e in required:
                m += startUC[uc][t] <= availE[e][t]

    # Budget constraints (start costs only)
    for t in T_local:
        m += pl.lpSum(data["cost"][e][t] * startE[e][t] for e in enablers) <= data["budget"][t]

    # FTE constraints (one-time when starting)
    for t in T_local:
        m += pl.lpSum(data["fte"][e][t] * startE[e][t] for e in enablers) <= data["fte_cap"][t]

    # Primary ROI objective components (ROI window)
    benefit_roi_expr = pl.lpSum(
        data["benefit"][uc][t] * activeUC[uc][t] for uc in use_cases for t in roi_periods
    )
    cost_roi_expr = pl.lpSum(
        data["cost"][e][t] * startE[e][t] for e in enablers for t in roi_periods
    )

    # Tie-breaker: prioritize higher-vision use cases being active (ROI window)
    vision_points_expr = pl.lpSum(
        float(data["vision"].get(uc, 0.0)) * activeUC[uc][t] for uc in use_cases for t in roi_periods
    )

    # IMPORTANT: Reported "Objective" remains Benefit - Cost.
    # We only add a *small* tie-breaker term to help choose among near-equal ROI solutions.
    m += (benefit_roi_expr - cost_roi_expr) + (vision_priority * vision_points_expr)

    m.solve(pl.PULP_CBC_CMD(msg=False))
    if pl.LpStatus[m.status] != "Optimal":
        return None

    def v(x):
        return float(pl.value(x) or 0.0)

    enabler_start = {e: next((t for t in T_local if v(startE[e][t]) > 0.5), None) for e in enablers}
    uc_start = {uc: next((t for t in T_local if v(startUC[uc][t]) > 0.5), None) for uc in use_cases}

    # Compute reported ROI totals (no vision in reported objective)
    roi_ben = sum(data["benefit"][uc][t] * v(activeUC[uc][t]) for uc in use_cases for t in roi_periods)
    roi_cost = sum(data["cost"][e][t] * v(startE[e][t]) for e in enablers for t in roi_periods)
    roi_obj = roi_ben - roi_cost

    summary = []
    for t in T_local:
        spend = sum(data["cost"][e][t] * v(startE[e][t]) for e in enablers)
        ben = sum(data["benefit"][uc][t] * v(activeUC[uc][t]) for uc in use_cases)
        fte_used = sum(data["fte"][e][t] * v(startE[e][t]) for e in enablers)
        summary.append({
            "Period": Lbl[t],
            "Spend ($M)": round(spend, 2),
            "FTE Used": round(fte_used, 1),
            "Benefit ($M)": round(ben, 2),
            "Net ($M)": round(ben - spend, 2),
            "In ROI Window": "✓" if t in roi_periods else ""
        })
    df_summary = pd.DataFrame(summary)

    return {
        "enabler_start_idx": enabler_start,
        "uc_start_idx": uc_start,
        "df_summary": df_summary,
        "roi_ben": roi_ben,
        "roi_cost": roi_cost,
        "roi_obj": roi_obj,  # Benefit - Cost (reported)
    }

# ------------------------------------------------------------
# What-if computation (reported Objective stays Benefit - Cost)
# ------------------------------------------------------------
def compute_whatif(data, enabler_start_idx, roi_periods):
    enablers = data["enablers"]
    use_cases = data["use_cases"]
    T_local = T

    # Build availability
    availE = {e: {t: 0 for t in T_local} for e in enablers}
    for e, t0 in enabler_start_idx.items():
        if t0 is not None:
            for t in T_local:
                if t >= t0:
                    availE[e][t] = 1

    # Dependency violations
    dep_violations = []
    for child, prereqs in data["prereq_ready"].items():
        if child not in enablers:
            continue
        child_start = enabler_start_idx.get(child)
        if child_start is None:
            continue
        for p in prereqs:
            if p not in enablers:
                continue
            p_start = enabler_start_idx.get(p)
            if p_start is None or child_start < p_start:
                dep_violations.append(
                    f"{data['en_name_map'].get(child, child)} ({child}) starts at {Lbl[child_start]}, "
                    f"but requires {data['en_name_map'].get(p, p)} ({p}) which starts at {Lbl[p_start] if p_start else 'never'}."
                )

    # Budget/FTE violations
    budget_violations, fte_violations = [], []
    for t in T_local:
        spend_t = sum(data["cost"][e][t] for e in enablers if enabler_start_idx.get(e) == t)
        if spend_t > data["budget"][t] + 1e-6:
            budget_violations.append(f"{Lbl[t]}: Spend ${spend_t:.2f}M exceeds budget ${data['budget'][t]:.2f}M")

        fte_t = sum(data["fte"][e][t] for e in enablers if enabler_start_idx.get(e) == t)
        if fte_t > data["fte_cap"][t] + 1e-6:
            fte_violations.append(f"{Lbl[t]}: FTE used {fte_t:.1f} exceeds cap {data['fte_cap'][t]:.1f}")

    # Use case start times (earliest feasible)
    uc_start_idx = {}
    for uc in use_cases:
        required = [e for e in enablers if data["req"].get(uc, {}).get(e, 0) == 1]
        start_t = None
        for t in T_local:
            if all(availE[e][t] == 1 for e in required):
                start_t = t
                break
        uc_start_idx[uc] = start_t

    activeUC = {uc: {t: 0 for t in T_local} for uc in use_cases}
    for uc, t0 in uc_start_idx.items():
        if t0 is not None:
            for t in T_local:
                if t >= t0:
                    activeUC[uc][t] = 1

    # ROI totals (reported objective Benefit - Cost)
    roi_ben = sum(data["benefit"][uc][t] * activeUC[uc][t] for uc in use_cases for t in roi_periods)

    # Cost is one-time at start, attributed to the start period
    roi_cost = 0.0
    for e in enablers:
        t0 = enabler_start_idx.get(e)
        if t0 is not None and t0 in roi_periods:
            roi_cost += data["cost"][e][t0]

    roi_obj = roi_ben - roi_cost

    # Tables
    df_E = pd.DataFrame([
        {"Enabler": e, "Name": data['en_name_map'].get(e, e), "Start": Lbl[t0] if t0 else "-"}
        for e, t0 in enabler_start_idx.items()
    ])
    df_UC = pd.DataFrame([
        {"Use Case": uc, "Name": data['uc_name_map'].get(uc, uc), "Start": Lbl[t0] if t0 else "-"}
        for uc, t0 in uc_start_idx.items()
    ])

    summary = []
    for t in T_local:
        spend = sum(data["cost"][e][t] for e in enablers if enabler_start_idx.get(e) == t)
        ben = sum(data["benefit"][uc][t] for uc in use_cases if activeUC[uc][t] == 1)
        fte_used = sum(data["fte"][e][t] for e in enablers if enabler_start_idx.get(e) == t)
        summary.append({
            "Period": Lbl[t],
            "Spend ($M)": round(spend, 2),
            "FTE Used": round(fte_used, 1),
            "Benefit ($M)": round(ben, 2),
            "Net ($M)": round(ben - spend, 2),
            "In ROI Window": "✓" if t in roi_periods else ""
        })
    df_summary = pd.DataFrame(summary)

    return {
        "dep_violations": dep_violations,
        "budget_violations": budget_violations,
        "fte_violations": fte_violations,
        "df_E": df_E,
        "df_UC": df_UC,
        "df_summary": df_summary,
        "roi_ben": roi_ben,
        "roi_cost": roi_cost,
        "roi_obj": roi_obj,
    }

# ------------------------------------------------------------
# Executive visuals helpers (Altair)
# ------------------------------------------------------------
def _df_util_from_summary(df_summary: pd.DataFrame, data: dict, roi_periods: list[int]) -> pd.DataFrame:
    df = df_summary.copy()
    df["t"] = df["Period"].map({Lbl[t]: t for t in T})
    df["Budget Cap ($M)"] = df["t"].map(data["budget"])
    df["FTE Cap"] = df["t"].map(data["fte_cap"])
    df["Budget Util %"] = (df["Spend ($M)"] / df["Budget Cap ($M)"]).replace([float("inf")], 0.0) * 100
    df["FTE Util %"] = (df["FTE Used"] / df["FTE Cap"]).replace([float("inf")], 0.0) * 100
    df["Cum Net ($M)"] = df["Net ($M)"].cumsum()
    df["In ROI Window"] = df["t"].apply(lambda x: "✓" if x in roi_periods else "")
    return df

def _gantt_df(plan_map: dict, name_map: dict) -> pd.DataFrame:
    rows = []
    for k, t0 in plan_map.items():
        if t0 is None:
            continue
        rows.append({
            "ID": k,
            "Name": name_map.get(k, k),
            "Start_t": t0,
            "End_t": T_MAX + 0.9,
            "Start": Lbl[t0],
        })
    if not rows:
        return pd.DataFrame(columns=["ID", "Name", "Start_t", "End_t", "Start"])
    return pd.DataFrame(rows).sort_values(["Start_t", "Name"]).reset_index(drop=True)

def _axis_periods():
    # Numeric periods with friendly labels (N, N+1, ...)
    return alt.Axis(
        title=None,
        values=T,
        labelExpr="datum.value==1 ? 'N' : 'N+' + (datum.value-1)"
    )

def chart_gantt(df_gantt: pd.DataFrame, title: str):
    if not _ALTAIR_OK or df_gantt.empty:
        return None
    height = max(220, 26 * len(df_gantt))
    bars = (
        alt.Chart(df_gantt)
        .mark_bar()
        .encode(
            y=alt.Y("Name:N", sort=alt.SortField("Start_t", order="ascending"), title=None),
            x=alt.X("Start_t:Q", axis=_axis_periods()),
            x2=alt.X2("End_t:Q"),
            tooltip=[alt.Tooltip("Name:N"), alt.Tooltip("Start:N")],
        )
        .properties(title=title, height=height)
    )
    labels = (
        alt.Chart(df_gantt)
        .mark_text(align="left", dx=4)
        .encode(
            y=alt.Y("Name:N", sort=alt.SortField("Start_t", order="ascending")),
            x=alt.X("Start_t:Q"),
            text=alt.Text("Start:N"),
        )
    )
    return bars + labels

def chart_budget(df_util: pd.DataFrame):
    if not _ALTAIR_OK:
        return None
    base = alt.Chart(df_util).encode(
        x=alt.X("Period:N", sort=LBL_ORDER, title=None),
        tooltip=[
            "Period:N",
            alt.Tooltip("Spend ($M):Q", format=".2f"),
            alt.Tooltip("Budget Cap ($M):Q", format=".2f"),
            alt.Tooltip("Budget Util %:Q", format=".1f"),
        ],
    )
    bars = base.mark_bar().encode(y=alt.Y("Spend ($M):Q", title="$M"))
    cap = base.mark_line(point=True).encode(y=alt.Y("Budget Cap ($M):Q"))
    return (bars + cap).properties(title="Budget: Spend vs Cap", height=240)

def chart_fte(df_util: pd.DataFrame):
    if not _ALTAIR_OK:
        return None
    base = alt.Chart(df_util).encode(
        x=alt.X("Period:N", sort=LBL_ORDER, title=None),
        tooltip=[
            "Period:N",
            alt.Tooltip("FTE Used:Q", format=".1f"),
            alt.Tooltip("FTE Cap:Q", format=".1f"),
            alt.Tooltip("FTE Util %:Q", format=".1f"),
        ],
    )
    bars = base.mark_bar().encode(y=alt.Y("FTE Used:Q", title="FTE"))
    cap = base.mark_line(point=True).encode(y=alt.Y("FTE Cap:Q"))
    return (bars + cap).properties(title="Capacity: FTE Used vs Cap", height=240)

def chart_value(df_util: pd.DataFrame):
    if not _ALTAIR_OK:
        return None
    base = alt.Chart(df_util).encode(
        x=alt.X("Period:N", sort=LBL_ORDER, title=None),
        tooltip=[
            "Period:N",
            alt.Tooltip("Benefit ($M):Q", format=".2f"),
            alt.Tooltip("Spend ($M):Q", format=".2f"),
            alt.Tooltip("Net ($M):Q", format=".2f"),
        ],
    )
    ben = base.mark_line(point=True).encode(y=alt.Y("Benefit ($M):Q", title="$M"))
    spend = base.mark_line(point=True).encode(y=alt.Y("Spend ($M):Q"))
    return (ben + spend).properties(title="Value: Benefit vs Spend", height=240)

def chart_cum_net(df_util: pd.DataFrame):
    if not _ALTAIR_OK:
        return None
    return (
        alt.Chart(df_util)
        .mark_line(point=True)
        .encode(
            x=alt.X("Period:N", sort=LBL_ORDER, title=None),
            y=alt.Y("Cum Net ($M):Q", title="$M"),
            tooltip=["Period:N", alt.Tooltip("Cum Net ($M):Q", format=".2f")],
        )
        .properties(title="Cumulative Net Value", height=240)
    )

def chart_requirements_heatmap(data: dict):
    if not _ALTAIR_OK:
        return None
    rows = []
    for uc in data["use_cases"]:
        for e in data["enablers"]:
            rows.append({
                "Use Case": data["uc_name_map"].get(uc, uc),
                "Enabler": data["en_name_map"].get(e, e),
                "Required": int(data["req"].get(uc, {}).get(e, 0)),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return None
    height = max(260, 28 * len(data["use_cases"]))
    return (
        alt.Chart(df)
        .mark_rect()
        .encode(
            x=alt.X("Enabler:N", title=None, sort=None),
            y=alt.Y("Use Case:N", title=None, sort=None),
            color=alt.Color("Required:Q", legend=None),
            tooltip=["Use Case:N", "Enabler:N", "Required:Q"],
        )
        .properties(title="Use Case → Enabler Requirements (1 = required)", height=height)
    )

def compute_exec_insights(res: dict, data: dict, roi_periods: list[int]):
    df_util = _df_util_from_summary(res["df_summary"], data, roi_periods)

    payback_label = "—"
    payback_row = df_util[df_util["Cum Net ($M)"] > 0].head(1)
    if not payback_row.empty:
        payback_label = payback_row.iloc[0]["Period"]

    roi_multiple = None
    if res["roi_cost"] > 1e-9:
        roi_multiple = res["roi_ben"] / res["roi_cost"]

    return {
        "df_util": df_util,
        "payback": payback_label,
        "peak_budget_util": float(df_util["Budget Util %"].max()) if not df_util.empty else 0.0,
        "peak_fte_util": float(df_util["FTE Util %"].max()) if not df_util.empty else 0.0,
        "roi_multiple": roi_multiple,
    }

# ------------------------------------------------------------
# Header (product-style)
# ------------------------------------------------------------
def render_header():
    # Optional logo support (place assets/logo.png). Safe if missing.
    logo_col, title_col, chip_col = st.columns([0.12, 0.68, 0.20], vertical_alignment="center")
    with logo_col:
        try:
            st.image("assets/logo.png", use_container_width=True)
        except Exception:
            st.markdown(
                '<div class="brand-dot" style="margin-left:6px;"></div>',
                unsafe_allow_html=True
            )
    with title_col:
        st.markdown(
            """
            <div style="margin-top:2px;">
              <div style="font-weight:800; font-size: 22px; line-height:1.15; color:#0B0F19;">
                Portfolio Optimization Tool
              </div>
              <div class="muted" style="margin-top:4px;">
                ROI roadmap optimizer — objective is to maximize ROI
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with chip_col:
        st.markdown(
            """
            <div style="display:flex; justify-content:flex-end;">
              <div class="header-chip" style="color:#0B0F19; border:1px solid #E7EAF0; background:#FFFFFF;">
                Horizon: 6 periods (N…N+5)
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

render_header()

# ------------------------------------------------------------
# Sidebar: ROI horizon + Vision tie-break strength (same as original)
# + adds scenario controls + reset
# ------------------------------------------------------------
with st.sidebar:
    st.markdown("## Settings")
    scenario = st.text_input("Scenario Name", value="Base Case")

    horizon = st.slider("ROI Horizon (years)", min_value=1, max_value=5, value=5)
    roi_periods = list(range(1, horizon + 1))
    st.caption(f"Maximizing ROI over periods: {', '.join([Lbl[t] for t in roi_periods])}")

    # Vision is used ONLY as a small tie-breaker in the solver (does not change reported Objective)
    vision_priority = st.slider("Vision Priority", 0.0, 0.5, 0.05, 0.01)
    st.caption("Objective remains Benefit - Cost; Higher vision weights steer the solver to prioritize the enablers needed to unlock higher‑vision use cases.")

    st.divider()
    colA, colB = st.columns(2)
    # with colA:
    #     if st.button("Reset", type="secondary", use_container_width=True):
    #         init_session_state(force_reset=True)
    #         st.success("Reset complete.")
    #         st.rerun()
    # with colB:
    #     st.caption(f"Last refresh: {datetime.now().strftime('%H:%M:%S')}")

    st.divider()
    # st.caption("Tip: add `assets/logo.png` to show your org logo in the header.")

# ------------------------------------------------------------
# Streamlit UI (KEEP original naming & flow)
# ------------------------------------------------------------
tab_input, tab_output = st.tabs(["📥 Input Data", "📈 Output"])

with tab_input:
    st.markdown(
        """
        <div class="section-card">
          <div style="font-weight:700; font-size:16px;">Input Parameters</div>
          <div class="muted" style="margin-top:4px;">
            Review and update the model inputs and constraints below, then run the solver to generate the recommended roadmap
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # #Quick stats row (nice for exec polish, doesn’t change flow)
    # c1, c2, c3, c4 = st.columns(4)
    # c1.metric("Use Cases", int(len(st.session_state.df_ben)) if "df_ben" in st.session_state else 0)
    # c2.metric("Enablers", int(len(st.session_state.df_cost)) if "df_cost" in st.session_state else 0)
    # c3.metric("Periods", len(T))
    # c4.metric("ROI Horizon", f"{horizon} yrs")

    input_tabs = st.tabs([
        "Constraints",
        "Use Case Benefits",
        "Enabler Costs",
        "FTE Needed",
        "Requirements",
        "Dependencies",
        "Vision",
    ])
    
    # Constraints
    with input_tabs[0]:
        st.markdown("</div>", unsafe_allow_html=True)
        #st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.caption("Define budget and FTE capacity for each period.")
        st.session_state.df_cons = st.data_editor(
            st.session_state.df_cons,
            column_config={
                "Period": st.column_config.TextColumn("Period", disabled=True),
                "Budget ($M)": st.column_config.NumberColumn("Budget ($M)", min_value=0.0, step=0.1),
                "FTE Cap": st.column_config.NumberColumn("FTE Cap", min_value=0, step=1),
            },
            use_container_width=True,
            hide_index=True,
            num_rows="fixed"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Use Case Benefits
    with input_tabs[1]:
        st.markdown("</div>", unsafe_allow_html=True)
        #st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.caption("Benefits accrue for active use cases across periods.")
        st.session_state.df_ben = st.data_editor(
            st.session_state.df_ben,
            column_config={
                "Use Case": st.column_config.TextColumn("Use Case", required=True),
                "Name": st.column_config.TextColumn("Name", required=True),
                **{Lbl[t]: st.column_config.NumberColumn(Lbl[t], min_value=0.0, step=0.1) for t in T}
            },
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Enabler Costs
    with input_tabs[2]:
        st.markdown("</div>", unsafe_allow_html=True)
        st.caption("One-time spend occurs when an enabler is started.")
        st.session_state.df_cost = st.data_editor(
            st.session_state.df_cost,
            column_config={
                "Enabler": st.column_config.TextColumn("Enabler", required=True),
                "Name": st.column_config.TextColumn("Name", required=True),
                **{Lbl[t]: st.column_config.NumberColumn(Lbl[t], min_value=0.0, step=0.1) for t in T}
            },
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # FTE Needed
    with input_tabs[3]:
        st.markdown("</div>", unsafe_allow_html=True)
        st.caption("FTE is consumed in the period an enabler is started.")
        st.session_state.df_fte = st.data_editor(
            st.session_state.df_fte,
            column_config={
                "Enabler": st.column_config.TextColumn("Enabler", required=True),
                "Name": st.column_config.TextColumn("Name", required=True),
                **{Lbl[t]: st.column_config.NumberColumn(Lbl[t], min_value=0.0, step=0.5) for t in T}
            },
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Requirements
    with input_tabs[4]:
        #st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.caption("Matrix indicating which enablers are required for each use case (0/1).")

        current_enablers = st.session_state.df_cost["Enabler"].tolist() if not st.session_state.df_cost.empty else []
        if not st.session_state.df_ben.empty:
            current_use_cases = st.session_state.df_ben["Use Case"].tolist()
            uc_name_map = dict(zip(st.session_state.df_ben["Use Case"], st.session_state.df_ben["Name"]))
        else:
            current_use_cases, uc_name_map = [], {}

        if "prev_enablers" not in st.session_state or st.session_state.prev_enablers != current_enablers:
            new_rows = []
            for uc in current_use_cases:
                row = {"Use Case": uc, "Name": uc_name_map.get(uc, uc)}
                for e in current_enablers:
                    if not st.session_state.df_req.empty and uc in st.session_state.df_req["Use Case"].values:
                        old_row = st.session_state.df_req[st.session_state.df_req["Use Case"] == uc].iloc[0]
                        row[e] = old_row.get(e, 0)
                    else:
                        row[e] = 0
                new_rows.append(row)
            st.session_state.df_req = pd.DataFrame(new_rows)
            st.session_state.prev_enablers = current_enablers

        col_config = {
            "Use Case": st.column_config.TextColumn("Use Case", disabled=True),
            "Name": st.column_config.TextColumn("Name", disabled=True),
        }
        for e in current_enablers:
            col_config[e] = st.column_config.NumberColumn(e, min_value=0, max_value=1, step=1)

        st.session_state.df_req = st.data_editor(
            st.session_state.df_req,
            column_config=col_config,
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Dependencies
    with input_tabs[5]:
        st.markdown("</div>", unsafe_allow_html=True)
        #st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.caption("Define enabler-to-enabler prerequisites (e.g., E3 requires E2).")
        enabler_options = st.session_state.df_cost["Enabler"].tolist() if not st.session_state.df_cost.empty else []
        st.session_state.df_dep = st.data_editor(
            st.session_state.df_dep,
            column_config={
                "Dependent": st.column_config.SelectboxColumn("Dependent", options=enabler_options, required=True),
                "Prerequisite": st.column_config.SelectboxColumn("Prerequisite", options=enabler_options, required=True),
            },
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Vision tab
    with input_tabs[6]:
        st.markdown("</div>", unsafe_allow_html=True)
        #st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.caption("Higher vision weights steer the solver to prioritize the enablers needed to unlock higher‑vision use cases.")

        if not st.session_state.df_ben.empty:
            current_use_cases = st.session_state.df_ben["Use Case"].tolist()
            uc_name_map = dict(zip(st.session_state.df_ben["Use Case"], st.session_state.df_ben["Name"]))
        else:
            current_use_cases, uc_name_map = [], {}

        if "prev_use_cases_for_vision" not in st.session_state or st.session_state.prev_use_cases_for_vision != current_use_cases:
            new_rows = []
            for uc in current_use_cases:
                old_score = None
                if not st.session_state.df_vision.empty and uc in st.session_state.df_vision["Use Case"].values:
                    old_row = st.session_state.df_vision[st.session_state.df_vision["Use Case"] == uc].iloc[0]
                    old_score = old_row.get("Vision Score (0-1)", None)

                score = float(old_score) if old_score is not None else float(default_vision.get(uc, 0.5))

                new_rows.append({
                    "Use Case": uc,
                    "Name": uc_name_map.get(uc, uc),
                    "Vision Score (0-1)": score,
                })
            st.session_state.df_vision = pd.DataFrame(new_rows)
            st.session_state.prev_use_cases_for_vision = current_use_cases

        st.session_state.df_vision = st.data_editor(
            st.session_state.df_vision,
            column_config={
                "Use Case": st.column_config.TextColumn("Use Case", disabled=True),
                "Name": st.column_config.TextColumn("Name", disabled=True),
                "Vision Score (0-1)": st.column_config.NumberColumn(
                    "Vision Score (0-1)",
                    min_value=0.0,
                    max_value=1.0,
                    step=0.05
                ),
            },
            use_container_width=True,
            hide_index=True,
            num_rows="fixed" if len(current_use_cases) > 0 else "dynamic"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    # KEEP original flow: Run Solver button is here in Input tab
    run_col1, run_col2 = st.columns([0.25, 0.75], vertical_alignment="center")
    with run_col1:
        run_solver = st.button("🚀 Run Solver", type="primary", use_container_width=True)
    with run_col2:
        st.markdown(
            '<div class="muted"></div>',
            unsafe_allow_html=True
        )

    if run_solver:
        data = get_current_data()
        with st.spinner("Solving…"):
            result = solve_model(data, roi_periods, vision_priority)

        if result:
            st.session_state.solver_result = result
            st.session_state.current_data = data
            st.success("Optimal solution found!")
        else:
            st.session_state.solver_result = None
            st.session_state.current_data = None
            st.error("No feasible solution. Adjust inputs or constraints.")

with tab_output:
    if st.session_state.get("solver_result") is None:
        st.info("Please run the solver from the Input tab first.")
    else:
        res = st.session_state.solver_result
        data = st.session_state.current_data

        out_tabs = st.tabs(["✅ Optimal Plan", "🔮 What-If"])

        # ------------------------------------------------------------
        # Optimal Plan (same as original, but with exec visuals + polish)
        # ------------------------------------------------------------
        with out_tabs[0]:
            st.markdown(
                f"""
                <div class="section-card">
                  <div style="display:flex; justify-content:space-between; align-items:center; gap:10px;">
                    <div>
                      <div style="font-weight:800; font-size:16px;">Executive Summary</div>
                      <div class="muted" style="margin-top:4px;">
                        Scenario: <b>{scenario}</b> • ROI Window: <b>{', '.join([Lbl[t] for t in roi_periods])}</b>
                      </div>
                    </div>
                    <div class="header-chip" style="color:#0B0F19; border:1px solid #E7EAF0; background:#FFFFFF;">
                      Objective = Benefit − Cost
                    </div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            col1, col2, col3, col4, col5 = st.columns(5)
            insights = compute_exec_insights(res, data, roi_periods)

            col1.metric("Total Benefit (ROI window)", f"${res['roi_ben']:.2f}M")
            col2.metric("Total Cost (ROI window)", f"${res['roi_cost']:.2f}M")
            col3.metric("Objective (Benefit - Cost)", f"${res['roi_obj']:.2f}M")
            col4.metric("ROI Multiple", "—" if insights["roi_multiple"] is None else f"{insights['roi_multiple']:.2f}×")
            col5.metric("Payback (cum net > 0)", insights["payback"])

            # Charts row
            df_util = insights["df_util"]

            if _ALTAIR_OK:
                ch1, ch2 = st.columns(2)
                with ch1:
                    st.altair_chart(chart_budget(df_util), use_container_width=True)
                with ch2:
                    st.altair_chart(chart_fte(df_util), use_container_width=True)

                ch3, ch4 = st.columns(2)
                with ch3:
                    st.altair_chart(chart_value(df_util), use_container_width=True)
                with ch4:
                    st.altair_chart(chart_cum_net(df_util), use_container_width=True)
            else:
                st.warning("Altair not installed — charts are disabled. Install with: `pip install altair`")

            st.divider()

            # Roadmap timelines
            if _ALTAIR_OK:
                left, right = st.columns(2)
                with left:
                    st.altair_chart(
                        chart_gantt(_gantt_df(res["enabler_start_idx"], data["en_name_map"]),
                                    "Roadmap Timeline: Enablers"),
                        use_container_width=True,
                    )
                with right:
                    st.altair_chart(
                        chart_gantt(_gantt_df(res["uc_start_idx"], data["uc_name_map"]),
                                    "Roadmap Timeline: Use Cases"),
                        use_container_width=True,
                    )

            st.divider()

            # KEEP original tables/sections (but polished)
            st.subheader("Enabler Start Plan")
            df_E_opt = pd.DataFrame([
                {"Enabler": e, "Name": data['en_name_map'].get(e, e),
                 "Start": Lbl[res['enabler_start_idx'][e]] if res['enabler_start_idx'][e] else "-"}
                for e in data['enablers']
            ])
            st.dataframe(df_E_opt, use_container_width=True, hide_index=True)

            st.subheader("Use Case Start Plan")
            df_UC_opt = pd.DataFrame([
                {"Use Case": uc, "Name": data['uc_name_map'].get(uc, uc),
                 "Start": Lbl[res['uc_start_idx'][uc]] if res['uc_start_idx'][uc] else "-"}
                for uc in data['use_cases']
            ])
            st.dataframe(df_UC_opt, use_container_width=True, hide_index=True)

            st.subheader("Period-by-Period Summary")
            st.dataframe(_blank_index(res["df_summary"]), use_container_width=True)

            st.divider()

            # Requirements heatmap (helpful for leadership narrative)
            if _ALTAIR_OK:
                st.subheader("Requirements Map (Why sequencing matters)")
                hm = chart_requirements_heatmap(data)
                if hm is not None:
                    st.altair_chart(hm, use_container_width=True)

            st.divider()

            # Export section (deck appendix / auditability)
            st.subheader("Export")
            zip_bytes = build_download_zip({
                "enablers_plan.csv": df_E_opt.to_csv(index=False).encode("utf-8"),
                "use_cases_plan.csv": df_UC_opt.to_csv(index=False).encode("utf-8"),
                "period_summary.csv": res["df_summary"].to_csv(index=False).encode("utf-8"),
                "inputs_constraints.csv": st.session_state.df_cons.to_csv(index=False).encode("utf-8"),
                "inputs_use_case_benefits.csv": st.session_state.df_ben.to_csv(index=False).encode("utf-8"),
                "inputs_enabler_costs.csv": st.session_state.df_cost.to_csv(index=False).encode("utf-8"),
                "inputs_enabler_fte.csv": st.session_state.df_fte.to_csv(index=False).encode("utf-8"),
                "inputs_requirements.csv": st.session_state.df_req.to_csv(index=False).encode("utf-8"),
                "inputs_dependencies.csv": st.session_state.df_dep.to_csv(index=False).encode("utf-8"),
                "inputs_vision.csv": st.session_state.df_vision.to_csv(index=False).encode("utf-8"),
            })

            st.download_button(
                "⬇️ Download Outputs + Inputs (ZIP)",
                data=zip_bytes,
                file_name="benefits_vs_cost_export.zip",
                mime="application/zip",
                use_container_width=True,
            )

        # ------------------------------------------------------------
        # What-If (same as original, plus polish + charts)
        # ------------------------------------------------------------
        with out_tabs[1]:
            st.markdown(
                """
                <div class="section-card">
                  <div style="font-weight:800; font-size:16px;">What‑If Scenario</div>
                  <div class="muted" style="margin-top:4px;">
                    Manually adjust enabler start times. The tool checks constraints and recomputes ROI.
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            edit_rows = []
            for e in data['enablers']:
                t0 = res['enabler_start_idx'].get(e)
                edit_rows.append({
                    "Enabler": e,
                    "Name": data['en_name_map'].get(e, e),
                    "Start Period": Lbl[t0] if t0 else "-"
                })
            edit_df = pd.DataFrame(edit_rows)
            period_options = ["-"] + [Lbl[t] for t in T]

            edited = st.data_editor(
                edit_df,
                column_config={
                    "Start Period": st.column_config.SelectboxColumn(
                        "Start Period",
                        options=period_options,
                        required=True
                    )
                },
                use_container_width=True,
                hide_index=True,
                disabled=["Enabler", "Name"],
                key="whatif_editor"
            )

            run_whatif = st.button("📊 Calculate What-If ROI", type="primary")

            if run_whatif:
                new_starts = {}
                for _, row in edited.iterrows():
                    e = row["Enabler"]
                    label = row["Start Period"]
                    new_starts[e] = None if label == "-" else period_label_to_t(label)

                whatif = compute_whatif(data, new_starts, roi_periods)
                st.session_state.whatif_result = whatif

            if "whatif_result" not in st.session_state:
                st.info("Adjust start periods and click **Calculate What‑If ROI**.")
            else:
                whatif = st.session_state.whatif_result

                # Violations
                if whatif["dep_violations"]:
                    st.warning("⚠️ Dependency violations:")
                    for v in whatif["dep_violations"]:
                        st.write(f"- {v}")
                if whatif["budget_violations"]:
                    st.error("💰 Budget violations:")
                    for v in whatif["budget_violations"]:
                        st.write(f"- {v}")
                if whatif["fte_violations"]:
                    st.error("👥 FTE violations:")
                    for v in whatif["fte_violations"]:
                        st.write(f"- {v}")

                if not any([whatif["dep_violations"], whatif["budget_violations"], whatif["fte_violations"]]):
                    st.success("All constraints satisfied.")

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Benefit (ROI window)", f"${whatif['roi_ben']:.2f}M")
                col2.metric("Total Cost (ROI window)", f"${whatif['roi_cost']:.2f}M")
                col3.metric(
                    "Objective (Benefit - Cost)",
                    f"${whatif['roi_obj']:.2f}M",
                    delta=f"{whatif['roi_obj'] - res['roi_obj']:+.2f}M vs optimal"
                )
                col4.metric("ROI Horizon", f"{horizon} yrs")

                # Charts for what-if
                if _ALTAIR_OK:
                    df_util_w = _df_util_from_summary(whatif["df_summary"], data, roi_periods)
                    c1, c2 = st.columns(2)
                    with c1:
                        st.altair_chart(chart_budget(df_util_w), use_container_width=True)
                    with c2:
                        st.altair_chart(chart_fte(df_util_w), use_container_width=True)

                    c3, c4 = st.columns(2)
                    with c3:
                        st.altair_chart(chart_value(df_util_w), use_container_width=True)
                    with c4:
                        st.altair_chart(chart_cum_net(df_util_w), use_container_width=True)

                st.subheader("Enabler Start Plan (What-If)")
                st.dataframe(whatif["df_E"], use_container_width=True, hide_index=True)

                st.subheader("Use Case Start Plan (What-If)")
                st.dataframe(whatif["df_UC"], use_container_width=True, hide_index=True)

                st.subheader("Period-by-Period Summary (What-If)")
                st.dataframe(_blank_index(whatif["df_summary"]), use_container_width=True)

# Bottom glossary / notes (optional, professional)
# with st.expander("Notes / Glossary"):
#     st.write(
#         "- **Benefit** accrues each period after a use case starts (use case stays active).\n"
#         "- **Cost** is one-time and charged in the period the enabler starts.\n"
#         "- **Objective** is always **Benefit − Cost** over the selected ROI window.\n"
#         "- **Vision Priority** is used only as a small tie-breaker to select among near-equal ROI solutions."
#     )



