import streamlit as st
import pulp as pl
import pandas as pd

# ------------------------------------------------------------
# 6-period horizon: N to N+5
# ------------------------------------------------------------
T = [1, 2, 3, 4, 5, 6]
Lbl = {t: ("N" if t == 1 else f"N+{t-1}") for t in T}

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
# Helper functions to convert between DataFrames and nested dicts
# ------------------------------------------------------------
def df_to_period_dict(df, id_col, value_cols):
    """Convert a DataFrame with period columns to nested dict: {id: {t: val}}."""
    result = {}
    for _, row in df.iterrows():
        id_val = row[id_col]
        result[id_val] = {}
        for t, label in Lbl.items():
            if label in value_cols:
                result[id_val][t] = row[label]
    return result

def period_dict_to_df(data, id_col, name_col, id_name_map):
    """Convert nested dict to DataFrame with period columns."""
    rows = []
    for id_val in data:
        row = {id_col: id_val, name_col: id_name_map[id_val]}
        for t, label in Lbl.items():
            row[label] = data[id_val][t]
        rows.append(row)
    return pd.DataFrame(rows)

def req_dict_to_df(req_dict, use_cases, enablers, uc_name_map):
    """Convert requirement dict to matrix DataFrame."""
    rows = []
    for uc in use_cases:
        row = {"Use Case": uc, "Name": uc_name_map.get(uc, uc)}
        for e in enablers:
            row[e] = int(req_dict.get(uc, {}).get(e, 0))
        rows.append(row)
    return pd.DataFrame(rows)

def df_to_req_dict(df, use_cases, enablers):
    """Convert requirement matrix DataFrame back to nested dict."""
    req = {}
    for _, row in df.iterrows():
        uc = row["Use Case"]
        req[uc] = {}
        for e in enablers:
            req[uc][e] = int(row[e])
    return req

def dep_df_to_dict(df):
    """Convert dependency DataFrame (Dependent, Prerequisite) to prereq_ready dict."""
    prereq = {}
    for _, row in df.iterrows():
        child = row["Dependent"]
        parent = row["Prerequisite"]
        if child not in prereq:
            prereq[child] = []
        if parent not in prereq[child]:
            prereq[child].append(parent)
    return prereq

def dict_to_dep_df(prereq_dict):
    """Convert prereq_ready dict to DataFrame with rows (Dependent, Prerequisite)."""
    rows = []
    for child, parents in prereq_dict.items():
        for p in parents:
            rows.append({"Dependent": child, "Prerequisite": p})
    return pd.DataFrame(rows)

# ------------------------------------------------------------
# Initialize session state with default editable dataframes
# ------------------------------------------------------------
def init_session_state():
    if "df_cons" not in st.session_state:
        st.session_state.df_cons = pd.DataFrame([
            {"Period": Lbl[t], "Budget ($M)": default_budget[t], "FTE Cap": default_fte_cap[t]}
            for t in T
        ])

    # Use Case Benefits (dynamic rows)
    if "df_ben" not in st.session_state:
        rows = []
        for uc in default_use_cases:
            row = {"Use Case": uc, "Name": default_uc_name[uc]}
            row.update({Lbl[t]: default_benefit[uc][t] for t in T})
            rows.append(row)
        st.session_state.df_ben = pd.DataFrame(rows)

    # Enabler Costs (dynamic rows)
    if "df_cost" not in st.session_state:
        rows = []
        for e in default_enablers:
            row = {"Enabler": e, "Name": default_en_name[e]}
            row.update({Lbl[t]: default_cost[e][t] for t in T})
            rows.append(row)
        st.session_state.df_cost = pd.DataFrame(rows)

    # FTE Needed (dynamic rows)
    if "df_fte" not in st.session_state:
        rows = []
        for e in default_enablers:
            row = {"Enabler": e, "Name": default_en_name[e]}
            row.update({Lbl[t]: default_fte[e][t] for t in T})
            rows.append(row)
        st.session_state.df_fte = pd.DataFrame(rows)

    # Requirements (will be rebuilt dynamically before display)
    if "df_req" not in st.session_state:
        st.session_state.df_req = req_dict_to_df(default_req, default_use_cases, default_enablers, default_uc_name)

    # Dependencies (dynamic rows)
    if "df_dep" not in st.session_state:
        st.session_state.df_dep = dict_to_dep_df(default_prereq_ready)

init_session_state()

# ------------------------------------------------------------
# Sidebar: ROI horizon
# ------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    horizon = st.slider("ROI Horizon (years)", min_value=1, max_value=5, value=5)
    roi_periods = list(range(1, horizon+1))
    st.caption(f"Maximizing ROI over periods: {', '.join([Lbl[t] for t in roi_periods])}")

# ------------------------------------------------------------
# Helper to extract current data from session state
# ------------------------------------------------------------
def get_current_data():
    # Constraints
    budget = {}
    fte_cap = {}
    for _, row in st.session_state.df_cons.iterrows():
        label = row["Period"]
        t = next(k for k, v in Lbl.items() if v == label)
        budget[t] = float(row["Budget ($M)"])
        fte_cap[t] = float(row["FTE Cap"])

    # Use cases and enablers from tables
    use_cases = st.session_state.df_ben["Use Case"].tolist()
    uc_name_map = dict(zip(st.session_state.df_ben["Use Case"], st.session_state.df_ben["Name"]))
    enablers = st.session_state.df_cost["Enabler"].tolist()
    en_name_map = dict(zip(st.session_state.df_cost["Enabler"], st.session_state.df_cost["Name"]))

    # Benefits
    benefit = df_to_period_dict(st.session_state.df_ben, "Use Case", [Lbl[t] for t in T])

    # Costs
    cost = df_to_period_dict(st.session_state.df_cost, "Enabler", [Lbl[t] for t in T])

    # FTE
    fte = df_to_period_dict(st.session_state.df_fte, "Enabler", [Lbl[t] for t in T])

    # Requirements
    req = df_to_req_dict(st.session_state.df_req, use_cases, enablers)

    # Dependencies
    prereq_ready = dep_df_to_dict(st.session_state.df_dep)

    return {
        "budget": budget,
        "fte_cap": fte_cap,
        "benefit": benefit,
        "cost": cost,
        "fte": fte,
        "req": req,
        "prereq_ready": prereq_ready,
        "use_cases": use_cases,
        "enablers": enablers,
        "uc_name_map": uc_name_map,
        "en_name_map": en_name_map,
    }

# ------------------------------------------------------------
# Solver function (uses current data and horizon)
# ------------------------------------------------------------
def solve_model(data, roi_periods):
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

    # FTE constraints: one‑time FTE when starting
    for t in T_local:
        m += pl.lpSum(data["fte"][e][t] * startE[e][t] for e in enablers) <= data["fte_cap"][t]

    # Objective: maximize ROI over selected periods
    benefit_roi = pl.lpSum(data["benefit"][uc][t] * activeUC[uc][t] for uc in use_cases for t in roi_periods)
    cost_roi = pl.lpSum(data["cost"][e][t] * startE[e][t] for e in enablers for t in roi_periods)
    m += benefit_roi - cost_roi

    m.solve(pl.PULP_CBC_CMD(msg=False))
    status = pl.LpStatus[m.status]
    if status != "Optimal":
        return None

    def v(x):
        return float(pl.value(x) or 0.0)

    enabler_start = {e: next((t for t in T_local if v(startE[e][t]) > 0.5), None) for e in enablers}
    uc_start = {uc: next((t for t in T_local if v(startUC[uc][t]) > 0.5), None) for uc in use_cases}

    # Build summary
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

    roi_ben = sum(data["benefit"][uc][t] * v(activeUC[uc][t]) for uc in use_cases for t in roi_periods)
    roi_cost = sum(data["cost"][e][t] * v(startE[e][t]) for e in enablers for t in roi_periods)

    return {
        "enabler_start_idx": enabler_start,
        "uc_start_idx": uc_start,
        "df_summary": df_summary,
        "roi_ben": roi_ben,
        "roi_cost": roi_cost,
        "roi_obj": roi_ben - roi_cost,
    }

# ------------------------------------------------------------
# What-if computation (uses current data, one‑time FTE)
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

    # Budget/FTE violations (FTE is one‑time when starting)
    budget_violations = []
    fte_violations = []
    for t in T_local:
        spend_t = sum(data["cost"][e][t] for e in enablers if enabler_start_idx.get(e) == t)
        if spend_t > data["budget"][t] + 1e-6:
            budget_violations.append(
                f"{Lbl[t]}: Spend ${spend_t:.2f}M exceeds budget ${data['budget'][t]:.2f}M"
            )

        fte_t = sum(data["fte"][e][t] for e in enablers if enabler_start_idx.get(e) == t)
        if fte_t > data["fte_cap"][t] + 1e-6:
            fte_violations.append(
                f"{Lbl[t]}: FTE used {fte_t:.1f} exceeds cap {data['fte_cap'][t]:.1f}"
            )

    # Use case start times
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

    # ROI totals (FIXED: correctly use the start period to fetch cost)
    roi_ben = sum(data["benefit"][uc][t] * activeUC[uc][t] for uc in use_cases for t in roi_periods)
    roi_cost = 0.0
    for e in enablers:
        t0 = enabler_start_idx.get(e)
        if t0 is not None and t0 in roi_periods:
            roi_cost += data["cost"][e][t0]

    # Build output tables
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
        "roi_obj": roi_ben - roi_cost,
    }

# ------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------
st.set_page_config(layout="wide", page_title="Benefits Vs. Cost Calculator")
st.title("📊 Benefits Vs. Cost Calculator")
#st.markdown("**Goal: Output which enabler to implement at what time**")

tab_input, tab_output = st.tabs(["📥 Input Data", "📈 Output"])

with tab_input:
    st.subheader("Input Parameters")
    input_tabs = st.tabs([
        "Constraints",
        "Use Case Benefits",
        "Enabler Costs",
        "FTE Needed",
        "Requirements",
        "Dependencies"
    ])

    # Constraints (fixed rows)
    with input_tabs[0]:
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

    # Use Case Benefits (dynamic rows)
    with input_tabs[1]:
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

    # Enabler Costs (dynamic rows)
    with input_tabs[2]:
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

    # FTE Needed (dynamic rows)
    with input_tabs[3]:
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

    # Requirements (dynamic rows and columns)
    with input_tabs[4]:
        # Get current enablers from df_cost
        if not st.session_state.df_cost.empty:
            current_enablers = st.session_state.df_cost["Enabler"].tolist()
        else:
            current_enablers = []
        # Get current use cases from df_ben
        if not st.session_state.df_ben.empty:
            current_use_cases = st.session_state.df_ben["Use Case"].tolist()
            uc_name_map = dict(zip(st.session_state.df_ben["Use Case"], st.session_state.df_ben["Name"]))
        else:
            current_use_cases = []
            uc_name_map = {}

        # Rebuild df_req if enablers changed (preserve existing data)
        if "prev_enablers" not in st.session_state or st.session_state.prev_enablers != current_enablers:
            # Create new df with all use cases and enabler columns
            new_rows = []
            for uc in current_use_cases:
                row = {"Use Case": uc, "Name": uc_name_map.get(uc, uc)}
                for e in current_enablers:
                    # Try to get old value if it existed
                    if not st.session_state.df_req.empty and uc in st.session_state.df_req["Use Case"].values:
                        old_row = st.session_state.df_req[st.session_state.df_req["Use Case"] == uc].iloc[0]
                        row[e] = old_row.get(e, 0)
                    else:
                        row[e] = 0
                new_rows.append(row)
            st.session_state.df_req = pd.DataFrame(new_rows)
            st.session_state.prev_enablers = current_enablers

        # Now display the editor with dynamic columns
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

    # Dependencies (dynamic rows)
    with input_tabs[5]:
        st.caption("Define enabler‑to‑enabler dependencies. Add/remove rows as needed.")
        # Get current enablers for dropdown options
        if not st.session_state.df_cost.empty:
            enabler_options = st.session_state.df_cost["Enabler"].tolist()
        else:
            enabler_options = []
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

    st.divider()
    if st.button("🚀 Run Solver", type="primary"):
        data = get_current_data()
        result = solve_model(data, roi_periods)
        if result:
            st.session_state.solver_result = result
            st.session_state.current_data = data
            st.success("Optimal solution found!")
        else:
            st.session_state.solver_result = None
            st.error("No feasible solution. Adjust inputs or constraints.")

with tab_output:
    if st.session_state.get("solver_result") is None:
        st.info("Please run the solver from the Input tab first.")
    else:
        res = st.session_state.solver_result
        data = st.session_state.current_data

        out_tabs = st.tabs(["✅ Optimal Plan", "🔮 What‑If"])

        with out_tabs[0]:
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Benefit (ROI window)", f"${res['roi_ben']:.2f}M")
            col2.metric("Total Cost (ROI window)", f"${res['roi_cost']:.2f}M")
            col3.metric("Objective (Benefit - Cost)", f"${res['roi_obj']:.2f}M")

            st.subheader("Enabler Start Plan")
            df_E_opt = pd.DataFrame([
                {"Enabler": e, "Name": data['en_name_map'].get(e, e), "Start": Lbl[res['enabler_start_idx'][e]] if res['enabler_start_idx'][e] else "-"}
                for e in data['enablers']
            ])
            st.dataframe(df_E_opt, use_container_width=True, hide_index=True)

            st.subheader("Use Case Start Plan")
            df_UC_opt = pd.DataFrame([
                {"Use Case": uc, "Name": data['uc_name_map'].get(uc, uc), "Start": Lbl[res['uc_start_idx'][uc]] if res['uc_start_idx'][uc] else "-"}
                for uc in data['use_cases']
            ])
            st.dataframe(df_UC_opt, use_container_width=True, hide_index=True)

            st.subheader("Period-by-Period Summary")
            st.dataframe(res["df_summary"], use_container_width=True, hide_index=True)

        with out_tabs[1]:
            st.markdown("Manually adjust enabler start times. The tool checks constraints and recomputes ROI.")
            # Build editor from current enablers
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

            if st.button("📊 Calculate What‑If ROI"):
                new_starts = {}
                for _, row in edited.iterrows():
                    e = row["Enabler"]
                    label = row["Start Period"]
                    if label == "-":
                        new_starts[e] = None
                    else:
                        new_starts[e] = next(t for t in T if Lbl[t] == label)

                whatif = compute_whatif(data, new_starts, roi_periods)

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

                col1, col2, col3 = st.columns(3)
                col1.metric("Total Benefit (ROI window)", f"${whatif['roi_ben']:.2f}M")
                col2.metric("Total Cost (ROI window)", f"${whatif['roi_cost']:.2f}M")
                col3.metric("Objective (Benefit - Cost)", f"${whatif['roi_obj']:.2f}M",
                            delta=f"{whatif['roi_obj'] - res['roi_obj']:.2f}M vs optimal")

                st.subheader("Enabler Start Plan (What‑If)")
                st.dataframe(whatif["df_E"], use_container_width=True, hide_index=True)

                st.subheader("Use Case Start Plan (What‑If)")
                st.dataframe(whatif["df_UC"], use_container_width=True, hide_index=True)

                st.subheader("Period-by-Period Summary (What‑If)")
                st.dataframe(whatif["df_summary"], use_container_width=True, hide_index=True)