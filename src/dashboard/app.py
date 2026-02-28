from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

from src.common.run_registry import list_runs, resolve_run

# If you defined these in experiment_log.py, import them.
# If not, keep the Path(...) values below and delete the import.
try:
    from src.common.experiment_log import TRAINING_CSV, GENERATION_CSV
except Exception:
    TRAINING_CSV = Path("artifacts/metrics/experiments.csv")
    GENERATION_CSV = Path("artifacts/metrics/generation.csv")

st.set_page_config(page_title="Fraud Ops Dashboard", layout="wide")

RUNS_ROOT = Path("artifacts/runs")
EXPERIMENTS_CSV = Path(TRAINING_CSV)
GEN_CSV = Path(GENERATION_CSV)


def read_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that are entirely empty and sort by timestamp if present."""
    # Drop all-null columns (your main request)
    df = df.dropna(axis=1, how="all")

    # Parse timestamp columns if present
    for ts_col in ["run_ts", "gen_ts", "ts"]:
        if ts_col in df.columns:
            df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")

    # Prefer run_ts sorting
    if "run_ts" in df.columns:
        df = df.sort_values("run_ts", ascending=False)
    elif "gen_ts" in df.columns:
        df = df.sort_values("gen_ts", ascending=False)

    return df


def load_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return _clean_df(df)


def money(x: Any) -> str:
    try:
        return f"${float(x):,.0f}"
    except Exception:
        return str(x)


def pct(x: Any) -> str:
    try:
        return f"{float(x) * 100:.2f}%"
    except Exception:
        return "N/A"


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

        :root {
            --bg-0: #f8fafc;
            --bg-1: #e2e8f0;
            --ink-0: #0f172a;
            --ink-1: #334155;
            --accent-0: #0ea5e9;
            --accent-1: #14b8a6;
            --card-bg: rgba(255, 255, 255, 0.80);
            --card-border: rgba(15, 23, 42, 0.08);
            --ok: #0f766e;
        }

        .stApp {
            background:
                radial-gradient(1200px 500px at -15% -20%, rgba(20, 184, 166, 0.22), transparent 60%),
                radial-gradient(900px 450px at 120% 10%, rgba(14, 165, 233, 0.20), transparent 55%),
                linear-gradient(165deg, var(--bg-0) 0%, var(--bg-1) 100%);
            color: var(--ink-0);
            font-family: "Space Grotesk", "Segoe UI", sans-serif;
        }

        h1, h2, h3 {
            letter-spacing: -0.01em;
        }

        div[data-testid="stMetric"] {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 14px;
            padding: 14px 16px;
            backdrop-filter: blur(6px);
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
        }

        div[data-testid="stMetricLabel"] p {
            color: var(--ink-1);
            font-weight: 600;
            font-size: 0.80rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        div[data-testid="stMetricValue"] {
            color: var(--ink-0);
            font-weight: 700;
        }

        div[data-baseweb="tab-list"] {
            gap: 8px;
        }

        button[data-baseweb="tab"] {
            border: 1px solid var(--card-border) !important;
            border-radius: 999px !important;
            padding: 8px 16px !important;
            background: rgba(255, 255, 255, 0.72) !important;
            color: var(--ink-1) !important;
            transition: all 0.2s ease;
        }

        button[data-baseweb="tab"][aria-selected="true"] {
            background: linear-gradient(90deg, var(--accent-0), var(--accent-1)) !important;
            color: #ffffff !important;
            border-color: transparent !important;
        }

        [data-testid="stDataFrame"], [data-testid="stJson"] {
            border-radius: 12px;
            border: 1px solid var(--card-border);
            overflow: hidden;
        }

        code, pre {
            font-family: "IBM Plex Mono", monospace !important;
        }

        .hero {
            border: 1px solid var(--card-border);
            border-radius: 18px;
            padding: 18px 20px;
            background: linear-gradient(
                135deg,
                rgba(14, 165, 233, 0.11) 0%,
                rgba(20, 184, 166, 0.10) 100%
            );
            margin-bottom: 0.5rem;
        }

        .hero-kicker {
            color: var(--ok);
            font-size: 0.80rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    inject_styles()
    st.title("Merchant Fraud Ops Intelligence")
    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">Live Monitoring</div>
            <div>Track model quality, threshold economics, and batch scoring outcomes for each experiment run.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    runs = list_runs(RUNS_ROOT)
    if not runs:
        st.error("No runs found in artifacts/runs. Train at least one run first.")
        return

    # --- Sidebar: run selection ---
    st.sidebar.header("Select Run")
    selected_run = st.sidebar.selectbox("run_id", runs, index=0)

    # Load run artifacts
    run = resolve_run(selected_run, RUNS_ROOT)
    metrics = read_json(run.metrics_path)
    thr = read_json(run.threshold_report_path)
    model_card = run.model_card_path.read_text(encoding="utf-8") if run.model_card_path.exists() else ""

    # Load tracking CSVs
    train_df = load_csv(EXPERIMENTS_CSV)
    gen_df = load_csv(GEN_CSV)

    # Optional: context from training CSV for selected run
    selected_row = None
    if train_df is not None and "run_id" in train_df.columns:
        match = train_df[train_df["run_id"] == selected_run]
        if len(match) > 0:
            selected_row = match.iloc[0].to_dict()

    # --- KPI Row ---
    k1, k2, k3, k4, k5, k6 = st.columns(6)

    best_t = metrics.get("best_threshold_valid_cost", thr.get("best_threshold", 0.0))
    net_cost = thr.get(
        "net_cost_lower_is_better",
        metrics.get("threshold_report_valid", {}).get("net_cost_lower_is_better", None),
    )

    k1.metric("Run ID", selected_run)
    k2.metric("Best Threshold", f"{float(best_t):.4f}")
    k3.metric("Valid PR-AUC", f"{float(metrics.get('valid_pr_auc', 0.0)):.4f}")
    k4.metric("Test PR-AUC", f"{float(metrics.get('test_pr_auc', 0.0)):.4f}")

    valid_alert_rate = None
    if isinstance(selected_row, dict):
        valid_alert_rate = selected_row.get("valid_alert_rate", None)

    k5.metric("Valid Net Cost", money(net_cost) if net_cost is not None else "N/A")
    k6.metric(
        "Valid Alert Rate",
        pct(valid_alert_rate) if valid_alert_rate not in [None, ""] else "N/A",
    )

    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Trends (Training)", "Trends (Generation)", "Selected Run Details", "Batch Score CSV"]
    )

    # ========== Tab 1: Training Trends ==========
    with tab1:
        st.subheader("Training Trends (experiments.csv)")

        if train_df is None:
            st.info("experiments.csv not found yet. Run training to generate it.")
        else:
            show_cols = [c for c in [
                "run_ts", "run_id", "log_tag", "dataset_id",
                "fp_cost", "topk_per_day", "fn_mult",
                "valid_pr_auc", "test_pr_auc",
                "best_threshold",
                "valid_recall_at_topk", "test_recall_at_topk",
                "valid_alert_rate", "test_alert_rate",
                "valid_net_cost", "valid_fp", "valid_fn",
            ] if c in train_df.columns]

            st.dataframe(train_df[show_cols], use_container_width=True, height=360)

            cA, cB = st.columns(2)

            if "valid_net_cost" in train_df.columns and "run_ts" in train_df.columns:
                tmp = train_df.dropna(subset=["run_ts"]).copy()
                tmp["valid_net_cost"] = pd.to_numeric(tmp["valid_net_cost"], errors="coerce")
                tmp = tmp.dropna(subset=["valid_net_cost"])
                if len(tmp) > 0:
                    cA.write("**Valid Net Cost (lower is better)**")
                    cA.line_chart(tmp.set_index("run_ts")[["valid_net_cost"]])

            if "best_threshold" in train_df.columns and "run_ts" in train_df.columns:
                tmp = train_df.dropna(subset=["run_ts"]).copy()
                tmp["best_threshold"] = pd.to_numeric(tmp["best_threshold"], errors="coerce")
                tmp = tmp.dropna(subset=["best_threshold"])
                if len(tmp) > 0:
                    cB.write("**Best Threshold trend**")
                    cB.line_chart(tmp.set_index("run_ts")[["best_threshold"]])

    # ========== Tab 2: Generation Trends ==========
    with tab2:
        st.subheader("Generation Trends (generation.csv)")

        if gen_df is None:
            st.info("generation.csv not found yet. Run generation logging to create it.")
        else:
            # keep only meaningful cols, and only those that exist
            show_cols = [c for c in [
                "run_ts", "stage", "dataset_id", "sigmoid_shift",
                "n_transactions", "fraud_rate", "raw_dir",
                "fp_cost", "topk_per_day", "fn_mult",
                "best_threshold",
                "baseline_alert_rate", "baseline_net_cost",
                "baseline_fp", "baseline_fn",
            ] if c in gen_df.columns]

            st.dataframe(gen_df[show_cols], use_container_width=True, height=360)

    # ========== Tab 3: Selected Run ==========
    with tab3:
        st.subheader("Selected Run - Artifacts & Evaluation")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Metrics")
            st.json(metrics)
        with c2:
            st.markdown("### Threshold Report")
            st.json(thr)

        st.markdown("### Model Card")
        if model_card:
            st.code(model_card, language="markdown")
        else:
            st.info("No model_card.md found for this run.")

    # ========== Tab 4: Batch Scoring ==========
    with tab4:
        st.subheader("Batch Score a CSV using the selected run")

        up = st.file_uploader("Upload CSV", type=["csv"])
        if up is not None:
            df = pd.read_csv(up)
            st.write("Preview:", df.head())

            import joblib
            bundle = joblib.load(run.model_path)
            model = bundle["model"]
            threshold = float(bundle["threshold"])
            spec = bundle.get("feature_spec", {})
            num_cols = list(spec.get("num_cols", []))
            cat_cols = list(spec.get("cat_cols", []))

            for col in (num_cols + cat_cols):
                if col not in df.columns:
                    df[col] = pd.NA

            probs = model.predict_proba(df)[:, 1]
            out = df.copy()
            out["fraud_probability"] = probs
            out["is_fraud_alert"] = (out["fraud_probability"] >= threshold).astype(int)

            st.write("Scored Output (top 50):")
            st.dataframe(out.head(50), use_container_width=True)

            st.download_button(
                "Download scored CSV",
                data=out.to_csv(index=False).encode("utf-8"),
                file_name=f"scored_{selected_run}.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()
