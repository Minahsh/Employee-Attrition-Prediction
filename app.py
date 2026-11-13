import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.express as px

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "gradient_boosting_model.pkl"
FEATURES_PATH = BASE_DIR / "models" / "feature_names.json"
TRAIN_PATH = BASE_DIR / "data" / "train.csv"
TEST_PATH = BASE_DIR / "data" / "test.csv"

TARGET = "Attrition"
ID_COLUMNS = ["Employee ID", "Employee_ID", "EmployeeID", "employee_id"]

SYNTHETIC_NOTE = (
    "This project uses a **synthetic** employee attrition dataset. Patterns shown "
    "here are properties of that dataset and the trained model. They are **not** "
    "real-world facts and **not** causal relationships."
)


METRICS = {
    "Accuracy": 76.12,
    "Precision": 74.84,
    "Recall": 74.42,
    "F1-Score": 74.63,
    "ROC-AUC": 85.31,
}
CONF_MATRIX = np.array([[6109, 1759], [1799, 5233]])

st.set_page_config(
    page_title="Employee Attrition Intelligence",
    page_icon="📊",
    layout="wide",
)


st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    .hero {
        padding: 2rem 2.2rem; border-radius: 18px; margin-bottom: 1.2rem;
        background: linear-gradient(135deg, #1f3c88 0%, #2b6cb0 55%, #38b2ac 100%);
        color: #ffffff;
    }
    .hero h1 {margin: 0; font-size: 2.2rem; color: #ffffff;}
    .hero p {margin: .5rem 0 0 0; font-size: 1.1rem; opacity: .95; color: #ffffff;}
    .kpi {
        border-radius: 14px; padding: 1rem 1.2rem; text-align: center;
        background: rgba(120, 140, 180, 0.12);
        border: 1px solid rgba(120, 140, 180, 0.30);
    }
    .kpi .label {font-size: .85rem; opacity: .75; text-transform: uppercase; letter-spacing: .04em;}
    .kpi .value {font-size: 1.7rem; font-weight: 700; margin-top: .2rem;}
    .risk-low {color: #2f9e44;} .risk-medium {color: #f08c00;} .risk-high {color: #e03131;}
    .note {
        border-left: 4px solid #f08c00; padding: .6rem .9rem; border-radius: 6px;
        background: rgba(240, 140, 0, 0.10); margin: .6rem 0 1rem 0; font-size: .92rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def kpi(label, value, css_class=""):
    st.markdown(
        f'<div class="kpi"><div class="label">{label}</div>'
        f'<div class="value {css_class}">{value}</div></div>',
        unsafe_allow_html=True,
    )


def synthetic_banner():
    st.markdown(f'<div class="note">⚠️ {SYNTHETIC_NOTE}</div>', unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_model():
    """Load the trained pipeline and show an actionable compatibility error."""
    try:
        import joblib
        return joblib.load(MODEL_PATH)
    except Exception as exc:
        import sklearn

        raise RuntimeError(
            "The saved model could not be loaded. "
            f"Installed scikit-learn version: {sklearn.__version__}. "
            "Install the project dependencies from requirements.txt "
            "and restart the app. Original error: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


@st.cache_data(show_spinner=False)
def load_feature_names():
    with open(FEATURES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("feature_names", "features", "columns"):
            if key in data and isinstance(data[key], list):
                return data[key]
        if all(isinstance(v, list) for v in data.values()):
            return [c for v in data.values() for c in v]
        return list(data.keys())
    raise ValueError("Unsupported feature_names.json format.")


@st.cache_data(show_spinner=False)
def load_train():
    df = pd.read_csv(TRAIN_PATH)
    return df


def prepare_target(df):
    df = df.copy()
    if TARGET not in df.columns:
        return df
    col = df[TARGET]
    if pd.api.types.is_numeric_dtype(col):
        df["Attrition_num"] = col
    else:
        mapping = {"stayed": 0, "left": 1, "no": 0, "yes": 1, "0": 0, "1": 1}
        df["Attrition_num"] = col.astype(str).str.strip().str.lower().map(mapping)
    df = df.dropna(subset=["Attrition_num"])
    df["Attrition_num"] = df["Attrition_num"].astype(int)
    df["Outcome"] = df["Attrition_num"].map({0: "Stayed", 1: "Left"})
    return df


def feature_columns(df):
    return [
        c
        for c in df.columns
        if c not in ID_COLUMNS + [TARGET, "Attrition_num", "Outcome"]
    ]


def risk_level(p):
    if p < 0.40:
        return "Low", "risk-low"
    if p < 0.65:
        return "Medium", "risk-medium"
    return "High", "risk-high"


LOW_WORDS = {"low", "very low", "poor", "fair", "bad"}


def _is_low(col, val, train):
    if isinstance(val, str):
        return val.strip().lower() in LOW_WORDS
    if train is not None and col in train and pd.api.types.is_numeric_dtype(train[col]):
        return val <= train[col].quantile(0.25)
    return False


def build_suggestions(inp, train):
    s = []
    g = inp.get

    def yes(v):
        return str(v).strip().lower() in {"yes", "true", "1"}

    if "Overtime" in inp and yes(g("Overtime")):
        s.append(
            "Review workload and overtime expectations with the employee's manager."
        )
    for col, msg in [
        ("Work-Life Balance", "Discuss work-life balance and scheduling flexibility."),
        (
            "Job Satisfaction",
            "Hold a stay conversation to understand job satisfaction.",
        ),
        (
            "Employee Recognition",
            "Consider whether recognition for the employee's contributions is sufficient.",
        ),
        (
            "Leadership Opportunities",
            "Explore leadership or stretch-assignment opportunities.",
        ),
        (
            "Innovation Opportunities",
            "Explore opportunities for the employee to contribute to innovation projects.",
        ),
        (
            "Company Reputation",
            "Check whether perceptions of the company reputation are a concern for this employee.",
        ),
    ]:
        if col in inp and _is_low(col, g(col), train):
            s.append(msg)
    if "Remote Work" in inp and not yes(g("Remote Work")):
        s.append("Review whether remote or hybrid work options could be offered.")
    if (
        "Number of Promotions" in inp
        and "Years at Company" in inp
        and pd.to_numeric(g("Number of Promotions"), errors="coerce") == 0
        and pd.to_numeric(g("Years at Company"), errors="coerce") >= 3
    ):
        s.append(
            "Review career path and promotion readiness (no promotions so far despite tenure)."
        )
    if "Monthly Income" in inp and train is not None and "Monthly Income" in train:
        if pd.to_numeric(g("Monthly Income"), errors="coerce") <= train[
            "Monthly Income"
        ].quantile(0.25):
            s.append(
                "Benchmark compensation: income is in the lower quartile of the training data."
            )
    if (
        "Distance from Home" in inp
        and train is not None
        and "Distance from Home" in train
    ):
        if pd.to_numeric(g("Distance from Home"), errors="coerce") >= train[
            "Distance from Home"
        ].quantile(0.75):
            s.append(
                "Consider commute-related support (flexible hours or remote days)."
            )
    if not s:
        s.append(
            "No specific review prompts triggered by the inputs. Continue regular check-ins."
        )
    return s


st.sidebar.title("📊 Attrition Intelligence")
page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🔮 Attrition Prediction",
        "📈 Analytics Dashboard",
        "🎯 Model Performance",
    ],
)
st.sidebar.markdown("---")
st.sidebar.caption(
    "Synthetic dataset. Findings are not real-world facts or causal claims."
)


try:
    train_raw = load_train()
    train = prepare_target(train_raw)
except Exception as e:
    train_raw, train = None, None
    st.error(f"Could not load training data at `{TRAIN_PATH}`: {e}")


def page_home():
    st.markdown(
        """
        <div class="hero">
            <h1>Employee Attrition Prediction &amp; Retention Intelligence</h1>
            <p>A machine learning application for estimating attrition risk and exploring workforce patterns.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    synthetic_banner()

    st.subheader("Project Description")
    st.write(
        "This application uses a trained Gradient Boosting classifier to estimate the "
        "probability that an employee will leave the company, based on job, personal and "
        "workplace attributes. It also provides an interactive analytics dashboard and a "
        "transparent evaluation of the model on an independent test set."
    )

    st.subheader("Dataset Overview")
    n_rows = len(train_raw) if train_raw is not None else 0
    n_feats = len(feature_columns(train_raw)) if train_raw is not None else 0
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi("Training Records", f"{n_rows:,}")
    with c2:
        kpi("Features", n_feats)
    with c3:
        kpi("Target Variable", TARGET)
    with c4:
        kpi("Dataset Type", "Synthetic")

    left, right = st.columns(2)
    with left:
        st.subheader("Project Objectives")
        st.markdown(
            "- Predict the likelihood of employee attrition from available features.\n"
            "- Support HR review with rule-based retention discussion prompts.\n"
            "- Explore attrition patterns across groups through interactive analytics.\n"
            "- Evaluate the model transparently on an independent test set."
        )
    with right:
        st.subheader("How Machine Learning Helps")
        st.write(
            "Instead of relying on a single factor, a machine learning model learns "
            "combinations of many attributes from historical records and outputs a risk "
            "probability for each employee. Teams can use this to prioritise conversations "
            "and review workplace conditions. Predictions are statistical estimates and "
            "should support, not replace, human judgement."
        )
    st.caption("Target encoding: Stayed = 0, Left = 1.")


def page_prediction():
    st.title("🔮 Attrition Prediction")
    synthetic_banner()

    try:
        model = load_model()
        features = load_feature_names()
    except Exception as e:
        st.error(f"Could not load model or feature names: {e}")
        return
    if train_raw is None:
        return

    st.subheader("Employee Details")
    inputs = {}
    with st.form("prediction_form"):
        cols = st.columns(3)
        for i, feat in enumerate(features):
            with cols[i % 3]:
                if feat in train_raw.columns:
                    series = train_raw[feat].dropna()
                    if pd.api.types.is_numeric_dtype(series):
                        is_int = pd.api.types.is_integer_dtype(
                            train_raw[feat]
                        ) or np.allclose(series, series.round())
                        if is_int:
                            inputs[feat] = st.number_input(
                                feat,
                                min_value=int(series.min()),
                                max_value=int(series.max()),
                                value=int(series.median()),
                                step=1,
                            )
                        else:
                            inputs[feat] = st.number_input(
                                feat,
                                min_value=float(series.min()),
                                max_value=float(series.max()),
                                value=float(series.median()),
                            )
                    else:
                        options = sorted(series.astype(str).unique().tolist())
                        default = options.index(series.astype(str).mode().iloc[0])
                        inputs[feat] = st.selectbox(feat, options, index=default)
                else:
                    inputs[feat] = st.text_input(feat)
        submitted = st.form_submit_button("Predict Attrition Risk", type="primary")

    if submitted:
        X = pd.DataFrame([inputs])[features]
        try:
            prob = float(model.predict_proba(X)[0][1])
            pred = int(model.predict(X)[0])
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            return
        level, css = risk_level(prob)
        st.session_state["result"] = {
            "inputs": inputs,
            "prob": prob,
            "pred": pred,
            "level": level,
            "css": css,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    res = st.session_state.get("result")
    if not res:
        st.info("Fill in the employee details and click **Predict Attrition Risk**.")
        return

    st.subheader("Prediction Results")
    c1, c2, c3 = st.columns(3)
    with c1:
        kpi("Attrition Probability", f"{res['prob']*100:.1f}%")
    with c2:
        kpi("Risk Level", res["level"], res["css"])
    with c3:
        kpi(
            "Model Prediction",
            "Likely to Leave" if res["pred"] == 1 else "Likely to Stay",
        )
    st.progress(min(max(res["prob"], 0.0), 1.0))
    st.caption(
        "Risk bands are illustrative application thresholds, not scientifically validated cutoffs: "
        "Low < 40%, Medium 40% to < 65%, High ≥ 65%."
    )

    st.subheader("Retention Review Suggestions")
    for tip in build_suggestions(res["inputs"], train_raw):
        st.markdown(f"- {tip}")
    st.caption("Suggestions are rule-based discussion prompts, not causal conclusions.")

    report = {
        **res["inputs"],
        "Attrition Probability (%)": round(res["prob"] * 100, 2),
        "Risk Level": res["level"],
        "Model Prediction": "Left" if res["pred"] == 1 else "Stayed",
        "Generated At": res["time"],
        "Note": "Synthetic data; illustrative risk bands",
    }
    st.download_button(
        "⬇️ Download Prediction Report (CSV)",
        pd.DataFrame([report]).to_csv(index=False).encode("utf-8"),
        file_name="attrition_prediction_report.csv",
        mime="text/csv",
    )


def rate_table(df, col, order=None):
    g = (
        df.groupby(col, observed=True)["Attrition_num"]
        .agg(["mean", "count"])
        .reset_index()
    )
    g.columns = [col, "Attrition Rate (%)", "Employees"]
    g["Attrition Rate (%)"] = (g["Attrition Rate (%)"] * 100).round(2)
    g[col] = g[col].astype(str)
    if order:
        g[col] = pd.Categorical(
            g[col], categories=[o for o in order if o in set(g[col])], ordered=True
        )
        g = g.sort_values(col)
    return g


def rate_chart(df, col, title, order=None, horizontal=False):
    t = rate_table(df, col, order)
    if t.empty:
        st.info("No data for the current filters.")
        return
    if HAS_PLOTLY:
        fig = px.bar(
            t,
            x="Attrition Rate (%)" if horizontal else col,
            y=col if horizontal else "Attrition Rate (%)",
            text="Attrition Rate (%)",
            hover_data=["Employees"],
            title=title,
            orientation="h" if horizontal else "v",
        )
        fig.update_traces(texttemplate="%{text:.1f}%", marker_color="#2b6cb0")
        fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=380)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown(f"**{title}**")
        st.bar_chart(t.set_index(col)["Attrition Rate (%)"])


def page_analytics():
    st.title("📈 Analytics Dashboard")
    synthetic_banner()
    if train is None or "Attrition_num" not in train.columns:
        st.error("Training data or the Attrition column is unavailable.")
        return

    df = train.copy()

    with st.expander("🔎 Filters", expanded=False):
        f1, f2, f3 = st.columns(3)
        if "Gender" in df.columns:
            sel = f1.multiselect(
                "Gender", sorted(df["Gender"].dropna().astype(str).unique())
            )
            if sel:
                df = df[df["Gender"].astype(str).isin(sel)]
        if "Job Role" in df.columns:
            sel = f2.multiselect(
                "Job Role", sorted(df["Job Role"].dropna().astype(str).unique())
            )
            if sel:
                df = df[df["Job Role"].astype(str).isin(sel)]
        if "Job Level" in df.columns:
            sel = f3.multiselect(
                "Job Level", sorted(df["Job Level"].dropna().astype(str).unique())
            )
            if sel:
                df = df[df["Job Level"].astype(str).isin(sel)]

    if df.empty:
        st.warning("No records match the selected filters.")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        kpi("Records (filtered)", f"{len(df):,}")
    with c2:
        kpi("Left", f"{int(df['Attrition_num'].sum()):,}")
    with c3:
        kpi("Overall Attrition Rate", f"{df['Attrition_num'].mean()*100:.1f}%")
    st.markdown("")

    st.subheader("1. Overall Attrition Distribution")
    dist = df["Outcome"].value_counts().reset_index()
    dist.columns = ["Outcome", "Count"]
    dist["Percentage (%)"] = (dist["Count"] / dist["Count"].sum() * 100).round(2)
    a, b = st.columns([2, 1])
    with a:
        if HAS_PLOTLY:
            fig = px.pie(
                dist,
                names="Outcome",
                values="Count",
                hole=0.5,
                title="Stayed vs Left",
                color="Outcome",
                color_discrete_map={"Stayed": "#38b2ac", "Left": "#e03131"},
            )
            fig.update_traces(textinfo="label+percent+value")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(dist.set_index("Outcome")["Count"])
    with b:
        st.dataframe(dist, hide_index=True, use_container_width=True)

    def pair(c_left, c_right):
        return st.columns(2)

    l, r = st.columns(2)
    with l:
        if "Gender" in df.columns:
            st.subheader("2. Attrition by Gender")
            rate_chart(df, "Gender", "Attrition Rate by Gender")
    with r:
        if "Job Role" in df.columns:
            st.subheader("3. Attrition by Job Role")
            rate_chart(df, "Job Role", "Attrition Rate by Job Role")

    l, r = st.columns(2)
    with l:
        if "Job Level" in df.columns:
            st.subheader("4. Attrition by Job Level")
            rate_chart(df, "Job Level", "Attrition Rate by Job Level")
    with r:
        if "Work-Life Balance" in df.columns:
            st.subheader("5. Attrition by Work-Life Balance")
            rate_chart(
                df,
                "Work-Life Balance",
                "Attrition Rate by Work-Life Balance",
                order=["Poor", "Fair", "Good", "Excellent"],
            )

    l, r = st.columns(2)
    with l:
        if "Remote Work" in df.columns:
            st.subheader("6. Attrition by Remote Work")
            rate_chart(df, "Remote Work", "Attrition Rate by Remote Work")
    with r:
        if "Overtime" in df.columns:
            st.subheader("7. Attrition by Overtime")
            rate_chart(df, "Overtime", "Attrition Rate by Overtime")

    l, r = st.columns(2)
    with l:
        if "Marital Status" in df.columns:
            st.subheader("8. Attrition by Marital Status")
            rate_chart(df, "Marital Status", "Attrition Rate by Marital Status")

    with r:
        if "Age" in df.columns:
            st.subheader("9. Attrition by Age Group")
            age = pd.to_numeric(df["Age"], errors="coerce")
            bins = [0, 24, 34, 44, 54, 200]
            labels = ["Under 25", "25–34", "35–44", "45–54", "55+"]
            tmp = df.assign(
                **{"Age Group": pd.cut(age, bins=bins, labels=labels)}
            ).dropna(subset=["Age Group"])
            rate_chart(tmp, "Age Group", "Attrition Rate by Age Group", order=labels)

    if "Monthly Income" in df.columns:
        st.subheader("10. Attrition by Monthly Income Range")
        inc = pd.to_numeric(df["Monthly Income"], errors="coerce")
        try:
            groups = pd.qcut(inc, q=5, duplicates="drop")
            labels = [
                f"{int(iv.left):,} – {int(iv.right):,}" for iv in groups.cat.categories
            ]
            groups = groups.cat.rename_categories(labels)
            tmp = df.assign(**{"Income Range": groups}).dropna(subset=["Income Range"])
            rate_chart(
                tmp,
                "Income Range",
                "Attrition Rate by Monthly Income Range (quintiles)",
                order=labels,
            )
        except ValueError:
            st.info("Not enough variation in Monthly Income to create ranges.")

    st.caption(
        "Rates are computed from the filtered synthetic training data. They describe the dataset only, "
        "not real employees, and do not imply causation."
    )


def page_performance():
    st.title("🎯 Model Performance")
    synthetic_banner()
    st.write(
        "Final results of the Gradient Boosting model on the **independent test set** "
        "(14,900 records)."
    )

    cols = st.columns(5)
    for c, (name, val) in zip(cols, METRICS.items()):
        with c:
            kpi(name, f"{val:.2f}%")

    st.subheader("Confusion Matrix")
    classes = ["Stayed (0)", "Left (1)"]
    l, r = st.columns([3, 2])
    with l:
        if HAS_PLOTLY:
            fig = px.imshow(
                CONF_MATRIX,
                text_auto=True,
                color_continuous_scale="Blues",
                x=[f"Predicted {c}" for c in classes],
                y=[f"Actual {c}" for c in classes],
                title="Confusion Matrix (Test Set)",
            )
            fig.update_layout(
                height=420, xaxis_title="Predicted Class", yaxis_title="Actual Class"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.dataframe(
                pd.DataFrame(
                    CONF_MATRIX,
                    index=[f"Actual {c}" for c in classes],
                    columns=[f"Predicted {c}" for c in classes],
                )
            )
    with r:
        tn, fp, fn, tp = CONF_MATRIX.ravel()
        st.markdown(
            f"- **True Negatives:** {tn:,} (correctly predicted Stayed)\n"
            f"- **False Positives:** {fp:,} (predicted Left, actually Stayed)\n"
            f"- **False Negatives:** {fn:,} (predicted Stayed, actually Left)\n"
            f"- **True Positives:** {tp:,} (correctly predicted Left)\n"
            f"- **Total:** {CONF_MATRIX.sum():,}"
        )

    st.subheader("Metric Explanations")
    st.markdown(
        "- **Accuracy:** share of all predictions that were correct.\n"
        "- **Precision:** of employees predicted to leave, the share who actually left.\n"
        "- **Recall:** of employees who actually left, the share the model identified.\n"
        "- **F1-Score:** harmonic mean of precision and recall, balancing the two.\n"
        "- **ROC-AUC:** how well the model ranks leavers above stayers across all thresholds "
        "(0.5 = random, 1.0 = perfect)."
    )

    st.subheader("Interpretation & Limitations")
    st.markdown(
        "- The model correctly separates leavers from stayers better than chance, but roughly one in four "
        "predictions is still wrong, so results should support human review rather than replace it.\n"
        "- The dataset is synthetic, so these scores may not carry over to real organisations.\n"
        "- The risk bands used on the prediction page are illustrative thresholds, not validated cutoffs."
    )


if page.endswith("Home"):
    page_home()
elif page.endswith("Attrition Prediction"):
    page_prediction()
elif page.endswith("Analytics Dashboard"):
    page_analytics()
else:
    page_performance()