from __future__ import annotations

import hashlib
from io import BytesIO
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="JobGuard | Fake Job Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
      :root { --ink:#102a32; --muted:#637780; --teal:#087f73; --mint:#dff4ed; --paper:#f5f8f6; }
      .stApp { background:var(--paper); color:var(--ink); font-family:'DM Sans',sans-serif; }
      h1,h2,h3 { font-family:'Manrope',sans-serif !important; letter-spacing:-.035em; }
      .block-container { padding-top:2rem; max-width:1440px; }
      [data-testid="stSidebar"] { background:#102a32; }
      [data-testid="stSidebar"] * { color:#eef8f4 !important; }
      .hero { background:linear-gradient(115deg,#102a32 0%,#14544e 68%,#0d8273 100%); padding:2.15rem 2.4rem; border-radius:22px; color:white; margin-bottom:1.4rem; box-shadow:0 14px 32px #102a321a; }
      .hero h1 { font-size:2.5rem; line-height:1.12; margin:.3rem 0 .65rem; color:white; }
      .hero p { color:#d7eae4; max-width:780px; font-size:1.03rem; margin:0; }
      .eyebrow { color:#8be0c7; text-transform:uppercase; letter-spacing:.13em; font-size:.73rem; font-weight:700; }
      .section-label { color:var(--teal); text-transform:uppercase; letter-spacing:.12em; font-size:.72rem; font-weight:700; margin-bottom:.25rem; }
      div[data-testid="stMetric"] { background:#fff; padding:1rem 1.15rem; border:1px solid #e2ebe7; border-radius:15px; box-shadow:0 5px 14px #102a3208; }
      div[data-testid="stMetricLabel"] p { color:var(--muted); font-weight:600; }
      .note { padding:1rem 1.1rem; border-radius:12px; background:#eaf5f1; color:#244c48; border:1px solid #d6e9e2; font-size:.91rem; }
      .small-note { color:#637780; font-size:.87rem; }
      .stButton button[kind="primary"] { background:#087f73; border:0; border-radius:10px; font-weight:700; }
      .stButton button[kind="primary"]:hover { background:#06675e; }
    </style>
    """,
    unsafe_allow_html=True,
)


TEXT_FIELDS = ["title", "company_profile", "description", "requirements", "benefits"]
FLAG_FIELDS = ["telecommuting", "has_company_logo", "has_questions"]
POSITIVE_WORDS = {"1", "fake", "fraudulent", "fraud", "scam", "yes", "true"}
NEGATIVE_WORDS = {"0", "genuine", "real", "legitimate", "legit", "no", "false"}


def read_csv(upload: Any) -> pd.DataFrame:
    raw = upload.getvalue()
    try:
        return pd.read_csv(BytesIO(raw))
    except UnicodeDecodeError:
        return pd.read_csv(BytesIO(raw), encoding="latin-1")


def clean_text_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series("", index=frame.index, dtype="object")
    return frame[column].fillna("").astype(str).replace({"nan": "", "None": ""})


def combine_text(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    parts = [clean_text_column(frame, column) for column in columns if column in frame]
    if not parts:
        return pd.Series("", index=frame.index, dtype="object")
    combined = parts[0]
    for part in parts[1:]:
        combined = combined + " " + part
    return combined.str.replace(r"\s+", " ", regex=True).str.strip()


def normalize_target(values: pd.Series) -> pd.Series:
    labels = values.astype(str).str.strip().str.lower()
    mapped = labels.map(lambda value: 1 if value in POSITIVE_WORDS else 0 if value in NEGATIVE_WORDS else np.nan)
    if mapped.notna().all() and mapped.nunique() == 2:
        return mapped.astype(int)
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().all() and set(numeric.unique()) == {0, 1}:
        return numeric.astype(int)
    raise ValueError("The selected target must contain two classes, such as 0/1 or genuine/fake.")


def engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=frame.index)
    for column in TEXT_FIELDS:
        values = clean_text_column(frame, column)
        label = column.replace("_", " ")
        result[f"{label} · words"] = values.str.split().str.len().astype(float)
        result[f"{label} · characters"] = values.str.len().astype(float)
    for column in FLAG_FIELDS:
        if column in frame.columns:
            result[column.replace("_", " ")] = pd.to_numeric(frame[column], errors="coerce")
    if "location" in frame.columns:
        result["location · provided"] = clean_text_column(frame, "location").str.strip().ne("").astype(float)
    result = result.replace([np.inf, -np.inf], np.nan)
    result = result.fillna(result.median(numeric_only=True)).fillna(0)
    return result


def make_model() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=3000, stop_words="english", ngram_range=(1, 2), min_df=2)),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear", random_state=42)),
        ]
    )


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig = px.imshow(
        matrix,
        text_auto=True,
        x=["Genuine", "Fake"],
        y=["Genuine", "Fake"],
        color_continuous_scale=[[0, "#dff4ed"], [0.55, "#32aa92"], [1, "#075b57"]],
        labels={"x": "Predicted label", "y": "Actual label", "color": "Listings"},
        aspect="auto",
    )
    fig.update_layout(margin=dict(l=10, r=10, t=20, b=10), height=360, coloraxis_showscale=False)
    return fig


def plot_top_terms(model: Pipeline) -> go.Figure:
    words = model.named_steps["tfidf"].get_feature_names_out()
    weights = model.named_steps["classifier"].coef_[0]
    order = np.argsort(weights)
    selected = np.concatenate([order[:12], order[-12:]])
    chart_data = pd.DataFrame({"Term": words[selected], "Weight": weights[selected]})
    chart_data["Signal"] = np.where(chart_data["Weight"] > 0, "Fake signal", "Genuine signal")
    fig = px.bar(chart_data.sort_values("Weight"), x="Weight", y="Term", orientation="h", color="Signal",
                 color_discrete_map={"Fake signal": "#e47f4e", "Genuine signal": "#087f73"},
                 labels={"Weight": "Logistic regression weight", "Term": "Text feature"})
    fig.update_layout(height=510, margin=dict(l=10, r=10, t=15, b=10), legend_title_text="")
    return fig


def render_overview(frame: pd.DataFrame, target: pd.Series) -> None:
    genuine = int((target == 0).sum())
    fake = int((target == 1).sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Job postings", f"{len(frame):,}")
    c2.metric("Genuine", f"{genuine:,}", f"{genuine / len(frame):.1%} of data")
    c3.metric("Flagged as fake", f"{fake:,}", f"{fake / len(frame):.1%} of data")
    c4.metric("Input columns", str(len(frame.columns)))

    left, right = st.columns([1, 1.25], gap="large")
    with left:
        st.markdown("#### Class balance")
        distribution = pd.DataFrame({"Class": ["Genuine", "Fake"], "Postings": [genuine, fake]})
        chart = px.pie(distribution, names="Class", values="Postings", hole=0.68,
                       color="Class", color_discrete_map={"Genuine": "#087f73", "Fake": "#f19b62"})
        chart.update_layout(height=330, margin=dict(l=0, r=0, t=10, b=10), legend_title_text="")
        chart.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(chart, use_container_width=True)
    with right:
        st.markdown("#### Where data is missing")
        missing = (frame.isna().mean() * 100).sort_values(ascending=False)
        missing = missing[missing > 0].head(12)
        if missing.empty:
            st.success("No missing values found in the uploaded columns.")
        else:
            missing_df = missing.rename("Missing (%)").rename_axis("Column").reset_index()
            chart = px.bar(missing_df, x="Missing (%)", y="Column", orientation="h", text_auto=".1f",
                           color="Missing (%)", color_continuous_scale=["#bfe8dc", "#087f73"])
            chart.update_layout(height=330, margin=dict(l=0, r=0, t=10, b=10), yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
            st.plotly_chart(chart, use_container_width=True)

    st.markdown("#### Numeric feature correlation")
    numeric = frame.select_dtypes(include=np.number).drop(columns=["job_id"], errors="ignore")
    numeric = numeric.loc[:, numeric.nunique(dropna=True) > 1]
    if numeric.shape[1] >= 2:
        corr = numeric.corr(numeric_only=True)
        heat = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                         labels={"x": "Feature", "y": "Feature", "color": "Correlation"}, aspect="auto")
        heat.update_layout(height=max(360, 38 * len(corr.columns)), margin=dict(l=5, r=5, t=15, b=5))
        st.plotly_chart(heat, use_container_width=True)
    else:
        st.info("Add at least two numeric columns to see feature correlations.")

    st.markdown("#### View labeled postings")
    selected_label = st.session_state.get("label_to_view", "Genuine")
    selected_value = 0 if selected_label == "Genuine" else 1
    selected_frame = frame.loc[target == selected_value]
    st.caption(f"Showing {len(selected_frame):,} {selected_label.lower()} postings. Training always uses both labels.")
    st.dataframe(selected_frame.head(12), use_container_width=True, hide_index=True)


def render_pca(frame: pd.DataFrame, target: pd.Series) -> None:
    features = engineered_features(frame)
    if len(features) < 3 or features.shape[1] < 2:
        st.info("At least three postings and two usable features are needed for PCA.")
        return
    sample_n = min(3500, len(features))
    if sample_n < len(features):
        sample = features.sample(sample_n, random_state=42)
        labels = target.loc[sample.index]
    else:
        sample = features
        labels = target
    scaled = StandardScaler().fit_transform(sample)
    pca = PCA(n_components=2, random_state=42)
    projection = pca.fit_transform(scaled)
    explained = pca.explained_variance_ratio_
    scatter_df = pd.DataFrame({"PC 1": projection[:, 0], "PC 2": projection[:, 1],
                               "Class": labels.map({0: "Genuine", 1: "Fake"}).values})
    fig = px.scatter(scatter_df, x="PC 1", y="PC 2", color="Class", opacity=.68,
                     color_discrete_map={"Genuine": "#087f73", "Fake": "#e47f4e"},
                     render_mode="webgl")
    fig.update_traces(marker={"size": 6})
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=15, b=10), legend_title_text="Posting label")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"PCA uses posting text lengths, word counts, and available listing flags. The two components explain {explained.sum():.1%} of variance in this feature set. Showing {sample_n:,} postings for readability.")


def render_results() -> None:
    result = st.session_state.get("training_result")
    if not result:
        return
    frame = result["frame"]
    target = result["target"]
    model = result["model"]
    columns = result["text_columns"]
    X_test_text = result["X_test_text"]
    y_test = result["y_test"]
    scores = result["scores"]

    st.markdown('<div class="section-label">Model results</div>', unsafe_allow_html=True)
    st.header("Explore your dataset")
    overview_tab, performance_tab, pca_tab, predict_tab = st.tabs(["Dataset overview", "Model performance", "PCA projection", "Try a prediction"])

    with overview_tab:
        render_overview(frame, target)

    with performance_tab:
        st.markdown("#### Held-out test performance")
        st.caption("The vectorizer is fitted on the training split only. Test postings stay unseen until evaluation.")
        threshold = st.slider("Fake classification threshold", 0.10, 0.90, 0.50, 0.01,
                              help="Higher values require stronger model confidence before a posting is labeled fake.")
        predictions = (scores >= threshold).astype(int)
        report = classification_report(y_test, predictions, labels=[0, 1], target_names=["Genuine", "Fake"], output_dict=True, zero_division=0)
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy", f"{accuracy_score(y_test, predictions):.1%}")
        m2.metric("Balanced accuracy", f"{balanced_accuracy_score(y_test, predictions):.1%}")
        m3.metric("Fake precision", f"{precision_score(y_test, predictions, zero_division=0):.1%}")
        m4.metric("Fake recall", f"{recall_score(y_test, predictions, zero_division=0):.1%}")
        m5.metric("Fake F1", f"{f1_score(y_test, predictions, zero_division=0):.1%}")
        left, right = st.columns(2, gap="large")
        with left:
            st.markdown("#### Confusion matrix heatmap")
            st.plotly_chart(plot_confusion(y_test, predictions), use_container_width=True)
        with right:
            st.markdown("#### ROC curve")
            fpr, tpr, _ = roc_curve(y_test, scores)
            roc = go.Figure()
            roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC {roc_auc_score(y_test, scores):.3f})",
                                     line={"color": "#087f73", "width": 3}, fill="tozeroy", fillcolor="rgba(8,127,115,.10)"))
            roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random baseline", line={"color": "#9aa9aa", "dash": "dash"}))
            roc.update_layout(xaxis_title="False positive rate", yaxis_title="True positive rate", height=360,
                              margin=dict(l=10, r=10, t=20, b=10), legend_title_text="")
            st.plotly_chart(roc, use_container_width=True)
        st.markdown("#### Precision, recall and F1 by class")
        report_df = pd.DataFrame(report).T.loc[["Genuine", "Fake"], ["precision", "recall", "f1-score"]].rename(columns={"f1-score": "F1 score"})
        st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)
        st.markdown("#### Text signals learned by the model")
        st.plotly_chart(plot_top_terms(model), use_container_width=True)
        st.caption("Positive weights push a prediction toward fake; negative weights push it toward genuine. These are associations learned from the uploaded labels, not proof of fraud.")
        with st.expander("How to read these results"):
            st.write("Fake recall measures the share of fake listings caught. Fake precision measures how many listings flagged by the model are actually fake in the labeled test set. Tune the threshold to explore that trade-off; this is a screening aid, not a guarantee.")

    with pca_tab:
        st.markdown("#### PCA: how postings spread across two components")
        st.write("Each dot is one posting projected from its text-length, word-count and available listing-flag features. Separation hints at patterns in the supplied data; it is not itself a prediction boundary.")
        render_pca(frame, target)

    with predict_tab:
        render_prediction(model, columns)


def render_prediction(model: Pipeline, text_columns: list[str]) -> None:
    st.markdown("#### Score a single posting")
    st.caption("Enter the fields used to train this model. Empty fields are allowed.")
    with st.form("single_prediction"):
        values: dict[str, str] = {}
        for column in text_columns:
            label = column.replace("_", " ").title()
            if column in {"description", "requirements", "benefits", "company_profile"}:
                values[column] = st.text_area(label, height=100 if column == "description" else 72)
            else:
                values[column] = st.text_input(label)
        submitted = st.form_submit_button("Check posting", type="primary")
    if submitted:
        row = pd.DataFrame([values])
        probability = float(model.predict_proba(combine_text(row, text_columns))[0, 1])
        if probability >= .5:
            st.error(f"Model assessment: likely fake · {probability:.1%} estimated probability")
        else:
            st.success(f"Model assessment: likely genuine · {(1 - probability):.1%} estimated probability")
        st.caption("Probability is a model score learned from the uploaded dataset, not a verification of the employer or job.")

    st.divider()
    st.markdown("#### Score a batch CSV")
    batch = st.file_uploader("Upload unlabeled postings as CSV", type=["csv"], key="batch_upload")
    if batch:
        try:
            batch_frame = read_csv(batch)
            text = combine_text(batch_frame, text_columns)
            probability = model.predict_proba(text)[:, 1]
            scored = batch_frame.copy()
            scored["fake_probability"] = probability
            scored["predicted_label"] = np.where(probability >= .5, "Fake", "Genuine")
            st.dataframe(scored.head(20), use_container_width=True, hide_index=True)
            st.download_button("Download scored CSV", scored.to_csv(index=False).encode("utf-8-sig"),
                               file_name="job_posting_predictions.csv", mime="text/csv")
        except Exception as exc:
            st.error(f"Could not score this file: {exc}")


def main() -> None:
    with st.sidebar:
        st.markdown("## 🛡️ JobGuard")
        st.caption("Fake job posting detection")
        st.markdown("---")
        st.markdown("**Workflow**")
        st.markdown("1. Upload a labeled CSV\n2. Choose the target column\n3. Train and review results\n4. Score new postings")
        st.markdown("---")
        st.caption("The uploaded dataset stays in this app session. Model quality depends on the labels and source data.")

    st.markdown(
        '<div class="hero"><div class="eyebrow">Machine learning · job safety</div>'
        '<h1>Spot risky job postings<br>before they reach applicants.</h1>'
        '<p>Train a transparent text classifier on your labeled dataset, inspect its performance, and score new listings in seconds.</p></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="section-label">Step 1 · Your data</div>', unsafe_allow_html=True)
    st.subheader("Upload a labeled job-posting dataset")
    st.markdown('<div class="note">Your CSV needs a label column plus at least one text field such as <b>title</b>, <b>description</b>, or <b>company_profile</b>. Labels can be 0/1 or genuine/fake.</div>', unsafe_allow_html=True)
    upload = st.file_uploader("Choose CSV", type=["csv"], help="For this project, use the fraudulent column as the target.")
    if upload is None:
        st.info("Upload a labeled CSV to begin. The supplied project dataset uses the `fraudulent` column, where 1 means fake and 0 means genuine.")
        with st.expander("What is included in the analysis?"):
            st.markdown("- Dataset summary, missing-value chart, class balance, and numeric correlation heatmap\n- TF-IDF + class-balanced logistic regression\n- Stratified train/test split, classification metrics, ROC curve, and confusion-matrix heatmap\n- PCA projection of engineered posting features\n- Single-posting and batch-CSV predictions")
        return

    try:
        frame = read_csv(upload)
    except Exception as exc:
        st.error(f"Could not read this CSV: {exc}")
        return
    if frame.empty:
        st.error("This CSV has no data rows.")
        return
    st.caption(f"Loaded {len(frame):,} rows and {len(frame.columns)} columns · {upload.name}")
    available_text = [column for column in TEXT_FIELDS if column in frame.columns]
    if not available_text:
        likely_text = [column for column in frame.select_dtypes(include=["object", "string"]).columns if column not in {"job_id"}]
        available_text = likely_text[:5]
    if not available_text:
        st.error("No text columns found. Include at least one posting text field.")
        return

    target_column = "fraudulent"
    if target_column not in frame.columns:
        st.error("This app requires a `fraudulent` column: 0 for Genuine and 1 for Fraudulent.")
        return
    st.markdown("#### Training labels")
    st.radio(
        "Choose the postings you want to view",
        options=["Genuine", "Fraudulent"],
        horizontal=True,
        key="label_to_view",
        format_func=lambda value: "0 — Genuine job posting" if value == "Genuine" else "1 — Fraudulent job posting",
        help="This selects the label shown in the Dataset Overview. The model still trains on both labels.",
    )
    text_columns = st.multiselect("Text fields to use", options=[column for column in frame.columns if column != target_column],
                                  default=[column for column in available_text if column != target_column],
                                  help="These are the job-posting details the model reads. They are input fields, not labels.")
    if not text_columns:
        st.warning("Choose at least one text field before training.")
        return
    st.caption("Model text fields: " + ", ".join(text_columns))

    file_bytes = upload.getvalue()
    file_id = f"{upload.name}:{hashlib.sha256(file_bytes).hexdigest()}"
    config_id = (file_id, target_column, tuple(text_columns))
    prior = st.session_state.get("training_config_id")
    if prior != config_id:
        st.session_state["training_config_id"] = config_id
        st.session_state.pop("training_result", None)

    if st.button("Train model and analyze dataset", type="primary"):
        try:
            y = normalize_target(frame[target_column])
            if y.value_counts().min() < 5:
                raise ValueError("For a stratified 80/20 split, each class needs at least five labeled examples.")
            raw_text = combine_text(frame, text_columns)
            if not raw_text.str.strip().any():
                raise ValueError("The selected text fields are empty for every row.")
            X_train, X_test, y_train, y_test = train_test_split(
                raw_text, y, test_size=0.2, random_state=42, stratify=y
            )
            model = make_model()
            model.fit(X_train, y_train)
            scores = model.predict_proba(X_test)[:, 1]
            st.session_state["training_result"] = {
                "frame": frame.copy(), "target": y, "text_columns": text_columns,
                "model": model, "X_test_text": X_test, "y_test": y_test.to_numpy(), "scores": scores,
            }
        except Exception as exc:
            st.error(f"Training could not complete: {exc}")

    if st.session_state.get("training_result"):
        render_results()

    st.markdown("---")
    st.caption("JobGuard is an educational AIML project. Review flagged listings independently before taking action.")


if __name__ == "__main__":
    main()
