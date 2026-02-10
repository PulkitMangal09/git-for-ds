# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import json
import os
import time
import optuna
import optuna.visualization as vis
from datetime import datetime

# -----------------------
# Page config + constants
# -----------------------
st.set_page_config(
    page_title="Iris — Streamlit + Optuna demo",
    layout="centered"
)

MODEL_PATH = "rf_iris_best.joblib"
META_PATH = "model_meta.json"
STUDY_DB = "rf_iris_study.db"

# Unique study name per run to avoid Optuna conflicts
STUDY_NAME = f"iris_optimization_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

RANDOM_STATE = 42

# -----------------------
# Utility: load Iris (cached)
# -----------------------
@st.cache_data(show_spinner=False)
def load_iris_df():
    iris = datasets.load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name="target")
    return X, y, iris.target_names.tolist(), iris.feature_names

X_all, y_all, target_names, feature_names = load_iris_df()

# -----------------------
# Train / Test split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_all,
    y_all,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y_all
)

# -----------------------
# Persistence helpers
# -----------------------
def save_model_and_meta(model, meta, model_path=MODEL_PATH, meta_path=META_PATH):
    joblib.dump(model, model_path)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

def load_model_and_meta(model_path=MODEL_PATH, meta_path=META_PATH):
    if os.path.exists(model_path) and os.path.exists(meta_path):
        model = joblib.load(model_path)
        with open(meta_path, "r") as f:
            meta = json.load(f)
        return model, meta
    return None, None

persisted_model, persisted_meta = load_model_and_meta()

# -----------------------
# App UI
# -----------------------
st.title("Interactive Iris Classifier — Streamlit + Optuna")
st.write(
    "Change feature values, train manually, or run Optuna hyperparameter tuning. "
    "This app is designed for demos and teaching — not production use."
)

# Dataset preview
with st.expander("Dataset preview & stats", expanded=False):
    st.dataframe(X_train.head())
    stats = X_train.describe().T[["min", "mean", "max", "std"]]
    st.dataframe(stats)

# -----------------------
# Sidebar inputs
# -----------------------
st.sidebar.header("Input features")

feature_mins = X_train.min()
feature_maxs = X_train.max()
feature_means = X_train.mean()

input_dict = {}
for feat in feature_names:
    input_dict[feat] = st.sidebar.slider(
        feat,
        float(feature_mins[feat]),
        float(feature_maxs[feat]),
        float(feature_means[feat]),
        step=0.01
    )

st.sidebar.markdown("---")
mode = st.sidebar.radio(
    "Mode",
    ["Predict only (use saved model)", "Manual Tuning", "Auto-Tuning (Optuna)"]
)

# -----------------------
# Prediction block
# -----------------------
st.header("Prediction")

if st.button("Predict current inputs"):
    X_input = np.array([list(input_dict.values())])

    model_to_use = None
    model_source = None

    if persisted_model is not None:
        model_to_use = persisted_model
        model_source = "saved best model"
    elif "manual_model" in st.session_state:
        model_to_use = st.session_state.manual_model
        model_source = "manual model (this session)"
    else:
        st.warning("No trained model available.")
        st.stop()

    pred = model_to_use.predict(X_input)[0]
    proba = model_to_use.predict_proba(X_input)[0]

    st.write(f"Using model: **{model_source}**")
    st.write(f"Predicted class: **{target_names[int(pred)]}**")

    st.subheader("Class probabilities")
    st.table(pd.DataFrame([proba], columns=target_names))

# -----------------------
# MODE: Manual Tuning
# -----------------------
if mode == "Manual Tuning":
    st.header("Manual Tuning")

    with st.form("manual_form"):
        n_estimators = st.slider("n_estimators", 10, 300, 100, step=10)
        max_depth = st.slider("max_depth (0 = None)", 0, 50, 5)
        criterion = st.selectbox("criterion", ["gini", "entropy"])
        save_model = st.checkbox("Save model persistently", value=False)
        submitted = st.form_submit_button("Train")

    if submitted:
        md = None if max_depth == 0 else int(max_depth)

        with st.spinner("Training RandomForest..."):
            clf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=md,
                criterion=criterion,
                random_state=RANDOM_STATE
            )
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
            acc = accuracy_score(y_test, preds)

        st.success(f"Test accuracy: {acc * 100:.2f}%")
        st.text(classification_report(y_test, preds, target_names=target_names))

        st.session_state.manual_model = clf

        if save_model:
            meta = {
                "saved_at": datetime.utcnow().isoformat(),
                "mode": "manual",
                "test_accuracy": acc,
                "feature_names": feature_names,
                "target_names": target_names
            }
            save_model_and_meta(clf, meta)
            st.success("Model saved.")

# -----------------------
# MODE: Auto-Tuning (Optuna)
# -----------------------
elif mode == "Auto-Tuning (Optuna)":
    st.header("Auto-Tuning with Optuna")

    n_trials = st.sidebar.slider("Number of trials", 5, 80, 20)

    if st.button("Run Optimization"):
        storage_str = f"sqlite:///{STUDY_DB}"

        study = optuna.create_study(
            direction="maximize",
            storage=storage_str,
            study_name=STUDY_NAME
        )

        progress = st.progress(0)
        status = st.empty()

        def objective(trial):
            clf = RandomForestClassifier(
                n_estimators=trial.suggest_int("n_estimators", 10, 300),
                max_depth=trial.suggest_int("max_depth", 2, 30),
                criterion=trial.suggest_categorical("criterion", ["gini", "entropy"]),
                random_state=RANDOM_STATE
            )
            clf.fit(X_train, y_train)
            return accuracy_score(y_test, clf.predict(X_test))

        def progress_cb(study, trial):
            completed = len(study.trials)
            progress.progress(min(completed / n_trials, 1.0))
            status.text(f"Trials completed: {completed}/{n_trials}")

        start = time.time()
        with st.spinner("Running Optuna optimization..."):
            study.optimize(objective, n_trials=n_trials, callbacks=[progress_cb])

        duration = time.time() - start
        st.success(
            f"Done in {duration:.1f}s — best accuracy: {study.best_value * 100:.2f}%"
        )

        st.json(study.best_params)

        best = study.best_params
        best_model = RandomForestClassifier(
            n_estimators=int(best["n_estimators"]),
            max_depth=int(best["max_depth"]),
            criterion=best["criterion"],
            random_state=RANDOM_STATE
        )
        best_model.fit(X_train, y_train)

        preds = best_model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        meta = {
            "saved_at": datetime.utcnow().isoformat(),
            "mode": "optuna",
            "test_accuracy": acc,
            "study_name": study.study_name,
            "feature_names": feature_names,
            "target_names": target_names
        }

        save_model_and_meta(best_model, meta)
        st.session_state.best_model = best_model

        st.write(f"Persisted best model — test accuracy: {acc * 100:.2f}%")

        # Trial history
        df = study.trials_dataframe()
        st.subheader("Top trials")
        st.dataframe(
            df.sort_values("value", ascending=False)[
                ["number", "value", "params_n_estimators", "params_max_depth"]
            ].head()
        )

        # Optuna plots
        st.subheader("Optimization history")
        try:
            st.plotly_chart(
                vis.plot_optimization_history(study),
                use_container_width=True
            )
        except Exception as e:
            st.write(e)

        st.subheader("Parameter importances")
        try:
            st.plotly_chart(
                vis.plot_param_importances(study),
                use_container_width=True
            )
        except Exception as e:
            st.write(e)

# -----------------------
# Footer
# -----------------------
st.markdown("---")
st.caption(
    "Educational demo only. For real HPO, run Optuna in a background worker "
    "and store results in a metrics database."
)

