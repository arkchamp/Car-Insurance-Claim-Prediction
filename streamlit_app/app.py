# streamlit_app/app.py
"""
Streamlit app for Car Insurance Claim Prediction
- Mirrors preprocessing_modeling_evaluation.ipynb steps exactly
- Pages: Intro, EDA Dashboard, Predict (Top 10), CSV Upload, About Me
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import re
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Car Insurance Claim Prediction", layout="wide")
ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(ROOT, "..")
DATA_PATH = os.path.join(PROJECT_ROOT, "data")
MODELS_PATH = os.path.join(PROJECT_ROOT, "models")

TRAIN_CSV = os.path.join(DATA_PATH, "train.csv")
TEST_CSV = os.path.join(DATA_PATH, "test.csv")
SAMPLE_SUB = os.path.join(DATA_PATH, "sample_submission.csv")

MODEL_FULL_PATH = os.path.join(MODELS_PATH, "final_model.pkl")
MODEL_TOP10_PATH = os.path.join(MODELS_PATH, "final_model_10f.pkl")
TOP10_CSV_PATH = os.path.join(MODELS_PATH, "top_10_features.csv")
SCALER_PATH = os.path.join(MODELS_PATH, "scaler.pkl")
FREQ_MAPS_PATH = os.path.join(MODELS_PATH, "freq_encoding_maps.pkl")

SEED = 42

# ---------------- CACHING LOADERS ----------------
@st.cache_data(show_spinner=False)
def load_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@st.cache_data(show_spinner=False)
def load_model(path):
    if os.path.exists(path):
        return joblib.load(path)
    return None

@st.cache_data(show_spinner=False)
def load_top10(path=TOP10_CSV_PATH):
    if os.path.exists(path):
        try:
            return pd.read_csv("../models/top_10_features.csv")['Feature'].tolist()
        except Exception:
            return pd.read_csv(path).columns.tolist()
    return None


@st.cache_data(show_spinner=False)
def load_freq_maps(path=FREQ_MAPS_PATH):
    if os.path.exists(path):
        return joblib.load(path)
    return None

@st.cache_data(show_spinner=False)
def load_scaler(path=SCALER_PATH):
    if os.path.exists(path):
        return joblib.load(path)
    return None




# ---------------- LOAD ARTIFACTS ----------------
train_df = load_csv(TRAIN_CSV)
model_full = load_model(MODEL_FULL_PATH)
model_top10 = load_model(MODEL_TOP10_PATH)
top10_features = load_top10()
freq_maps = load_freq_maps()
scaler = load_scaler()



# ---------------- SIDEBAR NAV ----------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Intro", "EDA Dashboard", "Predict (Top 10)", "CSV Upload", "About Me"])

# ---------------- PAGE: Intro ----------------
if page == "Intro":
    st.title("Car Insurance Claim Prediction")
    st.markdown(
        """
        **Objective:** Build a model to predict whether a policyholder will file a car insurance claim
        in the next policy period using demographic, vehicle and policy features.

        **This app provides:**
        - 🎯 Reproducible preprocessing and feature-engineering (same as notebooks)
        - 📊 Interactive EDA dashboard (Plotly)
        - 🤖 Live predictor using the saved model (Top-10 features)
        - 📁 Batch prediction via CSV upload (downloadable `policy_id, is_claim`)
        """
    )

    # Load reference datasets (cached loaders defined above)
    train_preview = load_csv(TRAIN_CSV)
    test_preview = load_csv(TEST_CSV)

    # Safely compute metrics
    train_rows = len(train_preview) if train_preview is not None else 0
    test_rows = len(test_preview) if test_preview is not None else 0

    # Claim distribution: prefer train file (has target); fall back to None
    claim_pct = None
    if train_preview is not None and 'is_claim' in train_preview.columns:
        claim_pct = train_preview['is_claim'].mean() * 100

    # Top-10 features (if loaded)
    top10_display = top10_features if top10_features is not None else []
    st.markdown("---")
    # Layout metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Train rows", f"{train_rows:,}")
    with c2:
        st.metric("Test rows", f"{test_rows:,}")
    with c3:
        if claim_pct is not None:
            st.metric("Claim % (train)", f"{claim_pct:.1f}%")
        else:
            st.metric("Claim % (train)", "N/A")
    with c4:
        if claim_pct is not None:
            st.metric("No-claim % (train)", f"{100 - claim_pct:.1f}%")
        else:
            st.metric("No-claim % (train)", "N/A")

    st.markdown("---")

    

# ---------------- PAGE: EDA DASHBOARD ----------------
elif page == "EDA Dashboard":
    st.title("EDA Dashboard (Plotly)")
    if train_df is None:
        st.warning("train.csv not found in data/ - place train.csv at data/train.csv")
    else:
        st.subheader("Dataset preview")
        st.dataframe(train_df.head(100))
        st.markdown("---")

        # Target distribution
        st.subheader("`[is_claim]` Distribution.")
        # Small target distribution figure (only when train target exists)
        train_preview = load_csv(TRAIN_CSV)
        if train_preview is not None and 'is_claim' in train_preview.columns:
            counts = train_preview['is_claim'].value_counts().rename_axis('is_claim').reset_index(name='count')
            counts['label'] = counts['is_claim'].map({0: 'No Claim', 1: 'Claim'})
            # Define custom colors
            color_map = {'No Claim': '#3e2723', 'Claim': '#a1887f'} 
            fig = px.pie(counts, names='label', values='count', title="Target distribution (train.csv)", hole=0.4, color='label', color_discrete_map=color_map)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Target distribution not available - `train.csv` missing or does not include `is_claim`.")
            st.markdown("---")

        # Correlation heatmap of numeric features (top by abs correlation with target)
        st.subheader("Correlation heatmap (`numeric features of raw data`)")
        numeric = train_df.select_dtypes(include=[np.number]).copy()
        if not numeric.empty:
            corr = numeric.corr()
            if 'is_claim' in corr.columns:
                corr_with_target = corr['is_claim'].abs().sort_values(ascending=False)
                top_corr_feats = corr_with_target.index[1:13].tolist()
            else:
                top_corr_feats = numeric.columns[:12].tolist()
            corr_sub = numeric[top_corr_feats + ['is_claim']] if 'is_claim' in numeric.columns else numeric[top_corr_feats]
            fig2 = px.imshow(corr_sub.corr(),color_continuous_scale="Rainbow", text_auto=True, aspect="auto", title="")
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown("---")

        # Boxplots for selected features
        st.subheader("Boxplots - Feature distribution vs 'is_claim'")

        # Feature list + color palette
        features = ["age_of_car", "population_density", "policy_tenure", "age_of_policyholder"]
        colors = ['crimson', 'steelblue', 'orchid', 'darkolivegreen']

        # Ensure features exist in dataset
        available_feats = [f for f in features if f in train_df.columns]

        # Two-column layout for side-by-side plots
        cols = st.columns(2)

        for i, (feat, color) in enumerate(zip(available_feats, colors)):
            with cols[i % 2]:
                fig = px.box(
                    train_df,
                    x='is_claim' if 'is_claim' in train_df.columns else None,
                    y=feat,
                    color='is_claim' if 'is_claim' in train_df.columns else None,
                    points='outliers',
                    color_discrete_sequence=[color],
                    title=f"Outlier Distribution: {feat.replace('_', ' ').title()}",
                    labels={'is_claim': 'Claim', feat: feat.replace('_', ' ').title()}
                )
                fig.update_layout(
                    xaxis_title="Claim (0 = No, 1 = Yes)",
                    yaxis_title=feat.replace('_', ' ').title(),
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        # Feature importance from model_full if available
        st.subheader("Feature Importance (from final model)")
        if model_full is not None:
            try:
                feat_names = getattr(model_full, "feature_names_in_", None)
                importances = model_full.feature_importances_

                #creating a dataframe for viz
                feature_imp_df = pd.DataFrame({
                    'Feature': feat_names,
                    'Importance':importances
                }).sort_values(by='Importance', ascending=False).head(10)

                fig_fi = px.bar(feature_imp_df.sort_values('Importance', ascending=True), x='Importance', y='Feature',
                                orientation='h', title="Top features by importance",color="Importance", color_continuous_scale="aggrnyl")
                st.plotly_chart(fig_fi, use_container_width=True)
            except Exception as e:
                st.error("Could not extract feature importances: " + str(e))
        st.markdown("---")

# ---------------- PAGE: PREDICT (TOP 10) ----------------
elif page == "Predict (Top 10)":
    st.title("Predict - Manual Input (Top 10 Features)")
    # st.markdown(load_top10())

    # Check model & artifacts
    if model_top10 is None or scaler is None or freq_maps is None or top10_features is None:
        st.error("Missing model/scaler/freq_encoding_maps/top_10_features file in models/. Please verify.")
    else:
        # Prepare top 10 features excluding engineered one
        if 'density_factor' in top10_features:
            top10_features = [f for f in top10_features if f != 'density_factor']

        st.markdown("Enter values for the 9 key features below -- the 10th feature (`density_factor`) will be auto-computed.")
        input_data = {}

        # Manual inputs (categorical + numeric)
        col1, col2 = st.columns(2)

        with col1:
            max_torque_opt = train_df['max_torque'].unique().tolist() if train_df is not None else ['60Nm@3500rpm','113Nm@4400rpm','91Nm@4250rpm','250Nm@2750rpm','200Nm@3000rpm']
            model_opts = train_df['model'].unique().tolist() if train_df is not None else ['M1','M2','M3','M4','M5']
            input_data['age_of_car'] = st.slider("Age of Car (normalized)",min_value=0.0, max_value=1.0, value=0.11, step=0.01)
            input_data['max_torque'] = st.selectbox("Max Torque", max_torque_opt, index=1)
            input_data['policy_tenure'] = st.slider("Policy Tenure (normalized)", min_value=0.002, max_value=1.39, value=0.82)
            input_data['model'] = st.selectbox("Model", model_opts, index=5)
            input_data['is_adjustable_steering'] = st.selectbox("Is Adjustable Steering", ['Yes', 'No'], index=0)

        with col2:
            area_cluster_opts = train_df['area_cluster'].unique().tolist() if train_df is not None else ['C1','C2','C3','C4','C5']
            max_power_opt = train_df['max_power'].unique().tolist() if train_df is not None else ['40.36bhp@6000rpm','88.50bhp@6000rpm','67.06bhp@5500rpm','113.45bhp@4000rpm','88.77bhp@4000rpm']
            input_data['population_density'] = st.slider("Population Density", min_value=290, max_value=73430, value=4500, step=1)
            input_data['age_of_policyholder'] = st.slider("Age of Policyholder (normalized)", min_value=0.28, max_value=1.0, value=0.35, step=0.01)
            input_data['area_cluster'] = st.selectbox("Area Cluster", area_cluster_opts, index=7)
            input_data['max_power'] = st.selectbox("Max Power", max_power_opt, index=5)
            

        # When user clicks Predict
        if st.button("Predict"):
            feat_names = getattr(model_top10, "feature_names_in_", None)
            st.info("Running preprocessing and prediction...")

            # Convert to DataFrame
            df_input = pd.DataFrame([input_data])

            
            # --- Clean numeric strings like '113Nm@4400rpm' ---
            def extract_numeric(val):
                try:
                    return float(re.findall(r'\d+\.?\d*', str(val))[0])
                except:
                    return np.nan

            df_input['max_power'] = df_input['max_power'].apply(extract_numeric)
            df_input['max_torque'] = df_input['max_torque'].apply(extract_numeric)

            # --- Convert Yes/No to 1/0 ---
            df_input['is_adjustable_steering'] = df_input['is_adjustable_steering'].replace({'Yes': 1, 'No': 0})

            # --- Frequency Encoding ---
            for col, fmap in freq_maps.items():
                if col in df_input.columns:
                    df_input[col] = df_input[col].map(fmap).fillna(0)

            # --- Engineer density_factor = population_density × encoded area_cluster ---
            if 'area_cluster' in df_input.columns and 'population_density' in df_input.columns:
                df_input['density_factor'] = df_input['population_density'] * df_input['area_cluster']

            

            # --- Add any missing columns ---
            expected_cols = list(model_full.feature_names_in_)
            for col in expected_cols:
                if col not in df_input.columns:
                    df_input[col] = 0.0

            # --- Ensure same column order ---
            df_input_to_display = df_input.copy()
            df_input_to_display = df_input_to_display[feat_names]
            st.subheader("`Data Preview`")
            st.dataframe(df_input_to_display.head())
            st.markdown("---")

            # --- Ensure same column order ---
            df_input = df_input[expected_cols]
            

            # --- Scale numeric features ---
            try:
                scaler_cols = getattr(scaler, "feature_names_in_", None)
                scaler_cols = scaler_cols.tolist()
                scaled_values = scaler.transform(df_input[scaler_cols].to_numpy(dtype=float))
                df_input.loc[:, scaler_cols] = scaled_values
            except Exception as e:
                st.warning(f"Scaling skipped or failed: {e}")

            
            # --- Prediction ---
            try:

                
                X_input = df_input[feat_names]
                pred_proba = model_top10.predict_proba(X_input)[:, 1][0]
                pred_class = int(model_top10.predict(X_input)[0])
                label = "Claim" if pred_class == 1 else "No Claim"

                st.success(f"✅ Prediction: **{label}** (Probability: {pred_proba:.4f})")

                # Optional: display processed input for transparency
                with st.expander("Show processed input data"):
                    st.dataframe(df_input)

            except Exception as e:
                st.error(f"Prediction failed: {e}")


# ---------------- PAGE: CSV UPLOAD ----------------
elif page == "CSV Upload":
    st.title("Batch Prediction - Upload CSV")
    st.markdown("Upload a CSV with the same schema as `test.csv` (must include `policy_id`).\n\nIf you don't have a file, you can use the project's `data/test.csv`.")

    # --- checkbox first (disable uploader when checked) ---
    use_example = st.checkbox("Use data/test.csv (example)", value=False)
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], disabled=use_example)

    input_df = None  # Initialize empty

    # --- load logic ---
    if use_example:
        st.info("Using built-in example file: data/test.csv")
        try:
            input_df = pd.read_csv(TEST_CSV)
        except Exception as e:
            st.error(f"Unable to read example test.csv: {e}")

    elif uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")
        except Exception as e:
            st.error(f"Unable to read uploaded CSV: {e}")

    else:
        st.info("Upload a CSV or check 'Use data/test.csv (example)' to proceed.")

    # --- Only continue if we have a dataframe ---
    if input_df is not None:
        # Drop target column if present (e.g., when user uploads train.csv)
        if 'is_claim' in input_df.columns:
            input_df.drop(columns=['is_claim'], inplace=True)
            st.warning("⚠️ Target column detected in uploaded file and removed automatically (`is_claim`).")
        st.subheader("Preview of uploaded data (first 10 rows)")
        st.dataframe(input_df.head(10))
        
        # Basic checks
        temp_df = pd.read_csv(TEST_CSV, nrows=1)
        cols_to_check = temp_df.columns.tolist()

        missing_cols = [col for col in cols_to_check if col not in input_df.columns]

        if missing_cols:
            st.error(f"Uploaded CSV is missing requrired columns: `{missing_cols}`")
            st.stop()

        if model_full is None:
            st.error("Full model (final_model.pkl) missing in models/. Can't run batch predictions.")
            st.stop()

        if scaler is None:
            st.warning("Scaler not found (scaler.pkl). Scaling will be skipped.")

        # Proceed with preprocessing
        with st.spinner("Preprocessing uploaded data..."):
            df = input_df.copy()
            # Keep policy_id aside
            policy_id_col = df['policy_id'].astype(str).copy()
            df.drop(columns=['policy_id'], inplace=True, errors='ignore')

            # --- Inspect columns ---
            # convert typical yes/no columns
            obj_cols = df.select_dtypes(include=['object']).columns.tolist()

            # Detect Yes/No columns (all values in {Yes,No} or subset)
            yes_no_cols = [c for c in obj_cols if set(df[c].dropna().unique()).issubset({'Yes', 'No', 'yes', 'no', 'YES', 'NO'})]

            # Replace Yes/No -> 1/0
            if yes_no_cols:
                df[yes_no_cols] = df[yes_no_cols].replace({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0, 'YES': 1, 'NO': 0})

            # Helper to extract numeric from strings like '113Nm@4400rpm' or '55.92bhp@5300rpm'
            def extract_numeric(value):
                try:
                    return float(re.findall(r'\d+\.?\d*', str(value))[0])
                except Exception:
                    return np.nan

            for c in ['max_power', 'max_torque']:
                if c in df.columns:
                    df[c] = df[c].apply(extract_numeric)

            # Frequency encoding using freq_maps if available
            if freq_maps is not None:
                for col, fmap in freq_maps.items():
                    if col in df.columns:
                        df[col] = df[col].map(fmap).fillna(0)
            else:
                st.warning("freq_encoding_maps.pkl not found. Skipping frequency encoding.")

            # Recompute object columns after freq-encoding & yes/no replacement
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            cat_unique_counts = {col: df[col].nunique() for col in categorical_cols}
            low_cardinality = [col for col, cnt in cat_unique_counts.items() if cnt <= 10]
            # high_cardinality = [col for col, cnt in cat_unique_counts.items() if cnt > 10]  # not used directly here

            # One-hot encode low-cardinality categorical columns (drop_first to avoid multicollinearity)
            if low_cardinality:
                df = pd.get_dummies(df, columns=low_cardinality, drop_first=True, dtype=int)

            # --- Feature engineering ---
            # power_to_weight, torque_to_weight, car_volume, age_gap, engine_efficiency, cylinder_to_power,
            # density_factor, tenure_to_car_age, tenure_to_owner_age
            try:
                if 'max_power' in df.columns and 'gross_weight' in df.columns:
                    df['power_to_weight'] = df['max_power'] / df['gross_weight']
                if 'max_torque' in df.columns and 'gross_weight' in df.columns:
                    df['torque_to_weight'] = df['max_torque'] / df['gross_weight']
                if all(c in df.columns for c in ['length', 'width', 'height']):
                    df['car_volume'] = (df['length'] * df['width'] * df['height']) / 1e9
                if 'age_of_policyholder' in df.columns and 'age_of_car' in df.columns:
                    df['age_gap'] = df['age_of_policyholder'] - df['age_of_car']
                if 'displacement' in df.columns and 'max_power' in df.columns:
                    df['engine_efficiency'] = df['displacement'] / df['max_power']
                if 'cylinder' in df.columns and 'max_power' in df.columns:
                    df['cylinder_to_power'] = df['cylinder'] / df['max_power']
                # density_factor uses encoded area_cluster (freq-encoded) if present, else will error silently
                if 'population_density' in df.columns and 'area_cluster' in df.columns:
                    df['density_factor'] = df['population_density'] * df['area_cluster']
                if 'policy_tenure' in df.columns and 'age_of_car' in df.columns:
                    df['tenure_to_car_age'] = df['policy_tenure'] / df['age_of_car']
                if 'policy_tenure' in df.columns and 'age_of_policyholder' in df.columns:
                    df['tenure_to_owner_age'] = df['policy_tenure'] / df['age_of_policyholder']
            except Exception as e:
                st.warning(f"Some engineered features could not be computed: {e}")

            # Replace inf/-inf and fill NaNs
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.fillna(0, inplace=True)

        # --- Scaling and alignment with model features ---
        with st.spinner("Scaling & aligning features..."):
            # Get model expected features
            model_feat_names = getattr(model_full, "feature_names_in_", None)
            if model_feat_names is None:
                st.warning("Model does not expose feature_names_in_. Falling back to columns present in preprocessed df.")
                model_feat_names = df.columns.tolist()

            # Ensure all expected columns exist (fill missing with 0)
            for col in model_feat_names:
                if col not in df.columns:
                    df[col] = 0.0

            # Keep only expected order
            df = df[model_feat_names]

            # Scaling: use scaler.feature_names_in_ if available
            if scaler is not None:
                try:
                    scaler_fnames = getattr(scaler, "feature_names_in_", None)
                    if scaler_fnames is not None:
                        # ensure str types, select intersection
                        scaler_cols = [str(c) for c in scaler_fnames if str(c) in df.columns]
                        if scaler_cols:
                            arr = df[scaler_cols].to_numpy(dtype=float)
                            scaled = scaler.transform(arr)
                            df.loc[:, scaler_cols] = scaled
                    else:
                        st.warning("Scaler has no feature_names_in_. Skipping strict-scaling match; attempting to scale all numeric columns.")
                        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                        if numeric_cols:
                            df[numeric_cols] = scaler.transform(df[numeric_cols].to_numpy(dtype=float))
                except Exception as e:
                    st.warning(f"Scaling failed or skipped: {e}")

        # --- Prediction ---
        with st.spinner("Running model predictions..."):
            try:
                preds = model_full.predict(df)
                # Build output dataframe
                out = pd.DataFrame({"policy_id": policy_id_col.values, "is_claim": preds.astype(int)})
                st.success(f"Predictions completed for {len(out)} rows.")
                st.dataframe(out.head(20))

                # Download button
                csv_out = out.to_csv(index=False).encode('utf-8')
                # --- Build dynamic filename for output ---
                if use_example:
                    base_name = "test"
                elif uploaded_file is not None:
                    # Remove extension safely
                    base_name = os.path.splitext(uploaded_file.name)[0]
                else:
                    base_name = "output"

                out_file_name = f"{base_name}_output.csv"

                # --- Download button ---
                st.download_button(
                    "Download predictions as CSV",
                    data=csv_out,
                    file_name=out_file_name,
                    mime="text/csv"
                )

            except Exception as e:
                st.error(f"Prediction failed: {e}")


# ---------------- PAGE: About Me ----------------
elif page == "About Me":
    st.title("About Me")
    st.markdown(
        """
        **Abdullah Khatri**  
        - Data Science learner - passionate about Machine Learning, Model Deployment & Data-driven applications.  
        - This project - **Car Insurance Claim Prediction** - demonstrates a complete ML lifecycle:
            - Data preprocessing & feature engineering  
            - Exploratory Data Analysis (EDA) with Plotly  
            - Model building (Logistic Regression, Random Forest, XGBoost, etc.)  
            - Feature importance extraction & Top-10 feature selection  
            - Deployment-ready pipeline using Streamlit  

        - The app includes both manual & batch prediction modes and an interactive EDA dashboard.  
        """
    )

    st.info("🌐 GitHub: [https://github.com/arkchamp](https://github.com/arkchamp)")
    st.caption("Developed as part of the GUVI Data Science program - Mentor-guided Final Project.")


# End of app.py
