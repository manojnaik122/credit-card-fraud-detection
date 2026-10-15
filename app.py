
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# -------------------------------
# Page configuration
# -------------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

MODEL_PATH = Path(__file__).parent / "fraud_detection_model.joblib"

# -------------------------------
# Load the trained pipeline
# -------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load the model: {e}")
    st.stop()

# -------------------------------
# Application interface
# -------------------------------
st.title("💳 Credit Card Fraud Detection")
st.write(
    "Upload transaction data and use a trained machine learning "
    "model to identify potentially fraudulent transactions."
)

st.info(
    "Upload a CSV containing Time, V1–V28, and Amount. "
    "The V features are anonymized PCA features from the dataset."
)

# -------------------------------
# File upload
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload your transaction CSV",
    type=["csv"]
)

expected_columns = [
    "Time",
    *[f"V{i}" for i in range(1, 29)],
    "Amount"
]

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        if df.empty:
            st.warning("The uploaded CSV is empty.")
            st.stop()

        missing_columns = [
            col for col in expected_columns
            if col not in df.columns
        ]

        if missing_columns:
            st.error(
                "The CSV is missing these required columns: "
                + ", ".join(missing_columns)
            )
            st.stop()

        # Use exactly the model's expected input features
        X = df[expected_columns].apply(
            pd.to_numeric, errors="coerce"
        )

        if X.isna().any().any():
            st.error(
                "Some required input values are missing or "
                "non-numeric. Please correct the CSV."
            )
            st.stop()

        if not X.map(lambda value: float("-inf") < value < float("inf")).all().all():
            st.error("The CSV contains infinite or invalid numeric values.")
            st.stop()

        st.subheader("Transaction preview")
        st.dataframe(df.head(10), use_container_width=True)

        st.caption(f"Total transactions uploaded: {len(X):,}")

        if st.button("🔍 Predict Fraud", type="primary"):
            with st.spinner("Analyzing transactions..."):
                predictions = model.predict(X)

                results = df.copy()
                results["Prediction"] = predictions
                results["Result"] = results["Prediction"].map({
                    0: "Legitimate",
                    1: "Potential Fraud"
                })

                # Show estimated probabilities when supported
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(X)
                    classes = list(model.classes_)

                    if 1 in classes:
                        fraud_index = classes.index(1)
                        results["Fraud Probability (%)"] = (
                            probabilities[:, fraud_index] * 100
                        ).round(2)

            fraud_count = int((predictions == 1).sum())
            legitimate_count = int((predictions == 0).sum())
            fraud_percentage = fraud_count / len(results) * 100

            st.subheader("Prediction summary")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Transactions analyzed",
                f"{len(results):,}"
            )
            col2.metric(
                "Potential fraud",
                f"{fraud_count:,}"
            )
            col3.metric(
                "Fraud percentage",
                f"{fraud_percentage:.2f}%"
            )

            st.subheader("Prediction results")

            if fraud_count:
                st.warning(
                    f"{fraud_count} transaction(s) flagged "
                    "as potentially fraudulent."
                )
            else:
                st.success(
                    "No transactions were flagged by the model."
                )

            st.dataframe(
                results,
                use_container_width=True
            )

            # Download the predictions
            csv_data = results.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="📥 Download prediction report",
                data=csv_data,
                file_name="fraud_predictions.csv",
                mime="text/csv"
            )

            st.caption(
                "These are model predictions, not proof of fraud. "
                "Fraud probabilities, if shown, may not be calibrated."
            )

    except Exception as e:
        st.error(f"An error occurred while processing the CSV: {e}")

st.divider()
st.caption(
    "Machine Learning Project | KNN + StandardScaler | "
    "Credit Card Fraud Detection"
)
