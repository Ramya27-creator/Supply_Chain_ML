import streamlit as st
import pandas as pd
import joblib
import zipfile
import os

st.set_page_config(page_title="🚀 Supply Chain ML Dashboard", layout="wide")

# -------------------------
# Cached Data Loader
# -------------------------
@st.cache_data(show_spinner=True)
def load_data():
    zip_path = "DataCo.zip"  # Ensure this file is in your repo root

    if not os.path.exists(zip_path):
        st.error("❌ DataCo.zip not found in repository.")
        return None

    with zipfile.ZipFile(zip_path, "r") as z:
        # List all files inside the zip
        file_list = z.namelist()
        st.write("📂 Files in ZIP:", file_list)  # Debugging help

        # Pick the first CSV automatically
        csv_files = [f for f in file_list if f.endswith(".csv")]
        if not csv_files:
            st.error("❌ No CSV file found inside DataCo.zip")
            return None

        csv_name = csv_files[0]  # Use the first CSV file
        with z.open(csv_name) as f:
            df = pd.read_csv(f, encoding="latin1", low_memory=False)
    return df

# -------------------------
# Cached Model Loader
# -------------------------
@st.cache_resource(show_spinner=True)
def load_models():
    try:
        # Special handling for delivery model (inside .zip)
        with zipfile.ZipFile("delivery_prediction_model.zip", "r") as z:
            file_list = z.namelist()
            model_files = [f for f in file_list if f.endswith(".joblib")]
            if not model_files:
                st.error("❌ No .joblib file found inside delivery_prediction_model.zip")
                delivery_model = None
            else:
                with z.open(model_files[0]) as f:
                    delivery_model = joblib.load(f)

        # Other models
        seg_model = joblib.load("customer_segmentation_model.joblib")
        seg_scaler = joblib.load("customer_segmentation_scaler.joblib")
        seg_personas = joblib.load("customer_segmentation_personas.joblib")
        forecast_model = joblib.load("demand_forecasting_model.joblib")

        return delivery_model, seg_model, seg_scaler, seg_personas, forecast_model

    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return None, None, None, None, None

# -------------------------
# Load Data + Models
# -------------------------
with st.spinner("📦 Loading dataset..."):
    df = load_data()

with st.spinner("🤖 Loading ML models..."):
    delivery_model, seg_model, seg_scaler, seg_personas, forecast_model = load_models()

if df is None:
    st.stop()

# -------------------------
# Streamlit Tabs
# -------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Data Overview", "🚚 Delivery Prediction", "👥 Customer Segmentation", "📈 Demand Forecasting"]
)

# --- Data Overview ---
with tab1:
    st.subheader("Data Preview")
    st.dataframe(df.head(20))
    st.write("Shape:", df.shape)

# --- Delivery Prediction ---
with tab2:
    st.subheader("Late Delivery Prediction")
    st.write("⚡ Model ready:", delivery_model is not None)

# --- Customer Segmentation ---
with tab3:
    st.subheader("Customer Segmentation")
    st.write("⚡ Model ready:", seg_model is not None)

# --- Demand Forecasting ---
with tab4:
    st.subheader("Product Demand Forecasting")
    st.write("⚡ Model ready:", forecast_model is not None)
