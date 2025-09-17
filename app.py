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
    zip_path = "DataCo.zip"  # Make sure this is in your repo root
    csv_name = "DataCoSupplyChainDataset.csv"

    if not os.path.exists(zip_path):
        st.error("❌ DataCo.zip not found in repository.")
        return None

    with zipfile.ZipFile(zip_path, "r") as z:
        with z.open(csv_name) as f:
            df = pd.read_csv(f, encoding="latin1", low_memory=False)
    return df

# -------------------------
# Cached Model Loader
# -------------------------
@st.cache_resource(show_spinner=True)
def load_models():
    try:
        delivery_model = joblib.load("delivery_prediction_model.zip")
        seg_model = joblib.load("customer_segmentation_model.joblib")
        seg_scaler = joblib.load("segmentation_scaler.joblib")
        seg_personas = joblib.load("segmentation_personas.joblib")
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
