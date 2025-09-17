# app.py
import streamlit as st
import pandas as pd
import joblib
import zipfile
import os

# -------------------------
# Streamlit Config
# -------------------------
st.set_page_config(page_title="Supply Chain ML Dashboard", layout="wide")
st.title("🚀 Supply Chain ML Dashboard")

# -------------------------
# Cached Dataset Loader
# -------------------------
@st.cache_data
def load_data(sample=True, nrows=10000):
    """Load dataset from DataCo.zip (sample for faster load)."""
    zip_path = "DataCo.zip"
    csv_name = "DataCoSupplyChainDataset.csv"

    if not os.path.exists(zip_path):
        st.error(f"{zip_path} not found in repo!")
        return None

    with zipfile.ZipFile(zip_path, "r") as z:
        if csv_name not in z.namelist():
            st.error(f"{csv_name} not found inside {zip_path}")
            return None
        with z.open(csv_name) as f:
            if sample:
                df = pd.read_csv(f, encoding="latin1", low_memory=False, nrows=nrows)
            else:
                df = pd.read_csv(f, encoding="latin1", low_memory=False)

    return df

# -------------------------
# Cached Model Loaders
# -------------------------
@st.cache_resource
def load_delivery_model():
    return joblib.load("delivery_prediction_model.joblib")

@st.cache_resource
def load_segmentation_models():
    seg_model = joblib.load("customer_segmentation_model.joblib")
    seg_scaler = joblib.load("segmentation_scaler.joblib")
    seg_personas = joblib.load("segmentation_personas.joblib")
    return seg_model, seg_scaler, seg_personas

@st.cache_resource
def load_forecast_model():
    return joblib.load("demand_forecasting_model.joblib")

# -------------------------
# Tabs
# -------------------------
tab1, tab2, tab3 = st.tabs(
    ["📦 Late Delivery Prediction", "👥 Customer Segmentation", "📈 Demand Forecasting"]
)

# -------------------------
# Tab 1: Late Delivery Prediction
# -------------------------
with tab1:
    st.header("📦 Late Delivery Prediction")
    st.info("Loading delivery model & sample dataset...")

    model = load_delivery_model()
    df = load_data(sample=True)

    if df is not None:
        st.success("✅ Delivery Model & Data loaded!")
        st.write("Sample Data:", df.head())
        # 👉 Add prediction inputs here

# -------------------------
# Tab 2: Customer Segmentation
# -------------------------
with tab2:
    st.header("👥 Customer Segmentation")
    st.info("Loading segmentation models & sample dataset...")

    seg_model, seg_scaler, seg_personas = load_segmentation_models()
    df = load_data(sample=True)

    if df is not None:
        st.success("✅ Segmentation Models & Data loaded!")
        st.write("Segmentation Personas:", seg_personas)
        # 👉 Add clustering inputs here

# -------------------------
# Tab 3: Demand Forecasting
# -------------------------
with tab3:
    st.header("📈 Product Demand Forecasting")
    st.info("Loading forecasting model & sample dataset...")

    forecast_model = load_forecast_model()
    df = load_data(sample=True)

    if df is not None:
        st.success("✅ Forecast Model & Data loaded!")
        st.write("Sample Data for Forecasting:", df.head())
        # 👉 Add forecasting plots here
