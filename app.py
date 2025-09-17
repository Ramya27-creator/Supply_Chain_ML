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
# Cached Loaders
# -------------------------
@st.cache_resource
def load_models():
    """Load ML models once and cache them."""
    delivery_model = joblib.load("delivery_prediction_model.joblib")
    seg_model = joblib.load("customer_segmentation_model.joblib")
    seg_scaler = joblib.load("segmentation_scaler.joblib")
    seg_personas = joblib.load("segmentation_personas.joblib")
    forecast_model = joblib.load("demand_forecasting_model.joblib")
    return delivery_model, seg_model, seg_scaler, seg_personas, forecast_model

@st.cache_data
def load_data():
    """Load dataset from DataCo.zip"""
    zip_path = "DataCo.zip"
    csv_name = "DataCoSupplyChainDataset.csv"  # inside the zip

    if not os.path.exists(zip_path):
        st.error(f"{zip_path} not found in the repo!")
        return None

    with zipfile.ZipFile(zip_path, "r") as z:
        if csv_name not in z.namelist():
            st.error(f"{csv_name} not found inside {zip_path}")
            return None
        with z.open(csv_name) as f:
            df = pd.read_csv(f, encoding="latin1", low_memory=False)
    return df

# -------------------------
# Sidebar Navigation
# -------------------------
option = st.sidebar.radio(
    "📊 Select Analysis",
    [
        "Late Delivery Prediction",
        "Customer Segmentation",
        "Demand Forecasting"
    ]
)

# -------------------------
# Late Delivery Prediction
# -------------------------
if option == "Late Delivery Prediction":
    st.header("📦 Late Delivery Prediction")
    st.info("Loading model & dataset... please wait")

    delivery_model, _, _, _, _ = load_models()
    df = load_data()

    if df is not None:
        st.success("✅ Data & Model loaded successfully!")
        st.write("Sample Data:", df.head())
        # 👉 Add your prediction code here
    else:
        st.error("Dataset could not be loaded!")

# -------------------------
# Customer Segmentation
# -------------------------
elif option == "Customer Segmentation":
    st.header("👥 Customer Segmentation")
    st.info("Loading segmentation model...")

    _, seg_model, seg_scaler, seg_personas, _ = load_models()
    df = load_data()

    if df is not None:
        st.success("✅ Segmentation Model & Data loaded!")
        st.write("Segmentation Personas:", seg_personas)
        # 👉 Add your clustering/segmentation visualization here
    else:
        st.error("Dataset could not be loaded!")

# -------------------------
# Demand Forecasting
# -------------------------
elif option == "Demand Forecasting":
    st.header("📈 Product Demand Forecasting")
    st.info("Loading forecasting model...")

    _, _, _, _, forecast_model = load_models()
    df = load_data()

    if df is not None:
        st.success("✅ Forecasting Model & Data loaded!")
        st.write("Sample Data for Forecasting:", df.head())
        # 👉 Add your forecasting plots here
    else:
        st.error("Dataset could not be loaded!")
