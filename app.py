# -*- coding: utf-8 -*-
"""app.py - Streamlit App for Supply Chain ML Dashboard"""

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
    zip_path = "DataCo.zip"
    csv_name = "DataCo.csv"   # inside your zip

    if not os.path.exists(zip_path):
        st.error("❌ DataCo.zip not found in repository.")
        return None

    with zipfile.ZipFile(zip_path, "r") as z:
        if csv_name not in z.namelist():
            st.error(f"❌ {csv_name} not found inside DataCo.zip")
            return None
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
        seg_scaler = joblib.load("customer_segmentation_scaler.joblib")
        seg_personas = joblib.load("customer_personas.joblib")
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
    st.subheader("🚚 Late Delivery Prediction")

    if delivery_model:
        st.success("✅ Delivery prediction model loaded")
        # Input form
        with st.form("delivery_form"):
            scheduled_days = st.slider("Days for Shipment (Scheduled)", 1, 30, 5)
            shipping_mode = st.selectbox("Shipping Mode", df["Shipping_Mode"].dropna().unique())
            region = st.selectbox("Order Region", df["Order_Region"].dropna().unique())
            state = st.selectbox("Order State", df["Order_State"].dropna().unique())
            quantity = st.slider("Order Item Quantity", 1, 50, 1)
            category = st.selectbox("Category Name", df["Category_Name"].dropna().unique())
            department = st.selectbox("Department Name", df["Department_Name"].dropna().unique())
            latitude = st.number_input("Latitude", value=0.0)
            longitude = st.number_input("Longitude", value=0.0)
            submitted = st.form_submit_button("Predict")

        if submitted:
            input_df = pd.DataFrame([{
                "Days_for_shipment_scheduled": scheduled_days,
                "Shipping_Mode": shipping_mode,
                "Order_Region": region,
                "Order_State": state,
                "Order_Item_Quantity": quantity,
                "Category_Name": category,
                "Department_Name": department,
                "Latitude": latitude,
                "Longitude": longitude,
            }])
            proba = delivery_model.predict_proba(input_df)[0]
            st.write("Prediction Probabilities:", {"Not Late (0)": proba[0], "Late (1)": proba[1]})
            st.write("🚨 Final Prediction:", "Late" if proba[1] > 0.5 else "On Time")
    else:
        st.error("❌ Delivery prediction model not loaded.")

# --- Customer Segmentation ---
with tab3:
    st.subheader("👥 Customer Segmentation")
    if seg_model and seg_scaler and seg_personas:
        st.success("✅ Segmentation model loaded")
        with st.form("segmentation_form"):
            total_sales = st.number_input("Total Sales", value=500.0)
            avg_benefit = st.number_input("Average Benefit per Order", value=50.0)
            purchase_freq = st.number_input("Purchase Frequency", value=5)
            submitted = st.form_submit_button("Predict Segment")
        if submitted:
            input_df = pd.DataFrame([[total_sales, avg_benefit, purchase_freq]],
                                    columns=['TotalSales', 'AverageBenefit', 'PurchaseFrequency'])
            scaled = seg_scaler.transform(input_df)
            cluster = seg_model.predict(scaled)[0]
            persona = seg_personas.get(cluster, "Unknown")
            st.write(f"Predicted Cluster: {cluster}")
            st.write(f"Persona: {persona}")
    else:
        st.error("❌ Segmentation model not loaded.")

# --- Demand Forecasting ---
with tab4:
    st.subheader("📈 Product Demand Forecasting")
    if forecast_model:
        st.success("✅ Forecast model loaded")
        days = st.slider("Days to Forecast", 7, 180, 30)
        if st.button("Generate Forecast"):
            forecast = forecast_model.get_forecast(steps=days)
            pred_mean = forecast.predicted_mean
            st.line_chart(pred_mean)
    else:
        st.error("❌ Forecast model not loaded.")
