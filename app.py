# -*- coding: utf-8 -*-
"""
Supply Chain ML Dashboard
- Model 1: Late Delivery Prediction
- Model 2: Customer Segmentation
- Model 3: Product Demand Forecasting
"""

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
    csv_name = "DataCo.csv"   # <- updated name

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
        delivery_model = joblib.load("delivery_prediction_model.joblib")  # <- updated ext
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
    st.dataframe(df.head(20), use_container_width=True)
    st.write("Shape:", df.shape)

# --- Delivery Prediction ---
with tab2:
    st.subheader("🚚 Late Delivery Prediction")

    if delivery_model is None:
        st.error("❌ Delivery prediction model not loaded.")
    else:
        st.markdown("Enter order details to check if delivery is likely to be **late or on time**.")

        col1, col2 = st.columns(2)

        with col1:
            days_for_shipment = st.number_input("Days for shipment (scheduled)", min_value=1, max_value=30, value=5)
            shipping_mode = st.selectbox("Shipping Mode", ["Standard Class", "Second Class", "First Class", "Same Day"])
            order_region = st.selectbox("Order Region", ["North", "South", "East", "West"])
            order_state = st.text_input("Order State", "California")

        with col2:
            order_item_qty = st.number_input("Order Item Quantity", min_value=1, max_value=100, value=2)
            category_name = st.text_input("Category Name", "Technology")
            department_name = st.text_input("Department Name", "Consumer")
            latitude = st.number_input("Latitude", value=37.77)
            longitude = st.number_input("Longitude", value=-122.41)

        if st.button("🔮 Predict Delivery Status"):
            input_df = pd.DataFrame([{
                "Days_for_shipment_scheduled": days_for_shipment,
                "Shipping_Mode": shipping_mode,
                "Order_Region": order_region,
                "Order_State": order_state,
                "Order_Item_Quantity": order_item_qty,
                "Category_Name": category_name,
                "Department_Name": department_name,
                "Latitude": latitude,
                "Longitude": longitude
            }])

            try:
                pred_prob = delivery_model.predict_proba(input_df)[0][1]
                pred_class = delivery_model.predict(input_df)[0]

                if pred_class == 1:
                    st.error(f"⚠️ High Risk of Late Delivery (Probability: {pred_prob:.2f})")
                else:
                    st.success(f"✅ On-Time Delivery Likely (Probability: {1 - pred_prob:.2f})")
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")

# --- Customer Segmentation ---
with tab3:
    st.subheader("👥 Customer Segmentation")

    if seg_model is None:
        st.error("❌ Segmentation model not loaded.")
    else:
        st.markdown("Enter customer details to assign a **segment/persona**.")

        col1, col2 = st.columns(2)

        with col1:
            recency = st.number_input("Recency (days since last purchase)", min_value=0, max_value=1000, value=30)
            frequency = st.number_input("Frequency (number of purchases)", min_value=1, max_value=100, value=5)

        with col2:
            monetary = st.number_input("Monetary Value (Total Spend $)", min_value=10, max_value=10000, value=500)

        if st.button("🔍 Predict Segment"):
            input_df = pd.DataFrame([[recency, frequency, monetary]], columns=["Recency", "Frequency", "Monetary"])

            try:
                # Scale input
                input_scaled = seg_scaler.transform(input_df)
                seg_label = seg_model.predict(input_scaled)[0]
                persona = seg_personas.get(seg_label, "Unknown")

                st.success(f"🧑 Customer assigned to Segment: **{seg_label} ({persona})**")
            except Exception as e:
                st.error(f"❌ Segmentation failed: {e}")

# --- Demand Forecasting ---
with tab4:
    st.subheader("📈 Product Demand Forecasting")

    if forecast_model is None:
        st.error("❌ Forecast model not loaded.")
    else:
        st.markdown("Select a product and forecast its demand for upcoming days.")

        product_name = st.text_input("Product Name", "Product A")
        days_ahead = st.slider("Days to Forecast", min_value=1, max_value=30, value=7)

        if st.button("📊 Forecast Demand"):
            try:
                # Dummy example input – adjust if your model needs more features
                future_dates = pd.DataFrame({"Days_Ahead": [days_ahead], "Product": [product_name]})
                forecast = forecast_model.predict(future_dates)[0]
                st.success(f"📦 Forecasted Demand for {product_name} in {days_ahead} days: **{forecast:.0f} units**")
            except Exception as e:
                st.error(f"❌ Forecasting failed: {e}")
