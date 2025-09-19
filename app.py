# -*- coding: utf-8 -*-
"""
Streamlit App for Supply Chain ML Dashboard
- Model 1: Late Delivery Prediction
- Model 2: Customer Segmentation
- Model 3: Product Demand Forecasting
"""

import streamlit as st
import pandas as pd
import joblib
import zipfile
import os
import matplotlib.pyplot as plt

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(page_title="Supply Chain ML Dashboard", layout="wide")
st.markdown("<h1 style='text-align:center; color:#4B0082;'>🚀 Supply Chain ML Dashboard</h1>", unsafe_allow_html=True)

# -------------------------
# Load Models
# -------------------------
@st.cache_resource
def load_models():
    try:
        # Delivery model
        if os.path.exists("delivery_prediction_model.joblib"):
            delivery_model = joblib.load("delivery_prediction_model.joblib")
        elif os.path.exists("delivery_prediction_model.zip"):
            with zipfile.ZipFile("delivery_prediction_model.zip", "r") as z:
                z.extractall()
            delivery_model = joblib.load("delivery_prediction_model.joblib")
        else:
            delivery_model = None

        # Customer segmentation models
        seg_model = joblib.load("customer_segmentation_model.joblib") if os.path.exists("customer_segmentation_model.joblib") else None
        seg_scaler = joblib.load("customer_segmentation_scaler.joblib") if os.path.exists("customer_segmentation_scaler.joblib") else None
        seg_personas = joblib.load("customer_personas.joblib") if os.path.exists("customer_personas.joblib") else {}

        # Forecasting model
        forecast_model = joblib.load("demand_forecasting_model.joblib") if os.path.exists("demand_forecasting_model.joblib") else None

        return delivery_model, seg_model, seg_scaler, seg_personas, forecast_model

    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return None, None, None, {}, None

delivery_model, seg_model, seg_scaler, seg_personas, forecast_model = load_models()

# -------------------------
# Helper Functions
# -------------------------
def predict_delivery(input_data: dict):
    df = pd.DataFrame([input_data])
    required_cols = delivery_model.feature_names_in_
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[required_cols]
    proba = delivery_model.predict_proba(df)[0]
    return proba

def predict_segmentation(total_sales, avg_benefit, purchase_freq):
    input_df = pd.DataFrame([[total_sales, avg_benefit, purchase_freq]], columns=["TotalSales","AverageBenefit","PurchaseFrequency"])
    scaled = seg_scaler.transform(input_df)
    cluster = seg_model.predict(scaled)[0]
    persona = seg_personas.get(cluster, "Unknown")
    return cluster, persona

# -------------------------
# Tabs
# -------------------------
tab1, tab2, tab3 = st.tabs(["🚚 Delivery Prediction", "👥 Customer Segmentation", "📈 Demand Forecasting"])

# =====================================================
# 🚚 Tab 1: Delivery Prediction
# =====================================================
with tab1:
    st.subheader("Late Delivery Prediction")
    if delivery_model is None:
        st.error("❌ Delivery prediction model not loaded.")
    else:
        # Inputs (same as training features, but no Lat/Long)
        Days_for_shipment_scheduled = st.number_input("Days for shipment scheduled", min_value=1, max_value=60, value=5)
        Order_Item_Quantity = st.number_input("Order Item Quantity", min_value=1, max_value=100, value=1)
        Shipping_Mode = st.selectbox("Shipping Mode", ["Standard Class", "Second Class", "First Class", "Same Day"])
        Order_Region = st.selectbox("Order Region", ["East", "West", "Central", "South"])
        Order_State = st.selectbox("Order State", ["California", "Texas", "New York", "Florida", "Other"])
        Category_Name = st.selectbox("Category Name", ["Furniture", "Office Supplies", "Technology"])
        Department_Name = st.selectbox("Department Name", ["Sales", "Operations", "Marketing", "Finance", "Other"])

        if st.button("Predict Delivery"):
            input_data = {
                "Days_for_shipment_scheduled": Days_for_shipment_scheduled,
                "Order_Item_Quantity": Order_Item_Quantity,
                "Shipping_Mode": Shipping_Mode,
                "Order_Region": Order_Region,
                "Order_State": Order_State,
                "Category_Name": Category_Name,
                "Department_Name": Department_Name,
            }
            proba = predict_delivery(input_data)
            st.progress(int(proba[1]*100))
            if proba[1] > 0.5:
                st.error(f"⚠️ High risk of Late Delivery ({proba[1]*100:.2f}%)")
            else:
                st.success(f"✅ Likely On-Time Delivery ({proba[0]*100:.2f}%)")

# =====================================================
# 👥 Tab 2: Customer Segmentation
# =====================================================
with tab2:
    st.subheader("Customer Segmentation")
    if seg_model is None:
        st.error("❌ Segmentation model not loaded.")
    else:
        total_sales = st.number_input("Total Sales ($)", min_value=0.0, value=500.0)
        avg_benefit = st.number_input("Average Benefit per Order ($)", value=50.0)
        purchase_freq = st.number_input("Purchase Frequency", min_value=1, value=5)

        if st.button("Predict Segment"):
            cluster, persona = predict_segmentation(total_sales, avg_benefit, purchase_freq)
            st.success(f"🧑 Customer assigned to Segment {cluster}: **{persona}**")

# =====================================================
# 📈 Tab 3: Demand Forecasting
# =====================================================
with tab3:
    st.subheader("Product Demand Forecasting")
    if forecast_model is None:
        st.error("❌ Forecasting model not loaded.")
    else:
        days_to_forecast = st.slider("Days to Forecast", min_value=7, max_value=180, value=30)
        if st.button("Generate Forecast"):
            try:
                forecast = forecast_model.get_forecast(steps=days_to_forecast)
                pred_mean = forecast.predicted_mean
                pred_ci = forecast.conf_int()

                fig, ax = plt.subplots(figsize=(10, 5))
                pred_mean.plot(ax=ax, label="Forecast", color="red")
                ax.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color="pink", alpha=0.3)
                ax.legend()
                st.pyplot(fig)
            except Exception as e:
                st.error(f"❌ Forecasting failed: {e}")
