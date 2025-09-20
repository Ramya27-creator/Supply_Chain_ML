# -*- coding: utf-8 -*-
"""
Streamlit App for Supply Chain ML Dashboard
- Model 1: Late Delivery Prediction
- Model 2: Product Demand Forecasting
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

        # Forecasting model
        forecast_model = joblib.load("demand_forecasting_model.joblib") if os.path.exists("demand_forecasting_model.joblib") else None

        return delivery_model, forecast_model

    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return None, None

delivery_model, forecast_model = load_models()

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

# -------------------------
# Tabs
# -------------------------
tab1, tab2 = st.tabs(["🚚 Delivery Prediction", "📈 Demand Forecasting"])

# =====================================================
# 🚚 Tab 1: Delivery Prediction (with dynamic dropdowns)
# =====================================================
with tab1:
    st.subheader("Late Delivery Prediction")
    if delivery_model is None:
        st.error("❌ Delivery prediction model not loaded.")
    else:
        # Load dropdown options dynamically from dataset
        region_options, dept_options = [], []
        if os.path.exists("DataCo.zip"):
            with zipfile.ZipFile("DataCo.zip", "r") as z:
                csv_files = [f for f in z.namelist() if f.endswith(".csv")]
                excel_files = [f for f in z.namelist() if f.endswith(".xlsx")]
                if csv_files:
                    with z.open(csv_files[0]) as f:
                        df_main = pd.read_csv(f)
                elif excel_files:
                    with z.open(excel_files[0]) as f:
                        df_main = pd.read_excel(f, engine="openpyxl")
                else:
                    df_main = None

                if df_main is not None:
                    if "Order_Region" in df_main.columns:
                        region_options = sorted(df_main["Order_Region"].dropna().unique().tolist())
                    if "Department_Name" in df_main.columns:
                        dept_options = sorted(df_main["Department_Name"].dropna().unique().tolist())

        # Fallback defaults if dataset not found
        region_options = region_options if region_options else ["East", "West", "Central", "South"]
        dept_options = dept_options if dept_options else ["Sales", "Operations", "Marketing", "Finance", "Other"]

        # Inputs
        Days_for_shipment_scheduled = st.number_input("Days for shipment scheduled", min_value=1, max_value=60, value=5)
        Order_Item_Quantity = st.number_input("Order Item Quantity", min_value=1, max_value=100, value=1)
        Shipping_Mode = st.selectbox("Shipping Mode", ["Standard Class", "Second Class", "First Class", "Same Day"])
        Order_Region = st.selectbox("Order Region", region_options)
        Order_State = st.text_input("Order State")   # free text if not standardized
        Category_Name = st.selectbox("Category Name", ["Furniture", "Office Supplies", "Technology"])
        Department_Name = st.selectbox("Department Name", dept_options)

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
# 📈 Tab 2: Demand Forecasting (with historical + forecast)
# =====================================================
with tab2:
    st.subheader("Product Demand Forecasting")
    if forecast_model is None:
        st.error("❌ Forecasting model not loaded.")
    else:
        Product_Name = st.selectbox("Select Product", product_list if product_list else ["Product A", "Product B", "Product C"])
        days_to_forecast = st.slider("Days to Forecast", min_value=7, max_value=180, value=30)

        if st.button("Generate Forecast"):
            try:
                # Historical data for selected product
                if df is not None and "Product_Name" in df.columns and "Order_Item_Quantity" in df.columns and "order_date" in df.columns:
                    product_data = df[df["Product_Name"] == Product_Name]
                    product_data["order_date"] = pd.to_datetime(product_data["order_date"])
                    ts = product_data.groupby("order_date")["Order_Item_Quantity"].sum().asfreq("D").fillna(0)

                    # Fit forecasting model on historical data
                    fitted_model = forecast_model.fit(ts)  # re-fit if using a generic model
                    forecast = fitted_model.get_forecast(steps=days_to_forecast)
                    pred_mean = forecast.predicted_mean
                    pred_ci = forecast.conf_int()

                    # Plot
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ts.plot(ax=ax, label="Historical Demand", color="blue")
                    pred_mean.plot(ax=ax, label=f"Forecast for {Product_Name}", color="red")
                    ax.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color="pink", alpha=0.3)
                    ax.legend()
                    st.pyplot(fig)
                else:
                    st.error("⚠️ Historical data for product not available in dataset.")

            except Exception as e:
                st.error(f"❌ Forecasting failed: {e}")

                st.error(f"❌ Forecasting failed: {e}")
