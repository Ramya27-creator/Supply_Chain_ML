# -*- coding: utf-8 -*-
"""
Supply Chain ML Dashboard (Streamlit)
- Model 1: Late Delivery Prediction (dynamic inputs)
- Model 2: Customer Segmentation
- Model 3: Product Demand Forecasting
"""

import streamlit as st
import pandas as pd
import joblib
import zipfile
import os
import warnings
import matplotlib.pyplot as plt
import statsmodels.api as sm

warnings.filterwarnings("ignore")
st.set_page_config(page_title="🚀 Supply Chain ML Dashboard", layout="wide")

st.markdown(
    """
    <h1 style='text-align:center; color: #4B0082;'>🚀 Supply Chain ML Dashboard</h1>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Load dataset from DataCo.zip
# -----------------------------
@st.cache_data(show_spinner=True)
def load_dataco(zip_path="DataCo.zip"):
    if not os.path.exists(zip_path):
        st.error(f"❌ {zip_path} not found in repo root.")
        return None

    with zipfile.ZipFile(zip_path, "r") as z:
        csv_files = [f for f in z.namelist() if f.endswith(".csv")]
        if not csv_files:
            st.error("❌ No CSV file found inside DataCo.zip")
            return None
        with z.open(csv_files[0]) as f:
            df = pd.read_csv(f)
    return df

# -----------------------------
# Load ML models
# -----------------------------
@st.cache_resource(show_spinner=True)
def load_models():
    try:
        # Delivery model (.zip containing a single .joblib inside)
        with zipfile.ZipFile("delivery_prediction_model.zip", "r") as z:
            model_file = z.namelist()[0]
            with z.open(model_file) as f:
                delivery_model = joblib.load(f)

        # Customer segmentation models
        seg_model = joblib.load("customer_segmentation_model.joblib")
        seg_scaler = joblib.load("customer_segmentation_scaler.joblib")
        seg_personas = joblib.load("customer_segmentation_personas.joblib")

        # Demand forecasting model
        forecast_model = joblib.load("demand_forecasting_model.joblib")

        return delivery_model, seg_model, seg_scaler, seg_personas, forecast_model

    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return None, None, None, None, None

# -----------------------------
# Load data and models
# -----------------------------
with st.spinner("📦 Loading dataset..."):
    df = load_dataco("DataCo.zip")

with st.spinner("🤖 Loading ML models..."):
    delivery_model, seg_model, seg_scaler, seg_personas, forecast_model = load_models()

if df is None:
    st.stop()

# -----------------------------
# Streamlit Tabs (ML tabs)
# -----------------------------
tab1, tab2, tab3 = st.tabs(
    ["🚚 Delivery Prediction", "👥 Customer Segmentation", "📈 Demand Forecasting"]
)

# ============================================================
# 🚚 Model 1: Late Delivery Prediction
# ============================================================
with tab1:
    st.subheader("🚚 Late Delivery Prediction")

    if delivery_model is None:
        st.error("❌ Delivery prediction model not loaded.")
    else:
        st.info("Enter shipment and order details below to predict delivery status.")

        # Dynamic dropdowns
        shipping_mode_options = sorted(df['Shipping_Mode'].dropna().unique())
        region_options = sorted(df['Order_Region'].dropna().unique())
        state_options = sorted(df['Order_State'].dropna().unique())
        category_options = sorted(df['Category_Name'].dropna().unique())
        department_options = sorted(df['Department_Name'].dropna().unique())

        col1, col2 = st.columns(2)

        with col1:
            days_for_shipment = st.number_input("Days for shipment (scheduled)", min_value=1, max_value=30, value=5)
            shipping_mode = st.selectbox("Shipping Mode", shipping_mode_options)
            order_region = st.selectbox("Order Region", region_options)
            order_state = st.selectbox("Order State", state_options)

        with col2:
            order_item_qty = st.number_input("Order Item Quantity", min_value=1, max_value=50, value=2)
            category_name = st.selectbox("Category Name", category_options)
            department_name = st.selectbox("Department Name", department_options)
            latitude = st.number_input("Latitude", value=37.77)
            longitude = st.number_input("Longitude", value=-122.41)

        if st.button("🔮 Predict Delivery Status"):
            input_df = pd.DataFrame([{
                'Days_for_shipment_scheduled': days_for_shipment,
                'Shipping_Mode': shipping_mode,
                'Order_Region': order_region,
                'Order_State': order_state,
                'Order_Item_Quantity': order_item_qty,
                'Category_Name': category_name,
                'Department_Name': department_name,
                'Latitude': latitude,
                'Longitude': longitude
            }])

            try:
                proba = delivery_model.predict_proba(input_df)[0]
                pred_class = delivery_model.predict(input_df)[0]

                if pred_class == 1:
                    st.error(f"⚠️ High Risk of Late Delivery (Probability: {proba[1]:.2f})")
                else:
                    st.success(f"✅ On-Time Delivery Likely (Probability: {proba[0]:.2f})")
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")

# ============================================================
# 👥 Model 2: Customer Segmentation
# ============================================================
with tab2:
    st.subheader("👥 Customer Segmentation")

    if seg_model is None:
        st.error("❌ Segmentation model not loaded.")
    else:
        st.info("Enter customer metrics to predict their segment/persona.")

        col1, col2 = st.columns(2)

        with col1:
            total_sales = st.number_input("Total Sales ($)", min_value=0.0, value=500.0)
            avg_benefit = st.number_input("Average Benefit per Order ($)", value=50.0)

        with col2:
            purchase_freq = st.number_input("Purchase Frequency", min_value=1, value=5)

        if st.button("🔍 Predict Segment"):
            input_df = pd.DataFrame([[total_sales, avg_benefit, purchase_freq]],
                                    columns=['TotalSales', 'AverageBenefit', 'PurchaseFrequency'])
            try:
                scaled = seg_scaler.transform(input_df)
                cluster = seg_model.predict(scaled)[0]
                persona = seg_personas.get(cluster, "Unknown")
                st.success(f"🧑 Customer assigned to Segment: **{cluster} ({persona})**")
            except Exception as e:
                st.error(f"❌ Segmentation failed: {e}")

# ============================================================
# 📈 Model 3: Product Demand Forecasting
# ============================================================
with tab3:
    st.subheader("📈 Product Demand Forecasting")

    if forecast_model is None:
        st.error("❌ Forecast model not loaded.")
    else:
        st.info("Forecast future demand based on historical order quantities.")

        days_to_forecast = st.slider("Days to Forecast", min_value=7, max_value=180, value=30)

        if st.button("📊 Generate Forecast"):
            try:
                forecast = forecast_model.get_forecast(steps=days_to_forecast)
                pred_mean = forecast.predicted_mean
                pred_ci = forecast.conf_int()

                fig, ax = plt.subplots(figsize=(10, 5))
                df['order_date'] = pd.to_datetime(df['order_date_DateOrders'], errors='coerce')
                daily_sales = df.groupby('order_date')['Order_Item_Quantity'].sum().reset_index()
                daily_sales.set_index('order_date', inplace=True)
                daily_sales = daily_sales.asfreq('D').fillna(0)

                daily_sales.last('90D').plot(ax=ax, label="Observed", color="blue")
                pred_mean.plot(ax=ax, label="Forecast", color="red")
                ax.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color="pink", alpha=0.3)
                ax.legend()
                st.pyplot(fig)

            except Exception as e:
                st.error(f"❌ Forecasting failed: {e}")
