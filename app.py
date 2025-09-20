# app.py
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import zipfile

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(page_title="Supply Chain Analysis Pipeline", layout="wide")
st.title("🚀 Supply Chain Analysis Pipeline")

# -------------------------
# Load Models
# -------------------------
@st.cache_resource
def load_models():
    try:
        late_delivery_model = joblib.load("delivery_prediction_model.zip")
        demand_forecast_model = joblib.load("demand_forecasting_model.joblib")
        return late_delivery_model, demand_forecast_model
    except Exception as e:
        st.error(f"⚠️ Error loading models: {e}")
        return None, None

late_delivery_model, demand_forecast_model = load_models()

# -------------------------
# Load Data
# -------------------------
@st.cache_data
def load_data():
    try:
        with zipfile.ZipFile("DataCo.zip") as z:
            # Assumes the first file inside is your CSV
            file_name = z.namelist()[0]
            df = pd.read_csv(z.open(file_name), encoding="ISO-8859-1")
        return df
    except Exception as e:
        st.error(f"⚠️ Error loading data: {e}")
        return pd.DataFrame()

df = load_data()

# Extract dropdown lists
product_list = df["Product Name"].dropna().unique().tolist() if not df.empty else []
state_list = df["Order State"].dropna().unique().tolist() if not df.empty else []

# -------------------------
# Tabs Navigation
# -------------------------
tab1, tab2 = st.tabs(["📦 Late Delivery Prediction", "📊 Product Demand Forecasting"])

# -------------------------
# Tab 1: Late Delivery Prediction
# -------------------------
with tab1:
    st.subheader("Predict Late Delivery Risk")

    col1, col2 = st.columns(2)

    with col1:
        days_for_shipping_scheduled = st.number_input("Days for shipment scheduled", min_value=1, max_value=30, value=5)
        shipping_mode = st.selectbox("Shipping Mode", df["Shipping Mode"].dropna().unique() if not df.empty else ["Standard Class", "Second Class", "First Class"])
        order_region = st.selectbox("Order Region", df["Order Region"].dropna().unique() if not df.empty else ["West", "East", "Central", "South"])

    with col2:
        order_state = st.selectbox("Order State", state_list if state_list else ["California", "Texas", "New York"])
        order_item_quantity = st.number_input("Order Item Quantity", min_value=1, value=1)
        department_name = st.selectbox("Department Name", df["Department Name"].dropna().unique() if not df.empty else ["Technology", "Furniture", "Office Supplies"])

    if st.button("Predict Late Delivery"):
        if late_delivery_model:
            # ⚠️ Make sure this matches the order of features your model was trained on
            features = [[days_for_shipping_scheduled, order_item_quantity]]
            prediction = late_delivery_model.predict(features)[0]
            st.success("🚚 Late Delivery" if prediction == 1 else "✅ On-Time Delivery")
        else:
            st.error("Late Delivery Model not loaded!")

# -------------------------
# Tab 2: Product Demand Forecasting
# -------------------------
with tab2:
    st.subheader("Forecast Product Demand")

    selected_product = st.selectbox("Select Product", product_list if product_list else ["Product A", "Product B", "Product C"])
    forecast_days = st.slider("Days to Forecast", min_value=7, max_value=365, value=30, step=7)

    if st.button("Generate Forecast"):
        if demand_forecast_model:
            product_data = df[df["Product Name"] == selected_product].groupby("order date (DateOrders)")["Order Item Quantity"].sum()
            product_data.index = pd.to_datetime(product_data.index)
            product_data = product_data.resample("D").sum().fillna(0)

            try:
                forecast = demand_forecast_model.forecast(steps=forecast_days)
                st.line_chart(forecast)
            except Exception as e:
                st.error(f"⚠️ Error generating forecast: {e}")
        else:
            st.error("Demand Forecast Model not loaded!")
