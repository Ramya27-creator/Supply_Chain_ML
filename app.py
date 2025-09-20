# app.py
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

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
        late_delivery_model = joblib.load("late_delivery_model.pkl")
        demand_forecast_model = joblib.load("demand_forecast_model.pkl")
        return late_delivery_model, demand_forecast_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

late_delivery_model, demand_forecast_model = load_models()

# -------------------------
# Load Data
# -------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("DataCoSupplyChainDataset.csv", encoding="ISO-8859-1")
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

df = load_data()

# Extract lists for dropdowns
product_list = df["Product Name"].dropna().unique().tolist() if not df.empty else []
state_list = df["Order State"].dropna().unique().tolist() if not df.empty else []

# -------------------------
# Sidebar Navigation
# -------------------------
menu = ["Late Delivery Prediction", "Product Demand Forecasting"]
choice = st.sidebar.radio("Select Analysis", menu)

# -------------------------
# Late Delivery Prediction
# -------------------------
if choice == "Late Delivery Prediction":
    st.header("📦 Late Delivery Risk Prediction")

    col1, col2 = st.columns(2)

    with col1:
        days_for_shipping_scheduled = st.number_input("Days for shipment scheduled", min_value=1, max_value=30, value=5)
        shipping_mode = st.selectbox("Shipping Mode", df["Shipping Mode"].dropna().unique() if not df.empty else ["Standard Class", "Second Class", "First Class"])
        order_region = st.selectbox("Order Region", df["Order Region"].dropna().unique() if not df.empty else ["West", "East", "Central", "South"])

    with col2:
        order_state = st.selectbox("Order State", state_list if state_list else ["California", "Texas", "New York"])  # ✅ Added dropdown
        order_item_quantity = st.number_input("Order Item Quantity", min_value=1, value=1)
        department_name = st.selectbox("Department Name", df["Department Name"].dropna().unique() if not df.empty else ["Technology", "Furniture", "Office Supplies"])

    if st.button("Predict Late Delivery"):
        features = [[days_for_shipping_scheduled, order_item_quantity]]
        prediction = late_delivery_model.predict(features)[0]
        st.success("Prediction: 🚚 Late Delivery" if prediction == 1 else "Prediction: ✅ On-Time Delivery")

# -------------------------
# Product Demand Forecasting
# -------------------------
elif choice == "Product Demand Forecasting":
    st.header("📊 Product Demand Forecasting")

    selected_product = st.selectbox("Select Product", product_list if product_list else ["Product A", "Product B", "Product C"])
    forecast_days = st.slider("Days to Forecast", min_value=7, max_value=365, value=30, step=7)

    if st.button("Generate Forecast"):
        product_data = df[df["Product Name"] == selected_product].groupby("order date (DateOrders)")["Order Item Quantity"].sum()
        product_data.index = pd.to_datetime(product_data.index)
        product_data = product_data.resample("D").sum().fillna(0)

        # Forecast
        try:
            forecast = demand_forecast_model.forecast(steps=forecast_days)
            st.line_chart(forecast)
        except Exception as e:
            st.error(f"Error generating forecast: {e}")
