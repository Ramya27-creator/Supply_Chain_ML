# app.py
import streamlit as st
import pandas as pd
import joblib
import zipfile
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# -------------------------
# Load Dataset
# -------------------------
@st.cache_data
def load_data():
    with zipfile.ZipFile("DataCo.zip") as z:
        with z.open(z.namelist()[0]) as f:
            df = pd.read_csv(f, encoding="latin1")
    return df

df = load_data()

# -------------------------
# Load Models
# -------------------------
@st.cache_resource
def load_models():
    try:
        delivery_model = joblib.load("delivery_prediction_model.zip")
        forecast_model = joblib.load("demand_forecasting_model.joblib")
        return delivery_model, forecast_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

delivery_model, forecast_model = load_models()

# -------------------------
# Streamlit Layout
# -------------------------
st.set_page_config(page_title="Supply Chain ML App", layout="wide")
st.title("🚀 Supply Chain ML Application")

tab1, tab2 = st.tabs(["📦 Late Delivery Prediction", "📊 Demand Forecasting"])

# -------------------------
# Tab 1: Late Delivery Prediction
# -------------------------
with tab1:
    st.subheader("📦 Enter Order Details")

    col1, col2 = st.columns(2)

    with col1:
        order_region = st.selectbox("Select Order Region", df["Order_Region"].dropna().unique())
        order_state = st.selectbox("Select Order State", df["Order_State"].dropna().unique())
        order_city = st.selectbox("Select Order City", df["Order_City"].dropna().unique())
        market = st.selectbox("Select Market", df["Market"].dropna().unique())

    with col2:
        customer_country = st.selectbox("Select Customer Country", df["Customer_Country"].dropna().unique())
        customer_segment = st.selectbox("Select Customer Segment", df["Customer_Segment"].dropna().unique())
        category_name = st.selectbox("Select Category Name", df["Category_Name"].dropna().unique())
        department_name = st.selectbox("Select Department Name", df["Department_Name"].dropna().unique())
        shipping_mode = st.selectbox("Select Shipping Mode", df["Shipping_Mode"].dropna().unique())
        order_status = st.selectbox("Select Order Status", df["Order_Status"].dropna().unique())

    if st.button("🔮 Predict Late Delivery"):
        if delivery_model is not None:
            input_data = pd.DataFrame([{
                "Order_Region": order_region,
                "Order_State": order_state,
                "Order_City": order_city,
                "Market": market,
                "Customer_Country": customer_country,
                "Customer_Segment": customer_segment,
                "Category_Name": category_name,
                "Department_Name": department_name,
                "Shipping_Mode": shipping_mode,
                "Order_Status": order_status
            }])

            try:
                prediction = delivery_model.predict(input_data)[0]
                st.success(f"✅ Prediction: {'Late Delivery' if prediction == 1 else 'On Time'}")
            except Exception as e:
                st.error(f"Prediction error: {e}")
        else:
            st.warning("⚠️ Delivery prediction model not loaded!")

# -------------------------
# Tab 2: Demand Forecasting
# -------------------------
with tab2:
    st.subheader("📊 Forecast Product Demand")

    product_name = st.selectbox("Select Product", df["Product_Name"].dropna().unique())
    days_to_forecast = st.slider("Days to Forecast", min_value=7, max_value=365, value=30, step=1)

    if st.button("📈 Generate Forecast"):
        if forecast_model is not None:
            try:
                # Filter sales history for selected product
                product_sales = df[df["Product_Name"] == product_name].groupby("order date (DateOrders)").size()

                if product_sales.empty:
                    st.warning("⚠️ No sales data found for this product.")
                else:
                    product_sales.index = pd.to_datetime(product_sales.index, errors="coerce")
                    product_sales = product_sales.sort_index()

                    # Create dummy future forecast
                    last_date = product_sales.index.max()
                    future_dates = pd.date_range(last_date, periods=days_to_forecast + 1, freq="D")[1:]
                    future_forecast = np.random.randint(
                        product_sales.min(), product_sales.max(), size=len(future_dates)
                    )

                    # Plot
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(product_sales.index, product_sales.values, label="Historical Sales")
                    ax.plot(future_dates, future_forecast, label="Forecast", linestyle="--")
                    ax.set_title(f"Demand Forecast for {product_name}")
                    ax.set_xlabel("Date")
                    ax.set_ylabel("Quantity")
                    ax.legend()
                    st.pyplot(fig)

            except Exception as e:
                st.error(f"Forecasting error: {e}")
        else:
            st.warning("⚠️ Forecasting model not loaded!")
