# app.py
import os

# -------------------------
# Limit threads to avoid sklearn / threadpoolctl error on Windows
# -------------------------
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

# -------------------------
# Imports
# -------------------------
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import statsmodels.api as sm
import zipfile
import gdown

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(page_title="Supply Chain ML Dashboard", layout="wide")
st.title("🚀 Supply Chain ML Dashboard")

# -------------------------
# Model URLs on Google Drive
# -------------------------
MODEL_URLS = {
    "delivery": "https://drive.google.com/uc?id=1BXZV305GFW7akNcXbCUvjTJfV8tFWvh7",
    "forecast": "https://drive.google.com/uc?id=1Zi_TYm438ougW-UVnIx54hlDG6WKFC-l"
}

# -------------------------
# Load Models (Cached)
# -------------------------
@st.cache_resource
def load_models():
    """Download and load ML models"""

    # --- Delivery Model ---
    if not os.path.exists("delivery_prediction_model.zip"):
        gdown.download(MODEL_URLS["delivery"], "delivery_prediction_model.zip", quiet=False)
    with zipfile.ZipFile("delivery_prediction_model.zip", "r") as z:
        with z.open("delivery_prediction_model.joblib") as f:
            delivery_model = joblib.load(f)

    # --- Forecasting Model ---
    if not os.path.exists("demand_forecasting_model.zip"):
        gdown.download(MODEL_URLS["forecast"], "demand_forecasting_model.zip", quiet=False)
    with zipfile.ZipFile("demand_forecasting_model.zip", "r") as z:
        with z.open("demand_forecasting_model.joblib") as f:
            forecast_model = joblib.load(f)

    # --- Customer Segmentation Models (Local) ---
    seg_model = joblib.load(r"C:\Users\Dell\Desktop\Internship_DataAnalyst_Projects\Supply Chain\project\customer_segmentation_model.joblib")
    seg_scaler = joblib.load(r"C:\Users\Dell\Desktop\Internship_DataAnalyst_Projects\Supply Chain\project\customer_segmentation_scaler.joblib")
    seg_personas = joblib.load(r"C:\Users\Dell\Desktop\Internship_DataAnalyst_Projects\Supply Chain\project\customer_segmentation_personas.joblib")

    return delivery_model, seg_model, seg_scaler, seg_personas, forecast_model


delivery_model, seg_model, seg_scaler, seg_personas, forecast_model = load_models()

# -------------------------
# Load Data (Cached)
# -------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("DataCo.csv", encoding="latin1", low_memory=False)
    df["order_date"] = pd.to_datetime(df["order_date_DateOrders"], errors="coerce")
    df.dropna(subset=["order_date"], inplace=True)
    return df

df = load_data()

# -------------------------
# Tabs (Same as before)
# -------------------------
tab1, tab2, tab3 = st.tabs([
    "🚚 Late Delivery Prediction",
    "👥 Customer Segmentation",
    "📈 Product Demand Forecasting"
])

# --- Tab 1: Late Delivery Prediction ---
with tab1:
    st.header("Late Delivery Prediction")
    st.markdown("### Enter Order Details")

    col1, col2 = st.columns(2)
    with col1:
        scheduled_days = st.slider("Days for Shipment (Scheduled)", 1, 30, 5)
        shipping_mode = st.selectbox("Shipping Mode", df["Shipping_Mode"].dropna().unique())
        region = st.selectbox("Order Region", df["Order_Region"].dropna().unique())
        state = st.selectbox("Order State", df["Order_State"].dropna().unique())
    with col2:
        quantity = st.slider("Order Item Quantity", 1, 50, 1)
        category = st.selectbox("Category Name", df["Category_Name"].dropna().unique())
        department = st.selectbox("Department Name", df["Department_Name"].dropna().unique())
        latitude = st.number_input("Latitude", value=0.0)
        longitude = st.number_input("Longitude", value=0.0)

    if st.button("🔮 Predict Delivery Risk", key="btn_delivery"):
        input_df = pd.DataFrame({
            "Days_for_shipment_scheduled": [scheduled_days],
            "Shipping_Mode": [shipping_mode],
            "Order_Region": [region],
            "Order_State": [state],
            "Order_Item_Quantity": [quantity],
            "Category_Name": [category],
            "Department_Name": [department],
            "Latitude": [latitude],
            "Longitude": [longitude]
        })
        proba = delivery_model.predict_proba(input_df)[0]
        st.success(f"✅ Probability Not Late (0): {proba[0]:.2f}")
        st.error(f"⚠️ Probability Late (1): {proba[1]:.2f}")

# --- Tab 2: Customer Segmentation ---
with tab2:
    st.header("Customer Segmentation")

    @st.cache_data
    def load_customer_data(df):
        df_customer = df.groupby("Customer_Id").agg({
            "Sales": "sum",
            "Benefit_per_order": "mean",
            "order_date_DateOrders": "count"
        }).reset_index()
        df_customer.columns = ["CustomerID", "TotalSales", "AverageBenefit", "PurchaseFrequency"]
        return df_customer

    customer_df = load_customer_data(df)

    option = st.radio("Choose Input Mode:", ["Manual Entry", "Select Customer from Data"])
    if option == "Manual Entry":
        total_sales = st.slider("Total Sales ($)", 0, 50000, 1000, step=100)
        avg_benefit = st.slider("Average Benefit per Order ($)", -500, 500, 0, step=10)
        purchase_freq = st.slider("Purchase Frequency (#Orders)", 1, 50, 1)
        input_df = pd.DataFrame([[total_sales, avg_benefit, purchase_freq]],
                                columns=["TotalSales", "AverageBenefit", "PurchaseFrequency"])
    else:
        selected_customer = st.selectbox("Select Customer", customer_df["CustomerID"].tolist())
        input_df = customer_df[customer_df["CustomerID"] == selected_customer][
            ["TotalSales", "AverageBenefit", "PurchaseFrequency"]
        ]

    if st.button("🔮 Predict Customer Segment", key="btn_segmentation"):
        input_scaled = seg_scaler.transform(input_df)
        cluster = seg_model.predict(input_scaled)[0]
        persona = seg_personas.get(cluster, "Unknown Segment")
        st.success(f"Predicted Cluster: {cluster}")
        st.info(f"Persona: {persona}")

    if st.checkbox("📊 Show Cluster Distribution"):
        customer_scaled = seg_scaler.transform(customer_df[["TotalSales", "AverageBenefit", "PurchaseFrequency"]])
        customer_df["Cluster"] = seg_model.predict(customer_scaled)
        fig, ax = plt.subplots()
        customer_df["Cluster"].value_counts().sort_index().plot(kind="bar", ax=ax, color="skyblue")
        ax.set_title("Customer Count per Cluster")
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Number of Customers")
        st.pyplot(fig)

# --- Tab 3: Product Demand Forecasting ---
with tab3:
    st.header("Product Demand Forecasting")

    product_list = ["All Products"] + sorted(df["Product_Name"].dropna().unique().tolist())
    selected_product = st.selectbox("Select Product", product_list)

    if selected_product != "All Products":
        product_sales = df[df["Product_Name"] == selected_product] \
                          .groupby("order_date")["Order_Item_Quantity"].sum().asfreq("D").fillna(0)
    else:
        product_sales = df.groupby("order_date")["Order_Item_Quantity"].sum().asfreq("D").fillna(0)

    days_to_forecast = st.slider("Days to Forecast", 7, 180, 30, key="forecast_days")

    if st.button("📈 Generate Forecast", key="btn_forecast"):
        try:
            if selected_product == "All Products":
                forecast = forecast_model.get_forecast(steps=days_to_forecast)
                pred_mean = forecast.predicted_mean
                pred_ci = forecast.conf_int()
            else:
                model = sm.tsa.statespace.SARIMAX(
                    product_sales,
                    order=(1, 1, 1),
                    seasonal_order=(1, 1, 1, 7),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                results = model.fit(disp=False)
                forecast = results.get_forecast(steps=days_to_forecast)
                pred_mean = forecast.predicted_mean
                pred_ci = forecast.conf_int()

            last_90d = product_sales.loc[product_sales.index >= (product_sales.index.max() - pd.Timedelta(days=90))]
            fig, ax = plt.subplots(figsize=(12, 6))
            last_90d.plot(ax=ax, label="Observed Sales", color="blue")
            pred_mean.plot(ax=ax, label="Forecast", color="red")
            ax.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1],
                            color="pink", alpha=0.5)
            ax.set_title(f"{selected_product} - Forecast for Next {days_to_forecast} Days")
            ax.set_xlabel("Date")
            ax.set_ylabel("Quantity")
            ax.legend()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error generating forecast: {e}")
