# app.py
import streamlit as st
import pandas as pd
import joblib
import zipfile
import os

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(page_title="Supply Chain Analysis Pipeline", layout="wide")
st.title("🚚 Late Delivery Prediction App")

# -------------------------
# Load Models
# -------------------------
@st.cache_resource
def load_models():
    try:
        model_path = "delivery_prediction_model.joblib"

        # If only .zip exists, extract it
        if not os.path.exists(model_path) and os.path.exists("delivery_prediction_model.zip"):
            with zipfile.ZipFile("delivery_prediction_model.zip", "r") as zip_ref:
                zip_ref.extractall(".")  # extract in current folder

        delivery_model = joblib.load(model_path)
        return delivery_model
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return None


delivery_model = load_models()

# -------------------------
# Prediction Functions
# -------------------------
def predict_delivery_single(input_data: dict):
    df = pd.DataFrame([input_data])
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

    required_cols = delivery_model.feature_names_in_

    # Add missing columns with default values
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0  # default fill

    df = df[required_cols]
    proba = delivery_model.predict_proba(df)[0]
    return f"On Time: {round(proba[0]*100,2)}%, Late: {round(proba[1]*100,2)}%"


def predict_delivery_file(df_file: pd.DataFrame):
    df_file = df_file.copy()
    df_file.columns = [c.strip().replace(" ", "_") for c in df_file.columns]

    required_cols = delivery_model.feature_names_in_

    # Add missing columns with default values
    for col in required_cols:
        if col not in df_file.columns:
            df_file[col] = 0

    df_file = df_file[required_cols]
    proba_list = delivery_model.predict_proba(df_file)

    # Show preview
    results = []
    for idx, p in enumerate(proba_list):
        if idx < 10:
            results.append(f"Row {idx+1}: On Time: {round(p[0]*100,2)}%, Late: {round(p[1]*100,2)}%")
    if len(proba_list) > 10:
        results.append(f"...and {len(proba_list)-10} more rows not displayed")

    return "\n".join(results)

# -------------------------
# App Tabs
# -------------------------
if delivery_model:
    tab1, tab2 = st.tabs(["📝 Manual Entry", "📂 Upload File"])

    # ---- Tab 1: Manual Entry ----
    with tab1:
        st.subheader("Manual Input for Prediction")

        col1, col2 = st.columns(2)
        with col1:
            Days_for_shipment_scheduled = st.number_input("Days for shipment scheduled", min_value=1, max_value=60, value=5)
            Shipping_Mode = st.selectbox("Shipping Mode", ["Standard Class", "Second Class", "First Class", "Same Day"])
            Order_Region = st.text_input("Order Region", "Central")
            Order_State = st.text_input("Order State", "California")
        with col2:
            Order_Item_Quantity = st.number_input("Order Item Quantity", min_value=1, max_value=100, value=2)
            Category_Name = st.text_input("Category Name", "Technology")
            Department_Name = st.text_input("Department Name", "Sales")

        if st.button("🔮 Predict Manually"):
            input_data = {
                "Days_for_shipment_scheduled": Days_for_shipment_scheduled,
                "Shipping_Mode": Shipping_Mode,
                "Order_Region": Order_Region,
                "Order_State": Order_State,
                "Order_Item_Quantity": Order_Item_Quantity,
                "Category_Name": Category_Name,
                "Department_Name": Department_Name
            }
            result = predict_delivery_single(input_data)
            st.success(result)

    # ---- Tab 2: Upload File ----
    with tab2:
        st.subheader("Upload a CSV File for Batch Prediction")
        uploaded_file = st.file_uploader("Upload your input CSV", type=["csv"])

        if uploaded_file is not None:
            try:
                df_upload = pd.read_csv(uploaded_file, encoding="utf-8", on_bad_lines="skip")
            except UnicodeDecodeError:
                df_upload = pd.read_csv(uploaded_file, encoding="latin1", on_bad_lines="skip")

            st.write("📄 File Preview:")
            st.dataframe(df_upload.head())
            result = predict_delivery_file(df_upload)
            st.success(result)
