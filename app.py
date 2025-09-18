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
# Load Model
# -------------------------
@st.cache_resource
def load_model():
    try:
        if os.path.exists("delivery_prediction_model.joblib"):
            model = joblib.load("delivery_prediction_model.joblib")
        elif os.path.exists("delivery_prediction_model.zip"):
            with zipfile.ZipFile("delivery_prediction_model.zip", "r") as z:
                z.extractall()
            model = joblib.load("delivery_prediction_model.joblib")
        else:
            st.error("❌ Model file not found! Please place 'delivery_prediction_model.joblib' or zip in app folder.")
            return None
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

delivery_model = load_model()

# -------------------------
# Prediction Functions
# -------------------------
def predict_delivery_single(input_data: dict):
    df = pd.DataFrame([input_data])
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

    required_cols = delivery_model.feature_names_in_

    # Fill missing columns with defaults
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0

    df = df[required_cols]
    proba = delivery_model.predict_proba(df)[0]
    return f"On Time: {round(proba[0]*100,2)}%, Late: {round(proba[1]*100,2)}%"

def predict_delivery_file(df_file: pd.DataFrame):
    df_file = df_file.copy()
    df_file.columns = [c.strip().replace(" ", "_") for c in df_file.columns]

    required_cols = delivery_model.feature_names_in_

    for col in required_cols:
        if col not in df_file.columns:
            df_file[col] = 0

    df_file = df_file[required_cols]
    proba_list = delivery_model.predict_proba(df_file)

    results = []
    for idx, p in enumerate(proba_list):
        if idx < 10:
            results.append(f"Row {idx+1}: On Time: {round(p[0]*100,2)}%, Late: {round(p[1]*100,2)}%")
    if len(proba_list) > 10:
        results.append(f"...and {len(proba_list)-10} more rows not displayed")

    return "\n".join(results)

# -------------------------
# UI
# -------------------------
if delivery_model is not None:
    st.sidebar.header("Choose Input Method")
    option = st.sidebar.radio("Select input type:", ["Manual Entry", "Upload CSV File"])

    # Manual Entry
    if option == "Manual Entry":
        st.subheader("Enter order details manually")

        # Numeric
        Days_for_shipment_scheduled = st.number_input("Days for shipment scheduled", min_value=1, max_value=60, value=5)
        Order_Item_Quantity = st.number_input("Order Item Quantity", min_value=1, max_value=100, value=1)

        # Dropdowns
        Shipping_Mode = st.selectbox("Shipping Mode", ["Standard Class", "Second Class", "First Class", "Same Day"])
        Order_Region = st.selectbox("Order Region", ["East", "West", "Central", "South"])
        Order_State = st.selectbox("Order State", ["California", "Texas", "New York", "Florida", "Other"])
        Category_Name = st.selectbox("Category Name", ["Furniture", "Office Supplies", "Technology"])
        Department_Name = st.selectbox("Department Name", ["Sales", "Operations", "Marketing", "Finance", "Other"])

        input_data = {
            "Days_for_shipment_scheduled": Days_for_shipment_scheduled,
            "Order_Item_Quantity": Order_Item_Quantity,
            "Shipping_Mode": Shipping_Mode,
            "Order_Region": Order_Region,
            "Order_State": Order_State,
            "Category_Name": Category_Name,
            "Department_Name": Department_Name,
        }

        if st.button("Predict"):
            result = predict_delivery_single(input_data)
            st.success(result)

    # File Upload
    elif option == "Upload CSV File":
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
        if uploaded_file is not None:
            try:
                # Try utf-8, fallback to latin1
                try:
                    df_upload = pd.read_csv(uploaded_file, encoding="utf-8", on_bad_lines="skip")
                except UnicodeDecodeError:
                    df_upload = pd.read_csv(uploaded_file, encoding="latin1", on_bad_lines="skip")

                st.write("Preview of uploaded data:")
                st.dataframe(df_upload.head())

                if st.button("Predict from File"):
                    result = predict_delivery_file(df_upload)
                    st.text_area("Prediction Results", result, height=250)

            except Exception as e:
                st.error(f"❌ Error reading file: {e}")
