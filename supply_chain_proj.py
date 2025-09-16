# -*- coding: utf-8 -*-
"""Supply_Chain_proj.py

Updated full pipeline with safe DataCo.zip loader
- Model 1: Late Delivery Prediction
- Model 2: Customer Segmentation
- Model 3: Product Demand Forecasting
"""

import pandas as pd
import joblib
import gradio as gr
import zipfile
import os
import warnings

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans
import statsmodels.api as sm
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# -------------------------------------------------------------------
# Helper function: Load DataCo dataset from DataCo.zip
# -------------------------------------------------------------------
def load_dataco(zip_path="DataCo.zip"):
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"{zip_path} not found")

    with zipfile.ZipFile(zip_path, "r") as z:
        file_list = z.namelist()
        print("Files inside ZIP:", file_list)

        # Prefer Excel first
        excel_files = [f for f in file_list if f.endswith(".xlsx")]
        if excel_files:
            with z.open(excel_files[0]) as f:
                df = pd.read_excel(f, engine="openpyxl")
                print(f"Loaded Excel file: {excel_files[0]}")
                return df

        # Else try CSV
        csv_files = [f for f in file_list if f.endswith(".csv")]
        if csv_files:
            with z.open(csv_files[0]) as f:
                df = pd.read_csv(f)
                print(f"Loaded CSV file: {csv_files[0]}")
                return df

        raise ValueError("No Excel or CSV file found in DataCo.zip.")

# ============================================================
# Model 1: Late Delivery Prediction
# ============================================================

print("--- Model 1: Late Delivery Prediction ---")

df = load_dataco("DataCo.zip")

features = [
    'Days_for_shipment_scheduled', 'Shipping_Mode',
    'Order_Region', 'Order_State', 'Order_Item_Quantity',
    'Category_Name', 'Department_Name', 'Latitude', 'Longitude'
]
target = 'Late_delivery_risk'

df_model = df[features + [target]].dropna()
X = df_model[features]
y = df_model[target]

categorical_features = X.select_dtypes(include=['object']).columns

preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)],
    remainder='passthrough'
)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training Random Forest...")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

joblib.dump(model, "delivery_prediction_model.joblib")

# --- Gradio Interface ---
def predict_delivery(scheduled_days, shipping_mode, region, state, quantity, category, department, latitude, longitude):
    input_data = pd.DataFrame({
        'Days_for_shipment_scheduled': [scheduled_days],
        'Shipping_Mode': [shipping_mode],
        'Order_Region': [region],
        'Order_State': [state],
        'Order_Item_Quantity': [quantity],
        'Category_Name': [category],
        'Department_Name': [department],
        'Latitude': [latitude],
        'Longitude': [longitude]
    })
    proba = model.predict_proba(input_data)[0]
    return {"Not Late (0)": float(proba[0]), "Late (1)": float(proba[1])}

shipping_mode_options = sorted(df['Shipping_Mode'].dropna().unique().tolist())
region_options = sorted(df['Order_Region'].dropna().unique().tolist())
state_options = sorted(df['Order_State'].dropna().unique().tolist())
category_options = sorted(df['Category_Name'].dropna().unique().tolist())
department_options = sorted(df['Department_Name'].dropna().unique().tolist())

with gr.Blocks(theme=gr.themes.Soft()) as iface1:
    gr.Markdown("# 🚚 Late Delivery Prediction")
    with gr.Row():
        with gr.Column():
            scheduled_days_input = gr.Slider(1, 30, step=1, label="Days for Shipment (Scheduled)")
            shipping_mode_input = gr.Dropdown(shipping_mode_options, label="Shipping Mode")
            region_input = gr.Dropdown(region_options, label="Order Region")
            state_input = gr.Dropdown(state_options, label="Order State")
            quantity_input = gr.Slider(1, 50, step=1, label="Order Item Quantity")
            category_input = gr.Dropdown(category_options, label="Category Name")
            department_input = gr.Dropdown(department_options, label="Department Name")
            latitude_input = gr.Number(label="Latitude")
            longitude_input = gr.Number(label="Longitude")
            predict_btn = gr.Button("Predict")
        with gr.Column():
            output_label = gr.Label(label="Prediction")
    predict_btn.click(
        predict_delivery,
        [scheduled_days_input, shipping_mode_input, region_input, state_input,
         quantity_input, category_input, department_input, latitude_input, longitude_input],
        output_label
    )

iface1.launch(debug=True)

# ============================================================
# Model 2: Customer Segmentation
# ============================================================

print("\n--- Model 2: Customer Segmentation ---")

df = load_dataco("DataCo.zip")

customer_df = df.groupby('Customer_Id').agg(
    TotalSales=('Sales_per_customer', 'sum'),
    AverageBenefit=('Benefit_per_order', 'mean'),
    PurchaseFrequency=('Order_Id', 'count')
).reset_index()

features_df = customer_df[['TotalSales', 'AverageBenefit', 'PurchaseFrequency']].dropna()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features_df)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10).fit(X_scaled)
customer_df['Cluster'] = kmeans.labels_

cluster_analysis = customer_df.groupby('Cluster').mean()[['TotalSales', 'AverageBenefit', 'PurchaseFrequency']]
personas = {}
for i, row in cluster_analysis.iterrows():
    if row['TotalSales'] > cluster_analysis['TotalSales'].quantile(0.75):
        personas[i] = "🏆 High-Value Loyalist"
    elif row['PurchaseFrequency'] > cluster_analysis['PurchaseFrequency'].quantile(0.75):
        personas[i] = "🛒 Frequent Shopper"
    elif row['AverageBenefit'] < 0:
        personas[i] = "💸 Low-Profit Customer"
    else:
        personas[i] = "🌱 Occasional Shopper"

joblib.dump(kmeans, "customer_segmentation_model.joblib")
joblib.dump(scaler, "customer_segmentation_scaler.joblib")
joblib.dump(personas, "customer_personas.joblib")

def predict_segment(total_sales, avg_benefit, purchase_freq):
    input_data = pd.DataFrame([[total_sales, avg_benefit, purchase_freq]],
                              columns=['TotalSales', 'AverageBenefit', 'PurchaseFrequency'])
    scaled = scaler.transform(input_data)
    cluster = kmeans.predict(scaled)[0]
    return {"Cluster": int(cluster), "Persona": personas.get(cluster, "Unknown")}

with gr.Blocks(theme=gr.themes.Soft()) as iface2:
    gr.Markdown("# 👥 Customer Segmentation Tool")
    with gr.Row():
        with gr.Column():
            total_sales_input = gr.Number(label="Total Sales")
            avg_benefit_input = gr.Number(label="Average Benefit per Order")
            purchase_freq_input = gr.Number(label="Purchase Frequency")
            predict_btn = gr.Button("Predict")
        with gr.Column():
            output = gr.Label(label="Segment")
    predict_btn.click(
        predict_segment,
        [total_sales_input, avg_benefit_input, purchase_freq_input],
        output
    )

iface2.launch(debug=True)

# ============================================================
# Model 3: Product Demand Forecasting
# ============================================================

print("\n--- Model 3: Product Demand Forecasting ---")

df = load_dataco("DataCo.zip")
df['order_date'] = pd.to_datetime(df['order_date_DateOrders'], errors='coerce')
daily_sales = df.groupby('order_date')['Order_Item_Quantity'].sum().reset_index()
daily_sales.set_index('order_date', inplace=True)
daily_sales = daily_sales.asfreq('D').fillna(0)

model = sm.tsa.statespace.SARIMAX(
    daily_sales,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 7),
    enforce_stationarity=False,
    enforce_invertibility=False
).fit(disp=False)

joblib.dump(model, "demand_forecasting_model.joblib")

def forecast_demand(days_to_forecast):
    forecast = model.get_forecast(steps=int(days_to_forecast))
    pred_mean = forecast.predicted_mean
    pred_ci = forecast.conf_int()
    fig, ax = plt.subplots(figsize=(12, 6))
    daily_sales.last('90D').plot(ax=ax, label='Observed', color='blue')
    pred_mean.plot(ax=ax, label='Forecast', color='red')
    ax.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='pink', alpha=0.3)
    ax.legend()
    return fig

with gr.Blocks(theme=gr.themes.Soft()) as iface3:
    gr.Markdown("# 📈 Product Demand Forecasting")
    with gr.Row():
        with gr.Column(scale=1):
            days_input = gr.Slider(7, 180, value=30, step=1, label="Days to Forecast")
            predict_btn = gr.Button("Generate Forecast")
        with gr.Column(scale=3):
            output_plot = gr.Plot(label="Forecast Plot")
    predict_btn.click(forecast_demand, days_input, output_plot)

iface3.launch(debug=True)
