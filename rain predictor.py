# Step 1: Import Libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Step 2: Load Dataset
df = pd.read_csv(r"C:\Users\sumit\python code of crt\rain_forecasting.csv")

# Step 3: Data Cleaning
df['RainToday'] = df['RainToday'].map({'No': 0, 'Yes': 1})
df['RainTomorrow'] = df['RainTomorrow'].map({'No': 0, 'Yes': 1})

# Drop rows with missing target or input data
df.dropna(subset=['RainTomorrow'], inplace=True)
df.dropna(inplace=True)

# Step 4: Select Features and Target
features = ['MinTemp', 'MaxTemp', 'Humidity9am', 'Humidity3pm',
            'Pressure9am', 'Pressure3pm', 'WindSpeed9am', 'WindSpeed3pm', 'RainToday']
X = df[features]
y = df['RainTomorrow']

# Step 5: Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train the Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 7: Evaluate the Model
y_pred = model.predict(X_test)

print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# Step 8: Save the Model
import joblib
joblib.dump(model, "rain_predictor_model.pkl")


# Load the model
model = joblib.load("rain_predictor_model.pkl")

# Load dataset (for charts in sidebar)
df = pd.read_csv(r"C:\Users\sumit\python code of crt\rain_forecasting.csv")
df['RainTomorrow'] = df['RainTomorrow'].map({'No': 0, 'Yes': 1})
df['RainToday'] = df['RainToday'].map({'No': 0, 'Yes': 1})
df.dropna(inplace=True)

# --- MAIN TITLE ---
st.set_page_config(page_title="Rain Forecasting App 🌧️", layout="wide")
st.title("☁️ Smart Rain Forecasting App")
st.markdown("""
Welcome to the **Rain Forecasting App** 🌦️  
Enter today's weather data to predict whether it will rain **tomorrow**.  
This helps farmers, gardeners, and water management planners to make sustainable decisions.  
""")

# --- SIDEBAR ANALYSIS ---
st.sidebar.title("📊 Data Insights")

if st.sidebar.checkbox("Show Rain Tomorrow Count"):
    rain_count = df['RainTomorrow'].value_counts()
    st.sidebar.bar_chart(rain_count)

# --- USER INPUT SECTION ---
st.header("🧪 Enter Today's Weather Data:")

col1, col2 = st.columns(2)

with col1:
    min_temp = st.number_input("Minimum Temperature (°C)", -10.0, 50.0, step=0.1)
    humidity_9am = st.slider("Humidity at 9am", 0.0, 100.0, step=0.01)
    pressure_9am = st.number_input("Pressure at 9am (hPa)", 950.0, 1050.0, step=0.1)
    wind_speed_9am = st.number_input("Wind Speed at 9am (km/h)", 0.0, 100.0, step=0.1)

with col2:
    max_temp = st.number_input("Maximum Temperature (°C)", -10.0, 60.0, step=0.1)
    humidity_3pm = st.slider("Humidity at 3pm", 0.0, 100.0, step=0.01)
    pressure_3pm = st.number_input("Pressure at 3pm (hPa)", 950.0, 1050.0, step=0.1)
    wind_speed_3pm = st.number_input("Wind Speed at 3pm (km/h)", 0.0, 100.0, step=0.1)

rain_today = st.radio("Did it rain today?", ['No', 'Yes'])
rain_today_binary = 1 if rain_today == "Yes" else 0

# --- PREDICTION ---
if st.button("🌧️ Predict Rain Tomorrow"):
    input_data = np.array([[min_temp, max_temp, humidity_9am, humidity_3pm,
                            pressure_9am, pressure_3pm, wind_speed_9am, wind_speed_3pm, rain_today_binary]])
    prediction = model.predict(input_data)

    st.subheader("🔍 Prediction Result:")
    if prediction[0] == 1:
        st.success("🌧️ Yes, it is likely to rain tomorrow.")
    else:
        st.info("☀️ No, it probably won't rain tomorrow.")


# --- INTERACTIVE WEATHER COMPARISON ---
st.markdown("---")
st.header("🔍 Interactive Weather Data Explorer")

# Allow users to select and compare weather variables
st.subheader("Compare Weather Variables")

col1, col2 = st.columns(2)

with col1:
    x_variable = st.selectbox(
        "Select X-axis variable:",
        ['MinTemp', 'MaxTemp', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'WindSpeed9am', 'WindSpeed3pm'],
        index=1
    )

with col2:
    y_variable = st.selectbox(
        "Select Y-axis variable:",
        ['MinTemp', 'MaxTemp', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'WindSpeed9am', 'WindSpeed3pm'],
        index=3
    )

if x_variable != y_variable:
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create scatter plot with rain information
    rain_colors = df['RainTomorrow'].map({0: 'lightblue', 1: 'red'})
    plt.scatter(df[x_variable], df[y_variable], c=rain_colors, alpha=0.6, s=50)
    
    plt.xlabel(x_variable)
    plt.ylabel(y_variable)
    plt.title(f'{x_variable} vs {y_variable} (Blue: No Rain, Red: Rain Tomorrow)')
    plt.grid(True, alpha=0.3)
    
    # Add legend
    import matplotlib.patches as mpatches
    blue_patch = mpatches.Patch(color='lightblue', label='No Rain Tomorrow')
    red_patch = mpatches.Patch(color='red', label='Rain Tomorrow')
    plt.legend(handles=[blue_patch, red_patch])
    
    st.pyplot(fig)
else:
    st.warning("Please select different variables for X and Y axes.")

