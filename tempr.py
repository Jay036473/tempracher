# ===============================
# STREAMLIT RAIN PREDICTION APP
# ===============================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# ===============================
# TITLE & LAYOUT
# ===============================

# Set layout to "wide" to use the full width of the screen
st.set_page_config(page_title="Rain Prediction", layout="wide")
st.title("🌧 Rain Prediction ML App")


# ===============================
# LOAD DATA (Cached to save time)
# ===============================
@st.cache_data
def load_data():
    df = pd.read_csv("D:/PythonProject/csv/weth.csv")
    if 'Date' in df.columns:
        df = df.drop(columns=['Date'])
    df = df.dropna()
    return df


df = load_data()

# ===============================
# SEPARATE FEATURES AND TARGET
# ===============================

target_col = "RainTomorrow"
X = df.drop(target_col, axis=1)
y = df[target_col]

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(exclude=['object']).columns.tolist()

# ===============================
# ENCODE CATEGORICAL FEATURES
# ===============================
label_encoders = {}
X_encoded = X.copy()

for col in categorical_cols:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X[col])
    label_encoders[col] = le

le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)


# ===============================
# TRAIN MODEL (Cached to avoid lag)
# ===============================
@st.cache_resource
def train_model(X_train_data, y_train_data):
    X_train, X_test, y_train, y_test = train_test_split(
        X_train_data, y_train_data, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


model = train_model(X_encoded, y_encoded)

# ===============================
# CREATE TABS FOR BETTER UI
# ===============================
# We create 3 tabs: App, Charts, and Data Preview
tab1, tab2, tab3 = st.tabs(["🔮 Prediction App", "📊 Data Visualizations (6 Charts)", "📋 Data Preview"])

# ===============================
# TAB 1: PREDICTION APP
# ===============================
with tab1:
    st.header("Enter Weather Information")

    user_input = {}

    col1, col2 = st.columns(2)
    all_features = categorical_cols + numerical_cols

    for i, col in enumerate(all_features):
        current_col = col1 if i % 2 == 0 else col2

        with current_col:
            if col in categorical_cols:
                options = label_encoders[col].classes_
                user_input[col] = st.selectbox(f"Select {col}", options)
            else:
                user_input[col] = st.number_input(f"Enter {col}", value=float(X[col].mean()))

    st.markdown("---")

    if st.button("Predict Rain Tomorrow", use_container_width=True):
        input_data = []

        for col in X_encoded.columns:
            if col in categorical_cols:
                encoded_val = label_encoders[col].transform([user_input[col]])[0]
                input_data.append(encoded_val)
            else:
                input_data.append(user_input[col])

        input_array = np.array([input_data])
        prediction = model.predict(input_array)
        predicted_label = le_target.inverse_transform(prediction)[0]

        if predicted_label == 'Yes':
            st.error("🌧 Yes, High Chances of Rain Tomorrow!")
        else:
            st.success("☀️ No Rain Tomorrow, Clear Weather Expected!")

# ===============================
# TAB 2: DATA VISUALIZATIONS (6 CHARTS)
# ===============================
with tab2:
    st.header("Exploratory Data Analysis (EDA)")

    # Create 2 columns for the charts so they display side-by-side
    chart_col1, chart_col2 = st.columns(2)

    # 1. Target Distribution (Yes vs No)
    with chart_col1:
        st.subheader("1. Rain Tomorrow Distribution")
        fig1, ax1 = plt.subplots()
        sns.countplot(data=df, x='RainTomorrow', palette='Set2', ax=ax1)
        ax1.set_title("How many days rained vs didn't rain?")
        st.pyplot(fig1)

    # 2. Humidity at 3pm vs Rain Tomorrow
    with chart_col2:
        st.subheader("2. Humidity (3pm) impact on Rain")
        fig2, ax2 = plt.subplots()
        sns.boxplot(data=df, x='RainTomorrow', y='Humidity3pm', palette='Set1', ax=ax2)
        ax2.set_title("Higher humidity usually means rain")
        st.pyplot(fig2)

    chart_col3, chart_col4 = st.columns(2)

    # 3. Min vs Max Temperature
    with chart_col3:
        st.subheader("3. Min vs Max Temperature")
        fig3, ax3 = plt.subplots()
        sns.scatterplot(data=df, x='MinTemp', y='MaxTemp', hue='RainTomorrow', alpha=0.5, ax=ax3)
        ax3.set_title("Temperature Correlation")
        st.pyplot(fig3)

    # 4. Feature Importance (From Model)
    with chart_col4:
        st.subheader("4. Top 10 Feature Importances")
        fig4, ax4 = plt.subplots()
        importances = pd.Series(model.feature_importances_, index=X.columns).nlargest(10)
        sns.barplot(x=importances.values, y=importances.index, palette='viridis', ax=ax4)
        ax4.set_title("Which factors help the model predict best?")
        st.pyplot(fig4)

    chart_col5, chart_col6 = st.columns(2)

    # 5. Top 10 Locations by Average Rainfall
    with chart_col5:
        st.subheader("5. Top 10 Rainiest Locations")
        fig5, ax5 = plt.subplots()
        top_locations = df.groupby('Location')['Rainfall'].mean().sort_values(ascending=False).head(10)
        sns.barplot(x=top_locations.values, y=top_locations.index, palette='magma', ax=ax5)
        ax5.set_xlabel("Average Rainfall (mm)")
        st.pyplot(fig5)

    # 6. Correlation Heatmap
    with chart_col6:
        st.subheader("6. Correlation Heatmap")
        fig6, ax6 = plt.subplots(figsize=(8, 6))
        # Selecting top numeric columns to keep the heatmap clean
        cols_to_plot = ['MinTemp', 'MaxTemp', 'Rainfall', 'Humidity9am', 'Humidity3pm', 'Pressure3pm']
        cols_to_plot = [c for c in cols_to_plot if c in df.columns]  # Ensure they exist
        sns.heatmap(df[cols_to_plot].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax6)
        ax6.set_title("How numerical features relate to each other")
        st.pyplot(fig6)

# ===============================
# TAB 3: DATA PREVIEW
# ===============================
with tab3:
    st.header("📋 Dataset Preview")

    # Display the total rows and columns dynamically
    st.write(f"This dataset contains **{df.shape[0]} rows** and **{df.shape[1]} columns**.")

    st.markdown("---")
    st.write("Below is the raw weather data used to train the machine learning model:")

    # Display the dataframe as an interactive table
    st.dataframe(df, use_container_width=True)
