import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time
from sklearn.metrics import classification_report, confusion_matrix
import plotly.express as px
from sklearn.model_selection import train_test_split

# Set wide layout and page config
st.set_page_config(page_title="Diabetes Prediction App", layout="wide", page_icon="🩺")

# Load trained model
model = pickle.load(open("trained_model.sav", "rb"))

# Load dataset
df = pd.read_csv("diabetes.csv")

# Sidebar navigation
st.sidebar.title("🔧 Navigation")
section = st.sidebar.radio("Go to", ("🏠 Home", "📂 Data Exploration", "📊 Visualizations", "🔮 Predict Diabetes", "📈 Model Performance"))

#  HOME 
if section == "🏠 Home":
    st.title("🩺 Diabetes Prediction App")
    st.markdown("""
    Welcome to the **Diabetes Prediction App** powered by Machine Learning.

    ### Key Features:
    - 📂 Explore the dataset
    - 📊 Visualize key insights
    - 🔮 Predict diabetes with a trained model
    - 📈 Review model performance 
    """)
# DATA EXPLORATION 
elif section == "📂 Data Exploration":
    st.header("📂 Data Overview")
    st.write("Shape of dataset:", df.shape)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🧾 Columns and Data Types")
        st.dataframe(df.dtypes)
    with col2:
        st.subheader("📋 Sample Data")
        st.dataframe(df.sample(5))

    st.subheader("⚠ Missing Values")
    st.write(df.isnull().sum())

    st.subheader("🔎 Interactive Filter: Glucose")
    glucose_filter = st.slider("Filter by Glucose level", int(df["Glucose"].min()), int(df["Glucose"].max()), 100)
    st.write(df[df["Glucose"] >= glucose_filter])

# VISUALIZATIONS 
elif section == "📊 Visualizations":
    st.header("📊 Visualizations")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Histogram: Age Distribution")
        fig1, ax1 = plt.subplots()
        sns.histplot(df["Age"], bins=30, kde=True, ax=ax1, color='purple')
        st.pyplot(fig1)

    with col2:
        st.subheader("🎯 Pie Chart: Outcome Distribution")
        pie_data = df["Outcome"].value_counts().reset_index()
        pie_data.columns = ['Outcome', 'Count']
        fig2 = px.pie(pie_data, names='Outcome', values='Count',
                      title="Outcome (0 = No Diabetes, 1 = Diabetes)",
                      color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig2)

    st.subheader("🧮 Correlation Heatmap")
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax3)
    st.pyplot(fig3)

# PREDICTION 
elif section == "🔮 Predict Diabetes":
    st.header("🔮 Diabetes Prediction")

    st.markdown("Enter patient details:")

    with st.form(key='prediction_form'):
        col1, col2, col3 = st.columns(3)

        with col1:
            pregnancies = st.number_input("Pregnancies", 0, 20, 1)
            glucose = st.number_input("Glucose", 0, 200, 120)
            blood_pressure = st.number_input("Blood Pressure", 0, 140, 70)
        with col2:
            skin_thickness = st.number_input("Skin Thickness", 0, 100, 20)
            insulin = st.number_input("Insulin", 0, 900, 80)
            bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
        with col3:
            dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
            age = st.number_input("Age", 10, 100, 33)

        submit = st.form_submit_button("Predict")

    if submit:
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
        with st.spinner("Making prediction..."):
            time.sleep(1)
            prediction = model.predict(input_data)
            probability = model.predict_proba(input_data)

        st.success(f"🩺 Prediction: {'Diabetic' if prediction[0] == 1 else 'Not Diabetic'}")
        st.info(f"🔒 Confidence: {round(np.max(probability) * 100, 2)}%")

#  MODEL PERFORMANCE 
elif section == "📈 Model Performance":
    st.header("📈 Model Performance Metrics")

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_pred = model.predict(X_test)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📋 Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

    with col2:
        st.subheader("🔁 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["No Diabetes", "Diabetes"],
                    yticklabels=["No Diabetes", "Diabetes"])
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
        st.pyplot(fig_cm)

    st.subheader("📊 Model Comparison (Simulated)")
    comparison_df = pd.DataFrame({
        "Model": ["SVM (Current)", "Random Forest", "Logistic Regression"],
        "Accuracy": [0.78, 0.81, 0.76],
        "Precision": [0.74, 0.79, 0.73],
        "Recall": [0.69, 0.76, 0.67]
    })
    st.dataframe(comparison_df)

    st.bar_chart(comparison_df.set_index("Model")[["Accuracy", "Precision", "Recall"]])
