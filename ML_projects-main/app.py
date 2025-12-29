import streamlit as st
import pandas as pd
import os
import dill

# Helper to load objects (model, preprocessor)
def load_object(file_path):
    with open(file_path, "rb") as file_obj:
        return dill.load(file_obj)

# Sidebar info
st.sidebar.title("Student Exam Performance ML Project")
st.sidebar.info("Predict student math scores based on demographic and academic features.")

st.title("Student Math Score Predictor")
st.write("Fill in the details below to get a math score prediction.")

# Input fields
gender = st.selectbox("Gender", ["female", "male"])
race_ethnicity = st.selectbox("Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
parental_level_of_education = st.selectbox(
    "Parental Level of Education",
    ["some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"]
)
lunch = st.selectbox("Lunch", ["standard", "free/reduced"])
test_preparation_course = st.selectbox("Test Preparation Course", ["none", "completed"])
reading_score = st.number_input("Reading Score", min_value=0, max_value=100, value=70)
writing_score = st.number_input("Writing Score", min_value=0, max_value=100, value=70)

if st.button("Predict Math Score"):
    # Prepare input as DataFrame
    input_df = pd.DataFrame({
        "gender": [gender],
        "race_ethnicity": [race_ethnicity],
        "parental_level_of_education": [parental_level_of_education],
        "lunch": [lunch],
        "test_preparation_course": [test_preparation_course],
        "reading_score": [reading_score],
        "writing_score": [writing_score],
    })

    # Load preprocessor and model
    preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
    model_path = os.path.join("artifacts", "model.pkl")
    try:
        preprocessor = load_object(preprocessor_path)
        model = load_object(model_path)
        X_scaled = preprocessor.transform(input_df)
        prediction = model.predict(X_scaled)[0]
        st.success(f"Predicted Math Score: {prediction:.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.markdown("---")
