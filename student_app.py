import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

st.write(
    """
# This app is to predict if students will pass or fail upcoming examinations
"""
)
st.sidebar.header("Student Factors")


# User input function
def student_input_features():
    options = [1, 2, 3, 4, 5]
    option_labels = {1: "Never", 2: "Rarely", 3: "Sometimes", 4: "Often", 5: "Always"}
    use_of_internet = st.sidebar.selectbox(
        "Internet Use", options, format_func=lambda x: option_labels[x]
    )
    class_participation = st.sidebar.selectbox(
        "Class Participation", ("Low", "Moderate", "High")
    )
    stress_levels = st.sidebar.selectbox(
        "Stress Level", options, format_func=lambda x: option_labels[x]
    )
    use_of_library_and_study_spaces = st.sidebar.selectbox(
        "use_of_library_and_study_spaces",
        options,
        format_func=lambda x: option_labels[x],
    )

    data = {
        "use_of_internet": use_of_internet,
        "class_participation": class_participation,
        "stress_levels": stress_levels,
        "use_of_library_and_study_spaces": use_of_library_and_study_spaces,
    }
    features = pd.DataFrame(data, index=[0])
    return features


# Get user inputs
input_df = student_input_features()

# Load the dataset for feature matching
students_raw = pd.read_csv("cleaned_data_model.csv")
Features = [
    "use_of_internet",
    "class_participation",
    "stress_levels",
    "use_of_library_and_study_spaces",
    "jamb_score",
]
students = students_raw[Features].drop(columns=["jamb_score"])

# Concatenate user input with the dataset for consistent encoding
df = pd.concat([input_df, students], axis=0)

# Encode the 'class_participation' column
le = LabelEncoder()
df["class_participation"] = le.fit_transform(df["class_participation"])

# Keep only the first row for prediction
df = df[:1]

# Displays the user input features
st.subheader("Student Factors")
st.write(df)

# Load the saved model
load_student = pickle.load(open("student_model.pkl", "rb"))

# Make predictions
prediction = load_student.predict(df)
prediction_proba = load_student.predict_proba(df)

# Display the results
st.subheader("Prediction")
student_performance = np.array(["Pass", "Fail"])
st.write(student_performance[prediction])

st.subheader("Prediction Probability")
st.write(prediction_proba)
