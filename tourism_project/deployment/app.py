import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Load the saved model from Hugging Face Model Hub
model_path = hf_hub_download(
    repo_id="rapidflow/wellness_tourism_model",
    filename="best_tourism_model_v1.joblib",
)
model = joblib.load(model_path)

st.title("Wellness Tourism Package Purchase Prediction")
st.write(
    """
    This application predicts whether a customer is likely to purchase the
    Wellness Tourism Package based on their profile and sales interaction details.
    Enter the customer information below to get a prediction.
    """
)

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
    typeof_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    duration_of_pitch = st.number_input(
        "Duration of Pitch (minutes)", min_value=1, max_value=60, value=10, step=1
    )
    occupation = st.selectbox(
        "Occupation",
        ["Salaried", "Free Lancer", "Small Business", "Large Business"],
    )
    gender = st.selectbox("Gender", ["Male", "Female"])
    num_persons = st.number_input(
        "Number of Persons Visiting", min_value=1, max_value=10, value=2, step=1
    )
    num_followups = st.number_input(
        "Number of Follow-ups", min_value=0, max_value=10, value=3, step=1
    )
    product_pitched = st.selectbox(
        "Product Pitched",
        ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"],
    )

with col2:
    preferred_star = st.number_input(
        "Preferred Property Star", min_value=1, max_value=5, value=3, step=1
    )
    marital_status = st.selectbox(
        "Marital Status", ["Single", "Married", "Divorced", "Unmarried"]
    )
    num_trips = st.number_input(
        "Number of Trips (annual average)", min_value=0, max_value=20, value=1, step=1
    )
    passport = st.selectbox("Passport", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    pitch_score = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
    own_car = st.selectbox("Own Car", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    num_children = st.number_input(
        "Number of Children Visiting", min_value=0, max_value=5, value=0, step=1
    )
    designation = st.selectbox(
        "Designation",
        ["Executive", "Manager", "Senior Manager", "AVP", "VP"],
    )
    monthly_income = st.number_input(
        "Monthly Income", min_value=1000, max_value=500000, value=20000, step=500
    )

# Assemble inputs into a DataFrame for prediction
input_data = pd.DataFrame([{
    "Age": age,
    "TypeofContact": typeof_contact,
    "CityTier": city_tier,
    "DurationOfPitch": duration_of_pitch,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": num_persons,
    "NumberOfFollowups": num_followups,
    "ProductPitched": product_pitched,
    "PreferredPropertyStar": preferred_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": num_trips,
    "Passport": passport,
    "PitchSatisfactionScore": pitch_score,
    "OwnCar": own_car,
    "NumberOfChildrenVisiting": num_children,
    "Designation": designation,
    "MonthlyIncome": monthly_income,
}])

if st.button("Predict Purchase"):
    purchase_prob = model.predict_proba(input_data)[0][1]
    prediction = 1 if purchase_prob >= 0.45 else 0
    if prediction == 1:
        st.success(f"Likely to purchase the Wellness Tourism Package (Probability: {purchase_prob:.2%})")
    else:
        st.warning(f"Unlikely to purchase the Wellness Tourism Package (Probability: {purchase_prob:.2%})")
