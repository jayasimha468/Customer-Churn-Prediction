
import streamlit as st
import pandas as pd
import joblib

# Load saved model and training features
model = joblib.load("customer_churn_model.pkl")
model_features = joblib.load("model_features.pkl")

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

st.title("Customer Churn Prediction")
st.write(
    "Enter customer details to predict whether the customer "
    "is likely to leave the telecom company."
)

st.subheader("Customer Information")

with st.form("churn_form"):

    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])

    tenure = st.number_input(
        "Tenure (Months)", min_value=0, max_value=100, value=12
    )

    phone = st.selectbox("Phone Service", ["Yes", "No"])
    multiple = st.selectbox(
        "Multiple Lines", ["No", "Yes", "No phone service"]
    )

    internet = st.selectbox(
        "Internet Service", ["DSL", "Fiber optic", "No"]
    )

    security = st.selectbox(
        "Online Security", ["Yes", "No", "No internet service"]
    )
    backup = st.selectbox(
        "Online Backup", ["Yes", "No", "No internet service"]
    )
    protection = st.selectbox(
        "Device Protection", ["Yes", "No", "No internet service"]
    )
    support = st.selectbox(
        "Tech Support", ["Yes", "No", "No internet service"]
    )
    tv = st.selectbox(
        "Streaming TV", ["Yes", "No", "No internet service"]
    )
    movies = st.selectbox(
        "Streaming Movies", ["Yes", "No", "No internet service"]
    )

    contract = st.selectbox(
        "Contract", ["Month-to-month", "One year", "Two year"]
    )

    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

    payment = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    monthly = st.number_input(
        "Monthly Charges", min_value=0.0, value=70.0
    )

    total = st.number_input(
        "Total Charges", min_value=0.0, value=840.0
    )

    submitted = st.form_submit_button("Predict Churn")


if submitted:

    customer_data = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": protection,
        "TechSupport": support,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }])

    customer_encoded = pd.DataFrame(
        0, index=[0], columns=model_features
    )

    for column, value in customer_data.iloc[0].items():

        if column in customer_encoded.columns:
            customer_encoded.at[0, column] = value

        else:
            feature_name = f"{column}_{value}"

            if feature_name in customer_encoded.columns:
                customer_encoded.at[0, feature_name] = 1

    prediction = model.predict(customer_encoded)[0]
    probability = model.predict_proba(customer_encoded)[0][1]

    st.subheader("Prediction Result")

    st.metric(
        "Churn Probability",
        f"{probability * 100:.2f}%"
    )

    if prediction == 1:
        st.error("This customer is predicted to churn.")
    else:
        st.success("This customer is predicted to stay.")

    st.caption(
        "Prediction is based on the trained Logistic Regression model."
    )
