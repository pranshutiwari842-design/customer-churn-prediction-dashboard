import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")

model = pickle.load(open("churn_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

df = pd.read_csv("data/Telco-Customer-Churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

st.title("📊 Customer Churn Prediction Dashboard")
st.write("This dashboard analyzes telecom customer churn and predicts whether a customer is likely to leave.")

st.sidebar.header("🔎 Dashboard Filters")

contract_filter = st.sidebar.multiselect(
    "Select Contract Type",
    options=df["Contract"].unique(),
    default=df["Contract"].unique()
)

internet_filter = st.sidebar.multiselect(
    "Select Internet Service",
    options=df["InternetService"].unique(),
    default=df["InternetService"].unique()
)

payment_filter = st.sidebar.multiselect(
    "Select Payment Method",
    options=df["PaymentMethod"].unique(),
    default=df["PaymentMethod"].unique()
)

filtered_df = df[
    (df["Contract"].isin(contract_filter)) &
    (df["InternetService"].isin(internet_filter)) &
    (df["PaymentMethod"].isin(payment_filter))
]

total_customers = filtered_df.shape[0]
churn_customers = filtered_df[filtered_df["Churn"] == "Yes"].shape[0]
retained_customers = filtered_df[filtered_df["Churn"] == "No"].shape[0]
churn_rate = (churn_customers / total_customers) * 100 if total_customers > 0 else 0

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Customers", total_customers)
col2.metric("Churn Customers", churn_customers)
col3.metric("Retained Customers", retained_customers)
col4.metric("Churn Rate", f"{churn_rate:.2f}%")

st.markdown("---")

chart1, chart2 = st.columns(2)

with chart1:
    st.subheader("Churn by Contract Type")
    fig1 = px.histogram(
        filtered_df,
        x="Contract",
        color="Churn",
        barmode="group",
        text_auto=True
    )
    st.plotly_chart(fig1, use_container_width=True)

with chart2:
    st.subheader("Churn by Internet Service")
    fig2 = px.histogram(
        filtered_df,
        x="InternetService",
        color="Churn",
        barmode="group",
        text_auto=True
    )
    st.plotly_chart(fig2, use_container_width=True)

chart3, chart4 = st.columns(2)

with chart3:
    st.subheader("Churn by Payment Method")
    fig3 = px.histogram(
        filtered_df,
        x="PaymentMethod",
        color="Churn",
        barmode="group",
        text_auto=True
    )
    st.plotly_chart(fig3, use_container_width=True)

with chart4:
    st.subheader("Monthly Charges vs Churn")
    fig4 = px.box(
        filtered_df,
        x="Churn",
        y="MonthlyCharges",
        color="Churn"
    )
    st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")

st.subheader("Customer Tenure Distribution")
fig5 = px.histogram(
    filtered_df,
    x="tenure",
    color="Churn",
    nbins=30,
    barmode="overlay"
)
st.plotly_chart(fig5, use_container_width=True)

st.markdown("---")

st.header("🤖 Predict Customer Churn")

p1, p2, p3 = st.columns(3)

with p1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])

with p2:
    tenure = st.slider("Tenure Months", 0, 72, 12)
    monthly = st.number_input("Monthly Charges", min_value=0.0, value=50.0)
    total = st.number_input("Total Charges", min_value=0.0, value=600.0)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

with p3:
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    payment = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )

input_data = pd.DataFrame(0, index=[0], columns=columns)

input_data["SeniorCitizen"] = senior
input_data["tenure"] = tenure
input_data["MonthlyCharges"] = monthly
input_data["TotalCharges"] = total

if "gender_Male" in columns and gender == "Male":
    input_data["gender_Male"] = 1

if "Partner_Yes" in columns and partner == "Yes":
    input_data["Partner_Yes"] = 1

if "Dependents_Yes" in columns and dependents == "Yes":
    input_data["Dependents_Yes"] = 1

if contract == "One year" and "Contract_One year" in columns:
    input_data["Contract_One year"] = 1
elif contract == "Two year" and "Contract_Two year" in columns:
    input_data["Contract_Two year"] = 1

if internet == "Fiber optic" and "InternetService_Fiber optic" in columns:
    input_data["InternetService_Fiber optic"] = 1
elif internet == "No" and "InternetService_No" in columns:
    input_data["InternetService_No"] = 1

if payment == "Electronic check" and "PaymentMethod_Electronic check" in columns:
    input_data["PaymentMethod_Electronic check"] = 1
elif payment == "Mailed check" and "PaymentMethod_Mailed check" in columns:
    input_data["PaymentMethod_Mailed check"] = 1
elif payment == "Credit card (automatic)" and "PaymentMethod_Credit card (automatic)" in columns:
    input_data["PaymentMethod_Credit card (automatic)"] = 1

input_scaled = scaler.transform(input_data)

if st.button("Predict Churn"):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Customer is likely to churn. Probability: {probability:.2f}")
    else:
        st.success(f"✅ Customer is not likely to churn. Probability: {probability:.2f}")

st.markdown("---")

st.subheader("Sample Data")
st.dataframe(filtered_df.head(20))