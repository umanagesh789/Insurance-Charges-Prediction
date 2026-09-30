import streamlit as st
from snowflake.snowpark.context import get_active_session

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="Insurance Charges Prediction",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Insurance Charges Prediction Dashboard")
st.write("Predict and analyze insurance charges using the Snowflake ML model.")

# Connect to Snowflake
session = get_active_session()

# -----------------------------
# Load prediction results
# -----------------------------
predictions = session.sql("""
    SELECT
        AGE,
        SEX,
        BMI,
        CHILDREN,
        SMOKER,
        REGION,
        PREDICTED_CHARGES
    FROM INSURANCE_ML.ML_PIPE.INSURANCE_PREDICTIONS
    ORDER BY PREDICTED_CHARGES DESC
""").to_pandas()

# -----------------------------
# Summary metrics
# -----------------------------
col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Predictions",
    len(predictions)
)

if len(predictions) > 0:
    col2.metric(
        "Average Predicted Charges",
        f"${predictions['PREDICTED_CHARGES'].mean():,.2f}"
    )

    col3.metric(
        "Highest Predicted Charges",
        f"${predictions['PREDICTED_CHARGES'].max():,.2f}"
    )
else:
    col2.metric("Average Predicted Charges", "$0.00")
    col3.metric("Highest Predicted Charges", "$0.00")

# -----------------------------
# Prediction chart
# -----------------------------
st.subheader("📊 Predicted Insurance Charges")

if len(predictions) > 0:
    chart_data = predictions.head(15)[
        ["AGE", "PREDICTED_CHARGES"]
    ]

    st.bar_chart(
        chart_data,
        x="AGE",
        y="PREDICTED_CHARGES"
    )
else:
    st.info("No prediction records available.")

# -----------------------------
# Prediction table
# -----------------------------
st.subheader("📋 Prediction Results")

st.dataframe(
    predictions,
    use_container_width=True
)

# -----------------------------
# New prediction
# -----------------------------
st.subheader("🔮 Predict Insurance Charges")

with st.form("prediction_form"):

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=100,
        value=30
    )

    sex = st.selectbox(
        "Sex",
        ["female", "male"]
    )

    bmi = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=60.0,
        value=27.5
    )

    children = st.number_input(
        "Children",
        min_value=0,
        max_value=10,
        value=0
    )

    smoker = st.selectbox(
        "Smoker",
        ["no", "yes"]
    )

    region = st.selectbox(
        "Region",
        ["southwest", "southeast", "northwest", "northeast"]
    )

    predict_button = st.form_submit_button(
        "🚀 Predict Charges"
    )

if predict_button:

    result = session.sql(f"""
        SELECT
            (
                MODEL(
                    INSURANCE_ML.ML_PIPE.INSURANCE_CHARGES_MODEL,
                    V2
                )!PREDICT(
                    {age},
                    '{sex}',
                    {bmi},
                    {children},
                    '{smoker}',
                    '{region}'
                ):output_feature_0
            )::FLOAT AS PREDICTED_CHARGES
    """).collect()

    predicted_value = result[0]["PREDICTED_CHARGES"]

    st.success(
        f"Predicted Insurance Charges: ${predicted_value:,.2f}"
    )

# -----------------------------
# Model information
# -----------------------------
st.subheader("🤖 Model Information")

info1, info2, info3 = st.columns(3)

info1.write("**Model:** INSURANCE_CHARGES_MODEL")
info2.write("**Version:** V2")
info3.write("**Model Type:** XGBoost Regression")
