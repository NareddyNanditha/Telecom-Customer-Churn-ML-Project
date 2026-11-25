import joblib
import streamlit as st
import pandas as pd
import numpy as np

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(page_title="Telco Churn Predictor", page_icon="📡", layout="wide")

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: linear-gradient(135deg,#081225 0%,#0b1830 55%,#101d38 100%); color:#eaf1ff; }
[data-testid="stSidebar"] { background:#0b1426; border-right:1px solid #233450; }
[data-testid="stHeader"] { background:transparent; }
.block-container { padding-top:2rem; padding-bottom:2.5rem; max-width:1250px; }
.hero { padding:1.6rem 1.8rem; border:1px solid rgba(123,170,255,.22); border-radius:22px;
 background:linear-gradient(120deg,rgba(24,49,86,.9),rgba(17,32,58,.75)); margin-bottom:1.4rem; }
.hero h1 { font-family:'Manrope',sans-serif; font-size:2.1rem; color:#f4f8ff; margin:0; letter-spacing:-.5px; }
.hero p { color:#a9bedf; margin:.45rem 0 0 0; font-size:1rem; }
.panel { background:rgba(16,30,51,.78); border:1px solid #263c5d; padding:1.1rem 1.3rem; border-radius:17px; }
.metric { background:linear-gradient(145deg,#132642,#102036); border:1px solid #294260; padding:1rem 1.2rem; border-radius:15px; }
.metric-label { color:#93aacb; font-size:.85rem; text-transform:uppercase; letter-spacing:1px; }
.metric-value { font-family:'Manrope',sans-serif; font-size:1.45rem; font-weight:800; color:#f4f7ff; margin-top:.25rem; }
div.stButton > button { background:linear-gradient(90deg,#2e86ff,#43c7d9); border:0; border-radius:12px; color:white; font-weight:700; padding:.7rem 1.3rem; width:100%; }
div.stButton > button:hover { border:1px solid #8ee9ff; color:white; box-shadow:0 0 18px rgba(48,156,255,.25); }
.stSelectbox label, .stSlider label, .stNumberInput label, .stRadio label { color:#d8e5f8 !important; }
[data-testid="stWidgetLabel"] p { color:#d8e5f8 !important; }
div[data-baseweb="select"] > div { background:#101f36; border-color:#2a4263; border-radius:10px; }
input { background:#101f36 !important; color:#eaf1ff !important; }
.success-card { background:rgba(25,137,110,.15); border:1px solid #2caa8b; border-radius:16px; padding:1.2rem; }
.warning-card { background:rgba(245,92,100,.13); border:1px solid #f06b77; border-radius:16px; padding:1.2rem; }
.small-note { color:#93a9c8; font-size:.85rem; }
hr { border-color:#263c5d !important; }
</style>""", unsafe_allow_html=True)

# -----------------------------
# Load saved model
# -----------------------------
@st.cache_resource
def load_model():
    with open("Telecustomer.pkl", "rb") as file:
        saved = joblib.load("Telecustomer.pkl")
    # Support either a bare estimator or a dictionary containing the estimator.
    if isinstance(saved, dict):
        return saved.get("model", saved.get("classifier", saved.get("estimator")))
    return saved


try:
    model = load_model()
    model_error = None
except Exception as ex:
    model = None
    model_error = str(ex)

# -----------------------------
# Header
# -----------------------------
st.markdown("""<div class="hero"><h1>📡 Telco Customer Churn Predictor</h1>
<p>Customer retention intelligence · Machine Learning classification</p></div>""", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("### 🧭 Customer profile")
st.sidebar.caption("Enter a customer’s details to estimate churn risk.")

# Demographic/customer profile
with st.sidebar:
    gender = st.selectbox("Gender", ["Female", "Male"])
    seniorcitizen = st.selectbox("Senior citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", min_value=0, max_value=72, value=12)
    monthlycharges = st.number_input("Monthly charges ($)", min_value=0.0, max_value=200.0, value=70.0, step=1.0)
    totalcharges = st.number_input("Total charges ($)", min_value=0.0, max_value=10000.0, value=840.0, step=10.0)

    st.markdown("---")
    phoneservice = st.selectbox("Phone service", ["Yes", "No"])
    multiplelines = st.selectbox("Multiple lines", ["Yes", "No", "No phone service"])
    internetservice = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
    onlinesecurity = st.selectbox("Online security", ["Yes", "No", "No internet service"])
    onlinebackup = st.selectbox("Online backup", ["Yes", "No", "No internet service"])
    deviceprotection = st.selectbox("Device protection", ["Yes", "No", "No internet service"])
    techsupport = st.selectbox("Tech support", ["Yes", "No", "No internet service"])
    streamingtv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streamingmovies = st.selectbox("Streaming movies", ["Yes", "No", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperlessbilling = st.selectbox("Paperless billing", ["Yes", "No"])
    paymentmethod = st.selectbox("Payment method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    predict_clicked = st.button("✨ Predict churn", use_container_width=True)

# Simple, editable Min-Max scaling based on the standard Telco dataset feature ranges.
# IMPORTANT: replace these ranges with the exact fitted scaler parameters from training if available.
NUMERIC_MIN_MAX = {
    "tenure": (0.0, 72.0),
    "monthlycharges": (18.25, 118.75),
    "totalcharges": (0.0, 8684.80),
}

def minmax_value(col, value):
    low, high = NUMERIC_MIN_MAX[col]
    if high == low:
        return 0.0
    return float(np.clip((float(value) - low) / (high - low), 0.0, 1.0))


def make_model_input():
    """Build an OHE row with the 43-feature layout expected by this saved model."""
    raw = {
        "gender": gender, "partner": partner, "dependents": dependents,
        "phoneservice": phoneservice, "multiplelines": multiplelines,
        "internetservice": internetservice, "onlinesecurity": onlinesecurity,
        "onlinebackup": onlinebackup, "deviceprotection": deviceprotection,
        "techsupport": techsupport, "streamingtv": streamingtv,
        "streamingmovies": streamingmovies, "paperlessbilling": paperlessbilling,
        "paymentmethod": paymentmethod,
    }
    # OHE is built from the same category spellings commonly present in the Telco churn dataset.
    categorical_options = {
        "gender": ["Female", "Male"],
        "partner": ["No", "Yes"],
        "dependents": ["No", "Yes"],
        "phoneservice": ["No", "Yes"],
        "multiplelines": ["No", "No phone service", "Yes"],
        "internetservice": ["DSL", "Fiber optic", "No"],
        "onlinesecurity": ["No", "No internet service", "Yes"],
        "onlinebackup": ["No", "No internet service", "Yes"],
        "deviceprotection": ["No", "No internet service", "Yes"],
        "techsupport": ["No", "No internet service", "Yes"],
        "streamingtv": ["No", "No internet service", "Yes"],
        "streamingmovies": ["No", "No internet service", "Yes"],
        "paperlessbilling": ["No", "Yes"],
        "paymentmethod": ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"],
    }
    row = {
        "tenure": minmax_value("tenure", tenure),
        "monthlycharges": minmax_value("monthlycharges", monthlycharges),
        "totalcharges": minmax_value("totalcharges", totalcharges),
        "seniorcitizen": int(seniorcitizen),
    }
    for column, categories in categorical_options.items():
        for category in categories:
            # The model's stored OHE feature names use raw category text, with spaces retained.
            feature = f"{column}_{category}"
            row[feature] = int(raw[column] == category)

    # The saved training object includes a 43-column feature layout. If feature names
    # exist on the estimator, align and fill missing columns automatically.
    feature_names = getattr(model, "feature_names_in_", None) if model is not None else None
    if feature_names is not None:
        aligned = pd.DataFrame([row]).reindex(columns=list(feature_names), fill_value=0)
        return aligned
    return pd.DataFrame([row])



# Main area: summary and prediction
left, right = st.columns([1.6, 1])
with left:
    st.markdown("### 📋 Profile overview")
    st.markdown('<div class="panel">Review the customer information entered in the sidebar. Numeric inputs are scaled and categorical fields are one-hot encoded before prediction.</div>', unsafe_allow_html=True)
    summary = pd.DataFrame({"Customer attribute": ["Tenure", "Monthly charges", "Total charges", "Contract", "Internet service", "Payment method"],
                          "Selected value": [f"{tenure} months", f"${monthlycharges:.2f}", f"${totalcharges:.2f}", contract, internetservice, paymentmethod]})
    st.dataframe(summary, use_container_width=True, hide_index=True)
with right:
    st.markdown("### 🎯 Prediction")
    st.markdown('<div class="metric"><div class="metric-label">Model</div><div class="metric-value">K-Nearest Neighbors</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="metric"><div class="metric-label">Selected tenure</div><div class="metric-value">' + str(tenure) + ' months</div></div>', unsafe_allow_html=True)

if predict_clicked:
    if model is None:
        st.error("Could not load the pickle model. Details: " + str(model_error))
        st.info("Keep Telecustomer.pkl in the same folder as app.py and install the matching scikit-learn version used to save the model.")
    else:
        try:
            input_df = make_model_input()
            prediction = model.predict(input_df)[0]
            prediction_label = str(prediction)
            # Typical Telco dataset labels: 0 = No churn, 1 = Churn.
            if prediction_label in ["1", "Yes", "True", "Churn"]:
                st.markdown('<div class="warning-card"><h2>⚠️ Customer likely to churn</h2><p>Consider a retention offer, proactive support, or plan review.</p></div>', unsafe_allow_html=True)
            elif prediction_label in ["0", "No", "False", "Stayed", "No Churn"]:
                st.markdown('<div class="success-card"><h2>✅ Customer likely to stay</h2><p>Continue positive engagement and monitor satisfaction.</p></div>', unsafe_allow_html=True)
            else:
                st.success(f"Prediction result: {prediction_label}")
            st.caption("Model prediction only; not a guarantee of future customer behavior.")
        except Exception as ex:
            st.error(f"Prediction failed: {ex}")
            st.write("Check that input columns, category encodings, and scaling match your model-training notebook exactly.")
else:
    st.info("Fill in the customer profile in the sidebar, then click **Predict churn**.")

st.markdown("---")
st.markdown('<p class="small-note">Built with Streamlit · Customer churn classification demo</p>', unsafe_allow_html=True)
