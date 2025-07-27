import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from streamlit_lottie import st_lottie
import requests

# --- Load resources ---
@st.cache_data
def load_model():
    return joblib.load("student_predictor_xg.pkl")  # Make sure this file exists

@st.cache_data
def load_background():
    return pd.read_csv("shap_background.csv").iloc[:100]  # Should match training structure

@st.cache_data
def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# --- Setup ---
model = load_model()
background = load_background()
feature_cols = [
    "G1_mat", "G2_mat", "studytime_mat", "failures_mat",
    "absences_mat", "higher_mat", "famsup_mat"
]
lottie_edu = load_lottie_url("https://assets10.lottiefiles.com/packages/lf20_kklehxgx.json")

# --- Streamlit Page Config ---
st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="wide")

# --- Sidebar Input ---
st.sidebar.image("sociolo.jpg", width=100)
st.sidebar.header("📋 Student Input")

G1 = st.sidebar.slider("First Period Grade (G1)", 0, 20, 10, help="Grade from first term exam")
G2 = st.sidebar.slider("Second Period Grade (G2)", 0, 20, 10, help="Grade from second term exam")
studytime = st.sidebar.slider("Study Time (1=low, 4=high)", 1, 4, 2, help="Weekly study time level")
failures = st.sidebar.slider("Past Class Failures", 0, 3, 0)
absences = st.sidebar.number_input("Absences", 0, 93, 5)
higher = st.sidebar.radio("Plans for Higher Education?", ["Yes", "No"])
famsup = st.sidebar.radio("Family Educational Support?", ["Yes", "No"])

higher = 1 if higher == "Yes" else 0
famsup = 1 if famsup == "Yes" else 0

input_data = pd.DataFrame([{
    'G1_mat': G1,
    'G2_mat': G2,
    'studytime_mat': studytime,
    'failures_mat': failures,
    'absences_mat': absences,
    'higher_mat': higher,
    'famsup_mat': famsup
}])

# --- Align input features with model ---
expected_features = model.get_booster().feature_names
for col in expected_features:
    if col not in input_data.columns:
        input_data[col] = 0
input_data = input_data[expected_features]

# --- Prediction ---
prediction = model.predict(input_data)[0]
confidence = model.predict_proba(input_data)[0][prediction]

# --- Tabs Layout ---
tab1, tab2, tab3 = st.tabs(["🎯 Prediction", "📊 Feature Importance", "🧠 SHAP Explanation"])

# 🎯 Prediction Tab
with tab1:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.header("🎯 Prediction Result")
        if prediction == 1:
            st.success(f"✅ Likely to PASS (Confidence: {confidence:.2%})")
        else:
            st.error(f"⚠️ At Risk of Failing (Confidence: {confidence:.2%})")
        st.metric("Confidence", f"{confidence:.2%}")
        st.markdown("Model: **XGBoost** | Threshold: G3 ≥ 10")
    with col2:
        if lottie_edu:
            st_lottie(lottie_edu, height=200)

# 📊 Feature Importance Tab
with tab2:
    st.header("📊 Feature Importances")
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        "Feature": expected_features,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)
    st.bar_chart(importance_df.set_index("Feature"))

# 🧠 SHAP Explanation Tab
with tab3:
    st.header("🧠 SHAP Explanation")
    explainer = shap.Explainer(model, background)
    shap_values = explainer(input_data)
    fig, ax = plt.subplots()
    shap.plots.bar(shap_values[0], ax=ax, show=False)
    st.pyplot(fig)

# --- Footer ---
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Made with ❤️ by Sudip & Sumit | "
    "<a href='https://github.com/YOUR_GITHUB_USERNAME' target='_blank'>GitHub</a>"
    "</div>",
    unsafe_allow_html=True
)
