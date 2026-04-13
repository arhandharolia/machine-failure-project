import streamlit as st
import numpy as np
import pickle
import pandas as pd
import time

# Set page config
st.set_page_config(page_title="Machine Failure Prediction", layout="wide", page_icon="⚙️")

# Custom CSS for a slightly more modern look
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
    }
    .metric-container {
        background-color: #1e2127;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Load the models
@st.cache_resource
def load_models():
    try:
        model = pickle.load(open("model.pkl", "rb"))
        scaler = pickle.load(open("scaler.pkl", "rb"))
        return model, scaler
    except Exception as e:
        return None, None

model, scaler = load_models()

# Initialize session state for navigation if not exists
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

# Screen: Login
def show_login():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h1 style='text-align: center;'>Predictive Maintenance Auth</h1>", unsafe_allow_html=True)
        st.write("---")
        email = st.text_input("Corporate Email", placeholder="user@company.com")
        password = st.text_input("Password", type="password")
        if st.button("Sign In", use_container_width=True, type="primary"):
            if email and password:
                st.session_state["logged_in"] = True
                st.success("Authenticated successfully!")
                time.sleep(0.5)
                st.rerun()

# Screen: Main App
def show_app():
    st.sidebar.title("Navigation")
    menu = st.sidebar.radio("Go to", ["Dashboard", "Sensor Data Charts", "Alerts & History"])
    
    st.sidebar.header("Live Sensor Inputs")
    air_temp = st.sidebar.slider("Air Temperature [K]", 290.0, 310.0, 298.1)
    process_temp = st.sidebar.slider("Process Temperature [K]", 300.0, 320.0, 308.6)
    rpm = st.sidebar.slider("Rotational Speed [rpm]", 1000, 3000, 1551)
    torque = st.sidebar.slider("Torque [Nm]", 10.0, 80.0, 42.8)
    tool_wear = st.sidebar.slider("Tool Wear [min]", 0, 250, 0)
    
    if st.sidebar.button("Logout"):
        st.session_state["logged_in"] = False
        st.rerun()

    # Prediction logic
    prob = 0
    failure = False
    if model and scaler:
        data = np.array([[air_temp, process_temp, rpm, torque, tool_wear]])
        data_scaled = scaler.transform(data)
        prob = model.predict_proba(data_scaled)[0][1] * 100
        failure = prob > 50

    if menu == "Dashboard":
        st.title("⚙️ Machine Health Dashboard")
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
            st.metric("Failure Probability", f"{prob:.2f}%", delta="High Risk" if failure else "Stable", delta_color="inverse")
            st.markdown("</div>", unsafe_allow_html=True)
        with col2:
            st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
            st.metric("System Status", "WARNING" if failure else "HEALTHY")
            st.markdown("</div>", unsafe_allow_html=True)
        with col3:
            st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
            st.metric("Tool Wear", f"{tool_wear} mins")
            st.markdown("</div>", unsafe_allow_html=True)
            
        if failure:
            st.error("⚠️ HIGH RISK OF MACHINE FAILURE DETECTED! Immediate maintenance recommended.")
        else:
            st.success("✅ Machine is operating within normal parameters.")
            
        st.subheader("Current Live Feed")
        st.write(pd.DataFrame({
            "Air Temp": [air_temp],
            "Process Temp": [process_temp],
            "RPM": [rpm],
            "Torque": [torque],
            "Tool Wear": [tool_wear]
        }))

    elif menu == "Sensor Data Charts":
        st.title("📈 Sensor Data Analytics")
        st.markdown("---")
        
        # Simulate some dummy historical data based on current inputs
        history_len = 50
        np.random.seed(42)
        
        chart_data = pd.DataFrame({
            "Air Temp": [air_temp + np.random.normal(0, 1) for _ in range(history_len)],
            "RPM": [rpm + np.random.normal(0, 50) for _ in range(history_len)],
            "Torque": [torque + np.random.normal(0, 2) for _ in range(history_len)]
        })
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Temperature Trends")
            st.line_chart(chart_data["Air Temp"])
        with col2:
            st.subheader("Performance Metrics (RPM & Torque)")
            st.line_chart(chart_data[["RPM", "Torque"]])

    elif menu == "Alerts & History":
        st.title("📜 Alerts & Maintenance History")
        st.markdown("---")
        
        # Static demo data
        alerts_data = pd.DataFrame({
            "Date/Time": ["2026-04-12 08:30:00", "2026-04-10 14:15:00", "2026-04-05 09:00:00", "Current"],
            "Event": ["Routine Inspection", "High Temperature Warning", "Tool Replacement", "Live Prediction"],
            "Severity": ["Info", "Warning", "Resolved", "Warning" if failure else "Healthy"]
        })
        
        def color_severity(val):
            if val == 'Warning': color = '#ff4b4b'
            elif val == 'Healthy' or val == 'Resolved': color = '#00c853'
            else: color = '#29b6f6'
            return f'color: {color}; font-weight: bold;'
            
        st.dataframe(alerts_data.style.map(color_severity, subset=['Severity']), use_container_width=True)


# Main router
if not st.session_state["logged_in"]:
    show_login()
else:
    show_app()