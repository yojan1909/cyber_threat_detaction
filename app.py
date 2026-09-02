import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI Cyber Threat Detection",
    page_icon="🛡️",
    layout="wide"
)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🛡️ AI-Powered Cyber Threat Detection System")
st.markdown(
    "### Machine Learning based Network Intrusion Detection Dashboard"
)

st.divider()


# --------------------------------------------------
# GENERATE SAMPLE CYBERSECURITY DATA
# --------------------------------------------------

@st.cache_data
def generate_data():

    np.random.seed(42)

    n = 2000

    data = pd.DataFrame({
        "packet_count": np.random.randint(10, 1000, n),
        "packet_size": np.random.randint(40, 1500, n),
        "connection_duration": np.random.randint(1, 300, n),
        "failed_logins": np.random.randint(0, 15, n),
        "port_scan_count": np.random.randint(0, 20, n),
        "bytes_sent": np.random.randint(100, 100000, n),
        "bytes_received": np.random.randint(100, 100000, n)
    })

    # Create threat labels
    data["threat"] = (
        (data["failed_logins"] > 8) |
        (data["port_scan_count"] > 12) |
        (data["packet_count"] > 800) |
        (data["bytes_sent"] > 80000)
    ).astype(int)

    return data


data = generate_data()


# --------------------------------------------------
# TRAIN MACHINE LEARNING MODEL
# --------------------------------------------------

X = data.drop("threat", axis=1)
y = data["threat"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=150,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.header("⚙️ System Information")

st.sidebar.metric(
    "Model Accuracy",
    f"{accuracy * 100:.2f}%"
)

st.sidebar.info(
    "The system uses a Random Forest machine learning "
    "model to identify suspicious network activity."
)


# --------------------------------------------------
# DASHBOARD METRICS
# --------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Network Records",
        len(data)
    )

with col2:
    st.metric(
        "Normal Traffic",
        int((data["threat"] == 0).sum())
    )

with col3:
    st.metric(
        "Threat Records",
        int((data["threat"] == 1).sum())
    )

with col4:
    st.metric(
        "Model Accuracy",
        f"{accuracy * 100:.1f}%"
    )


st.divider()


# --------------------------------------------------
# THREAT DISTRIBUTION
# --------------------------------------------------

st.subheader("📊 Threat Distribution")

threat_counts = data["threat"].value_counts()

fig, ax = plt.subplots()

ax.bar(
    ["Normal", "Threat"],
    [
        threat_counts.get(0, 0),
        threat_counts.get(1, 0)
    ]
)

ax.set_ylabel("Number of Records")
ax.set_title("Network Traffic Classification")

st.pyplot(fig)


# --------------------------------------------------
# USER INPUT FOR REAL-TIME DETECTION
# --------------------------------------------------

st.divider()

st.subheader("🔍 Real-Time Cyber Threat Detection")

st.write(
    "Enter network activity information below "
    "to check whether it is normal or suspicious."
)


col1, col2 = st.columns(2)

with col1:

    packet_count = st.number_input(
        "Packet Count",
        min_value=0,
        max_value=10000,
        value=500
    )

    packet_size = st.number_input(
        "Average Packet Size",
        min_value=40,
        max_value=1500,
        value=500
    )

    connection_duration = st.number_input(
        "Connection Duration (seconds)",
        min_value=1,
        max_value=5000,
        value=100
    )

    failed_logins = st.number_input(
        "Failed Login Attempts",
        min_value=0,
        max_value=100,
        value=2
    )


with col2:

    port_scan_count = st.number_input(
        "Port Scan Count",
        min_value=0,
        max_value=100,
        value=2
    )

    bytes_sent = st.number_input(
        "Bytes Sent",
        min_value=0,
        max_value=1000000,
        value=20000
    )

    bytes_received = st.number_input(
        "Bytes Received",
        min_value=0,
        max_value=1000000,
        value=30000
    )


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if st.button("🚨 Analyze Network Activity", use_container_width=True):

    input_data = pd.DataFrame({
        "packet_count": [packet_count],
        "packet_size": [packet_size],
        "connection_duration": [connection_duration],
        "failed_logins": [failed_logins],
        "port_scan_count": [port_scan_count],
        "bytes_sent": [bytes_sent],
        "bytes_received": [bytes_received]
    })

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0]

    threat_probability = probability[1] * 100


    st.divider()

    if prediction == 1:

        st.error("🚨 CYBER THREAT DETECTED!")

        st.warning(
            f"Threat Probability: {threat_probability:.2f}%"
        )

        st.write("### Recommended Actions")

        st.write("• Investigate the source IP address")
        st.write("• Check failed login attempts")
        st.write("• Monitor unusual network traffic")
        st.write("• Review firewall logs")
        st.write("• Consider temporarily blocking suspicious traffic")

    else:

        st.success("✅ NETWORK ACTIVITY IS NORMAL")

        st.info(
            f"Threat Probability: {threat_probability:.2f}%"
        )


# --------------------------------------------------
# FEATURE IMPORTANCE
# --------------------------------------------------

st.divider()

st.subheader("🧠 AI Feature Importance")

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

st.dataframe(
    importance,
    use_container_width=True
)


# --------------------------------------------------
# DATASET
# --------------------------------------------------

st.divider()

st.subheader("📁 Network Traffic Dataset")

st.dataframe(
    data.head(100),
    use_container_width=True
)


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "AI-Powered Cyber Threat Detection System | "
    "Python + Machine Learning + Streamlit"
)
