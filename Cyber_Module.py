import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import plotly.express as px

def run_cyber_dashboard():
    st.markdown("""
        <style>
        .main { background-color: #0e1117; color: white; }
        .stApp { background-color: #0e1117; }
        h1, h2, h3, h4, h5 { color: #00f0ff; }
        .metric-label, .metric-value { color: white !important; }
        .block-container { padding: 2rem 2rem 2rem 2rem; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div style='background-color: #1f2937; padding: 20px; border-radius: 10px; text-align: center;'>
            <h1 style='color: #ffffff;'>🛡️ Cybersecurity Threat Detection</h1>
            <p style='color: #d1d5db;'>Detect suspicious web traffic using anomaly detection & ML models</p>
        </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📁 Upload Web Traffic CSV", type=["csv"], key="cyber")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        df.drop_duplicates(inplace=True)
        for col in ['creation_time', 'end_time', 'time']:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        df['src_ip_country_code'] = df['src_ip_country_code'].str.upper()

        df['session_duration'] = (df['end_time'] - df['creation_time']).dt.total_seconds().replace(0, 1)
        df['avg_packet_size'] = (df['bytes_in'] + df['bytes_out']) / df['session_duration']

        features = df[['bytes_in', 'bytes_out', 'session_duration', 'avg_packet_size']].copy()
        for col in features.columns:
            median_val = features[col].replace([np.inf, -np.inf], np.nan).median()
            features[col] = features[col].replace([np.inf, -np.inf], median_val).fillna(median_val)

        # Anomaly Detection Models
        iso = IsolationForest(contamination=0.05, random_state=42)
        df['anomaly_iso'] = iso.fit_predict(features)
        df['anomaly_iso'] = df['anomaly_iso'].map({-1: 'Suspicious', 1: 'Normal'})

        svm_model = OneClassSVM(kernel='rbf', nu=0.05, gamma='auto')
        svm_pred = svm_model.fit_predict(features)
        df['anomaly_svm'] = pd.Series(svm_pred).map({-1: 'Suspicious', 1: 'Normal'})

        lof_model = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
        lof_pred = lof_model.fit_predict(features)
        df['anomaly_lof'] = pd.Series(lof_pred).map({-1: 'Suspicious', 1: 'Normal'})

        df['label'] = (df['detection_types'] == 'waf_rule').astype(int)
        X = features.copy()
        y = df['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        col1, col2 = st.columns([1, 2])
        col1.metric("✅ Accuracy", f"{acc:.2f}")
        col2.success("Random Forest model trained successfully")

        st.markdown("---")
        st.subheader("📊 Feature Importance")
        importances = clf.feature_importances_
        fig, ax = plt.subplots()
        sns.barplot(x=importances, y=X.columns, ax=ax, palette="mako")
        ax.set_title("Feature Importance", fontsize=14, fontweight='bold')
        st.pyplot(fig)

        st.subheader("📉 Anomaly Overview")
        col1, col2 = st.columns(2)

        with col1:
            fig2 = px.scatter(df, x='bytes_in', y='bytes_out', color='anomaly_iso',
                              title="Bytes In vs Bytes Out (ISO Detection)",
                              labels={'bytes_in': 'Bytes In', 'bytes_out': 'Bytes Out'},
                              template="plotly_dark")
            st.plotly_chart(fig2)

        with col2:
            st.markdown("### 🔎 Top Suspicious IPs")
            top_ips = df[df['anomaly_iso'] == 'Suspicious']['src_ip'].value_counts().head(5).reset_index()
            top_ips.columns = ['IP Address', 'Suspicious Count']
            st.table(top_ips)

        st.markdown("---")
        st.subheader("🔥 Protocol vs Destination Port Heatmap")
        if 'protocol' in df.columns and 'dst_port' in df.columns:
            heatmap_data = df.pivot_table(index='protocol', columns='dst_port', aggfunc='size', fill_value=0)
            fig3, ax3 = plt.subplots(figsize=(12, 6))
            sns.heatmap(heatmap_data, cmap='YlGnBu', linewidths=0.5, ax=ax3)
            ax3.set_title("Protocol vs Destination Port", fontsize=14)
            st.pyplot(fig3)

        st.subheader("🌍 Suspicious Traffic by Country")
        if 'src_ip_country_code' in df.columns:
            country_anomalies = df[df['anomaly_iso'] == 'Suspicious']['src_ip_country_code'].value_counts().head(10)
            fig4, ax4 = plt.subplots(figsize=(10, 4))
            sns.barplot(x=country_anomalies.index, y=country_anomalies.values, ax=ax4, palette="flare")
            ax4.set_title("Top 10 Countries with Suspicious Traffic")
            ax4.set_ylabel("Suspicious Count")
            ax4.set_xlabel("Country Code")
            st.pyplot(fig4)

        st.subheader("⏰ Hourly Anomaly Trend")
        if 'time' in df.columns:
            df['hour'] = pd.to_datetime(df['time'], errors='coerce').dt.hour
            hourly_trend = df[df['anomaly_iso'] == 'Suspicious']['hour'].value_counts().sort_index()
            fig5, ax5 = plt.subplots()
            sns.lineplot(x=hourly_trend.index, y=hourly_trend.values, marker='o', ax=ax5)
            ax5.set_title("Suspicious Activity by Hour", fontsize=14)
            ax5.set_xlabel("Hour of Day")
            ax5.set_ylabel("Anomaly Count")
            st.pyplot(fig5)

        st.markdown("---")
        st.subheader("📄 Data Preview")
        st.dataframe(df.head(20), height=300, use_container_width=True)
