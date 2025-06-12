import streamlit as st
from Insta_Module import run_insta_dashboard
from Cyber_Module import run_cyber_dashboard

st.set_page_config(page_title="AI Dashboards", layout="wide")
st.sidebar.title(" Project Selector")

project = st.sidebar.radio("Choose a project", ["Instagram Influencer", "Cyber Threat Detection"])

if project == "Instagram Influencer":
    run_insta_dashboard()

elif project == "Cyber Threat Detection":
    run_cyber_dashboard()
