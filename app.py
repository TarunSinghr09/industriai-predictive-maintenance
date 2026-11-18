"""INDUSTRIAI entry point: defines the sidebar navigation."""
import streamlit as st

pages = [
    st.Page("pages/0_Fleet_Overview.py", title="Fleet Overview", icon="🏭", default=True),
    st.Page("pages/1_Live_Monitoring.py", title="Live Monitoring", icon="📡"),
    st.Page("pages/2_Machine_Check.py", title="Machine Check", icon="🧪"),
    st.Page("pages/3_Maintenance_Planner.py", title="Maintenance Planner", icon="🗓️"),
    st.Page("pages/4_Model_Performance.py", title="Model Performance", icon="📊"),
    st.Page("pages/5_Data_Explorer.py", title="Data Explorer", icon="🔎"),
    st.Page("pages/6_About.py", title="About", icon="ℹ️"),
]
st.navigation(pages).run()
