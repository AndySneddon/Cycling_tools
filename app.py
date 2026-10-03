"""Entry point: navigation for the Sneddon in Sport cycling tools. Run with `streamlit run app.py`."""

import streamlit as st

pages = [
    st.Page("home.py", title="Home", icon=":material/home:", default=True),
    st.Page("pages/1_CdA_Estimator.py", title="CdA Estimator", icon=":material/air:", url_path="cda"),
    st.Page("pages/2_Gearing.py", title="Gearing", icon=":material/settings:", url_path="gearing"),
    st.Page("pages/3_Race_Planner.py", title="Race Planner", icon=":material/flag:", url_path="race-planner"),
]
st.navigation(pages).run()
