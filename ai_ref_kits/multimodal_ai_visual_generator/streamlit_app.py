import streamlit as st
from streamlit_helper import (
    init_session, check_fastapi_running, render_landing_page, render_scene_generation_page, render_image_generation_page,
    apply_custom_styling 
)

# ------------------ Page Config ------------------
st.set_page_config(page_title="Imagine Your Story", layout="centered")
apply_custom_styling()  # <-- call it here
init_session()

# ------------------ FastAPI Check ------------------
if "fastapi_ready" not in st.session_state:
    st.session_state.fastapi_ready = check_fastapi_running()
if not st.session_state.fastapi_ready:
    st.error("\u274c FastAPI backend is not running. Please start it first using:\n\n`uvicorn main:app --host 0.0.0.0 --port 8000`")
    st.stop()

# ------------------ Page Routing ------------------
if st.session_state.page == "landing":
    render_landing_page()
elif st.session_state.page == "scenes":
    render_scene_generation_page()
elif st.session_state.page == "images":
    render_image_generation_page()
