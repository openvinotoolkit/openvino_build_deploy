import streamlit as st
import requests
import base64
import time
from PIL import Image
from io import BytesIO
from fpdf import FPDF
import os
import yaml
import uuid
import streamlit.components.v1 as components

# ------------------ Dark Theme Styling ------------------
def apply_custom_styling():
    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(180deg, #0f0f1c, #1a1a2e) !important;
        color: white !important;
    }
    html, body, [class*="css"] {
        color: #ffffff !important;
    }
    .stTextInput>div>div>input {
        background-color: #222;
        color: #fff;
        border: 1px solid #444;
        border-radius: 8px;
        padding: 0.6rem;
    }
    .stTextInput>div>div>input::placeholder {
        color: #999 !important;
        opacity: 1 !important;
    }
    div[data-testid="column"] button {
        background-color: #2a2a2a !important;
        color: #dddddd !important;
        border: 1px solid #444 !important;
        border-radius: 999px !important;
        font-size: 0.9rem !important;
        padding: 0.5rem 1.2rem !important;
        margin: 0.4rem 0.6rem !important;
        white-space: normal !important;
        text-align: center !important;
    }
    div.stButton>button {
        background: linear-gradient(90deg, #6a11cb 0%, #2575fc 100%) !important;
        color: white !important;
        border-radius: 10px;
        font-size: 1.1rem;
        font-weight: bold;
        padding: 0.75rem 2rem;
        margin-top: 1rem;
    }
    .story-download-section {
        padding: 1.5rem 2rem;
        border: 1px solid #444;
        border-radius: 12px;
        background-color: #1f1f2e;
        margin-top: 2.5rem;
        text-align: center;
        max-width: 600px;
        margin-left: auto;
        margin-right: auto;
    }
    .story-download-section h3 {
        font-size: 1.8rem;
        font-weight: bold;
        margin-bottom: 0.75rem;
    }
    .story-download-section p {
        font-size: 1.1rem;
        margin-bottom: 1.2rem;
    }
    .button-wrapper {
        text-align: center;
        margin-top: 1rem;
    }
    label, .stTextInput label {
        color: #ffffff !important;
    }
    [data-baseweb="select"] {
        background-color: #222 !important;
        color: #fff !important;
        border: 1px solid #444 !important;
        border-radius: 8px !important;
        padding: 0.6rem !important;
        font-size: 1rem !important;
    }
    [data-baseweb="select"] div[role="button"] {
        color: #fff !important;
    }
    [data-baseweb="menu"] {
        background-color: #1a1a2e !important;
        color: #fff !important;
        border: 1px solid #444 !important;
        border-radius: 8px !important;
    }
    [data-baseweb="option"] {
        background-color: transparent !important;
        color: #fff !important;
        padding: 0.6rem 1rem !important;
    }
    [data-baseweb="option"]:hover {
        background-color: #333 !important;
        color: #fff !important;
    }
    </style>
    """, unsafe_allow_html=True)

# ------------------ Session State Initialization ------------------
def init_session():
    defaults = {
        "page": "landing",
        "story_idea": "",
        "scene_animation_complete": False,
        "images_generated": False,
        "images_displayed": False,
        "generated_images": [],
        "pdf_data": None,
        "scenes": None,
        "edited_scenes": None,
        "mode_param": "illustration"
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# ------------------ Reset Logic ------------------
def reset_to_landing():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    init_session()
    st.rerun()

# ------------------ FastAPI Health Check ------------------
def check_fastapi_running():
    try:
        res = requests.get("http://localhost:8000/docs", timeout=3)
        return res.status_code == 200
    except:
        return False

# ------------------ Load Prompts from YAML ------------------
def load_config_prompts(mode):
    config_path = f"config/{mode}.yaml"
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
    return data.get("prompts", [])

# ------------------ PDF Download Button ------------------
def render_download_button(pdf_bytes, filename):
    b64 = base64.b64encode(pdf_bytes).decode()
    return f"""
    <div class='button-wrapper'>
        <a href="data:application/pdf;base64,{b64}" download="{filename}" target="_self">
            <button style="
                background: linear-gradient(90deg, #6a11cb 0%, #2575fc 100%);
                color: white;
                padding: 0.75rem 2rem;
                font-size: 1.1rem;
                font-weight: bold;
                border: none;
                border-radius: 10px;
                cursor: pointer;
                margin-top: 1rem;
            ">Download Your Story as PDF</button>
        </a>
    </div>
    """

# ------------------ Landing Page ------------------
def render_landing_page():
    mode_display = st.selectbox("🛠️ Choose a generation mode", ["🖼️ Story Illustration mode", "👕 Branding mode"], index=0)
    new_mode = "illustration" if "Illustration" in mode_display else "branding"
    if st.session_state.get("mode_param") != new_mode:
        st.session_state.mode_param = new_mode
        st.rerun()

    mode_param = st.session_state.mode_param
    title = "🎩 Imagine Your Story" if mode_param == "illustration" else "👕 Design Your Brand"
    subtitle = "Craft a magical 4-scene story from a single idea" if mode_param == "illustration" else "Generate 4 artistic T-shirt designs from a single prompt"
    st.markdown(f"<h1>{title}</h1><p>{subtitle}</p>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("<p style='font-size:1.2rem;'>Enter your idea below or choose from our suggestions</p>", unsafe_allow_html=True)
    placeholder = "e.g., A robot building a sandcastle" if mode_param == "illustration" else "e.g., A cat surfing a rainbow"
    story_input = st.text_input("Your creative idea", value=st.session_state.story_idea, placeholder=placeholder)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        btn_label = "🎬 Generate Story Scenes" if mode_param == "illustration" else "🎨 Generate Design Concepts"
        generate_button = st.button(btn_label, use_container_width=True)

    st.markdown("---")
    st.markdown("<p style='font-size:1.2rem;'>Or pick a suggestion:</p>", unsafe_allow_html=True)
    prompts = load_config_prompts(mode_param)
    prompt_cols = st.columns(2)
    for idx, prompt in enumerate(prompts):
        with prompt_cols[idx % 2]:
            if st.button(prompt, key=f"prompt_{idx}"):
                st.session_state.story_idea = prompt
                st.rerun()

    if generate_button and story_input:
        st.session_state.page = "scenes"
        st.session_state.story_idea = story_input
        st.rerun()

# ------------------ Scene Generation Page ------------------
def render_scene_generation_page():
    st.markdown("---")
    mode_param = st.session_state.mode_param
    title = "🖍️ Your Story Scenes" if mode_param == "illustration" else "🎨 Your Design Concepts"
    st.markdown(f"<h2>{title}</h2>", unsafe_allow_html=True)

    labels = ["Beginning", "Rising Action", "Climax", "Resolution"] if mode_param == "illustration" else [f"Design {i+1}" for i in range(4)]
    cols = st.columns(2)
    placeholders = [cols[i % 2].empty() for i in range(4)]

    if st.session_state.scenes is None:
        with st.spinner("Generating story scenes..."):
            res = requests.post(f"http://localhost:8000/generate_story_prompts?config={mode_param}", json={"prompt": st.session_state.story_idea})
            scenes = res.json()["scenes"]
            st.session_state.scenes = scenes
            st.session_state.edited_scenes = scenes.copy()

    if not st.session_state.scene_animation_complete:
        for idx, text in enumerate(st.session_state.scenes):
            display = ""
            for word in text.split(" "):
                display += word + " "
                placeholders[idx].text_area(label=labels[idx], value=display, height=150)
                time.sleep(0.03)
        st.session_state.scene_animation_complete = True
    else:
        for idx, text in enumerate(st.session_state.edited_scenes):
            placeholders[idx].text_area(label=labels[idx], value=text, height=150)

    next_btn = "📸 Tell Your Story" if mode_param == "illustration" else "📸 Show Me the Designs"
    if st.button(next_btn, use_container_width=True):
        st.session_state.page = "images"
        st.rerun()

# ------------------ Image Generation Page ------------------
def render_image_generation_page():
    st.markdown("---")
    mode_param = st.session_state.mode_param
    title = "🎮 Your Visual Story" if mode_param == "illustration" else "🎨 Your Design Showcase"
    st.markdown(f"<h2>{title}</h2>", unsafe_allow_html=True)

    if not st.session_state.images_generated:
        progress_bar = st.progress(0)
        cols = st.columns(2)
        st.session_state.generated_images.clear()

        for idx, prompt in enumerate(st.session_state.edited_scenes):
            with cols[idx % 2]:
                with st.spinner(f"🖼️ Generating image {idx+1}/4..."):
                    res = requests.post("http://localhost:8000/generate_images", json={"prompt": prompt})
                img_b64 = res.json()["image"]
                img = Image.open(BytesIO(base64.b64decode(img_b64)))
                st.session_state.generated_images.append((img, prompt))
                st.image(img, caption=prompt, use_container_width=True)
                progress_bar.progress((idx + 1) / 4)

        # Generate PDF
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_fill_color(31, 31, 46)
        pdf.rect(0, 0, 210, 297, 'F')
        pdf.set_font("Arial", "B", 28)
        pdf.set_text_color(106, 17, 203)
        pdf.cell(0, 80, "", ln=True)
        pdf_title = "Your Visual Story" if st.session_state.mode_param == "illustration" else "Your T-Shirt Designs"
        pdf.cell(0, 20, pdf_title, ln=True, align='C')
        pdf.set_font("Arial", "", 16)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(0, 15, "Crafted with AI Magic", ln=True, align='C')
        pdf.ln(20)
        pdf.set_font("Arial", "I", 14)
        pdf.set_text_color(200, 200, 200)
        pdf.multi_cell(0, 10, f'Story Idea:\n"{st.session_state.story_idea}"', align='C')


        image_paths = []
        for img, caption in st.session_state.generated_images:
            pdf.add_page()
            img_path = f"temp_image_{uuid.uuid4().hex}.png"
            img.save(img_path)
            image_paths.append(img_path)
            pdf.image(img_path, w=180)
            pdf.ln(5)
            pdf.set_font("Arial", size=12)
            pdf.set_text_color(0, 0, 0)
            pdf.multi_cell(0, 10, caption)

        pdf_bytes = BytesIO()
        pdf_bytes.write(pdf.output(dest='S').encode('latin1'))
        pdf_bytes.seek(0)
        st.session_state.pdf_data = pdf_bytes.read()
        for path in image_paths:
            os.remove(path)

        st.session_state.images_generated = True
        st.session_state.images_displayed = True

    elif not st.session_state.images_displayed:
        cols = st.columns(2)
        for idx, (img, cap) in enumerate(st.session_state.generated_images):
            with cols[idx % 2]:
                st.image(img, caption=cap, use_container_width=True)
        st.session_state.images_displayed = True

    if st.session_state.pdf_data:
        pdf_heading = "📘 Your Story is Ready to Save" if mode_param == "illustration" else "👕 Your Designs are Ready to Save"
        pdf_desc = "Click the button below to download your illustrated PDF story." if mode_param == "illustration" else "Click the button below to download your T-shirt design concepts."
        st.markdown(f"""<div class="story-download-section">
            <h3>{pdf_heading}</h3>
            <p>{pdf_desc}</p>
        </div>""", unsafe_allow_html=True)
        cols = st.columns([1, 2, 1])
        with cols[1]:
            pdf_filename = "visual_story.pdf" if mode_param == "illustration" else "tshirt_designs.pdf"
            components.html(render_download_button(st.session_state.pdf_data, pdf_filename), height=100)

    button_cols = st.columns([1, 2, 1])
    with button_cols[1]:
        reset_label = "📖 Create a new story" if mode_param == "illustration" else "🖌️ Start a new design"
        if st.button(reset_label, use_container_width=True):
    
            reset_to_landing()
