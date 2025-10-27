import streamlit as st
import joblib
import time

# Page config MUST be the first Streamlit command
st.set_page_config(
    page_title="Scamurai Suite",
    page_icon="⚔️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- SESSION STATE ----------------
if "page" not in st.session_state:
    st.session_state.page = "About"

if "models_dropdown" not in st.session_state:
    st.session_state.models_dropdown = False

# ---------------- NAVIGATION ----------------


def go_page(page_name):
    st.session_state.page = page_name


def toggle_models():
    st.session_state.models_dropdown = not st.session_state.models_dropdown


# ---------------- GLOBAL STYLES ----------------
st.markdown("""
    <style>
    /* General App Styling */
    .stApp {
        background-color: #0e0e0e;
        color: #f0f0f0;
    }
    [data-testid="stSidebar"] { background-color: #141414 !important; border-right: 1px solid #00e0ff33; }
    .stButton button { background-color: #1f1f1f; color: #fff; border: 1px solid #444; border-radius: 10px; font-weight: 500; padding: 10px; width: 100%; transition: background-color 0.3s, border-color 0.3s; }
    .stButton button:hover { background-color: #00e0ff22; border: 1px solid #00e0ff; }
    .stButton button[kind="primary"] { background-color: #007bff; border-color: #007bff; }
    .stButton button[kind="primary"]:hover { background-color: #0056b3; border-color: #0056b3; }
    .models-container { background-color: #1a1a1a; border-radius: 10px; padding: 8px; margin-top: 6px; box-shadow: 0 0 10px #00e0ff33; }
    .content-container { background-color: #0b0b0b; border: 1px solid #00e0ff55; border-radius: 16px; padding: 2rem 3rem; margin: 2rem auto; width: 90%; max-width: 950px; box-shadow: 0 0 20px #00e0ff33; }
    .content-title { font-size: 2.3rem; color: #00e0ff; text-align: center; font-weight: 700; margin-bottom: 1rem; }
    .content-subtitle { color: #aaa; font-size: 1.1rem; text-align: center; margin-bottom: 2rem; }
    .content-section { color: #f0f0f0; line-height: 1.6; font-size: 1.1rem; margin-top: 1.2rem; }
    .highlight { color: #00e0ff; font-weight: bold; }
    .content-section ul { list-style-type: none; margin-left: 1.5rem; padding-left: 0; }
    .content-section ul li { margin-bottom: 0.5rem; }
    .content-section ul li::before { content: "⚔️ "; margin-right: 0.5rem; }
    .result-container { margin-top: 2rem; padding: 1.5rem; border-radius: 10px; text-align: center; }
    .result-scam { border: 1px solid #ff4b4b; background-color: #ff4b4b22; }
    .result-safe { border: 1px solid #28a745; background-color: #28a74522; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("<h2 style='color:#00e0ff; text-align: center;'>⚔️ Scamurai Suite</h2>",
                unsafe_allow_html=True)
    st.markdown("---")

    st.button("ℹ️ About / Home", on_click=go_page,
              args=("About",), use_container_width=True)

    models_button_label = "💠 Models ▴" if st.session_state.models_dropdown else "💠 Models ▾"
    st.button(models_button_label, on_click=toggle_models,
              use_container_width=True)

    if st.session_state.models_dropdown:
        with st.container():
            st.markdown("<div class='models-container'>",
                        unsafe_allow_html=True)
            st.button("⚔️ Text Katana (SMS)", on_click=go_page,
                      args=("SMS",), use_container_width=True)
            st.button("📧 Mail Shuriken (Email)", on_click=go_page,
                      args=("Email",), use_container_width=True)
            st.button("🎙️ Voice Tanto (Voice)", on_click=go_page,
                      args=("Voice",), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    st.button("📞 Contact", on_click=go_page, args=(
        "Contact",), use_container_width=True)

# ---------------- LOAD MODEL ----------------


@st.cache_resource
def load_model():
    try:
        model = joblib.load('Model/scam_lr_model.pkl')
        vectorizer = joblib.load('Model/scam_vectorizer.pkl')
        return model, vectorizer
    except FileNotFoundError:
        st.sidebar.error(
            "⚠️ Model files not found. Please check the 'Model' directory.")
        return None, None
    except Exception as e:
        st.sidebar.error(f"An error occurred while loading models: {e}")
        return None, None


model, vectorizer = load_model()

# ---------------- PAGE CONTENT ROUTING ----------------
if st.session_state.page == "About":
    st.markdown("""
    <div class="content-container">
        <div class="content-title">Scamurai Suite</div>
        <div class="content-subtitle">Your Cyber-Defense Dojo, Powered by AI</div>
        <div class="content-section">
            Welcome to <span class="highlight">Scamurai Suite</span>: a skilled collection of AI-driven models 
            that will slice through scams and keeps everything safe.
        </div>
        <div class="content-section">
            <b>Models (Active & Upcoming):</b>
            <ul>
                <li><span class="highlight">Text Katana (SMS)</span>: Detects scam or spam text messages in real time.</li>
                <li><span class="highlight">Mail Shuriken (Email)</span>: (Coming soon) Scans emails for phishing attempts.</li>
                <li><span class="highlight">Voice Tanto (Voice)</span>: (Coming soon) Analyzes scam call transcripts or audio.</li>
            </ul>
        </div>
        <div class="content-section">
            <span class="highlight">Developed by:</span> <b>Sreehan Adigopula</b><br>
            Year: <b>2025</b><br>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center; color:#aaa; font-size:0.9rem;'>⚠️ Note: This model is not always 100% accurate. It’s meant for educational and experimental use only.</p>",
        unsafe_allow_html=True
    )

elif st.session_state.page == "SMS":
    st.markdown("<h1 style='color:#00e0ff;'>⚔️ Text Katana (SMS Scam Detector)</h1>",
                unsafe_allow_html=True)
    st.markdown("---", unsafe_allow_html=True)
    if not model or not vectorizer:
        st.error(
            "Model is not loaded. Cannot perform analysis. Please check the sidebar for error messages.")
    else:
        user_input = st.text_area("Enter a message to analyze:",
                                  height=150, placeholder="Paste your SMS message here...")
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            check_button = st.button(
                "🔍 Analyze Message", use_container_width=True, type="primary")
        if check_button and user_input.strip():
            with st.spinner("Analyzing..."):
                time.sleep(1)  # Simulate analysis time
                vec = vectorizer.transform([user_input])
                prediction = model.predict(vec)[0]
                probas = model.predict_proba(vec)[0]
                confidence = max(probas)
            st.markdown("---", unsafe_allow_html=True)
            if prediction == 1:
                st.markdown(
                    f"""
                    <div class="result-container result-scam">
                        <h2 style='color:#ff4b4b;'>⚠️ SCAM DETECTED ⚠️</h2>
                        <p>Confidence Level: <strong>{confidence:.1%}</strong></p>
                        <p>This message shows characteristics of a scam. Be cautious and do not click any links or provide personal information.</p>
                    </div>
                    """, unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div class="result-container result-safe">
                        <h2 style='color:#28a745;'>✅ MESSAGE APPEARS SAFE ✅</h2>
                        <p>Confidence Level: <strong>{confidence:.1%}</strong></p>
                        <p>This message appears to be legitimate. However, always be cautious and watch out.</p>
                    </div>
                    """, unsafe_allow_html=True
                )
        elif check_button:
            st.warning("⚠️ Please enter a message to analyze.")

elif st.session_state.page == "Email":
    st.markdown("<h1 style='color:#00e0ff;'>📧 Mail Shuriken (Email Scam Detector)</h1>",
                unsafe_allow_html=True)
    st.markdown("---", unsafe_allow_html=True)
    st.info("🚧 **Coming Soon** — This model will analyze suspicious emails for phishing intent and malicious content.")

elif st.session_state.page == "Voice":
    st.markdown("<h1 style='color:#00e0ff;'>🎙️ Voice Tanto (Voice Scam Detector)</h1>",
                unsafe_allow_html=True)
    st.markdown("---", unsafe_allow_html=True)
    st.info("🚧 **Coming Soon** — This model will detect scam or fraudulent voice messages and calls.")


elif st.session_state.page == "Contact":
    st.markdown("<h1 style='color:#00e0ff;'>📞 Contact</h1>",
                unsafe_allow_html=True)
    st.markdown("---", unsafe_allow_html=True)
    st.markdown("""
    <div class="content-container">
        <div class="content-section">
            <h3>Get in Touch</h3>
            If you have any inquiries, please feel free to reach out.
            <br><br>
            <b>Email:</b> <a href="mailto:asreehan4u@gmail.com" style="color: #00e0ff;">asreehan4u@gmail.com</a><br>
            <b>GitHub:</b> <a href="https://github.com/SreehanAdigopula" target="_blank" style="color: #00e0ff;">SreehanAdigopula</a><br>
            <b>Linkedin:</b> <a href="https://www.linkedin.com/in/asreehan/" target="_blank" style="color: #00e0ff;">Sreehan Adigopula</a>
        </div>
    </div>
    """, unsafe_allow_html=True)

# A consistent footer for all pages
st.markdown("---", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: #666;'>© 2025 Sreehan Adigopula | Scamurai Suite | Educational Purpose Only</p>",
    unsafe_allow_html=True
)
