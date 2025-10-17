import streamlit as st
import joblib

# ---------- Sidebar State ----------
if "page" not in st.session_state:
    st.session_state.page = "About"

if "show_models" not in st.session_state:
    st.session_state.show_models = False


# ---------- Navigation Helper ----------
def go_page(name):
    st.session_state.page = name


# ---------- Custom Styling ----------
st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background-color: #0e0e0e !important;
        color: white;
        padding-top: 1rem;
    }

    .sidebar-title {
        font-size: 22px;
        font-weight: 700;
        color: #00e0ff;
        margin-bottom: 1rem;
        text-align: left;
    }

    .sidebar-btn {
        background-color: #1a1a1a;
        color: #e6e6e6;
        padding: 10px 14px;
        margin: 6px 0;
        border-radius: 8px;
        border: 1px solid #222;
        width: 100%;
        text-align: left;
        font-size: 15px;
        transition: 0.2s;
        cursor: pointer;
    }

    .sidebar-btn:hover {
        background-color: #00e0ff;
        color: black;
        border-color: #00e0ff;
    }

    .dropdown-container {
        background-color: #141414;
        border: 1px solid #00e0ff;
        border-radius: 8px;
        padding: 8px;
        margin-top: 6px;
        margin-bottom: 10px;
        animation: slideDown 0.25s ease;
    }

    @keyframes slideDown {
        from {opacity: 0; transform: translateY(-5px);}
        to {opacity: 1; transform: translateY(0);}
    }

    .dropdown-item {
        background-color: #1f1f1f;
        color: #e6e6e6;
        padding: 8px 10px;
        border-radius: 6px;
        border: none;
        width: 100%;
        margin: 4px 0;
        text-align: left;
        font-size: 14px;
        cursor: pointer;
        transition: 0.2s;
    }

    .dropdown-item:hover {
        background-color: #00e0ff;
        color: black;
    }
    </style>
""", unsafe_allow_html=True)


# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("<div class='sidebar-title'>⚔️ Scamurai Suite</div>", unsafe_allow_html=True)

    if st.button("ℹ️ About / Home", key="about", use_container_width=True):
        go_page("About")

    # Models dropdown toggle
    if st.button("💠 Models ▼", key="models", use_container_width=True):
        st.session_state.show_models = not st.session_state.show_models

    if st.session_state.show_models:
        st.markdown("<div class='dropdown-container'>", unsafe_allow_html=True)
        if st.button("Text Katana (SMS)", key="text", use_container_width=True):
            go_page("Text Katana")
        if st.button("Mail Shuriken (Email)", key="mail", use_container_width=True):
            go_page("Mail Shuriken")
        if st.button("Voice Tanto (Voice)", key="voice", use_container_width=True):
            go_page("Voice Tanto")
        st.markdown("</div>", unsafe_allow_html=True)

    if st.button("📞 Contact", key="contact", use_container_width=True):
        go_page("Contact")


# ---------- Load Model ----------
model = joblib.load('Model/scam_lr_model.pkl')
vectorizer = joblib.load('Model/scam_vectorizer.pkl')


# ---------- Pages ----------
if st.session_state.page == "About":
    st.title("🏯 Scamurai Suite")
    st.markdown("""
        Welcome to **Scamurai Suite**, a cyber defense dojo powered by AI ⚔️.  
        This platform detects scams across different forms of communication.
        
        ### Modules
        - **Text Katana (SMS)** — Detects scam text messages  
        - **Mail Shuriken (Email)** — (Coming soon) Catches phishing emails  
        - **Voice Tanto (Voice)** — (Coming soon) Analyzes scam calls  

        ---
        **Developer:** Sreehan Adigopula (2025)  
        *For educational and cybersecurity research.*
    """)

elif st.session_state.page == "Text Katana":
    st.title("💬 Text Katana — SMS Scam Detector")
    user_input = st.text_area("Enter a message to test:")

    if st.button("Slice Scam!"):
        if user_input.strip():
            vectorized = vectorizer.transform([user_input])
            probas = model.predict_proba(vectorized)[0]
            ham_prob, spam_prob = probas[0], probas[1]

            confidence = max(ham_prob, spam_prob)
            prediction = model.predict(vectorized)[0]
            if confidence < 0.8:
                prediction = 1 if prediction == 0 else 0

            if prediction == 1:
                st.error("⚠️ This message is a **SCAM**.")
            else:
                st.success("✅ This message is **SAFE**.")

            st.subheader("Confidence Levels")
            st.write(f"Safe: {ham_prob:.2%}")
            st.progress(int(ham_prob * 100))
            st.write(f"Scam: {spam_prob:.2%}")
            st.progress(int(spam_prob * 100))
        else:
            st.warning("Please enter a message first.")

elif st.session_state.page == "Mail Shuriken":
    st.title("📧 Mail Shuriken — Email Scam Detector")
    st.write("Coming soon: detect phishing and malicious emails ⚙️")

elif st.session_state.page == "Voice Tanto":
    st.title("🎙️ Voice Tanto — Voice Scam Detector")
    st.write("Coming soon: analyze scam voice messages 🥷")

elif st.session_state.page == "Contact":
    st.title("📞 Contact")
    st.markdown("""
        **Reach out at:**  
        - 📧 `asreehan4u@gmail.com`  
        - 💻 [GitHub: SreehanAdigopula](https://github.com/SreehanAdigopula)
    """)


# ---------- Footer ----------
st.markdown("---")
st.caption("© 2025 Scamurai Suite | Crafted with ⚔️ by Sreehan Adigopula")
