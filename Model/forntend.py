import streamlit as st
import joblib

# ---------- Sidebar State ----------
if "page" not in st.session_state:
    st.session_state.page = "Home"  # Default page

if "history" not in st.session_state:
    st.session_state.history = []  # To keep track of past checks

# ---------- Page Navigation Functions ----------
def go_home():
    st.session_state.page = "Home"

def go_about():
    st.session_state.page = "About"

def go_contact():
    st.session_state.page = "Contact"

# ---------- Small Button Styling ----------
st.markdown("""
    <style>
    .sidebar-btn {
        background-color: #f0f2f6;
        padding: 6px 10px;
        margin: 3px 0;
        border-radius: 6px;
        font-size: 14px;
        text-align: left;
        border: none;
        cursor: pointer;
        width: 100%;
    }
    .sidebar-btn:hover {
        background-color: #e0e2e6;
    }
    .active-btn {
        background-color: #4cafef !important;
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.header("📋 Menu")
    st.button("🏠 Home", key="home", on_click=go_home)
    st.button("ℹ️ About", key="about", on_click=go_about)
    st.button("📞 Contact", key="contact", on_click=go_contact)

# ---------- Load Model ----------
model = joblib.load('Model/scam_lr_model.pkl')
vectorizer = joblib.load('Model/scam_vectorizer.pkl')

# ---------- Page Content ----------
if st.session_state.page == "Home":
    st.title("🥷 Scamurai-Shield: Scam Message Detector")
    st.markdown("Enter a message to check if it's a scam:")

    user_input = st.text_area("")

    if st.button("Check"):
        if user_input.strip():  # only check if not empty
            vectorized = vectorizer.transform([user_input])

            probas = model.predict_proba(vectorized)[0]
            ham_prob, spam_prob = probas[0], probas[1]

            confidence = max(ham_prob, spam_prob)
            prediction = model.predict(vectorized)[0]
            if confidence < 0.8:
                prediction = 1 if prediction == 0 else 0

            # Show result
            if prediction == 1:
                st.error("⚠️ This message is a **SCAM**.")
            else:
                st.success("✅ This message is **SAFE**.")

            # Show probability bars
            st.subheader("Confidence Levels")
            st.write(f"Safe: {ham_prob:.2%}")
            st.progress(int(ham_prob * 100))
            st.write(f"Scam: {spam_prob:.2%}")
            st.progress(int(spam_prob * 100))

            # Save to history
            st.session_state.history.append({
                "message": user_input,
                "result": "SCAM" if prediction == 1 else "SAFE",
                "confidence": f"{confidence:.2%}"
            })
        else:
            st.warning("Please enter a message first.")

    # Show history if available
    if st.session_state.history:
        st.subheader("📝 History")
        for idx, entry in enumerate(reversed(st.session_state.history[-5:]), 1):
            st.write(f"{idx}. **{entry['result']}** (confidence: {entry['confidence']})")
            st.caption(f"Message: {entry['message']}")

elif st.session_state.page == "About":
    st.title("ℹ️ About")
    st.write("This app uses a trained machine learning model to detect scam messages.")
    st.write("This model is in its early stages and **can make mistakes**.")
    st.write("I'm still working on improving the model.")
    st.write("Developed by **Sreehan Adigopula**, 2025.")

elif st.session_state.page == "Contact":
    st.title("📞 Contact")
    st.write("For inquiries, reach out via:")
    st.markdown("- Email: `asreehan4u@gmail.com`")
    st.markdown("- GitHub: [My GitHub](https://github.com/SreehanAdigopula)")

# ---------- Footer ----------
st.markdown("#")
st.markdown("#")
st.markdown("#")
st.write("© 2025 Sreehan Adigopula")
