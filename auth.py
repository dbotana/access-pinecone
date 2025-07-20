import hmac
import streamlit as st

def check_password():
    """Returns True if the user entered the correct password."""
    
    def password_entered():
        """Check if entered password matches stored password."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["app_password"]):
            st.session_state["password_correct"] = True
            # Clear password from session for security
            del st.session_state["password"]  
        else:
            st.session_state["password_correct"] = False

    # Return True if password already validated
    if st.session_state.get("password_correct", False):
        return True

    # Show password input
    st.text_input(
        "Enter Password to Access Application", 
        type="password", 
        on_change=password_entered, 
        key="password"
    )
    
    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
        st.error("🚫 Incorrect password")
    
    return False

def get_api_key():
    """Get the OpenAI API key from secrets."""
    return st.secrets["openai_api_key"]