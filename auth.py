import hmac
import streamlit as st

import bcrypt

def check_password():
    """Returns True if the user entered the correct password."""
    
    def password_entered():
        """Check if entered password matches the bcrypt hash in secrets."""
        entered_password = st.session_state["password"]
        stored_hash = st.secrets["app_password"]
        
        # Verify bcrypt hash
        if bcrypt.checkpw(entered_password.encode('utf-8'), stored_hash.encode('utf-8')):
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
        "🔐 Enter Application Password", 
        type="password", 
        on_change=password_entered, 
        key="password",
        help="Enter the password to access the Disability Science Research Assistant"
    )
    
    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
        st.error("🚫 Incorrect password - please try again")
    
    return False

def get_api_key():
    """Get the OpenAI API key from secrets."""
    return st.secrets["openai_api_key"]