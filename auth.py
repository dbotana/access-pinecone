import streamlit as st
import streamlit_authenticator as stauth
import bcrypt

def initialize_authenticator():
    """Initialize the authenticator with single special user."""
    
    try:
        # Load single user credentials from secrets
        username = st.secrets["special_user"]["username"]
        name = st.secrets["special_user"]["name"]
        password = st.secrets["special_user"]["password"]
        
        # Create authenticator for single user
        authenticator = stauth.Authenticate(
            names=[name],
            usernames=[username], 
            passwords=[password],
            cookie_name=st.secrets["auth"]["cookie_name"],
            key=st.secrets["auth"]["cookie_key"],
            cookie_expiry_days=st.secrets["auth"]["cookie_expiry_days"]
        )
        
        return authenticator, username
        
    except Exception as e:
        st.error(f"Authentication configuration error: {e}")
        return None, None

def get_special_user_api_key():
    """Get the special user's API key from secrets."""
    try:
        return st.secrets["openai"]["special_user_api_key"]
    except Exception as e:
        st.error(f"Error loading special user API key: {e}")
        return None

def hash_password(plain_password):
    """Helper function to hash a password for initial setup."""
    hashed = bcrypt.hashpw(plain_password.encode('utf-8'), bcrypt.gensalt())
    return hashed.decode('utf-8')