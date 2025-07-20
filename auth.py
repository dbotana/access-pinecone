import streamlit as st
import streamlit_authenticator as stauth
import bcrypt

def initialize_authenticator():
    """Initialize the authenticator with RAB member credentials."""
    
    # Load credentials from secrets
    try:
        usernames = st.secrets["credentials"]["usernames"]
        names = st.secrets["credentials"]["names"] 
        passwords = st.secrets["credentials"]["passwords"]
        roles = st.secrets["credentials"]["roles"]
        
        # Create authenticator
        authenticator = stauth.Authenticate(
            names=names,
            usernames=usernames, 
            passwords=passwords,
            cookie_name=st.secrets["auth"]["cookie_name"],
            key=st.secrets["auth"]["cookie_key"],
            cookie_expiry_days=st.secrets["auth"]["cookie_expiry_days"]
        )
        
        return authenticator, dict(zip(usernames, roles))
        
    except Exception as e:
        st.error(f"Authentication configuration error: {e}")
        return None, {}

def hash_passwords(plain_passwords):
    """Helper function to hash passwords for initial setup."""
    hashed_passwords = []
    for password in plain_passwords:
        hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        hashed_passwords.append(hashed.decode('utf-8'))
    return hashed_passwords

def get_user_api_key(username, user_roles):
    """Get appropriate API key based on user role."""
    user_role = user_roles.get(username, "member")
    
    if user_role == "admin":
        return st.secrets["openai"]["rab_api_key"]
    else:
        return st.secrets["openai"]["general_api_key"]

def check_rab_membership(username, user_roles):
    """Verify if user is a valid RAB member."""
    return username in user_roles