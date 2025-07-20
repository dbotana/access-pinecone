import streamlit as st
import streamlit_authenticator as stauth
import bcrypt

def initialize_authenticator():
    """Initialize the authenticator with enhanced error handling."""
    
    try:
        # Load credentials with explicit error checking
        if "special_user" not in st.secrets:
            return None, None
            
        if "auth" not in st.secrets:
            return None, None
        
        username = st.secrets["special_user"]["username"]
        name = st.secrets["special_user"]["name"]
        password = st.secrets["special_user"]["password"]
        
        # Create credentials dictionary in the correct format
        credentials = {
            "usernames": {
                username: {
                    "name": name,
                    "password": password
                }
            }
        }
        
        # Create authenticator with correct parameter order
        authenticator = stauth.Authenticate(
            credentials,
            st.secrets["auth"]["cookie_name"],
            st.secrets["auth"]["cookie_key"],
            cookie_expiry_days=st.secrets["auth"]["cookie_expiry_days"]
        )
        
        return authenticator, username
        
    except Exception as e:
        st.error(f"Authentication setup error: {str(e)}")
        return None, None

def get_special_user_api_key():
    """Get the special user's API key from secrets."""
    try:
        # Debug: Check if secrets are accessible
        st.write("🔍 Checking API key access...")
        
        if "openai" not in st.secrets:
            st.error("❌ 'openai' section not found in secrets")
            return None
            
        if "special_user_api_key" not in st.secrets["openai"]:
            st.error("❌ 'special_user_api_key' not found in openai section")
            return None
            
        api_key = st.secrets["openai"]["special_user_api_key"]
        
        if api_key:
            st.write(f"✅ API key loaded (ends with: ...{api_key[-8:]})")
            return api_key
        else:
            st.error("❌ API key is empty")
            return None
            
    except Exception as e:
        st.error(f"❌ Error loading special user API key: {e}")
        return None


def hash_password(plain_password):
    """Helper function to hash a password for initial setup."""
    hashed = bcrypt.hashpw(plain_password.encode('utf-8'), bcrypt.gensalt())
    return hashed.decode('utf-8')
