import streamlit as st
import google.generativeai as genai
import rag_engine
import os
import base64

# --- 1. CONFIGURATION & SETUP ---
st.set_page_config(page_title="Genesis", layout="wide")

# Securely load API Key
if "GOOGLE_API_KEY" in st.secrets:
    API_KEY = st.secrets["GOOGLE_API_KEY"]
elif os.getenv("GOOGLE_API_KEY"):
    API_KEY = os.getenv("GOOGLE_API_KEY")
else:
    st.error("❌ API Key missing. Please ensure you have a file named `.streamlit/secrets.toml` containing `GOOGLE_API_KEY = 'AIza...'`")
    st.stop()

# --- 2. BACKGROUND & STYLING SETUP ---
ASSETS_DIR = "assets"
BACKGROUND_IMAGE = "supper.jpg"  # Options: "supper.jpg", "creation.jpg", "prodigal.jpg"

def get_base64_image(image_path):
    """Convert image to base64 for embedding in CSS"""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def set_background():
    """Apply fixed background image, dark input, and elegant typography"""
    # 1. Get the absolute path to ensure image loading works in all environments
    base_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(base_dir, ASSETS_DIR, BACKGROUND_IMAGE)
    
    # 2. Check if file exists
    if not os.path.exists(img_path):
        st.error(f"❌ Could not find image at: {img_path}")
        return
    
    # 3. Convert to Base64
    b64_img = get_base64_image(img_path)
    
    # 4. Inject CSS
    st.markdown(f"""
    <style>
    /* IMPORT ELEGANT FONTS */
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;700&family=EB+Garamond:ital,wght@0,400;0,600;1,400&display=swap');

    /* MAIN BACKGROUND */
    [data-testid="stAppViewContainer"] {{
        background-image: linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.7)), 
                          url("data:image/jpeg;base64,{b64_img}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    
    /* REMOVE HEADER BACKGROUND */
    [data-testid="stHeader"] {{ background-color: rgba(0,0,0,0); }}

    /* TYPOGRAPHY */
    h1, h2, h3 {{
        font-family: 'Cinzel', serif !important; /* Roman-style headers */
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 2px !important;
        color: #e5e5e5 !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.8);
    }}
    
    p, div, span, i {{
        font-family: 'EB Garamond', serif !important; /* Elegant body text */
        font-size: 19px !important;
        color: #f0f0f0 !important;
    }}

    /* DARK TRANSPARENT INPUT BOX */
    .stTextInput > div > div > input {{
        background-color: rgba(0, 0, 0, 0.7) !important;
        color: #fff !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 4px !important; /* Sharper corners for a classic look */
        padding: 15px !important;
        font-family: 'EB Garamond', serif !important;
        font-size: 20px !important;
    }}

    /* INPUT FOCUS STATE */
    .stTextInput > div > div > input:focus {{
        background-color: rgba(0, 0, 0, 0.9) !important;
        border: 1px solid #d4af37 !important; /* Gold border */
        box-shadow: 0 0 15px rgba(212, 175, 55, 0.3) !important;
    }}
    
    .stTextInput > div > div > input::placeholder {{
        color: rgba(255, 255, 255, 0.5) !important;
    }}

    /* RESULT CARD STYLING */
    .result-container {{
        background-color: rgba(12, 12, 12, 0.85);
        padding: 40px;
        border: 1px solid rgba(212, 175, 55, 0.3); /* Subtle Gold Border */
        border-radius: 6px;
        margin-top: 30px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.6);
    }}
    
    .gold-text {{
        color: #d4af37 !important;
        font-weight: bold;
        font-size: 21px !important;
    }}
    
    /* SPINNER COLOR */
    .stSpinner > div {{
        border-top-color: #d4af37 !important;
    }}
    </style>
    """, unsafe_allow_html=True)

set_background()

# --- 3. LOAD DATABASE ---
@st.cache_resource
def load_db():
    return rag_engine.load_data()

try:
    docs, vectors = load_db()
except:
    st.error("Database missing! Please run 'python build_db.py' in the terminal first.")
    st.stop()

# --- 4. UI ---
st.title("GENESIS") 

query = st.text_input("", placeholder="Seek guidance...")

if query:
    # A. Search verses
    results = rag_engine.find_verses(query, docs, vectors, top_k=3)
    
    # B. Generate AI response
    genai.configure(api_key=API_KEY)
    
    try:
        model = genai.GenerativeModel('gemini-2.0-flash') 
        
        context = "\n".join([f"{r['ref']}: {r['text']}" for r in results])
        prompt = f"User: {query}\nVerses: {context}\nProvide a comforting, biblical response using these verses. Keep it solemn, elegant, and wise."
        
        with st.spinner("Discernment..."):
            response = model.generate_content(prompt)

            # C. Display results (Scripture FIRST, then Guidance)
            st.markdown(f"""
            <div class="result-container">
                <h3 style="color: #d4af37; border-bottom: 1px solid rgba(212,175,55,0.3); padding-bottom: 10px; margin-bottom: 25px;">
                    📖 Scripture
                </h3>
                {''.join([f'<div style="margin-bottom:25px;"><span class="gold-text">{r["ref"]}</span><br/><i style="opacity: 0.85; font-size: 18px;">"{r["text"]}"</i></div>' for r in results])}
                
                <br>
                
                <h3 style="color: #d4af37; border-bottom: 1px solid rgba(212,175,55,0.3); padding-bottom: 10px; margin-bottom: 25px;">
                    ✨ Guidance
                </h3>
                <p style="line-height: 1.8; font-size: 19px; text-align: justify; opacity: 0.95;">
                    {response.text}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"Error connecting to AI: {e}")