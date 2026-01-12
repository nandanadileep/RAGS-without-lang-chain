import streamlit as st
import google.generativeai as genai
import rag_engine
import os
import base64
import html

# --- 1. CONFIGURATION & SETUP ---
st.set_page_config(page_title="Genesis", layout="wide", page_icon="🕊️")

# Securely load API Key
if "GOOGLE_API_KEY" in st.secrets:
    API_KEY = st.secrets["GOOGLE_API_KEY"]
elif os.getenv("GOOGLE_API_KEY"):
    API_KEY = os.getenv("GOOGLE_API_KEY")
else:
    st.error("❌ API Key missing. Please check your secrets configuration.")
    st.stop()

# --- 2. BACKGROUND & STYLING SETUP ---
ASSETS_DIR = "assets"
BACKGROUND_IMAGE = "supper.jpg" 

def get_base64_image(image_path):
    """Convert image to base64 for embedding in CSS"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        return None

def set_background():
    """Apply fixed background image, dark input, and elegant typography"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(base_dir, ASSETS_DIR, BACKGROUND_IMAGE)
    b64_img = get_base64_image(img_path)
    
    # Fallback if image missing
    bg_style = ""
    if b64_img:
        bg_style = f"""
        background-image: linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.7)), 
                          url("data:image/jpeg;base64,{b64_img}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        """
    else:
        bg_style = "background-color: #0e0e0e;"

    st.markdown(f"""
    <style>
    /* IMPORT ELEGANT FONTS */
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;700&family=EB+Garamond:ital,wght@0,400;0,600;1,400&display=swap');

    /* MAIN BACKGROUND */
    [data-testid="stAppViewContainer"] {{
        {bg_style}
    }}
    
    /* REMOVE HEADER BACKGROUND */
    [data-testid="stHeader"] {{ background-color: rgba(0,0,0,0); }}

    /* TYPOGRAPHY */
    h1, h2, h3 {{
        font-family: 'Cinzel', serif !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 2px !important;
        color: #e5e5e5 !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.8);
    }}
    
    p, div, span, i {{
        font-family: 'EB Garamond', serif !important;
        font-size: 19px !important;
        color: #f0f0f0 !important;
    }}

    /* DARK TRANSPARENT INPUT BOX */
    .stTextInput > div > div > input {{
        background-color: rgba(0, 0, 0, 0.7) !important;
        color: #fff !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 4px !important;
        padding: 15px !important;
        font-family: 'EB Garamond', serif !important;
        font-size: 20px !important;
    }}

    /* INPUT FOCUS STATE */
    .stTextInput > div > div > input:focus {{
        background-color: rgba(0, 0, 0, 0.9) !important;
        border: 1px solid #d4af37 !important;
        box-shadow: 0 0 15px rgba(212, 175, 55, 0.3) !important;
    }}
    
    /* RESULT CARD STYLING */
    .result-container {{
        background-color: rgba(12, 12, 12, 0.85);
        padding: 40px;
        border: 1px solid rgba(212, 175, 55, 0.3);
        border-radius: 6px;
        margin-top: 30px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.6);
        animation: fadeIn 1.5s ease-in-out;
    }}

    @keyframes fadeIn {{
        0% {{ opacity: 0; transform: translateY(20px); }}
        100% {{ opacity: 1; transform: translateY(0); }}
    }}

    .gold-text {{
        color: #d4af37 !important;
        font-weight: bold;
        font-size: 21px !important;
    }}
    
    /* GUIDANCE TEXT STYLING */
    .guidance-text {{
        font-size: 19px; 
        text-align: justify; 
        opacity: 0.95;
    }}

    /* ADD MARGIN TO PARAGRAPHS INSIDE GUIDANCE */
    .guidance-text p {{
        margin-bottom: 20px;
        line-height: 1.8;
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
        prompt = (
            f"User: {query}\n"
            f"Verses: {context}\n"
            "Provide a comforting, biblical response connecting the user's query to these verses. "
            "Structure your response in clear, distinct paragraphs. "
            "Keep the tone solemn, elegant, and wise."
        )
        
        with st.spinner("Discernment..."):
            response = model.generate_content(prompt)

            # C. Construct HTML safely
            
            # 1. Process Guidance Text (Paragraphs)
            raw_text = (response.text or "")
            raw_text = raw_text.replace("**", "") # Clean bolding
            escaped_text = html.escape(raw_text)
            
            # Split by double newline to create distinct paragraphs
            paragraphs = [f"<p>{p.strip()}</p>" for p in escaped_text.split("\n\n") if p.strip()]
            guidance_html = "".join(paragraphs)

            # 2. Process Scripture Text
            verses_html = ""
            for r in results:
                s_ref = html.escape(r.get('ref', ''))
                s_txt = html.escape(r.get('text', ''))
                verses_html += f"""<div style="margin-bottom:25px;">
<span class="gold-text">{s_ref}</span><br/>
<i style="opacity: 0.85; font-size: 18px;">"{s_txt}"</i>
</div>"""

            # D. Display results
            st.markdown(f"""
<div class="result-container">
<h3 style="color: #d4af37; border-bottom: 1px solid rgba(212,175,55,0.3); padding-bottom: 10px; margin-bottom: 25px;">
📖 Scripture
</h3>
{verses_html}
<br>
<h3 style="color: #d4af37; border-bottom: 1px solid rgba(212,175,55,0.3); padding-bottom: 10px; margin-bottom: 25px;">
✨ Guidance
</h3>
<div class="guidance-text">
{guidance_html}
</div>
</div>
""", unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"Error connecting to AI: {e}")