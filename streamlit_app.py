import streamlit as st
import os

# -- Secret Management --
st.sidebar.title("API Keys")
openai_key = st.sidebar.text_input("OpenAI API Key", type="password")
gemini_key = st.sidebar.text_input("Gemini API Key", type="password")
vertex_key = st.sidebar.text_input("Vertex API Key", type="password")

selected_api = st.sidebar.radio("Which API to use?", ["OpenAI", "Gemini", "Vertex"])

api_keys = {
    "OpenAI": openai_key,
    "Gemini": gemini_key,
    "Vertex": vertex_key
}

def get_api_key():
    key = api_keys[selected_api]
    if not key:
        st.error(f"Please provide a {selected_api} API key.")
        st.stop()
    return key

# -- Input Section --
st.title("Text Extract Optimizer Demo")
input_text = st.text_area("Paste your input text here:", height=240)

if st.button("Run") and input_text.strip():
    key = get_api_key()
    
    # --- Initial Extraction and Visualization ---
    def extract_structure(text, key, api):
        # TODO: Replace with real API call/logic
        # This is a demo placeholder
        import hashlib
        data = {
            "length": len(text),
            "unique_words": len(set(text.split())),
            "hash": hashlib.md5(text.encode()).hexdigest()[:8],
            "sample_struct": [{"type": "entity", "value": w} for w in set(text.split()[:5])]
        }
        return data
    
    orig_struct = extract_structure(input_text, key, selected_api)
    
    st.subheader("Original Text Structure Visualization")
    st.json(orig_struct)
    
    # --- Auto Optimization ---
    def optimize_text(text, key, api):
        # TODO: Replace with actual optimization logic
        # This is a demo placeholder
        words = text.split()
        richer_text = " | ".join(sorted(set(words), key=lambda x: -len(x)))
        return richer_text
    
    optimized_text = optimize_text(input_text, key, selected_api)
    st.subheader("Auto-Optimized Text")
    st.text_area("Optimized Text", optimized_text, height=160)
    
    opt_struct = extract_structure(optimized_text, key, selected_api)
    st.subheader("Optimized Text Structure Visualization")
    st.json(opt_struct)
    
    # --- Comparison Visualization ---
    st.subheader("Comparison of Structured Information")
    import pandas as pd
    compare_df = pd.DataFrame({
        "Metric": ["Length", "Unique Words", "Sample Struct Entities"],
        "Original": [orig_struct["length"], orig_struct["unique_words"], len(orig_struct["sample_struct"])],
        "Optimized": [opt_struct["length"], opt_struct["unique_words"], len(opt_struct["sample_struct"])]
    })
    st.dataframe(compare_df)

else:
    st.info("Paste text above and click Run to start.")

st.markdown("---")
st.caption("Demo for google/langextract: Visualize, optimize, and compare text extracts.")
