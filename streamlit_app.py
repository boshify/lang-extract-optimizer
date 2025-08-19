import streamlit as st
import streamlit.components.v1 as components
import tempfile
import os

# --- Sidebar for secrets ---
st.sidebar.title("API Keys")
openai_key = st.sidebar.text_input("OpenAI API Key", type="password")
gemini_key = st.sidebar.text_input("Gemini API Key", type="password")
vertex_key = st.sidebar.text_input("Vertex API Key", type="password")
selected_api = st.sidebar.radio("Which API to use?", ["OpenAI", "Gemini", "Vertex"])
api_keys = {"OpenAI": openai_key, "Gemini": gemini_key, "Vertex": vertex_key}

def get_api_key():
    key = api_keys[selected_api]
    if not key:
        st.error(f"Please provide a {selected_api} API key.")
        st.stop()
    return key

# --- Main UI ---
st.title("LangExtract Optimizer & Visualizer Demo")
input_text = st.text_area("Paste your input text here:", height=240)

if st.button("Run") and input_text.strip():
    key = get_api_key()
    
    # --- Extraction & Visualization ---
    import langextract as lx  # Use correct import

    # Step 1: Extract entities from input
    result = lx.extract(input_text, api_key=key)
    
    # Step 2: Save annotated extraction to jsonl
    with tempfile.TemporaryDirectory() as tmpdir:
        jsonl_path = os.path.join(tmpdir, "orig_extraction.jsonl")
        lx.io.save_annotated_documents([result], output_name="orig_extraction.jsonl", output_dir=tmpdir)
        
        # Step 3: Generate HTML visualization
        html_content = lx.visualize(jsonl_path)
        st.subheader("Original Text Visualization")
        components.html(html_content.data if hasattr(html_content, 'data') else html_content, height=450, scrolling=True)
    
    # --- Optimization ---
    optimized_text = lx.optimize(input_text, api_key=key)
    st.subheader("Optimized Text")
    st.text_area("Optimized Text Output", optimized_text, height=160)
    
    # --- Extract & Visualize Optimized ---
    optimized_result = lx.extract(optimized_text, api_key=key)
    with tempfile.TemporaryDirectory() as tmpdir2:
        jsonl_path2 = os.path.join(tmpdir2, "opt_extraction.jsonl")
        lx.io.save_annotated_documents([optimized_result], output_name="opt_extraction.jsonl", output_dir=tmpdir2)
        html_content2 = lx.visualize(jsonl_path2)
        st.subheader("Optimized Text Visualization")
        components.html(html_content2.data if hasattr(html_content2, 'data') else html_content2, height=450, scrolling=True)
    
    # --- Comparison (structured info) ---
    def get_struct_info(result):
        entities = result.get("entities", [])
        return {
            "Entity Count": len(entities),
            "Types": list(set(e.get("type", "") for e in entities)),
            "Attributes": [e.get("attributes", {}) for e in entities]
        }
    st.subheader("Structured Information Comparison")
    st.write("Original:", get_struct_info(result))
    st.write("Optimized:", get_struct_info(optimized_result))

else:
    st.info("Paste text above and click Run to start.")

st.markdown("---")
st.caption("Demo for langextract: Visualize, optimize, and compare text extracts interactively.")
