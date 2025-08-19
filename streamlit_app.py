import streamlit as st
import streamlit.components.v1 as components
import tempfile
import os
import json

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

st.title("LangExtract Optimizer & Visualizer Demo")
input_text = st.text_area("Paste your input text here:", height=240)

# ----------------------
# Optimization Function
# ----------------------
def optimize_text(text, api_key, provider):
    """Rewrite text for SEO and LLM visibility improvement."""
    prompt = (
        "Rewrite the following text to improve clarity, structure, and SEO visibility. "
        "Make it more readable and useful for large language models while preserving meaning. "
        "Return only the improved text.\n\n"
        f"Text:\n{text}\n\nImproved Text:"
    )
    try:
        if provider == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()

        elif provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(prompt)
            return response.text.strip()

        elif provider == "vertex":
            return text + "\n\n[Vertex optimization not implemented]"
        else:
            return text
    except Exception as e:
        st.warning(f"Optimization error: {e}")
        return text

if st.button("Run") and input_text.strip():
    key = get_api_key()
    provider = selected_api.lower()

    import langextract as lx
    from langextract.data import ExampleData, Extraction

    # ----------------------
    # Run LangExtract on Original
    # ----------------------
    extract_kwargs = {}
    if provider == "openai":
        extract_kwargs.update({
            "model_id": "gpt-4o-mini",
            "api_key": openai_key,
            "fence_output": True,
            "use_schema_constraints": False,
        })
    elif provider == "gemini":
        extract_kwargs.update({"model_id": "gemini-pro", "api_key": gemini_key})
    elif provider == "vertex":
        extract_kwargs.update({"model_id": "vertex-model", "api_key": vertex_key})

    result = lx.extract(input_text, **extract_kwargs)

    with tempfile.TemporaryDirectory() as tmpdir:
        jsonl_path = os.path.join(tmpdir, "orig_extraction.jsonl")
        lx.io.save_annotated_documents([result], output_name="orig_extraction.jsonl", output_dir=tmpdir)
        html_content = lx.visualize(jsonl_path)
        st.subheader("Original Text Visualization")
        components.html(html_content.data if hasattr(html_content, 'data') else html_content, height=450, scrolling=True)

    # ----------------------
    # Optimize Text (custom prompt)
    # ----------------------
    optimized_text = optimize_text(input_text, key, provider)
    st.subheader("Optimized Text")
    st.text_area("Optimized Text Output", optimized_text, height=160)

    # ----------------------
    # Run LangExtract on Optimized Text
    # ----------------------
    optimized_result = lx.extract(optimized_text, **extract_kwargs)

    with tempfile.TemporaryDirectory() as tmpdir2:
        jsonl_path2 = os.path.join(tmpdir2, "opt_extraction.jsonl")
        lx.io.save_annotated_documents([optimized_result], output_name="opt_extraction.jsonl", output_dir=tmpdir2)
        html_content2 = lx.visualize(jsonl_path2)
        st.subheader("Optimized Text Visualization")
        components.html(html_content2.data if hasattr(html_content2, 'data') else html_content2, height=450, scrolling=True)

else:
    st.info("Paste text above and click Run to start.")

st.markdown("---")
st.caption("Demo for langextract: Extract, visualize, and optimize text interactively for SEO & LLM visibility.")
