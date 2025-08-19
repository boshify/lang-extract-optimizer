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

def get_extractions_from_ai(text, api_key, provider):
    """
    Calls the chosen AI provider to extract entities from the input text.
    Returns a list of dicts: [{"type": ..., "text": ..., "attributes": {...}}, ...]
    """
    if provider == "openai":
        try:
            import openai
            openai.api_key = api_key
            prompt = (
                "Extract all named entities from the following text. "
                "Return a JSON list of objects, each with 'type' (entity class), 'text' (entity mention), and 'attributes' (dictionary, optional).\n\n"
                f"Text:\n{text}\n\nEntities:"
            )
            response = openai.chat.completions.create(
                model="gpt-4",  # <--- use gpt-4, not gpt-3.5-turbo
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            entities_json_str = response.choices[0].message.content.strip()
            try:
                if entities_json_str.startswith("```"):
                    entities_json_str = entities_json_str.split("```")[1].strip()
                entities = json.loads(entities_json_str)
            except Exception:
                entities = []
            return entities
        except Exception as e:
            st.warning(f"OpenAI entity extraction error: {e}")
            return []
    elif provider == "gemini":
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            prompt = (
                "Extract all named entities from the following text. "
                "Return a JSON list of objects, each with 'type' (entity class), 'text' (entity mention), and 'attributes' (dictionary, optional).\n\n"
                f"Text:\n{text}\n\nEntities:"
            )
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(prompt)
            entities_json_str = response.text.strip()
            try:
                if entities_json_str.startswith("```"):
                    entities_json_str = entities_json_str.split("```")[1].strip()
                entities = json.loads(entities_json_str)
            except Exception:
                entities = []
            return entities
        except Exception as e:
            st.warning(f"Gemini entity extraction error: {e}")
            return []
    elif provider == "vertex":
        st.warning("Vertex AI extraction not implemented in this demo. Falling back to basic example.")
        return []
    return []

if st.button("Run") and input_text.strip():
    key = get_api_key()
    provider = selected_api.lower()
    import langextract as lx
    from langextract.data import ExampleData, Extraction

    # Get structured entities from AI provider
    entities = get_extractions_from_ai(input_text, key, provider)
    extractions = []
    if entities and isinstance(entities, list):
        for ent in entities:
            extraction_class = ent.get("type") or ent.get("extraction_class") or "entity"
            extraction_text = ent.get("text") or ent.get("extraction_text") or input_text
            attributes = ent.get("attributes", {})
            try:
                extraction = Extraction(
                    extraction_class=extraction_class,
                    extraction_text=extraction_text,
                    attributes=attributes
                )
                extractions.append(extraction)
            except Exception as e:
                st.warning(f"Error creating Extraction object: {e}")
        if not extractions:
            st.warning("AI output could not be converted to Extraction objects. Using fallback.")
    else:
        st.warning("Could not extract entities using the AI provider. Falling back to treating full text as a single entity.")
        extractions = [
            Extraction(
                extraction_class="TEXT",
                extraction_text=input_text,
                attributes={}
            )
        ]
    examples = [
        ExampleData(
            text=input_text,
            extractions=extractions
        )
    ]

    # Extraction API call -- pass api_key, model_id, fence_output, use_schema_constraints for OpenAI
    extract_kwargs = dict(examples=examples)
    if provider == "openai":
        extract_kwargs.update({
            "model_id": "gpt-4",  # <--- use gpt-4
            "api_key": openai_key,
            "fence_output": True,
            "use_schema_constraints": False,
        })
    elif provider == "gemini":
        extract_kwargs.update({"model_id": "gemini-pro", "api_key": gemini_key})
    elif provider == "vertex":
        extract_kwargs.update({"model_id": "vertex-model", "api_key": vertex_key})

    result = lx.extract(input_text, **extract_kwargs)

    # Save annotated extraction to jsonl
    with tempfile.TemporaryDirectory() as tmpdir:
        jsonl_path = os.path.join(tmpdir, "orig_extraction.jsonl")
        lx.io.save_annotated_documents([result], output_name="orig_extraction.jsonl", output_dir=tmpdir)
        html_content = lx.visualize(jsonl_path)
        st.subheader("Original Text Visualization")
        components.html(html_content.data if hasattr(html_content, 'data') else html_content, height=450, scrolling=True)

    # Optimization
    optimized_text = lx.optimize(input_text, examples=examples)
    st.subheader("Optimized Text")
    st.text_area("Optimized Text Output", optimized_text, height=160)

    # Extract & Visualize Optimized
    optimized_result = lx.extract(optimized_text, **extract_kwargs)
    with tempfile.TemporaryDirectory() as tmpdir2:
        jsonl_path2 = os.path.join(tmpdir2, "opt_extraction.jsonl")
        lx.io.save_annotated_documents([optimized_result], output_name="opt_extraction.jsonl", output_dir=tmpdir2)
        html_content2 = lx.visualize(jsonl_path2)
        st.subheader("Optimized Text Visualization")
        components.html(html_content2.data if hasattr(html_content2, 'data') else html_content2, height=450, scrolling=True)

    def get_struct_info(res):
        if hasattr(res, "extractions"):
            entities = res.extractions or []
            return {
                "Entity Count": len(entities),
                "Types": list(set(getattr(e, "extraction_class", "") for e in entities)),
                "Attributes": [getattr(e, "attributes", {}) for e in entities]
            }
        elif isinstance(res, dict) and "entities" in res:
            entities = res.get("entities", [])
            return {
                "Entity Count": len(entities),
                "Types": list(set(e.get("type", "") for e in entities)),
                "Attributes": [e.get("attributes", {}) for e in entities]
            }
        else:
            return {"Entity Count": 0, "Types": [], "Attributes": []}

    st.subheader("Structured Information Comparison")
    st.write("Original:", get_struct_info(result))
    st.write("Optimized:", get_struct_info(optimized_result))

else:
    st.info("Paste text above and click Run to start.")

st.markdown("---")
st.caption("Demo for langextract: Visualize, optimize, and compare text extracts interactively.")
