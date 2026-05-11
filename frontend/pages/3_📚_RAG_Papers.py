import streamlit as st
import requests

st.set_page_config(page_title="RAG Multimodal", page_icon="📚")
st.title("📚 RAG Multimodal para Papers")

API_UPLOAD_URL = "http://localhost:8000/rag/upload"
API_ASK_URL = "http://localhost:8000/rag/ask"

# Sección de subida de archivos
st.subheader("1. Ingestar nuevo Documento")
archivo = st.file_uploader("Sube un paper en formato PDF", type=["pdf"])

if archivo is not None:
    if st.button("Indexar Documento"):
        with st.spinner("Procesando, extrayendo imágenes y resumiendo... (Puede tardar)"):
            files = {"file": (archivo.name, archivo.getvalue(), "application/pdf")}
            response = requests.post(API_UPLOAD_URL, files=files)
            if response.status_code == 200:
                st.success("¡Documento indexado en ChromaDB con éxito!")
            else:
                st.error(f"Error: {response.text}")

st.divider()

# Sección de Chat / Q&A
st.subheader("2. Consultar al RAG")
pregunta = st.text_input("Haz una pregunta sobre los documentos indexados:")

if st.button("Preguntar"):
    if pregunta:
        with st.spinner("Buscando en la base vectorial y generando respuesta..."):
            response = requests.post(API_ASK_URL, json={"pregunta": pregunta})
            if response.status_code == 200:
                data = response.json()
                st.write("**Respuesta de la IA:**")
                st.info(data.get("answer"))
                
                with st.expander("Ver fragmento recuperado (Transparencia)"):
                    st.write(f"**Score de similitud:** {data.get('score')}")
                    st.text(data.get("best_chunk"))
            else:
                st.error("Error en la consulta.")