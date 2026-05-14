import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Ubicación Óptima", page_icon="🗺️", layout="wide")

st.title("🗺️ Recomendador de Ubicaciones")
st.markdown("Encuentra la provincia o comunidad óptima para construir o establecer un nuevo equipamiento cultural.")

API_URL = "http://localhost:8000/recomendar/ubicacion"

with st.form("form_ubicacion"):
    c1, c2 = st.columns(2)
    with c1:
        escala = st.selectbox("Escala del proyecto", ["nacional", "regional", "local", "barrio"])
        publicos = st.multiselect(
            "Público objetivo", 
            ["joven", "infantil", "adulto", "mayor", "familia", "turista"], 
            default=["adulto"]
        )
    with c2:
        renta_alta = st.checkbox("Requiere zona de renta alta")
        turismo = st.checkbox("Fuertemente orientado al turismo")
        deficit = st.checkbox("Priorizar zonas con déficit de oferta cultural")
        top_n = st.slider("Resultados a mostrar", 1, 10, 3)

    submit = st.form_submit_button("Buscar Territorios Óptimos", type="primary")

if submit:
    payload = {
        "escala": escala,
        "publico_objetivo": publicos,
        "requiere_renta_alta": renta_alta,
        "orientado_turismo": turismo,
        "prioridad_deficit": deficit,
        "top_n": top_n
    }
    
    with st.spinner("Analizando territorios..."):
        try:
            response = requests.post(API_URL, json=payload)
            if response.status_code == 200:
                recs = response.json().get("recomendaciones", [])
                if recs:
                    st.success("¡Territorios encontrados!")
                    df = pd.DataFrame(recs)
                    df.columns = ["Provincia", "CCAA", "Afinidad (%)", "Puntos Fuertes (Justificación)"]
                    st.dataframe(df, use_container_width=True, hide_index=True)
                else:
                    st.warning("No se encontraron resultados.")
            else:
                st.error(f"Error {response.status_code}: {response.text}")
        except Exception as e:
            st.error(f"No se pudo conectar con la API: {e}")