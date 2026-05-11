import streamlit as st
import requests
import pandas as pd

# Configuración de la página
st.set_page_config(page_title="Recomendador Cultural", page_icon="🎯", layout="wide")

# Valores válidos extraídos del modelo y la base de datos 
ESTUDIOS_VALIDOS = ["primaria", "secundaria", "universitario", "posgrado"]
INTERESES_VALIDOS = [
    "arte", "escenicas", "musica", "cine", "diseno",
    "arquitectura", "literatura", "patrimonio", "digital",
    "hibrido", "ciencia", "infantil", "comunidad"
]
PERFIL_SOCIAL_VALIDO = ["individual", "familiar", "grupo"]
EPOCAS_PREFERENCIA = ["historico", "moderno", "reciente", "sin_preferencia"]
TIPOLOGIAS_VALIDAS = ["museo", "biblioteca", "teatro", "auditorio", "centro_cultural", "cine", "otro"]
GESTIONES_VALIDAS = ["pública", "privada", "mixta", "no_consta"]
CCAA_VALIDAS = [
    "Andalucía", "Aragón", "Asturias", "Baleares", "Canarias", "Cantabria", 
    "Castilla y León", "Castilla-La Mancha", "Cataluña", "Comunitat Valenciana", 
    "Extremadura", "Galicia", "Madrid", "Murcia", "Navarra", "País Vasco", "La Rioja"
]

st.title("🎯 Recomendador Personalizado de Equipamientos")
st.markdown("""
Introduce tus preferencias para que nuestro algoritmo de **similitud de coseno** encuentre los 
espacios culturales que mejor se adaptan a tu perfil.
""")

API_URL = "http://localhost:8000/recomendar/equipamiento"

# --- FORMULARIO DE ENTRADA ---
with st.form("form_recomendador"):
    st.subheader("👤 Perfil de Usuario")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        edad = st.number_input("Edad", min_value=0, max_value=120, value=25)
        perfil_social = st.selectbox("Perfil social", PERFIL_SOCIAL_VALIDO)
        
    with col2:
        estudios = st.selectbox("Nivel de estudios", ESTUDIOS_VALIDOS)
        comunidad_res = st.selectbox("Comunidad de residencia", CCAA_VALIDAS)
        
    with col3:
        preferencia_epoca = st.selectbox("Preferencia de época", EPOCAS_PREFERENCIA)
        top_n = st.slider("Número de recomendaciones", 1, 20, 5)

    st.divider()
    
    st.subheader("🎨 Intereses y Accesibilidad")
    c1, c2 = st.columns(2)
    
    with c1:
        intereses = st.multiselect("Tus intereses (selecciona al menos uno)", INTERESES_VALIDOS, default=["arte"])
        
    with c2:
        movilidad = st.checkbox("Movilidad reducida")
        digital = st.checkbox("Preferencia por espacios digitalizados")

    st.divider()
    
    with st.expander("🔍 Filtros Avanzados (Opcional)"):
        f1, f2, f3 = st.columns(3)
        with f1:
            filtro_ccaa = st.multiselect("Filtrar por CCAA", CCAA_VALIDAS)
        with f2:
            filtro_tipo = st.multiselect("Filtrar por Tipología", TIPOLOGIAS_VALIDAS)
        with f3:
            filtro_gest = st.selectbox("Filtrar por Gestión", [None] + GESTIONES_VALIDAS)

    submit = st.form_submit_button("Obtener Recomendaciones", type="primary")

# --- PROCESAMIENTO DE RESPUESTA ---
if submit:
    if not intereses:
        st.error("Debes seleccionar al menos un interés cultural.")
    else:
        # Construcción del payload según el modelo Pydantic 
        payload = {
            "edad": edad,
            "nivel_estudios": estudios,
            "comunidad_residencia": comunidad_res,
            "intereses": intereses,
            "movilidad_reducida": movilidad,
            "preferencia_digital": digital,
            "perfil_social": perfil_social,
            "preferencia_epoca": preferencia_epoca,
            "top_n": top_n,
            "filtro_comunidad": filtro_ccaa if filtro_ccaa else None,
            "filtro_tipologia": filtro_tipo if filtro_tipo else None,
            "filtro_gestion": filtro_gest
        }

        with st.spinner("Calculando afinidad..."):
            try:
                response = requests.post(API_URL, json=payload)
                
                if response.status_code == 200:
                    res = response.json()
                    recs = res.get("recomendaciones", [])
                    total = res.get("total_evaluados", 0)
                    
                    st.success(f"Se han analizado {total} equipamientos.")
                    
                    if recs:
                        # Convertir a DataFrame para visualización limpia
                        df = pd.DataFrame(recs)
                        # Reordenar y renombrar columnas para el usuario
                        df_vista = df[["posicion", "nombre", "similitud_pct", "tipologia", "municipio", "comunidad", "epoca"]]
                        df_vista.columns = ["Rango", "Nombre", "Afinidad (%)", "Tipo", "Municipio", "CCAA", "Época"]
                        
                        st.dataframe(df_vista, use_container_width=True, hide_index=True)
                        
                        # Mostrar detalle visual del ganador
                        st.balloons()
                        st.info(f"🏆 Tu mejor opción es: **{recs[0]['nombre']}** con un **{recs[0]['similitud_pct']}%** de coincidencia.")
                    else:
                        st.warning("No se encontraron resultados con los filtros aplicados.")
                else:
                    st.error(f"Error {response.status_code}: {response.text}")
                    
            except Exception as e:
                st.error(f"No se pudo conectar con el backend: {e}")