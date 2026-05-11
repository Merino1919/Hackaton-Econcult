
import streamlit as st
import requests
import pandas as pd
import os

# Configuración de la página
st.set_page_config(page_title="Impacto Cultural AI", page_icon="🏛️", layout="wide")

st.title("🏛️ Catalogación de Impactos Culturales")
st.markdown("""
Esta herramienta analiza las últimas noticias de un equipamiento cultural y las clasifica 
automáticamente en 7 dimensiones de impacto utilizando Inteligencia Artificial.
""")

# Input del usuario
centro_cultural = st.text_input("Introduce el nombre del centro cultural:", placeholder="Ej. IVAM Valencia, Museo del Prado, Matadero Madrid...")

# URL de la API (por defecto localhost)
API_URL = "http://localhost:8000/noticias/analizar"

if st.button("Analizar Noticias", type="primary"):
    if not centro_cultural.strip():
        st.warning("Por favor, introduce el nombre de un centro cultural.")
    else:
        with st.spinner(f"Buscando y analizando noticias para '{centro_cultural}'... (Esto puede tardar unos segundos)"):
            try:
                # Hacer la petición a nuestro endpoint FastAPI
                respuesta = requests.post(API_URL, json={"centro_cultural": centro_cultural})
                
                if respuesta.status_code == 200:
                    datos = respuesta.json()
                    resultados = datos.get("resultados", [])
                    
                    if not resultados:
                        st.info(f"No se han encontrado noticias recientes para '{centro_cultural}'.")
                    else:
                        st.success("¡Análisis completado!")
                        
                        # Convertir a DataFrame de Pandas para visualización
                        df = pd.DataFrame(resultados)
                        
                        # Formatear la lista de impactos a un string separado por comas para que se vea mejor en la tabla
                        df['impactos'] = df['impactos'].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)
                        
                        # Renombrar columnas para la interfaz
                        df.columns = ['Fecha', 'Titular', 'Impactos', 'Razonamiento']
                        
                        # Mostrar el DataFrame interactivo
                        st.dataframe(df, use_container_width=True)
                else:
                    st.error(f"Error en la API: {respuesta.status_code} - {respuesta.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("No se pudo conectar con la API. Asegúrate de que el servidor FastAPI está ejecutándose en el puerto 8000.")
            except Exception as e:
                st.error(f"Ocurrió un error inesperado: {str(e)}")

# Sección de las dimensiones (para referencia visual del usuario)
with st.expander("ℹ️ Ver leyenda de las dimensiones de impacto"):
    st.markdown("""
    * **D1**: Transformación del espacio y del territorio
    * **D2**: Generación de conocimiento y conexión entre expertos
    * **D3**: Notoriedad, marca territorial y soft power
    * **D4**: Influencia en la agenda política y opinión pública
    * **D5**: Articulación de redes y comunidades sociales
    * **D6**: Robustecimiento de sectores culturales
    * **D7**: Satisfacción de los derechos culturales
    """)
