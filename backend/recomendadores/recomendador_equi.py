import os
import warnings
import numpy as np
import pandas as pd
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

EPOCAS = [
    "Histórico (Pre-1900)",
    "Moderno (1900-1975)",
    "Reciente (Post-1975)",
    "Sin datos",
]

EPOCA_OHE_COLS = ["epoca_historico", "epoca_moderno", "epoca_reciente", "epoca_sin_datos"]

FEATURE_COLS = [
    "municipio_hab", "campo_cult_artes_visuales", "campo_cult_artes_escenicas",
    "campo_cult_musica", "campo_cult_cine_audiovisual", "campo_cult_diseno",
    "campo_cult_arquitectura_urbanismo", "campo_cult_literatura_pensamiento",
    "campo_cult_patrimonio_cultural", "campo_cult_cultura_digital",
    "campo_cult_practicas_hibridas", "campo_cult_otro",
    "actividad_exhibicion_programacion", "actividad_formacion",
    "actividad_residencias", "actividad_produccion_cultural",
    "actividad_mediacion_publicos", "actividad_investigacion_laboratorio",
    "actividad_archivo_documentacion", "actividad_publicaciones_edicion",
    "actividad_cesion_espacios", "actividad_comunitarias_territoriales",
    "publico_general", "publico_especializados", "publico_comunidades_locales",
    "publico_infancia", "publico_mayores", "publico_colectivos_vulnerables",
    "publico_creadores_profesionales", "publico_turismo_cultural",
    "publico_otros", "accesibilidad", "digitalizado", "participacion",
    "epoca_historico", "epoca_moderno", "epoca_reciente", "epoca_sin_datos"
]

INTERESES_VALIDOS = [
    "arte", "escenicas", "musica", "cine", "diseno",
    "arquitectura", "literatura", "patrimonio", "digital",
    "hibrido", "ciencia", "infantil", "comunidad",
]
TIPOLOGIAS_VALIDAS = ["museo", "biblioteca", "teatro", "auditorio", "centro_cultural", "cine", "otro"]
GESTIONES_VALIDAS  = ["pública", "privada", "mixta", "no_consta"]
ESTUDIOS_VALIDOS   = ["primaria", "secundaria", "universitario", "posgrado"]
PERFIL_SOCIAL_VALIDO = ["individual", "familiar", "grupo"]
EPOCAS_PREFERENCIA  = ["historico", "moderno", "reciente", "sin_preferencia"]

class PerfilUsuario(BaseModel):
    edad: int = Field(..., ge=0, le=120, description="Edad del usuario (0-120)")
    nivel_estudios: str = Field(..., description=f"Nivel de estudios: {ESTUDIOS_VALIDOS}")
    comunidad_residencia: str = Field(..., description="Comunidad Autónoma de residencia del usuario")
    intereses: List[str] = Field(..., min_length=1, description=f"Lista de intereses culturales. Valores válidos: {INTERESES_VALIDOS}")
    movilidad_reducida: bool = Field(..., description="¿Tiene movilidad reducida?")
    preferencia_digital: bool = Field(..., description="¿Prefiere equipamientos digitalizados?")
    perfil_social: str = Field(..., description=f"Perfil social: {PERFIL_SOCIAL_VALIDO}")
    preferencia_epoca: str = Field(default="sin_preferencia", description=f"Preferencia de época: {EPOCAS_PREFERENCIA}")
    top_n: int = Field(default=5, ge=1, le=50, description="Número de recomendaciones a devolver (1-50)")
    filtro_comunidad: Optional[List[str]] = Field(default=None, description="Filtrar solo equipamientos en estas CCAA (None = todas)")
    filtro_tipologia: Optional[List[str]] = Field(default=None, description=f"Filtrar por tipología. Valores: {TIPOLOGIAS_VALIDAS}")
    filtro_gestion: Optional[str] = Field(default=None, description=f"Filtrar por tipo de gestión. Valores: {GESTIONES_VALIDAS}")

    @field_validator("nivel_estudios")
    @classmethod
    def validar_estudios(cls, v):
        if v not in ESTUDIOS_VALIDOS: raise ValueError(f"Debe ser uno de {ESTUDIOS_VALIDOS}")
        return v

    @field_validator("perfil_social")
    @classmethod
    def validar_perfil_social(cls, v):
        if v not in PERFIL_SOCIAL_VALIDO: raise ValueError(f"Debe ser uno de {PERFIL_SOCIAL_VALIDO}")
        return v

    @field_validator("preferencia_epoca")
    @classmethod
    def validar_epoca(cls, v):
        if v not in EPOCAS_PREFERENCIA: raise ValueError(f"Debe ser uno de {EPOCAS_PREFERENCIA}")
        return v

    @field_validator("intereses")
    @classmethod
    def validar_intereses(cls, v):
        invalidos = [i for i in v if i not in INTERESES_VALIDOS]
        if invalidos: raise ValueError(f"Intereses no válidos: {invalidos}. Válidos: {INTERESES_VALIDOS}")
        return v

    @field_validator("filtro_tipologia")
    @classmethod
    def validar_filtro_tipologia(cls, v):
        if v is not None:
            invalidos = [t for t in v if t not in TIPOLOGIAS_VALIDAS]
            if invalidos: raise ValueError(f"Tipologías no válidas: {invalidos}")
        return v

    @field_validator("filtro_gestion")
    @classmethod
    def validar_filtro_gestion(cls, v):
        if v is not None and v not in GESTIONES_VALIDAS:
            raise ValueError(f"Debe ser uno de {GESTIONES_VALIDAS}")
        return v

class Equipamiento(BaseModel):
    posicion: int
    id: str | int | None
    nombre: str | None
    municipio: str | None
    comunidad: str | None
    tipologia: str | None
    gestion: str | None
    epoca: str | None
    similitud_pct: float

class RespuestaRecomendacion(BaseModel):
    total_evaluados: int
    recomendaciones: List[Equipamiento]

class RecomendadorCultural:
    def __init__(self, db_path: str = "./data/BDD_equipamientos_v8.xlsx - Sheet1.csv"):
        self.db_path = db_path
        self.df = None
        self.matrix = None
        self._cargar_y_preparar()

    def _cargar_y_preparar(self):
        if not os.path.exists(self.db_path):
            print(f"Advertencia: No se encontró la base de datos en {self.db_path}")
            return
            
        # Dependiendo del formato, cargamos con read_csv o read_excel
        if self.db_path.endswith('.csv'):
            df = pd.read_csv(self.db_path)
        else:
            df = pd.read_excel(self.db_path, index_col=0)
        
        # Asegurarnos de que todas las columnas numéricas existen en el df antes de crear la matriz. 
        for col in FEATURE_COLS:
            if col not in df.columns:
                df[col] = 0.0
                
        self.df = df
        self.matrix = df[FEATURE_COLS].values.astype(float)

    def _construir_vector_usuario(self, perfil: PerfilUsuario) -> np.ndarray:
        intereses = perfil.intereses
        municipio_hab = 0.5
        campo_artes_visuales  = 1.0 if "arte" in intereses else 0.0
        campo_artes_escenicas = 1.0 if "escenicas" in intereses else 0.0
        campo_musica          = 1.0 if "musica" in intereses else 0.0
        campo_cine            = 1.0 if "cine" in intereses else 0.0
        campo_diseno          = 1.0 if "diseno" in intereses else 0.0
        campo_arquitectura    = 1.0 if "arquitectura" in intereses else 0.0
        campo_literatura      = 1.0 if "literatura" in intereses else 0.0
        campo_patrimonio      = 1.0 if "patrimonio" in intereses else 0.0
        campo_digital         = 1.0 if "digital" in intereses else 0.0
        campo_hibrido         = 1.0 if "hibrido" in intereses else 0.0
        campo_otro            = 0.0

        act_exhibicion    = 1.0 if any(i in intereses for i in ["arte", "patrimonio", "ciencia"]) else 0.0
        act_formacion     = 1.0 if any(i in intereses for i in ["arte", "musica", "ciencia", "infantil"]) else 0.0
        act_residencias   = 1.0 if perfil.nivel_estudios in ("universitario", "posgrado") else 0.0
        act_produccion    = 1.0 if any(i in intereses for i in ["arte", "escenicas", "musica", "digital"]) else 0.0
        act_mediacion     = 1.0 if any(i in intereses for i in ["patrimonio", "comunidad", "infantil"]) else 0.0
        act_investigacion = 1.0 if perfil.nivel_estudios in ("universitario", "posgrado") else 0.0
        act_archivo       = 1.0 if any(i in intereses for i in ["patrimonio", "literatura"]) else 0.0
        act_publicaciones = 1.0 if "literatura" in intereses else 0.0
        act_cesion        = 0.0
        act_comunitarias  = 1.0 if "comunidad" in intereses or perfil.perfil_social == "grupo" else 0.0

        pub_general       = 1.0
        pub_especializados= 1.0 if perfil.nivel_estudios in ("universitario", "posgrado") else 0.0
        pub_comunidades   = 1.0 if "comunidad" in intereses else 0.5
        pub_infancia      = 1.0 if perfil.edad < 12 or "infantil" in intereses else 0.0
        pub_mayores       = 1.0 if perfil.edad >= 60 else 0.0
        pub_vulnerables   = 1.0 if perfil.movilidad_reducida else 0.0
        pub_creadores     = 1.0 if perfil.nivel_estudios == "posgrado" else 0.0
        pub_turismo       = 0.5
        pub_otros         = 0.0

        accesibilidad = 1.0 if perfil.movilidad_reducida else 0.5
        digitalizado  = 1.0 if perfil.preferencia_digital else 0.3
        participacion = 0.8 if perfil.perfil_social in ("familiar", "grupo") else 0.3

        mapa_ohe = {
            "historico":       [1, 0, 0, 0],
            "moderno":         [0, 1, 0, 0],
            "reciente":        [0, 0, 1, 0],
            "sin_preferencia": [0, 0, 0, 1],
        }
        epoca_ohe = mapa_ohe.get(perfil.preferencia_epoca, [0, 0, 0, 1])

        return np.array([
            municipio_hab,
            campo_artes_visuales, campo_artes_escenicas, campo_musica,
            campo_cine, campo_diseno, campo_arquitectura, campo_literatura,
            campo_patrimonio, campo_digital, campo_hibrido, campo_otro,
            act_exhibicion, act_formacion, act_residencias, act_produccion,
            act_mediacion, act_investigacion, act_archivo, act_publicaciones,
            act_cesion, act_comunitarias,
            pub_general, pub_especializados, pub_comunidades,
            pub_infancia, pub_mayores, pub_vulnerables, pub_creadores,
            pub_turismo, pub_otros,
            accesibilidad, digitalizado, participacion,
            *epoca_ohe,
        ], dtype=float)

    def recomendar(self, perfil: PerfilUsuario) -> RespuestaRecomendacion:
        if self.df is None or self.matrix is None:
            raise ValueError("La base de datos no está cargada correctamente.")

        df = self.df
        matrix = self.matrix

        mask = pd.Series([True] * len(df), index=df.index)
        if perfil.filtro_comunidad:
            mask &= df["comunidad"].isin(perfil.filtro_comunidad)
        if perfil.filtro_tipologia:
            mask &= df["tipologia"].isin(perfil.filtro_tipologia)
        if perfil.filtro_gestion:
            # Revisa si gestion_norm existe, sino usa gestion
            col_gestion = "gestion_norm" if "gestion_norm" in df.columns else "gestion"
            mask &= df[col_gestion] == perfil.filtro_gestion

        df_f = df[mask].copy()
        mat_f = matrix[mask.values]

        if len(df_f) == 0:
            return RespuestaRecomendacion(total_evaluados=0, recomendaciones=[])

        vector = self._construir_vector_usuario(perfil)
        sim = cosine_similarity(vector.reshape(1, -1), mat_f).flatten()

        df_f["similitud"] = sim
        top = df_f.sort_values("similitud", ascending=False).head(perfil.top_n).reset_index(drop=True)

        recomendaciones = []
        for pos, (_, row) in enumerate(top.iterrows(), start=1):
            recomendaciones.append(
                Equipamiento(
                    posicion=pos,
                    id=row.get("id"),
                    nombre=row.get("nombre"),
                    municipio=row.get("nombre_municipio_api") if "nombre_municipio_api" in row else row.get("municipio"),
                    comunidad=row.get("comunidad") if "comunidad" in row else row.get("com_aut"),
                    tipologia=row.get("tipologia"),
                    gestion=row.get("gestion_norm") if "gestion_norm" in row else row.get("gestion"),
                    epoca=row.get("epoca_construccion"),
                    similitud_pct=round(float(row["similitud"]) * 100, 1),
                )
            )

        return RespuestaRecomendacion(
            total_evaluados=len(df_f),
            recomendaciones=recomendaciones,
        )
