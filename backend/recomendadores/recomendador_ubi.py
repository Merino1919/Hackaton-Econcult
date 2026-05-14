import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import os
from typing import List, Optional
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore")

FEATURE_COLS = [
    "pob_norm", "pct_jovenes", "pct_mayores", "pct_extranjeros",
    "renta_norm", "paro_norm", "dot_cult_inv_prov", 
    "gasto_hogar_cultura_norm", "financiacion_publica_norm", "turismo_norm"
]

ETIQUETAS = {
    "pob_norm": "Masa poblacional", "pct_jovenes": "Público joven", "pct_mayores": "Público senior",
    "pct_extranjeros": "Población internacional", "renta_norm": "Nivel de renta",
    "paro_norm": "Necesidad de empleo", "dot_cult_inv_prov": "Déficit de oferta cultural",
    "gasto_hogar_cultura_norm": "Gasto doméstico en cultura", "financiacion_publica_norm": "Inversión pública",
    "turismo_norm": "Atractivo turístico"
}

# --- MODELOS PYDANTIC EXCLUSIVOS ---
class PerfilEquipamiento(BaseModel):
    escala: str = Field(..., description="nacional, regional, local, barrio")
    publico_objetivo: List[str] = Field(..., description="joven, infantil, adulto, mayor, familia, turista")
    requiere_renta_alta: bool = Field(False)
    orientado_turismo: bool = Field(False)
    prioridad_deficit: bool = Field(False)
    top_n: int = Field(5, ge=1, le=50)
    filtro_ca: Optional[str] = Field(None, description="Comunidad Autónoma a filtrar")

class UbicacionRecomendada(BaseModel):
    provincia: str
    comunidad: str
    similitud_pct: float
    justificacion: str

class RespuestaUbicacion(BaseModel):
    total_evaluados: int
    recomendaciones: List[UbicacionRecomendada]

# --- CLASE PRINCIPAL ---
class RecomendadorUbicacion:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.df = None
        self.matrix = None
        self._cargar_datos()

    def _cargar_datos(self):
        if not os.path.exists(self.db_path):
            print(f"Advertencia: No se encontró {self.db_path}")
            return
            
        if self.db_path.endswith('.csv'):
            self.df = pd.read_csv(self.db_path)
        else:
            self.df = pd.read_excel(self.db_path)
            
        for col in FEATURE_COLS:
            if col not in self.df.columns:
                self.df[col] = 0.0
                
        self.matrix = self.df[FEATURE_COLS].values.astype(float)

    def _construir_vector(self, perfil: PerfilEquipamiento) -> np.ndarray:
        escala_pob = {"nacional": 0.9, "regional": 0.6, "local": 0.4, "barrio": 0.2}
        pob_norm = escala_pob.get(perfil.escala.lower(), 0.5)
        
        pct_jovenes = 1.0 if any(p in perfil.publico_objetivo for p in ["joven", "infantil"]) else 0.4
        pct_mayores = 1.0 if "mayor" in perfil.publico_objetivo else 0.3
        pct_extranjeros = 0.8 if "turista" in perfil.publico_objetivo or perfil.orientado_turismo else 0.4

        renta_norm = 0.8 if perfil.requiere_renta_alta else 0.3
        paro_norm = 0.4 if perfil.requiere_renta_alta else 0.7  

        dot_cult_inv_prov = 0.9 if perfil.prioridad_deficit else 0.3
        gasto_hogar_cultura_norm = 0.8 if perfil.escala in ["nacional", "regional"] else 0.4
        financiacion_publica_norm = 0.5 
        turismo_norm = 0.9 if perfil.orientado_turismo else 0.2

        return np.array([
            pob_norm, pct_jovenes, pct_mayores, pct_extranjeros,
            renta_norm, paro_norm, dot_cult_inv_prov,
            gasto_hogar_cultura_norm, financiacion_publica_norm, turismo_norm
        ], dtype=float)

    def recomendar(self, perfil: PerfilEquipamiento) -> RespuestaUbicacion:
        if self.df is None or self.matrix is None:
            raise ValueError("La base de datos no está cargada.")
            
        df_f = self.df.copy()
        mat_f = self.matrix.copy()

        if perfil.filtro_ca:
            mask = df_f["comunidad"] == perfil.filtro_ca
            df_f = df_f[mask]
            mat_f = mat_f[mask]

        if len(df_f) == 0:
            return RespuestaUbicacion(total_evaluados=0, recomendaciones=[])

        vector = self._construir_vector(perfil)
        sims = cosine_similarity(vector.reshape(1, -1), mat_f).flatten()
        df_f["similitud_pct"] = (sims * 100).round(1)
        
        resultados = df_f.sort_values("similitud_pct", ascending=False).head(perfil.top_n)
        
        recomendaciones = []
        for _, row in resultados.iterrows():
            ter_vec = row[FEATURE_COLS].values.astype(float)
            contribucion = vector * ter_vec
            top_indices = np.argsort(contribucion)[::-1][:3]
            
            razones = [f"{ETIQUETAS[FEATURE_COLS[i]]} ({ter_vec[i]:.2f})" for i in top_indices]
            
            recomendaciones.append(
                UbicacionRecomendada(
                    provincia=row.get("nombre_provincia", "Desconocida"),
                    comunidad=row.get("comunidad", "Desconocida"),
                    similitud_pct=row["similitud_pct"],
                    justificacion=" | ".join(razones)
                )
            )
            
        return RespuestaUbicacion(total_evaluados=len(df_f), recomendaciones=recomendaciones)