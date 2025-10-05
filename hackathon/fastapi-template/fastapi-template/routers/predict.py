# routers/predict.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from starlette.responses import FileResponse
from starlette.background import BackgroundTask
import pandas as pd
import numpy as np
import joblib
import tempfile
import os

router = APIRouter(prefix="/api", tags=["Predicción desde CSV"])

# === Ruta del modelo ===
MODEL_PATH = os.path.join("models", "gradientboost_exoplanets.pkl")

# === Cargar modelo con joblib ===
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"No se pudo cargar el modelo: {e}")

# === Columnas requeridas ===
FEATURE_COLS = [
    "koi_score",
    "koi_fpflag_nt",
    "koi_fpflag_ss",
    "koi_fpflag_co",
    "koi_fpflag_ec",
    "koi_period",
    "koi_impact",
    "koi_duration",
    "koi_depth",
    "koi_prad",
    "koi_teq",
    "koi_insol",
    "koi_model_snr",
    "koi_steff",
    "koi_slogg",
    "koi_srad",
]

@router.post("/predict/from_csv")
async def predict_from_csv(file: UploadFile = File(...)):
    """
    Recibe un CSV con los datos de las estrellas y devuelve un nuevo CSV
    con las predicciones del modelo (una por fila).
    """
    try:
        if not file.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail="El archivo debe ser un CSV.")

        df = pd.read_csv(file.file)

        # Obtener las features esperadas por el modelo (como en Colab)
        try:
            feature_names = model.named_steps['imputer'].feature_names_in_
        except Exception:
            raise HTTPException(status_code=500, detail="El modelo cargado no contiene 'imputer' en el pipeline.")

        # Validar columnas requeridas
        missing = [c for c in feature_names if c not in df.columns]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Faltan columnas requeridas: {', '.join(missing)}"
            )

        # Seleccionar las columnas correctas
        X_pred = df[list(feature_names)].copy()

        # Predicciones usando el pipeline completo
        y_pred_proba = model.predict_proba(X_pred)[:, 1]
        y_pred = model.predict(X_pred)

        # Agregar resultados al DataFrame
        df["prob_confirme_planeta"] = y_pred_proba
        df["prediccion"] = y_pred

        # Guardar resultados
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        df.to_csv(tmp_file.name, index=False)

        return FileResponse(
            tmp_file.name,
            media_type="text/csv",
            filename="predicciones_exoplanetas.csv",
            background=BackgroundTask(lambda p=tmp_file.name: os.remove(p) if os.path.exists(p) else None)
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la predicción: {e}")
    finally:
        file.file.close()

