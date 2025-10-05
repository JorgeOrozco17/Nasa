# routers/graph.py
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tempfile
from typing import Optional, List
import re
import os

router = APIRouter()


def _parse_separators(s: Optional[str]) -> List[float]:
    """
    Parser tolerante: extrae todos los números de una cadena cualquiera.
    Soporta: "370,730", "370 730", "370|730", etc.
    Si no hay números, retorna [].
    """
    if not s:
        return []
    # normaliza coma por espacio
    s = s.replace(",", " ")
    nums = re.findall(r"[+-]?\d+(?:\.\d+)?", s)
    try:
        return [float(x) for x in nums]
    except Exception:
        return []


def detect_separators(df: pd.DataFrame, gap_threshold: float = 20.0) -> List[float]:
    """
    Detecta separadores automáticamente a partir de 'gaps' en el eje de tiempo.
    - gap_threshold: diferencia mínima (en días) para considerar un salto real.
    Retorna los valores X donde empieza el nuevo bloque (líneas verticales).
    """
    t = df["time"].sort_values().to_numpy()
    if t.size < 2:
        return []
    diffs = np.diff(t)
    gaps = t[1:][diffs > gap_threshold]
    return gaps.tolist()


@router.post("/graph/from_csv")
async def csv_to_graph(
    file: UploadFile = File(...),
    title: str = Form("Kepler Full-Mission Light Curve"),
    x_label: str = Form("Time [BKJD]"),
    y_label: str = Form("Relative Flux"),
    separators: Optional[str] = Form(None),           # si viene vacío → usar auto-detección
    sep_color: str = Form("black"),
    auto_gap_threshold: float = Form(20.0),           # umbral de gap (días) para auto-detección
):
    """
    Sube un CSV con columnas 'time' y 'flux' y devuelve una PNG panorámica en blanco y negro.
    - Si 'separators' está vacío → se detectan automáticamente por gaps en 'time'.
    - 'auto_gap_threshold' controla la sensibilidad de esa detección (default: 20 días).
    """
    try:
        if not file.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail="El archivo debe ser un CSV.")

        # Cargar CSV
        df = pd.read_csv(file.file)
        if not {"time", "flux"}.issubset(df.columns):
            raise HTTPException(
                status_code=400,
                detail="El CSV debe contener las columnas 'time' y 'flux'.",
            )

        # Ordenar por seguridad
        df = df.sort_values("time").reset_index(drop=True)

        # Separadores: manuales (si se enviaron) o automáticos (por gaps)
        xs = _parse_separators(separators)
        if not xs:
            xs = detect_separators(df, gap_threshold=float(auto_gap_threshold))

        # Archivo temporal de salida
        tmp_png = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tmp_path = tmp_png.name
        tmp_png.close()  # cerramos el handle; matplotlib solo necesita la ruta

        # === Estilo blanco y negro en formato panorámico ===
        plt.style.use("grayscale")
        fig = plt.figure(figsize=(16, 2.4), dpi=150)
        fig.patch.set_facecolor("#d9d9d9")  # fondo exterior gris claro
        ax = plt.gca()
        ax.set_facecolor("white")           # área del gráfico en blanco

        # Dispersión de puntos (look Kepler)
        ax.scatter(df["time"], df["flux"], s=4, c="black", alpha=0.9, linewidths=0)

        # Etiquetas
        ax.set_title(title, color="black", pad=10)
        ax.set_xlabel(x_label, color="black")
        ax.set_ylabel(y_label, color="black")

        # Cuadrícula tenue
        ax.grid(True, color="gray", linestyle="--", linewidth=0.4, alpha=0.5)

        # Dibujar separadores detectados o manuales y etiquetar Q1..Qn
        for i, x in enumerate(xs, start=1):
            ax.axvline(x=x, color=sep_color, linestyle="--", linewidth=1)
            ax.text(
                x, ax.get_ylim()[1] * 0.94,
                f"Q{i}",
                color=sep_color,
                fontsize=8,
                ha="center",
                va="top",
                rotation=0,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.6, pad=1.5),
            )

        plt.tight_layout()
        plt.savefig(tmp_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)

        # Borrar el archivo temporal después de enviar la respuesta
        return FileResponse(
            tmp_path,
            media_type="image/png",
            filename="lightcurve_panorama.png",
            background=BackgroundTask(lambda p=tmp_path: os.remove(p) if os.path.exists(p) else None),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar el CSV: {str(e)}")
    finally:
        file.file.close()
