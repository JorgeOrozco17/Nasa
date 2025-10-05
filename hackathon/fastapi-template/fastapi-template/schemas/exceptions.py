# schemas/exceptions.py
from fastapi import HTTPException, status

class FileFormatException(HTTPException):
    """Error cuando el archivo no es válido."""
    def __init__(self, detail="El archivo debe ser un CSV válido."):
        super().__init__(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)


class MissingColumnsException(HTTPException):
    """Error cuando faltan columnas requeridas."""
    def __init__(self, missing_cols: list[str]):
        detail = f"Faltan columnas requeridas: {', '.join(missing_cols)}"
        super().__init__(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)


class ModelLoadException(HTTPException):
    """Error al cargar el modelo .pkl"""
    def __init__(self, error: str):
        super().__init__(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                         detail=f"No se pudo cargar el modelo: {error}")


class PredictionException(HTTPException):
    """Error durante la predicción"""
    def __init__(self, error: str):
        super().__init__(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                         detail=f"Error durante la predicción: {error}")
