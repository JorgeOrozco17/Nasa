from pydantic import BaseModel
from typing import List

class Song(BaseModel):
    artista: str  
    titulo: str  
    album: str 
    
class SongResponse(BaseModel):
    message: str
    