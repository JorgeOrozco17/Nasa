from fastapi import APIRouter
from services.songs import SongsService 
from schemas.songs import Song, SongResponse

router = APIRouter(prefix="/songs", tags=["Canciones"])

@router.post("/", response_model=SongResponse)
def get_favorite_song(item: Song):
    return SongsService().get_favorite_song(item)