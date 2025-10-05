# from dao.songs_dao import SongsCRUD
from dao.songs_dao import SongsCRUD
from schemas.songs import Song

class SongsService():
    def get_favorite_song(self, item: Song) -> dict:
        fav_song = SongsCRUD().get_favorite_song(item)
        return {"message": fav_song}

