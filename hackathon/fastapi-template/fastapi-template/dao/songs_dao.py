from schemas.songs import Song

class SongsCRUD():
    def get_favorite_song(self, item: Song):
        results = f"Tú cancióm favorita es: Artista: {item.artista}, Titulo: {item.titulo}, Album: {item.album}"
        return results