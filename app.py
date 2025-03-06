import os
import numpy as np
import librosa
import librosa.display
import streamlit as st
import time
from io import BytesIO
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Cargar variables de entorno
load_dotenv()

# Configurar Spotipy con las credenciales de Spotify Developer
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=os.getenv("SPOTIPY_CLIENT_ID"),
    client_secret=os.getenv("SPOTIPY_CLIENT_SECRET")
))

# Diccionario de artistas por género con características clave
artistas_por_genero = {
    "Trap": {"bpm": (140, 160), "artistas": ["Travis Scott", "Drake"]},
    "Reggaeton": {"bpm": (90, 110), "artistas": ["Bad Bunny", "J Balvin"]},
    "Hip-Hop": {"bpm": (110, 140), "artistas": ["Kendrick Lamar", "J. Cole"]},
    "Drill": {"bpm": (160, 180), "artistas": ["Pop Smoke", "Fivio Foreign"]},
    "Afrobeat": {"bpm": (70, 90), "artistas": ["Burna Boy", "Wizkid"]}
}

# Función para analizar el audio
def analizar_audio(archivo):
    try:
        # Leer el archivo directamente
        y, sr = librosa.load(BytesIO(archivo.read()), sr=22050, offset=30, duration=30)
        
        # Obtener BPM
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = round(float(tempo), 2) if tempo else 0
        
        # Obtener Key y escala
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        key_index = np.argmax(np.mean(chroma, axis=1))
        key_list = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        key = key_list[key_index]
        
        # Determinar género basado en BPM
        genero_detectado = "Unknown"
        for genero, data in artistas_por_genero.items():
            if data["bpm"][0] <= tempo < data["bpm"][1]:
                genero_detectado = genero
                break
        
        # Buscar artistas similares en Spotify
        artistas_sugeridos = buscar_artistas_similares(genero_detectado, tempo, key)
        
        return tempo, key, genero_detectado, artistas_sugeridos
    except Exception as e:
        return 0, "Unknown", "Unknown", []

# Buscar en Spotify los artistas más compatibles
def buscar_artistas_similares(genero, bpm, key):
    if genero == "Unknown":
        return ["No disponible"]
    
    artistas_base = artistas_por_genero.get(genero, {}).get("artistas", [])
    
    artistas_ordenados = sorted(artistas_base, key=lambda artista: abs(obtener_bpm_spotify(artista) - bpm))
    return artistas_ordenados[:2]

# Obtener BPM promedio de un artista en Spotify
def obtener_bpm_spotify(artista):
    resultados = sp.search(q=artista, type='track', limit=5)
    track_ids = [track['id'] for track in resultados['tracks']['items']]
    
    if not track_ids:
        return 0
    
    features = sp.audio_features(track_ids)
    bpms = [f['tempo'] for f in features if f and 'tempo' in f]
    
    return sum(bpms) / len(bpms) if bpms else 0

# Interfaz en Streamlit
st.title("🎵 Your Type Beat")
st.markdown("Sube un beat para analizar su género, BPM, tonalidad y qué artistas son similares a él.")
archivo_audio = st.file_uploader("Sube tu beat en formato WAV o MP3", type=["wav", "mp3"])

if archivo_audio is not None:
    st.audio(archivo_audio, format="audio/wav")
    with st.spinner("🔍 Analizando beat..."):
        time.sleep(2)
        tempo, key, genre, artistas_sugeridos = analizar_audio(archivo_audio)
    
    st.write(f"**🎶 Género detectado:** {genre}")
    st.write(f"**📊 BPM:** {tempo}")
    st.write(f"**🎼 Key:** {key}")
    st.write(f"**🔥 Artistas similares:** {', '.join(artistas_sugeridos)}")
