import os
import numpy as np
import librosa
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
    "Trap": {"bpm": (140, 160), "artistas": ["Travis Scott", "Drake", "Future", "Lil Baby", "21 Savage"]},
    "Reggaeton": {"bpm": (90, 110), "artistas": ["Bad Bunny", "J Balvin", "Rauw Alejandro", "Feid", "Ozuna"]},
    "Hip-Hop": {"bpm": (110, 140), "artistas": ["Kendrick Lamar", "J. Cole", "Nas", "Jay-Z", "Eminem"]},
    "Drill": {"bpm": (160, 180), "artistas": ["Pop Smoke", "Fivio Foreign", "Central Cee", "Sheff G", "Headie One"]},
    "Afrobeat": {"bpm": (70, 90), "artistas": ["Burna Boy", "Wizkid", "Davido", "Rema", "Tems"]}
}

# Función para analizar el audio
def analizar_audio(archivo):
    try:
        # Leer el archivo desde BytesIO
        y, sr = librosa.load(BytesIO(archivo.getvalue()), sr=22050, offset=30, duration=30)
        
        # Obtener BPM
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = round(float(tempo), 2) if tempo else 0
        
        # Obtener Key y escala
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        key_index = np.argmax(np.mean(chroma, axis=1))
        keys = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        key = keys[key_index]
        scale = "Minor" if np.mean(chroma[key_index]) < 0.5 else "Major"
        
        # Determinar género basado en BPM
        genero_detectado = "Unknown"
        for genero, data in artistas_por_genero.items():
            if data["bpm"][0] <= tempo < data["bpm"][1]:
                genero_detectado = genero
                break
        
        # Buscar artista en Spotify
        artista_sugerido = buscar_artista_spotify(genero_detectado, tempo, key, scale)
        
        return tempo, key, scale, genero_detectado, artista_sugerido
    except Exception as e:
        return 0, "Unknown", "Unknown", "Unknown", "No disponible"

# Buscar en Spotify el artista más compatible
def buscar_artista_spotify(genero, bpm, key, scale):
    if genero == "Unknown":
        return "No disponible"
    
    posibles_artistas = artistas_por_genero.get(genero, {}).get("artistas", [])
    
    for artista in posibles_artistas:
        results = sp.search(q=artista, type="artist", limit=1)
        if results["artists"]["items"]:
            artist_id = results["artists"]["items"][0]["id"]
            top_tracks = sp.artist_top_tracks(artist_id, country="US")["tracks"]
            
            for track in top_tracks:
                audio_features = sp.audio_features(track["id"])[0]
                if audio_features:
                    track_bpm = round(audio_features["tempo"], 2)
                    track_key = audio_features["key"]
                    track_scale = "Minor" if audio_features["mode"] == 0 else "Major"
                    
                    if abs(track_bpm - bpm) <= 5 and track_key == key and track_scale == scale:
                        return artista
    return "No disponible"

# Interfaz en Streamlit
st.title("🎵 Your Type Beat")
st.markdown("Sube un beat para analizar su género, BPM, tonalidad y qué artista quedaría bien en él.")
archivo_audio = st.file_uploader("Sube tu beat en formato WAV o MP3", type=["wav", "mp3"])

if archivo_audio is not None:
    st.audio(archivo_audio, format="audio/wav")
    with st.spinner("🔍 Analizando beat..."):
        time.sleep(2)
        tempo, key, scale, genre, artista = analizar_audio(archivo_audio)
    
    st.write(f"**🎶 Género detectado:** {genre}")
    st.write(f"**📊 BPM:** {tempo}")
    st.write(f"**🎼 Key/Scale:** {key} {scale}")
    st.write(f"**🔥 Artista recomendado:** {artista}")
