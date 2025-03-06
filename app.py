import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Cargar credenciales desde las variables de entorno (GitHub Secrets)
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

# Configurar autenticación con Spotify API
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=SPOTIFY_CLIENT_ID,
    client_secret=SPOTIFY_CLIENT_SECRET
))

import streamlit as st
import librosa
import numpy as np
import time

# Diccionario de artistas por género con características clave
artistas_por_genero = {
    "Trap": {"bpm": (140, 160), "artistas": ["Travis Scott", "Drake", "Future", "Lil Baby", "21 Savage", "Don Toliver", "Playboi Carti", "Roddy Ricch", "Gunna", "Lil Uzi Vert"]},
    "Reggaeton": {"bpm": (90, 110), "artistas": ["Bad Bunny", "J Balvin", "Rauw Alejandro", "Feid", "Ozuna", "Anuel AA", "Karol G", "Daddy Yankee", "Myke Towers", "Jhay Cortez"]},
    "Hip-Hop": {"bpm": (110, 140), "artistas": ["Kendrick Lamar", "J. Cole", "Nas", "Jay-Z", "Eminem", "Kanye West", "Pusha T", "Tyler, The Creator", "A$AP Rocky", "Big Sean"]},
    "Drill": {"bpm": (160, 180), "artistas": ["Pop Smoke", "Fivio Foreign", "Central Cee", "Sheff G", "Headie One", "Digga D", "G Herbo", "Lil Durk", "King Von", "Stormzy"]},
    "Afrobeat": {"bpm": (70, 90), "artistas": ["Burna Boy", "Wizkid", "Davido", "Rema", "Tems", "Omah Lay", "CKay", "Fireboy DML", "Joeboy", "Mr Eazi"]}
}

# Función para analizar el audio
def analizar_audio(ruta_audio):
    try:
        # Cargar fragmento más largo (30s-60s) para mejorar la detección de key/scale
        y, sr = librosa.load(ruta_audio, sr=22050, offset=30, duration=30)
        
        # Obtener BPM
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = round(float(tempo), 2) if tempo else 0
        
        # Obtener Key y si es menor o mayor
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
        
        # Seleccionar 1 o 2 artistas más adecuados del género
        artistas = np.random.choice(artistas_por_genero.get(genero_detectado, {}).get("artistas", ["No disponible"]), min(2, len(artistas_por_genero.get(genero_detectado, {}).get("artistas", []))), replace=False)
        
        return tempo, key, scale, genero_detectado, artistas
    except Exception as e:
        return 0, "Unknown", "Unknown", "Unknown", ["No disponible"]

# Interfaz de usuario en Streamlit
st.title("🎵 Your Type Beat")
st.markdown("Sube un beat para analizar su género, BPM, tonalidad y qué artista quedaría bien en él.")
archivo_audio = st.file_uploader("Sube tu beat en formato WAV o MP3", type=["wav", "mp3"])

if archivo_audio is not None:
    st.audio(archivo_audio, format="audio/wav")
    with st.spinner("🔍 Analizando beat..."):
        time.sleep(2)  # Simulación de carga
        tempo, key, scale, genre, artistas = analizar_audio(archivo_audio)
    
    st.write(f"**🎶 Género detectado:** {genre}")
    st.write(f"**📊 BPM:** {tempo}")
    st.write(f"**🎼 Key/Scale:** {key} {scale}")
    st.write(f"**🔥 Artistas recomendados:** {', '.join(artistas)}")
