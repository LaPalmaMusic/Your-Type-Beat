import streamlit as st
import librosa
import numpy as np
import time

# Diccionario de artistas por género
top_artists = {
    "Trap": ["Travis Scott", "Drake", "Future", "Lil Baby", "21 Savage", "Don Toliver", "Playboi Carti", "Roddy Ricch", "Gunna", "Lil Uzi Vert"],
    "Reggaeton": ["Bad Bunny", "J Balvin", "Rauw Alejandro", "Feid", "Ozuna", "Anuel AA", "Karol G", "Daddy Yankee", "Myke Towers", "Jhay Cortez"],
    "Hip-Hop": ["Kendrick Lamar", "J. Cole", "Nas", "Jay-Z", "Eminem", "Kanye West", "Pusha T", "Tyler, The Creator", "A$AP Rocky", "Big Sean"],
    "Drill": ["Pop Smoke", "Fivio Foreign", "Central Cee", "Sheff G", "Headie One", "Digga D", "G Herbo", "Lil Durk", "King Von", "Stormzy"],
    "Afrobeat": ["Burna Boy", "Wizkid", "Davido", "Rema", "Tems", "Omah Lay", "CKay", "Fireboy DML", "Joeboy", "Mr Eazi"]
}

# Función para analizar el audio
def analizar_audio(ruta_audio):
    try:
        # Cargar solo el fragmento entre 30s y 40s con menor calidad para optimizar
        y, sr = librosa.load(ruta_audio, sr=22050, offset=30, duration=10)
        
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
        if tempo < 90:
            genre = "Afrobeat"
        elif 90 <= tempo < 110:
            genre = "Reggaeton"
        elif 110 <= tempo < 140:
            genre = "Hip-Hop"
        elif 140 <= tempo < 160:
            genre = "Trap"
        else:
            genre = "Drill"
        
        # Seleccionar 1 o 2 artistas del género
        artistas = np.random.choice(top_artists.get(genre, ["Desconocido"]), min(2, len(top_artists.get(genre, []))), replace=False)
        
        return tempo, key, scale, genre, artistas
    except Exception as e:
        return 0, "Unknown", "Unknown", "Unknown", ["No disponible"]

# Interfaz de usuario en Streamlit
st.title("🎵 Your Type Beat")
st.markdown("Sube un beat para analizar su género, BPM, tonalidad y qué artista quedaría bien en él.")
archivo_audio = st.file_uploader("Sube tu beat en formato WAV o MP3", type=["wav", "mp3"])

if archivo_audio is not None:
    st.audio(archivo_audio, format="audio/wav")
    with st.spinner("Analizando beat..."):
        time.sleep(2)  # Simulación de carga
        tempo, key, scale, genre, artistas = analizar_audio(archivo_audio)
    
    st.write(f"**🎶 Género detectado:** {genre}")
    st.write(f"**📊 BPM:** {tempo}")
    st.write(f"**🎼 Key/Scale:** {key} {scale}")
    st.write(f"**🔥 Artistas recomendados:** {', '.join(artistas)}")
