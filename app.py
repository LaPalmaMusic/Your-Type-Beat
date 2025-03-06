import streamlit as st
import librosa
import numpy as np

# Función para analizar el audio
def analizar_audio(ruta_audio):
    try:
        # Cargar solo el fragmento entre 30s y 40s con una frecuencia de 22.05 kHz
        y, sr = librosa.load(ruta_audio, sr=22050, offset=30, duration=10)
        
        # Obtener el tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        # Calcular la energía media con menor carga de cómputo
        energy = np.mean(librosa.feature.rms(y=y, frame_length=2048, hop_length=1024))
        
        # Obtener el chroma reducido para mejor rendimiento
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=8)
        chroma_mean = np.mean(chroma, axis=1)
        
        return tempo, energy, chroma_mean
    except Exception as e:
        return None, None, None

# Interfaz de usuario en Streamlit
st.title("Análisis de Beat")
archivo_audio = st.file_uploader("Sube tu beat en formato WAV o MP3", type=["wav", "mp3"])

if archivo_audio is not None:
    st.audio(archivo_audio, format="audio/wav")
    tempo, energy, chroma = analizar_audio(archivo_audio)
    
    if tempo is not None:
        st.write(f"**BPM:** {tempo:.2f}")
        st.write(f"**Energía promedio:** {energy:.5f}")
        st.write("**Perfil cromático:**")
        st.bar_chart(chroma)
    else:
        st.error("Error al analizar el audio. Asegúrate de subir un archivo válido.")
