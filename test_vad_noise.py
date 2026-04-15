import webrtcvad
import numpy as np

vad = webrtcvad.Vad(3)

t = np.linspace(0, 0.03, 480)
loud_noise = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
pcm_int16 = (np.clip(loud_noise, -1.0, 1.0) * 32767.0).astype(np.int16)

print(f"Loud tone in WebRTC VAD: {vad.is_speech(pcm_int16.tobytes(), 16000)}")
