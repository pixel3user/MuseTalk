import webrtcvad
import numpy as np

vad = webrtcvad.Vad(3)
# generate some small white noise
noise = np.random.normal(0, 0.01, 480).astype(np.float32)
pcm_int16 = (np.clip(noise, -1.0, 1.0) * 32767.0).astype(np.int16)
print(f"Noise detected as speech: {vad.is_speech(pcm_int16.tobytes(), 16000)}")

# generate silence
silence = np.zeros(480, dtype=np.float32)
pcm_int16 = (np.clip(silence, -1.0, 1.0) * 32767.0).astype(np.int16)
print(f"Silence detected as speech: {vad.is_speech(pcm_int16.tobytes(), 16000)}")
