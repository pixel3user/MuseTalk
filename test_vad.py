import webrtcvad
import numpy as np

vad = webrtcvad.Vad(3)

def test_chunk(val):
    noise = np.full(480, val, dtype=np.float32)
    pcm_int16 = (np.clip(noise, -1.0, 1.0) * 32767.0).astype(np.int16)
    return vad.is_speech(pcm_int16.tobytes(), 16000)

print("0.0:", test_chunk(0.0))
print("0.01:", test_chunk(0.01))
print("0.1:", test_chunk(0.1))
print("0.5:", test_chunk(0.5))
