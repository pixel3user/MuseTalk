import torch
import numpy as np

model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

noise = np.random.normal(0, 0.05, 16000).astype(np.float32)
tensor = torch.from_numpy(noise)

timestamps = get_speech_timestamps(tensor, model, sampling_rate=16000)
print(f"Noise timestamps: {timestamps}")

silence = np.zeros(16000, dtype=np.float32)
tensor = torch.from_numpy(silence)
timestamps = get_speech_timestamps(tensor, model, sampling_rate=16000)
print(f"Silence timestamps: {timestamps}")
