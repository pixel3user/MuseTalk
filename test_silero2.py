import torch
import numpy as np

model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

# A very loud continuous tone
t = np.linspace(0, 1, 16000)
loud_noise = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

tensor = torch.from_numpy(loud_noise)
timestamps = get_speech_timestamps(tensor, model, sampling_rate=16000)
print(f"Loud tone timestamps: {timestamps}")
