import av
frame = av.AudioFrame(format='s16', layout='stereo', samples=960)
channels = len(frame.layout.channels)
print(channels)
print(frame.format.is_planar)
