"""Optional aiortc/av imports exposed behind a stable interface."""

try:
    import av
    from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription
    from aiortc.mediastreams import AudioStreamTrack, MediaStreamError, VideoStreamTrack

    AIORTC_AVAILABLE = True
except Exception:  # pragma: no cover - import availability depends on runtime image.
    AIORTC_AVAILABLE = False
    av = None
    RTCConfiguration = None
    RTCIceServer = None
    RTCPeerConnection = None
    RTCSessionDescription = None
    AudioStreamTrack = object
    MediaStreamError = RuntimeError
    VideoStreamTrack = object

__all__ = [
    "AIORTC_AVAILABLE",
    "av",
    "RTCConfiguration",
    "RTCIceServer",
    "RTCPeerConnection",
    "RTCSessionDescription",
    "AudioStreamTrack",
    "MediaStreamError",
    "VideoStreamTrack",
]
