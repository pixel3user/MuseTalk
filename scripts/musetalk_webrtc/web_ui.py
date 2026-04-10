"""Embedded browser client HTML used by GET /."""

HTML_PAGE = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>MuseTalk WebRTC</title>
</head>
<body style="margin:0;background:#121212;color:#e5e5e5;font-family:system-ui,sans-serif">
  <div style="padding:12px;font-size:14px">MuseTalk In-Memory WebRTC Preview</div>
  <div style="display:flex;gap:12px;padding:0 12px 12px 12px;flex-wrap:wrap">
    <video id="v" autoplay playsinline muted style="width:min(96vw,960px);background:black;border-radius:10px"></video>
    <audio id="a" autoplay controls style="width:min(96vw,960px)"></audio>
  </div>
  <div style="padding:0 12px 12px 12px">
    <button id="start">Start</button>
    <button id="stop" disabled>Stop</button>
    <span id="state" style="margin-left:8px">idle</span>
    <div id="mic" style="margin-top:8px;font-size:12px;color:#8fd48f">mic: idle</div>
    <div id="dbg" style="margin-top:8px;font-size:12px;white-space:pre-wrap;color:#b0b0b0"></div>
  </div>
<script>
let pc = null;
let localStream = null;
let sessionId = null;
let sessionToken = null;
const startBtn = document.getElementById('start');
const stopBtn = document.getElementById('stop');
const stateEl = document.getElementById('state');
const micEl = document.getElementById('mic');

// Update textual connection state in UI.
function setState(next) {
  stateEl.textContent = next;
}

// Update mic permission/status label in UI.
function setMic(next) {
  micEl.textContent = 'mic: ' + next;
}

// Wait for non-trickle ICE gathering to complete before POSTing /offer.
async function waitIceGatheringComplete(pc, timeoutMs = 4000) {
  if (pc.iceGatheringState === 'complete') return;
  await new Promise((resolve) => {
    let done = false;
    const finish = () => {
      if (done) return;
      done = true;
      pc.removeEventListener('icegatheringstatechange', onState);
      resolve();
    };
    const onState = () => {
      if (pc.iceGatheringState === 'complete') finish();
    };
    pc.addEventListener('icegatheringstatechange', onState);
    setTimeout(finish, timeoutMs);
  });
}

// Close browser RTCPeerConnection and stop local media tracks.
async function cleanupPeer() {
  if (pc) {
    try { pc.ontrack = null; } catch (e) {}
    try { pc.onconnectionstatechange = null; } catch (e) {}
    try { pc.close(); } catch (e) {}
    pc = null;
  }
  if (localStream) {
    for (const track of localStream.getTracks()) {
      try { track.stop(); } catch (e) {}
    }
    localStream = null;
  }
}

// Delete backend session (if created) using returned session token.
async function cleanupSession() {
  if (!sessionId || !sessionToken) {
    sessionId = null;
    sessionToken = null;
    return;
  }
  try {
    await fetch('/v1/sessions/' + sessionId, {
      method: 'DELETE',
      headers: { 'x-session-token': sessionToken },
    });
  } catch (e) {}
  sessionId = null;
  sessionToken = null;
}

// Start local mic capture, negotiate WebRTC, and call POST /offer.
async function start() {
  if (pc) {
    await stop();
  }
  startBtn.disabled = true;
  stopBtn.disabled = false;
  try {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error('Browser does not support getUserMedia');
    }
    setState('requesting-mic');
    setMic('requesting permission');

    const rtcCfg = await (await fetch('/config')).json();
    localStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
      video: false,
    });
    setMic('granted');

    pc = new RTCPeerConnection(rtcCfg);
    pc.addTransceiver('video', { direction: 'recvonly' });
    pc.addTransceiver('audio', { direction: 'recvonly' });
    for (const track of localStream.getAudioTracks()) {
      pc.addTrack(track, localStream);
    }
    pc.onconnectionstatechange = () => {
      if (pc) setState(pc.connectionState);
    };
    pc.ontrack = (ev) => {
      if (ev.track.kind === 'video') {
        document.getElementById('v').srcObject = ev.streams[0];
      } else if (ev.track.kind === 'audio') {
        document.getElementById('a').srcObject = ev.streams[0];
      }
    };

    setState('connecting');
    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);
    // Non-trickle flow: wait for ICE gathering so TURN/relay candidates are included.
    await waitIceGatheringComplete(pc, 5000);
    const resp = await fetch('/offer', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ sdp: pc.localDescription.sdp, type: pc.localDescription.type }),
    });
    if (!resp.ok) {
      throw new Error('offer failed: ' + await resp.text());
    }
    const answer = await resp.json();
    sessionId = answer.session_id || null;
    sessionToken = answer.session_token || null;
    await pc.setRemoteDescription(answer);
    setState('connected');
  } catch (e) {
    console.error(e);
    setState('error');
    setMic('error');
    await cleanupPeer();
    await cleanupSession();
    stopBtn.disabled = true;
  } finally {
    startBtn.disabled = false;
  }
}

// Fully stop peer/session and reset media elements/UI.
async function stop() {
  stopBtn.disabled = true;
  setState('stopping');
  await cleanupPeer();
  await cleanupSession();
  document.getElementById('v').srcObject = null;
  document.getElementById('a').srcObject = null;
  setMic('idle');
  setState('idle');
}

startBtn.onclick = start;
stopBtn.onclick = stop;

setInterval(async () => {
  try {
    const s = await (await fetch('/status')).json();
    document.getElementById('dbg').textContent = JSON.stringify(s, null, 2);
  } catch (e) {}
}, 1000);

window.addEventListener('beforeunload', () => {
  try {
    if (localStream) {
      for (const track of localStream.getTracks()) {
        track.stop();
      }
    }
    if (pc) {
      pc.close();
    }
  } catch (e) {}
});
</script>
</body>
</html>
"""
