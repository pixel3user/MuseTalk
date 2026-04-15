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
    <div style="margin-top:4px;height:10px;width:100%;max-width:300px;background:#333;border-radius:5px;overflow:hidden;">
      <div id="mic-level" style="height:100%;width:0%;background:#4caf50;transition:width 0.1s;"></div>
    </div>
    <div id="dbg" style="margin-top:8px;font-size:12px;white-space:pre-wrap;color:#b0b0b0"></div>
  </div>
<script>
let pc = null;
let localStream = null;
let sessionId = null;
let sessionToken = null;
let audioContext = null;
let analyser = null;

const startBtn = document.getElementById('start');
const stopBtn = document.getElementById('stop');
const stateEl = document.getElementById('state');
const micEl = document.getElementById('mic');
const micLevelEl = document.getElementById('mic-level');

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

function updateMicLevel() {
  if (!analyser) return;
  const data = new Uint8Array(analyser.frequencyBinCount);
  analyser.getByteTimeDomainData(data);
  let sum = 0;
  for (let i = 0; i < data.length; i++) {
    const val = (data[i] - 128) / 128;
    sum += val * val;
  }
  let rms = Math.sqrt(sum / data.length);
  // Scale so normal talking is visible
  let db = 20 * Math.log10(rms || 0.0001);
  let pct = Math.max(0, Math.min(100, (db + 60) * (100/60)));
  micLevelEl.style.width = pct + '%';
  if (localStream) requestAnimationFrame(updateMicLevel);
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
  if (audioContext) {
    audioContext.close();
    audioContext = null;
    analyser = null;
  }
  micLevelEl.style.width = '0%';
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

    // Enable browser audio processing enhancements
    localStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
        channelCount: 1,
      },
      video: false,
    });
    setMic('granted (processed audio)');

    // IMPORTANT: When you route the microphone stream through the Web Audio API
    // to mix in silence, the browser's native WebRTC stack automatically STRIPS
    // the noiseSuppression and echoCancellation properties off the track.
    // So we have to pass the pristine, unmixed microphone stream straight
    // into the RTCPeerConnection so the WebRTC DSP processes it properly.
    audioContext = new (window.AudioContext || window.webkitAudioContext)();

    // Setup Visualizer (just tapping the original stream, not replacing it)
    const micSource = audioContext.createMediaStreamSource(localStream);
    analyser = audioContext.createAnalyser();
    analyser.fftSize = 2048;
    micSource.connect(analyser);
    requestAnimationFrame(updateMicLevel);

    pc = new RTCPeerConnection(rtcCfg);
    pc.addTransceiver('video', { direction: 'recvonly' });
    pc.addTransceiver('audio', { direction: 'recvonly' });

    // Send the original mic stream directly to WebRTC so DSP works properly
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
  cleanupPeer();
});
</script>
</body>
</html>
"""