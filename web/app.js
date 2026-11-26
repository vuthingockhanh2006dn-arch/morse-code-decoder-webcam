"use strict";
// Presentation and input only. Detection, timing, Morse lookup and text live in Python.
const $ = id => document.getElementById(id);
const videoEl = $("webcam-video"), canvasEl = $("output-canvas");
const canvasCtx = canvasEl.getContext("2d"), cameraBtn = $("camera-btn");
const statusTextEl = $("camera-status-text"), cameraPlaceholder = $("camera-placeholder");
let backendOnline = false, requestChain = Promise.resolve(), frameBusy = false;
let virtualPose = "IDLE", isStreaming = false, isProcessing = false, mediaStream = null;
let handsInstance = null, audioCtx = null, morseMap = {}, config = {}, latest = {};
let generation = 0, lastFrameTime = performance.now(), frameCount = 0, bufferTimer = null;

function setConnected(online) {
  backendOnline = online;
  $("conn-dot").classList.toggle("online", online);
  $("conn-text").textContent = online ? "Python đang kết nối" : "Mất kết nối Python · kiểm tra máy chủ";
}

function api(path, body) {
  // All writes share one ordered stream; clear/cancel cannot overtake a frame.
  const request = requestChain.then(async () => {
    const controller = new AbortController(), timer = setTimeout(() => controller.abort(), 4000);
    try {
      const response = await fetch(path, {method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify(body), signal: controller.signal});
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Lệnh không hợp lệ");
      setConnected(true);
      return data;
    } finally { clearTimeout(timer); }
  });
  requestChain = request.catch(() => {});
  return request;
}

function showError(error) {
  $("conn-text").textContent = error.message || "Không kết nối được Python";
}

function unlockAudio() {
  try {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    if (!Ctx) return;
    audioCtx ||= new Ctx();
    if (audioCtx.state === "suspended") audioCtx.resume();
  } catch (error) { console.warn(error); }
}

function playBeep(kind) {
  const isDash = kind === "DASH" || kind === 240;
  const duration = typeof kind === "number" ? kind : (isDash ? 240 : 85);
  if (!$("audio-toggle").checked) return 0;
  try {
    unlockAudio();
    if (audioCtx) {
      const now = audioCtx.currentTime;
      const osc = audioCtx.createOscillator(), gain = audioCtx.createGain();
      // "Tích" (DOT): âm cao, gọn (920 Hz, 85 ms); "Tè" (DASH): âm trầm, ngân dài (560 Hz, 240 ms)
      osc.type = isDash ? "triangle" : "sine";
      osc.frequency.setValueAtTime(isDash ? 560 : 920, now);
      gain.gain.setValueAtTime(.18, now);
      gain.gain.setValueAtTime(.18, now + (duration * .75) / 1000);
      gain.gain.exponentialRampToValueAtTime(.001, now + duration / 1000);
      osc.connect(gain); gain.connect(audioCtx.destination);
      osc.start(now); osc.stop(now + duration / 1000);
    }
  } catch (error) { console.warn(error); }
  return duration;
}

function applyBufferView(data) {
  const buffer = data.morse_buffer || "";
  $("morse-buffer-view").textContent = buffer;
  $("preview-char").textContent = data.preview ? `Ký tự tiếp theo: ${data.preview}` : "";
  for (const cell of $("morse-table").children) {
    cell.style.background = buffer && cell.dataset.code === buffer ? "#235f49" :
      buffer && cell.dataset.code.startsWith(buffer) ? "#493362" : "";
  }
}

function render(data) {
  latest = data;
  $("decoded-text-view").textContent = data.decoded_text || "";
  $("fsm-badge").textContent = data.state || "IDLE";
  $("fsm-badge").className = `fsm-badge ${data.state || "IDLE"}`;
  if ("pose" in data) {
    $("hud-pose").textContent = data.pose || "Không thấy tay";
    $("hud-conf").textContent = `${Math.round((data.confidence || 0) * 100)}%`;
    $("hud-latency").textContent = `${(data.latency_ms || 0).toFixed(1)} ms`;
    $("signal-led").classList.toggle("active", (data.pose || "").startsWith("PINCH_"));
  }
  if (data.distances) {
    for (const [name, label] of [["thumb", "PINCH_THUMB"], ["ring", "PINCH_RING"], ["pinky", "PINCH_PINKY"]]) {
      const value = data.distances[label];
      $("val-" + name).textContent = typeof value === "number" ? value.toFixed(3) : "--";
      $("bar-" + name).style.width = `${Math.min(100, (value || 0) * 100)}%`;
      $("bar-" + name).classList.toggle("pinched", value < config.touch_threshold);
    }
  }
  const events = data.events || [];
  let beepDelay = 0;
  for (const event of events) {
    if (event === "DOT" || event === "DASH") {
      beepDelay = Math.max(beepDelay, playBeep(event));
    }
  }
  // Phát âm thanh tích/tè trước, sau đó mới nhảy ký tự dot/dash tương ứng lên màn hình
  if (beepDelay > 0) {
    if (bufferTimer) clearTimeout(bufferTimer);
    bufferTimer = setTimeout(() => {
      bufferTimer = null;
      applyBufferView(latest);
    }, beepDelay);
  } else if (events.length > 0 || !bufferTimer) {
    if (bufferTimer) {
      clearTimeout(bufferTimer);
      bufferTimer = null;
    }
    applyBufferView(data);
  }
}

async function checkBackend() {
  try {
    const response = await fetch("/api/status", {signal: AbortSignal.timeout(3000)});
    if (!response.ok) throw Error("Máy chủ chưa sẵn sàng");
    const data = await response.json();
    setConnected(true); config = data.config;
    if (!Object.keys(morseMap).length) {
      morseMap = data.morse_table;
      for (const [code, char] of Object.entries(morseMap)) {
        const cell = document.createElement("div"), bold = document.createElement("b");
        cell.className = "morse-cell"; cell.dataset.code = code;
        bold.textContent = char; cell.append(bold, document.createTextNode(code));
        $("morse-table").append(cell);
      }
      render(data);
    }
    for (const option of $("algo-select").options) {
      option.disabled = !["auto", "heuristic", ...data.loaded_models].includes(option.value);
    }
  } catch (error) { setConnected(false); }
}

async function sendPose(pose) {
  if (frameBusy || !backendOnline) return;
  frameBusy = true;
  try { render(await api("/api/fsm_step", {pose, confidence: 1.})); }
  catch (error) { showError(error); }
  finally { frameBusy = false; }
}

function startVirtualPose(pose) {
  unlockAudio();
  if (isStreaming) { $("conn-text").textContent = "Tắt camera trước khi dùng nút mô phỏng."; return; }
  virtualPose = pose;
  sendPose(pose);
}
function stopVirtualPose() { virtualPose = "IDLE"; }
for (const [id, pose] of [["pad-thumb", "PINCH_THUMB"], ["pad-ring", "PINCH_RING"], ["pad-pinky", "PINCH_PINKY"]]) {
  $(id).addEventListener("pointerdown", event => {
    event.preventDefault(); $(id).setPointerCapture(event.pointerId); startVirtualPose(pose);
  });
  $(id).addEventListener("pointerup", stopVirtualPose);
  $(id).addEventListener("pointercancel", stopVirtualPose);
}
window.addEventListener("blur", stopVirtualPose);
$("pad-rest").addEventListener("click", stopVirtualPose);
// Continuous rest frames also expire single ring taps and unlock held commands.
setInterval(() => { if (!isStreaming) sendPose(virtualPose); }, 33);

async function stopCamera() {
  generation++;
  isStreaming = false;
  virtualPose = "IDLE";
  mediaStream?.getTracks().forEach(track => track.stop());
  mediaStream = null; videoEl.srcObject = null;
  cameraBtn.textContent = "Bật Camera"; cameraBtn.className = "btn primary";
  cameraPlaceholder.style.display = "flex";
  canvasCtx.clearRect(0, 0, canvasEl.width, canvasEl.height);
  try { render(await api("/api/cancel", {})); } catch (error) { showError(error); }
}

async function onResults(results) {
  if (!isStreaming || frameBusy) return;
  frameBusy = true;
  const currentGeneration = generation;
  try {
    frameCount++;
    if (performance.now() - lastFrameTime > 1000) {
      $("hud-fps").textContent = `${frameCount} fps`;
      frameCount = 0; lastFrameTime = performance.now();
    }
    canvasEl.width = videoEl.videoWidth || 640; canvasEl.height = videoEl.videoHeight || 480;
    canvasCtx.clearRect(0, 0, canvasEl.width, canvasEl.height);
    const landmarks = results.multiHandLandmarks?.[0];
    const handedness = results.multiHandedness?.[0]?.label || "Right";
    if (landmarks) drawHandSkeleton(canvasCtx, landmarks, canvasEl.width, canvasEl.height, handedness);
    else drawTargetGuide(canvasCtx, canvasEl.width, canvasEl.height);
    const data = await api("/api/predict", {landmarks: landmarks ? landmarks.map(p => [p.x, p.y, p.z]) : null,
      handedness, algo: $("algo-select").value, width: canvasEl.width, height: canvasEl.height});
    if (generation === currentGeneration && isStreaming) render(data);
  } catch (error) { showError(error); }
  finally { frameBusy = false; }
}

async function videoProcessingLoop() {
  if (!isStreaming) return;
  if (videoEl.readyState >= 2 && !isProcessing) {
    isProcessing = true;
    try { await handsInstance.send({image: videoEl}); }
    catch (error) { showError(error); }
    finally { isProcessing = false; }
  }
  if (isStreaming) requestAnimationFrame(videoProcessingLoop);
}

cameraBtn.addEventListener("click", async () => {
  unlockAudio();
  if (isStreaming) { await stopCamera(); return; }
  if (!backendOnline) { showError(Error("Hãy chạy Python và mở http://localhost:8000")); return; }
  cameraBtn.disabled = true;
  try {
    await api("/api/cancel", {});
    if (!handsInstance) {
      handsInstance = new Hands({locateFile: file => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`});
      handsInstance.setOptions({maxNumHands: 1, modelComplexity: 1, minDetectionConfidence: .6, minTrackingConfidence: .5});
      handsInstance.onResults(onResults);
    }
    const metadata = new Promise(resolve => { videoEl.onloadedmetadata = resolve; });
    mediaStream = await navigator.mediaDevices.getUserMedia({video: {width: {ideal: 640}, height: {ideal: 480}}, audio: false});
    videoEl.srcObject = mediaStream;
    await metadata; await videoEl.play();
    generation++; isStreaming = true; virtualPose = "IDLE";
    cameraBtn.textContent = "Tắt Camera"; cameraBtn.className = "btn danger";
    cameraPlaceholder.style.display = "none";
    requestAnimationFrame(videoProcessingLoop);
  } catch (error) {
    await stopCamera();
    statusTextEl.textContent = "Không mở được camera: " + error.message;
  } finally { cameraBtn.disabled = false; }
});

$("algo-select").addEventListener("change", async () => {
  generation++;
  try { render(await api("/api/cancel", {})); } catch (error) { showError(error); }
});
$("clear-btn").addEventListener("click", async () => {
  generation++; virtualPose = "IDLE";
  try { render(await api("/api/reset", {})); } catch (error) { showError(error); }
});
$("copy-btn").addEventListener("click", async () => {
  try { await navigator.clipboard.writeText(latest.decoded_text || ""); $("conn-text").textContent = "Đã sao chép thông điệp"; }
  catch { $("conn-text").textContent = "Không sao chép được. Hãy chọn văn bản và nhấn Ctrl+C."; }
});

// Dịch từ chữ sang mã Morse (Text -> Morse)
const DEFAULT_CHAR_TO_MORSE = {
  A: ".-", B: "-...", C: "-.-.", D: "-..", E: ".", F: "..-.", G: "--.", H: "....", I: "..",
  J: ".---", K: "-.-", L: ".-..", M: "--", N: "-.", O: "---", P: ".--.", Q: "--.-", R: ".-.",
  S: "...", T: "-", U: "..-", V: "...-", W: ".--", X: "-..-", Y: "-.--", Z: "--..",
  0: "-----", 1: ".----", 2: "..---", 3: "...--", 4: "....-",
  5: ".....", 6: "-....", 7: "--...", 8: "---..", 9: "----."
};

function encodeTextToMorse(raw) {
  const reverseMap = Object.keys(morseMap).length
    ? Object.fromEntries(Object.entries(morseMap).map(([code, ch]) => [ch, code]))
    : DEFAULT_CHAR_TO_MORSE;
  const cleaned = String(raw || "")
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/đ/gi, "D")
    .toUpperCase()
    .trim();
  if (!cleaned) return "";
  return cleaned
    .split(/\s+/)
    .map(word => Array.from(word).map(ch => reverseMap[ch] || "?").join(" "))
    .join(" / ");
}

let morsePlayTimers = [];
function stopMorsePlayback() {
  for (const t of morsePlayTimers) clearTimeout(t);
  morsePlayTimers = [];
  $("play-morse-btn").textContent = "🔊 Phát âm";
}

$("text-to-morse-input").addEventListener("input", () => {
  stopMorsePlayback();
  $("text-to-morse-output").textContent = encodeTextToMorse($("text-to-morse-input").value);
});

$("clear-morse-btn").addEventListener("click", () => {
  stopMorsePlayback();
  $("text-to-morse-input").value = "";
  $("text-to-morse-output").textContent = "";
});

$("copy-morse-btn").addEventListener("click", async () => {
  const code = $("text-to-morse-output").textContent || "";
  try { await navigator.clipboard.writeText(code); $("conn-text").textContent = "Đã sao chép mã Morse"; }
  catch { $("conn-text").textContent = "Không sao chép được mã Morse."; }
});

$("play-morse-btn").addEventListener("click", () => {
  unlockAudio();
  if (morsePlayTimers.length) { stopMorsePlayback(); return; }
  const code = $("text-to-morse-output").textContent || encodeTextToMorse($("text-to-morse-input").value);
  if (!code) return;
  $("text-to-morse-output").textContent = code;
  $("play-morse-btn").textContent = "⏹ Dừng phát";
  let offset = 0;
  for (const ch of code) {
    if (ch === "." || ch === "-") {
      const kind = ch === "." ? "DOT" : "DASH";
      const dur = ch === "." ? 85 : 240;
      morsePlayTimers.push(setTimeout(() => playBeep(kind), offset));
      offset += dur + 90;
    } else if (ch === " ") {
      offset += 140;
    } else if (ch === "/") {
      offset += 260;
    }
  }
  morsePlayTimers.push(setTimeout(stopMorsePlayback, offset));
});
window.addEventListener("pagehide", () => {
  mediaStream?.getTracks().forEach(track => track.stop());
  navigator.sendBeacon("/api/cancel", new Blob(["{}"], {type: "application/json"}));
});
if (location.protocol === "file:") $("protocol-warning").style.display = "block";
checkBackend();
setInterval(checkBackend, 5000);

    function drawTargetGuide(ctx, w, h) {
      if (!w || !h) return;
      ctx.save();
      ctx.strokeStyle = "rgba(56, 189, 248, 0.4)";
      ctx.lineWidth = 2;
      ctx.setLineDash([8, 8]);
      const boxW = w * 0.45;
      const boxH = h * 0.6;
      const boxX = (w - boxW) / 2;
      const boxY = (h - boxH) / 2;
      ctx.strokeRect(boxX, boxY, boxW, boxH);

      ctx.fillStyle = "rgba(56, 189, 248, 0.8)";
      ctx.font = "14px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("✋ Đưa bàn tay vào vùng này", w / 2, boxY + boxH / 2);
      ctx.restore();
    }

    function drawHandSkeleton(ctx, landmarks, w, h, handedness) {
      const connections = [
        [0, 1], [1, 2], [2, 3], [3, 4],
        [0, 5], [5, 6], [6, 7], [7, 8],
        [5, 9], [9, 10], [10, 11], [11, 12],
        [9, 13], [13, 14], [14, 15], [15, 16],
        [13, 17], [17, 18], [18, 19], [19, 20],
        [0, 17]
      ];

      // Khớp nối
      ctx.lineWidth = 3;
      ctx.strokeStyle = "rgba(56, 189, 248, 0.85)";
      for (const [a, b] of connections) {
        ctx.beginPath();
        ctx.moveTo(landmarks[a].x * w, landmarks[a].y * h);
        ctx.lineTo(landmarks[b].x * w, landmarks[b].y * h);
        ctx.stroke();
      }

      // Điểm mốc
      for (let i = 0; i < landmarks.length; i++) {
        const pt = landmarks[i];
        ctx.beginPath();
        ctx.arc(pt.x * w, pt.y * h, (i === 4 || i === 8 || i === 16 || i === 20) ? 6 : 4, 0, 2 * Math.PI);
        if (i === 4 || i === 8 || i === 16 || i === 20) {
          ctx.fillStyle = "#22c55e"; // Fingertips
        } else {
          ctx.fillStyle = "#ffffff";
        }
        ctx.fill();
      }

      // Nhãn tay
      ctx.fillStyle = "#22c55e";
      ctx.font = "bold 13px monospace";
      ctx.fillText(`${handedness} Hand`, landmarks[0].x * w - 30, landmarks[0].y * h + 24);
    }

    // Vòng lặp camera trực tiếp native
