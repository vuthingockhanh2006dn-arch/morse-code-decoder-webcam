"use strict";
// Presentation and input only. Detection, timing, Morse lookup and text live in Python.
const $ = id => document.getElementById(id);
const videoEl = $("webcam-video"), canvasEl = $("output-canvas");
const canvasCtx = canvasEl.getContext("2d"), cameraBtn = $("camera-btn");
const statusTextEl = $("camera-status-text"), cameraPlaceholder = $("camera-placeholder");
let backendOnline = false, requestChain = Promise.resolve(), frameBusy = false;
let virtualPose = "IDLE", isStreaming = false, isProcessing = false, mediaStream = null;
let handsInstance = null, audioCtx = null, morseMap = {}, config = {}, latest = {};
let generation = 0, lastFrameTime = performance.now(), frameCount = 0;
let imageLearning = {}, recordingImages = false, lastSavedImage = 0;
let lastTrainingMessage = "";
const cropCanvas = document.createElement("canvas");
cropCanvas.width = cropCanvas.height = 128;
const cropContext = cropCanvas.getContext("2d");
const runMode = () => $("run-mode").value;

function updateImageStatus(data) {
  if (!data) return;
  imageLearning = data;
  const report = data.report;
  $("image-model-status").textContent = data.trained ?
    `Đã học từ ảnh: ${report.model} · Test Accuracy ${(report.test_accuracy*100).toFixed(1)}% · Macro F1 ${report.test_macro_f1.toFixed(3)} · ${report.dataset_protocol?.subject_ids_available === false ? "nhóm theo tên file; chưa phải đánh giá người mới/webcam" : "chia theo " + report.split_by}` :
    "Chưa có mô hình ảnh. Chạy import_pose_images.py rồi train_image_models.py để dùng ảnh chạm ngón có sẵn, hoặc thu ảnh bổ sung trên web.";
  if (data.error) $("image-model-status").textContent += " Lỗi nạp: " + data.error;
  if (data.shape_report) {
    const shape = data.shape_report;
    $("image-model-status").textContent = `Nhận diện hình dạng học từ ảnh: ${shape.model} · Test ${(shape.test_accuracy*100).toFixed(1)}% · Tách tay nghỉ/chạm ngón · Điểm test chưa phải độ chính xác webcam/người mới.`;
  }
  $("image-counts").textContent = Object.entries(data.dataset.counts).map(([label, count])=>`${label}: ${count}`).join(" · ") + ` · ${data.dataset.sessions} phiên thật + ${data.dataset.proxy_groups || 0} nhóm file / ${data.dataset.subjects} người có mã xác nhận`;
  $("train-images").disabled = data.training.state === "running" || isStreaming;
  if (data.training.state !== "idle" && data.training.message !== lastTrainingMessage) {
    $("image-result").textContent = data.training.message;
    lastTrainingMessage = data.training.message;
  }
}

function handImage(landmarks) {
  const w = videoEl.videoWidth, h = videoEl.videoHeight;
  let side = Math.min(w, h), x = (w-side)/2, y = (h-side)/2;
  if (landmarks) {
    const xs = landmarks.map(p=>p.x*w), ys = landmarks.map(p=>p.y*h);
    const x0=Math.min(...xs), x1=Math.max(...xs), y0=Math.min(...ys), y1=Math.max(...ys);
    side = Math.min(Math.min(w,h), Math.max(x1-x0,y1-y0)*1.45);
    side = Math.max(32, side);
    x = Math.max(0, Math.min(w-side, (x0+x1-side)/2));
    y = Math.max(0, Math.min(h-side, (y0+y1-side)/2));
  }
  cropContext.drawImage(videoEl, x, y, side, side, 0, 0, 128, 128);
  return cropCanvas.toDataURL("image/jpeg", .90);
}

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

function playBeep(duration) {
  if (!$("audio-toggle").checked) return;
  try {
    audioCtx ||= new (window.AudioContext || window.webkitAudioContext)();
    audioCtx.resume();
    const osc = audioCtx.createOscillator(), gain = audioCtx.createGain();
    osc.frequency.value = 800;
    gain.gain.setValueAtTime(.12, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(.001, audioCtx.currentTime + duration / 1000);
    osc.connect(gain); gain.connect(audioCtx.destination);
    osc.start(); osc.stop(audioCtx.currentTime + duration / 1000);
  } catch (error) { console.warn(error); }
}

function render(data) {
  latest = data;
  const eventNames = {DOT:"DOT (.)", DASH:"DASH (-)", CONFIRM:"Chốt chữ", SPACE:"Dấu cách", DELETE:"Xóa"};
  $("input-status").textContent = `${data.ready ? "Đã mở khóa nhập" : "Chưa mở khóa nhập"} · FSM: ${data.state || "IDLE"} · Sự kiện cuối: ${eventNames[data.last_event] || "chưa có"} · Tổng lệnh: ${data.event_count || 0}`;
  $("morse-buffer-view").textContent = data.morse_buffer || "";
  $("decoded-text-view").textContent = data.decoded_text || "";
  $("preview-char").textContent = data.preview ? `Ký tự tiếp theo: ${data.preview}` : "";
  $("fsm-badge").textContent = data.state || "IDLE";
  $("fsm-badge").className = `fsm-badge ${data.state || "IDLE"}`;
  if ("pose" in data) {
    $("hud-pose").textContent = data.pose || "Không thấy tay";
    $("hud-conf").textContent = `${Math.round((data.confidence || 0) * 100)}%`;
    $("hud-latency").textContent = `${(data.latency_ms || 0).toFixed(1)} ms`;
    $("signal-led").classList.toggle("active", (data.pose || "").startsWith("PINCH_"));
  }
  if (data.predicted_class) {
    const names = {PINCH_THUMB:"Cái–trỏ", PINCH_RING:"Cái–áp út", PINCH_PINKY:"Cái–út", UNKNOWN:"Tay mở/nghỉ"};
    $("recognition-detail").textContent = `Dự đoán: ${names[data.predicted_class] || data.predicted_class} · Confidence ${(data.confidence*100).toFixed(1)}% / ngưỡng ${(data.confidence_threshold*100).toFixed(0)}% · ${data.accepted ? "Đạt ngưỡng" : "Chưa đạt ngưỡng, không nhập lệnh"}. ${data.recognition_hint || ""}`;
    if (data.recognition_model) $("recognition-detail").textContent += ` · ${data.recognition_model}`;
  } else if ("pose" in data && data.pose === null) {
      $("recognition-detail").textContent = data.recognition_hint || "MediaPipe chưa thấy tay. Đưa đủ bàn tay vào camera và tránh che các đầu ngón.";
  }
  if (data.distances) {
    for (const [name, label] of [["thumb", "PINCH_THUMB"], ["ring", "PINCH_RING"], ["pinky", "PINCH_PINKY"]]) {
      const value = data.distances[label];
      $("val-" + name).textContent = typeof value === "number" ? value.toFixed(3) : "--";
      $("bar-" + name).style.width = `${Math.min(100, (value || 0) * 100)}%`;
      $("bar-" + name).classList.toggle("pinched", value < config.touch_threshold);
    }
  }
  for (const event of data.events || []) {
    if (event === "DOT") playBeep(80);
    if (event === "DASH") playBeep(240);
  }
  for (const cell of $("morse-table").children) {
    const buffer = data.morse_buffer || "";
    cell.style.background = buffer && cell.dataset.code === buffer ? "#235f49" :
      buffer && cell.dataset.code.startsWith(buffer) ? "#493362" : "";
  }
}

async function checkBackend() {
  try {
    const response = await fetch("/api/status", {signal: AbortSignal.timeout(3000)});
    if (!response.ok) throw Error("Máy chủ chưa sẵn sàng");
    const data = await response.json();
    setConnected(true); config = data.config;
    updateImageStatus(data.image_learning);
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
  if (frameBusy || !backendOnline || runMode() !== "legacy") return;
  frameBusy = true;
  try { render(await api("/api/fsm_step", {pose, confidence: 1.})); }
  catch (error) { showError(error); }
  finally { frameBusy = false; }
}

function startVirtualPose(pose) {
  if (runMode() !== "legacy") { showError(Error("Nút mô phỏng chỉ dùng ở chế độ tham khảo, không phải dự đoán từ ảnh.")); return; }
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
  $("arm-input").disabled = true;
  recordingImages = false;
  $("record-images").textContent = "Bắt đầu lưu ảnh";
  generation++;
  isStreaming = false;
  virtualPose = "IDLE";
  mediaStream?.getTracks().forEach(track => track.stop());
  mediaStream = null; videoEl.srcObject = null;
  cameraBtn.textContent = "Bật Camera"; cameraBtn.className = "btn primary";
  $("train-images").disabled = imageLearning.training?.state === "running";
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
    let data;
    if (runMode() === "collect") {
      const image = handImage(landmarks);
      $("image-crop-preview").src = image;
      if (recordingImages && performance.now()-lastSavedImage >= 1000) {
        const label = $("image-label").value;
        if (landmarks || label === "UNKNOWN") {
          lastSavedImage = performance.now();
          const saved = await api("/api/images/collect", {image, label,
            subject: $("image-subject").value, session: $("image-session").value,
            train_only: $("image-train-only").checked});
          $("image-result").textContent = saved.message;
          if (imageLearning.dataset) updateImageStatus({...imageLearning, dataset: saved.dataset});
        }
      }
      return;
    } else if (runMode() === "image") {
      const image = landmarks ? handImage(landmarks) : null;
      if (image) $("image-crop-preview").src = image;
      data = await api("/api/predict_image", {image, hand_present: !!landmarks,
        landmarks: landmarks ? landmarks.map(p => [p.x,p.y,p.z]) : null,
        width: canvasEl.width, height: canvasEl.height});
    } else {
      data = await api("/api/predict", {landmarks: landmarks ? landmarks.map(p => [p.x, p.y, p.z]) : null,
        handedness, algo: $("algo-select").value, width: canvasEl.width, height: canvasEl.height});
    }
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
  if (isStreaming) { await stopCamera(); return; }
  if (!backendOnline) { showError(Error("Hãy chạy Python và mở http://localhost:8000")); return; }
  if (runMode() === "image" && !imageLearning.trained) {
    showError(Error("Chưa có mô hình ảnh. Chọn chế độ Thu ảnh, thu đủ nhãn rồi bấm Huấn luyện.")); return;
  }
  cameraBtn.disabled = true;
  try {
    await api("/api/cancel", {});
    if (!handsInstance) {
      handsInstance = new Hands({locateFile: file => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`});
      handsInstance.setOptions({maxNumHands: 1, modelComplexity: 1, minDetectionConfidence: .5, minTrackingConfidence: .35});
      handsInstance.onResults(onResults);
    }
    const metadata = new Promise(resolve => { videoEl.onloadedmetadata = resolve; });
    mediaStream = await navigator.mediaDevices.getUserMedia({video: {width: {ideal: 640}, height: {ideal: 480}}, audio: false});
    videoEl.srcObject = mediaStream;
    await metadata; await videoEl.play();
    generation++; isStreaming = true; virtualPose = "IDLE";
    $("arm-input").disabled = runMode() === "collect";
    $("train-images").disabled = true;
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
window.addEventListener("pagehide", () => {
  mediaStream?.getTracks().forEach(track => track.stop());
  navigator.sendBeacon("/api/cancel", new Blob(["{}"], {type: "application/json"}));
});
if (location.protocol === "file:") $("protocol-warning").style.display = "block";
$("run-mode").addEventListener("change", async () => {
  await stopCamera();
  $("collection-panel").hidden = runMode() !== "collect";
  $("algo-select").disabled = runMode() !== "legacy";
});
$("algo-select").disabled = true;
$("record-images").addEventListener("click", () => {
  if (runMode() !== "collect" || !isStreaming) { showError(Error("Chọn Thu ảnh và bật camera trước.")); return; }
  recordingImages = !recordingImages;
  $("record-images").textContent = recordingImages ? "Dừng lưu ảnh" : "Bắt đầu lưu ảnh";
});
for (const id of ["image-label", "image-subject", "image-session", "image-train-only"]) {
  $(id).addEventListener("change", () => {
    recordingImages = false;
    $("record-images").textContent = "Bắt đầu lưu ảnh";
    $("image-result").textContent = "Đã tạm dừng lưu. Giữ đúng tư thế nhãn mới rồi bấm Bắt đầu lưu ảnh.";
  });
}
$("train-images").addEventListener("click", async () => {
  if (isStreaming) { showError(Error("Dừng camera trước khi huấn luyện để cố định dữ liệu.")); return; }
  try { updateImageStatus(await api("/api/images/train", {split_by: $("image-split").value})); }
  catch (error) { showError(error); }
});
$("test-image").addEventListener("change", async event => {
  const file = event.target.files[0];
  if (!file) return;
  if (isStreaming) { showError(Error("Dừng camera trước khi thử ảnh riêng.")); return; }
  try {
    const bitmap = await createImageBitmap(file);
    const testCanvas = document.createElement("canvas"); testCanvas.width = testCanvas.height = 128;
    testCanvas.getContext("2d").drawImage(bitmap, 0, 0, 128, 128); bitmap.close();
    const result = await api("/api/predict_image", {image: testCanvas.toDataURL("image/jpeg", .90), apply: false, crop_input: true});
    $("image-result").textContent = `Mô hình dự đoán: ${result.predicted_class}, xác suất ${(result.confidence*100).toFixed(1)}% · ${result.accepted ? "đạt ngưỡng" : "chưa chắc chắn"}. Không nhập lệnh khi thử ảnh.`;
  } catch (error) { showError(error); }
});
$("download-crop").addEventListener("click", () => {
  const src = $("image-crop-preview").src;
  if (!src?.startsWith("data:image/")) { showError(Error("Bật camera và đưa tay vào trước khi lưu ảnh kiểm tra.")); return; }
  const link = document.createElement("a");
  link.href = src; link.download = "morse-hand-diagnostic.jpg"; link.click();
});
$("arm-input").addEventListener("click", async () => {
  if (!isStreaming || runMode() === "collect") { showError(Error("Bật camera ở chế độ nhận diện trước khi mở khóa nhập.")); return; }
  try {
    const data = await api("/api/arm", {source: runMode() === "image" ? "image" : "landmark"});
    render(data);
    $("recognition-detail").textContent = data.recognition_hint;
  } catch (error) { showError(error); }
});
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
