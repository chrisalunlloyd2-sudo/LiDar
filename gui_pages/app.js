"use strict";

const $ = (id) => document.getElementById(id);
const clamp01 = (v) => Math.max(0, Math.min(1, v));

const state = {
  stream: null,
  intrinsics: null,
  pose: { lat: 0, lon: 0, alt_m: 0, heading_deg: 0, pitch_deg: 0, roll_deg: 0, accuracy_m: 8, has_gps: false },
  fCal: 0,
  detection: null,
  sendTimer: null,
  mapPoints: [],
  lastResult: null,
};

const PRESETS = [
  ["2x4", 88.9], ["2x6", 139.7], ["2x8", 184.2], ["4x4", 88.9], ["1x4", 88.9],
  ["credit_card", 85.6], ["a4_paper", 210.0], ["brick", 200.0],
  ["paint_can", 168.0], ["box", 400.0], ["square_100", 100.0],
  ["cube_100", 100.0], ["rect_200x100", 200.0], ["custom", 88.9],
];

const PAGES = ["controls", "camera", "measure", "geo", "map", "inventory", "illusions", "settings"];

// ── modular page loading (mmap-backed fragments) ──────────────────────
async function loadPages() {
  await Promise.all(PAGES.map(async (name) => {
    try {
      const res = await fetch(`/page/${name}.html`, { cache: "no-store" });
      const slot = $(`slot-${name}`);
      if (res.ok && slot) slot.innerHTML = await res.text();
    } catch (e) { /* leave slot empty */ }
  }));
}

function fillPresets() {
  const sel = $("preset");
  if (!sel) return;
  sel.innerHTML = PRESETS.map(([tag, w]) =>
    `<option value="${tag}" data-w="${w}">${tag} (${w} mm)</option>`).join("");
  sel.onchange = () => {
    const opt = sel.selectedOptions[0];
    if (opt) $("known").value = opt.dataset.w;
  };
}

// ── sensors ───────────────────────────────────────────────────────────
function watchGPS() {
  if (!navigator.geolocation) return;
  navigator.geolocation.watchPosition((pos) => {
    state.pose.lat = pos.coords.latitude;
    state.pose.lon = pos.coords.longitude;
    state.pose.alt_m = pos.coords.altitude || 0;
    state.pose.accuracy_m = pos.coords.accuracy || 8;
    state.pose.has_gps = true;
    updateSensorBadge();
  }, (err) => setStatus("GPS: " + err.message, "err"), {
    enableHighAccuracy: true, maximumAge: 1000, timeout: 12000,
  });
}

function onOrientation(e) {
  if (e.webkitCompassHeading != null) state.pose.heading_deg = e.webkitCompassHeading;
  else if (e.alpha != null) state.pose.heading_deg = (360 - e.alpha) % 360;
  if (e.beta != null) state.pose.pitch_deg = e.beta - 90;
  if (e.gamma != null) state.pose.roll_deg = e.gamma;
  updateSensorBadge();
}

function enableOrientation() {
  if (typeof DeviceOrientationEvent !== "undefined" &&
      typeof DeviceOrientationEvent.requestPermission === "function") {
    DeviceOrientationEvent.requestPermission().then((p) => {
      if (p === "granted") window.addEventListener("deviceorientation", onOrientation);
    }).catch(() => {});
  } else {
    window.addEventListener("deviceorientationabsolute", onOrientation);
    window.addEventListener("deviceorientation", onOrientation);
  }
}

function updateSensorBadge() {
  const el = $("sensor");
  if (!el) return;
  const g = state.pose.has_gps ? `GPS ±${state.pose.accuracy_m.toFixed(0)}m` : "GPS: —";
  el.innerHTML = `${g} · compass <b>${state.pose.heading_deg.toFixed(0)}°</b>`;
}

// ── camera ────────────────────────────────────────────────────────────
async function toggleCamera() {
  if (state.stream) {
    state.stream.getTracks().forEach((t) => t.stop());
    state.stream = null;
    $("btnStart").textContent = "Start camera";
    $("btnStart").classList.remove("on");
    if (state.sendTimer) { clearInterval(state.sendTimer); state.sendTimer = null; }
    setStatus("Camera stopped");
    return;
  }
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment", width: { ideal: 720 } },
      audio: false,
    });
    const video = $("video");
    video.srcObject = state.stream;
    $("btnStart").textContent = "Stop camera";
    $("btnStart").classList.add("on");
    video.onloadedmetadata = () => {
      const fov = parseFloat($("fov").value) || 60;
      state.intrinsics = makeIntrinsics(video.videoWidth, video.videoHeight, fov);
      resizeOverlay();
      setStatus(`Camera ${video.videoWidth}×${video.videoHeight} — hold a preset object in the reticle, then Calibrate.`, "ok");
      if (!state.sendTimer) state.sendTimer = setInterval(sendObservation, 700);
    };
  } catch (e) {
    setStatus("Camera error: " + e.message, "err");
  }
}

function makeIntrinsics(w, h, fov) {
  const f = (w / 2) / Math.tan((fov * Math.PI / 180) / 2);
  return { f_px: f, cx: w / 2, cy: h / 2, image_w: w, image_h: h, fov_deg: fov };
}

function resizeOverlay() {
  const ov = $("overlay");
  if (ov) { ov.width = ov.offsetWidth; ov.height = ov.offsetHeight; }
  const map = $("map");
  if (map) { map.width = map.offsetWidth; map.height = map.offsetHeight; drawMap(); }
}

// ── frame analysis ────────────────────────────────────────────────────
const work = document.createElement("canvas");
const wctx = work.getContext("2d", { willReadFrequently: true });

function analyzeFrame() {
  const video = $("video");
  if (!video || !video.videoWidth || !state.intrinsics) return null;
  const W = video.videoWidth, H = video.videoHeight;
  work.width = W; work.height = H;
  wctx.drawImage(video, 0, 0, W, H);
  let data;
  try { data = wctx.getImageData(0, 0, W, H).data; } catch (e) { return null; }

  const y0 = Math.floor(H * 0.32), y1 = Math.floor(H * 0.68);
  const rows = y1 - y0;
  const colLum = new Float64Array(W);
  const colGrad = new Float64Array(W);
  let specularCount = 0, lumSum = 0, lumSq = 0, total = 0;

  for (let y = y0; y < y1; y++) {
    for (let x = 0; x < W - 1; x++) {
      const i = (y * W + x) * 4, j = (y * W + x + 1) * 4;
      const r = data[i], g = data[i + 1], b = data[i + 2];
      const rn = data[j], gn = data[j + 1], bn = data[j + 2];
      const lum = (r + g + b) / 3, lumN = (rn + gn + bn) / 3;
      colLum[x] += lum; colGrad[x] += Math.abs(lum - lumN);
      if (lum > 235 && r > 220 && g > 220 && b > 220) specularCount++;
      lumSum += lum; lumSq += lum * lum; total++;
    }
  }
  for (let x = 0; x < W - 1; x++) { colLum[x] /= rows; colGrad[x] /= rows; }

  const mean = lumSum / Math.max(1, total);
  const variance = Math.max(0, lumSq / Math.max(1, total) - mean * mean);
  const contrast = Math.sqrt(variance) / 128;

  let sum = 0;
  for (let x = 0; x < W; x++) sum += colLum[x];
  const colMean = sum / W;
  let vsum = 0;
  for (let x = 0; x < W; x++) vsum += (colLum[x] - colMean) ** 2;
  const colStd = Math.sqrt(vsum / W);

  const left = Math.floor(W * 0.12), right = Math.floor(W * 0.88);
  const thresh = colMean - colStd * 1.15;
  let bestStart = -1, bestLen = 0, curStart = -1, curLen = 0;
  for (let x = left; x <= right; x++) {
    if (colLum[x] < thresh) {
      if (curStart < 0) curStart = x;
      curLen++;
      if (curLen > bestLen) { bestLen = curLen; bestStart = curStart; }
    } else { curStart = -1; curLen = 0; }
  }
  if (bestLen < 10) return null;
  const centerX = bestStart + bestLen / 2;

  const cx = Math.max(0, Math.min(W - 1, Math.round(centerX)));
  let top = -1, bottom = -1;
  for (let y = Math.floor(H * 0.1); y < Math.floor(H * 0.9); y++) {
    const i = (y * W + cx) * 4;
    const lum = (data[i] + data[i + 1] + data[i + 2]) / 3;
    if (lum < thresh) { if (top < 0) top = y; bottom = y; }
  }
  const pxHeight = (top >= 0 && bottom > top) ? (bottom - top) : bestLen * 0.45;
  const centerY = (top >= 0 && bottom > top) ? (top + bottom) / 2 : H * 0.5;

  let gsum = 0, gsq = 0, gcount = 0;
  for (let x = left; x <= right; x++) { gsum += colGrad[x]; gsq += colGrad[x] ** 2; gcount++; }
  const gradMean = gsum / Math.max(1, gcount);
  const gradVar = Math.max(0, gsq / Math.max(1, gcount) - gradMean * gradMean);
  const textureDensity = clamp01(gradMean / 40);
  const sharpness = clamp01(Math.sqrt(gradVar) / 50);

  let eL = 0, eR = 0;
  const band = (right - left) * 0.2;
  for (let x = left; x < left + band; x++) eL += colGrad[x];
  for (let x = right - band; x <= right; x++) eR += colGrad[x];
  const convergence = clamp01(1 - Math.min(eL, eR) / Math.max(eL, eR, 1e-6));

  return {
    width: bestLen, height: pxHeight, centerX, centerY,
    textureDensity, sharpness, convergence,
    specular: clamp01(specularCount / Math.max(1, total)),
    contrast: clamp01(contrast),
  };
}

// ── calibration ───────────────────────────────────────────────────────
function calibrate() {
  const det = analyzeFrame();
  if (!det) { setStatus("Calibrate: no object detected in the reticle.", "err"); return; }
  const K = parseFloat($("known").value);
  const calD = parseFloat($("caldist").value);
  if (!K || !calD) { setStatus("Calibrate: set known width + distance.", "err"); return; }
  state.fCal = (calD * 10 * det.width) / K;
  state.intrinsics.f_px = state.fCal;
  setStatus(`Calibrated: f = ${state.fCal.toFixed(1)} px (object ${det.width.toFixed(1)} px).`, "ok");
}

// ── observation → backend ─────────────────────────────────────────────
async function sendObservation() {
  if (!state.stream || !state.intrinsics) return;
  const det = analyzeFrame();
  state.detection = det;
  drawOverlay(det);
  if (!det) return;

  const payload = {
    object_px_width: det.width,
    object_px_height: det.height,
    center_x: det.centerX,
    center_y: det.centerY,
    intrinsics: state.intrinsics,
    pose: state.pose,
    known_width_mm: parseFloat($("known").value) || null,
    object_class: $("preset").value,
    shape_mode: $("shapemode").value,
    camera_height_m: parseFloat($("camh").value) || 1.5,
    texture_density: det.textureDensity,
    sharpness: det.sharpness,
    convergence: det.convergence,
    specular: det.specular,
    contrast: det.contrast,
    room: $("room").value || "unknown",
    level: 0.95,
  };

  try {
    const res = await fetch("/api/observe", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok || data.error) { setStatus(data.error || ("HTTP " + res.status), "err"); return; }
    state.lastResult = data;
    renderResult(data);
    if (data.accepted) refreshInventory();
  } catch (e) {
    setStatus("observe failed: " + e.message, "err");
  }
}

// ── rendering ─────────────────────────────────────────────────────────
function renderResult(d) {
  const est = d.estimate;
  $("depth").innerHTML = `${est.depth_m.toFixed(2)} <small>m</small>`;
  const half = (est.ci_high_m - est.ci_low_m) / 2;
  $("ci").innerHTML = `±${half.toFixed(2)} m (${(est.level * 100).toFixed(0)}%) `
    + `<span class="range">[${est.ci_low_m.toFixed(2)}, ${est.ci_high_m.toFixed(2)}]</span>`;
  const pct = Math.round(est.confidence * 100);
  $("confpct").textContent = pct + "%";
  const bar = $("confbar");
  bar.style.width = pct + "%";
  bar.style.background = pct > 70 ? "var(--acc)" : pct > 40 ? "var(--warn)" : "var(--bad)";

  $("tag").textContent = d.tag.label || d.tag.tag;
  const raw = d.dimensions_mm;
  const corr = d.corrected_dimensions_mm || raw;
  const fmtDims = (x) => `${x.w} × ${x.h}` + (x.d != null ? ` × ${x.d}` : "") + " mm";
  $("dims").textContent = fmtDims(raw);
  $("dimsCorr").textContent = fmtDims(corr);
  const cp = d.corrections ? d.corrections.correction_percent : 0;
  $("corrPct").textContent = `${cp.toFixed(2)} %`;
  $("shape").textContent = d.shape ? `${d.shape.shape} (aspect ${d.shape.aspect})` : "—";
  $("cues").textContent = est.cues.map((c) => `${c.name}:${c.depth_m.toFixed(2)}`).join("  ");

  const flagBox = $("flags");
  flagBox.innerHTML = "";
  if (d.illusions.rejected) {
    const r = document.createElement("span");
    r.className = "flag hard";
    r.textContent = "REJECTED — " + (d.illusions.reason || "illusion risk");
    flagBox.appendChild(r);
  }
  (d.illusions.flags || []).slice(0, 4).forEach((f) => {
    const el = document.createElement("span");
    el.className = "flag" + (f.severity > 0.7 ? " hard" : "");
    el.textContent = `${f.label} ${f.severity.toFixed(2)}`;
    flagBox.appendChild(el);
  });

  const g = d.geo;
  if (g) {
    $("lat").textContent = g.lat.toFixed(7);
    $("lon").textContent = g.lon.toFixed(7);
    $("alt").textContent = g.alt_m.toFixed(2) + " m";
    $("radius").textContent = "± " + g.radius_m.toFixed(2) + " m";
    $("bearing").textContent = `${g.bearing_deg.toFixed(0)}° / ${g.distance_m.toFixed(2)} m`;
    addMapPoint(g.east_m, g.north_m);
  } else {
    $("lat").textContent = "no GPS";
  }
  setStatus(d.accepted ? `accepted · ${d.tag.tag}` : "rejected by illusion filter", d.accepted ? "ok" : "err");
}

function drawOverlay(det) {
  const ov = $("overlay");
  if (!ov) return;
  const ctx = ov.getContext("2d");
  ctx.clearRect(0, 0, ov.width, ov.height);
  if (!det || !state.intrinsics) return;
  const sx = ov.width / state.intrinsics.image_w;
  const sy = ov.height / state.intrinsics.image_h;
  const px = det.centerX - det.width / 2;
  const py = det.centerY - det.height / 2;
  ctx.strokeStyle = "rgba(0,255,170,0.85)";
  ctx.lineWidth = 2;
  ctx.setLineDash([6, 4]);
  ctx.strokeRect(px * sx, py * sy, det.width * sx, det.height * sy);
  ctx.setLineDash([]);
  ctx.fillStyle = "rgba(0,255,170,0.9)";
  ctx.font = "11px ui-monospace, monospace";
  ctx.fillText(`${det.width.toFixed(0)}px`, px * sx + 3, Math.max(12, py * sy - 4));
}

// ── mini map ──────────────────────────────────────────────────────────
function addMapPoint(east, north) {
  if (east == null || north == null) return;
  state.mapPoints.push({ east, north });
  if (state.mapPoints.length > 600) state.mapPoints.shift();
  drawMap();
}

function drawMap() {
  const c = $("map");
  if (!c) return;
  const ctx = c.getContext("2d");
  ctx.clearRect(0, 0, c.width, c.height);
  const cx = c.width / 2, cy = c.height / 2;
  const pts = state.mapPoints;
  let maxR = 5;
  for (const p of pts) maxR = Math.max(maxR, Math.hypot(p.east, p.north));
  const scale = (Math.min(c.width, c.height) / 2 - 12) / maxR;

  ctx.strokeStyle = "#132038";
  ctx.lineWidth = 1;
  for (let r = 1; r <= 4; r++) {
    ctx.beginPath();
    ctx.arc(cx, cy, (Math.min(c.width, c.height) / 2 - 12) * r / 4, 0, Math.PI * 2);
    ctx.stroke();
  }
  ctx.beginPath();
  ctx.moveTo(cx, 6); ctx.lineTo(cx, c.height - 6);
  ctx.moveTo(6, cy); ctx.lineTo(c.width - 6, cy);
  ctx.strokeStyle = "#0e1930";
  ctx.stroke();

  for (const p of pts) {
    const x = cx + p.east * scale;
    const y = cy - p.north * scale;
    ctx.fillStyle = "rgba(0,255,170,0.75)";
    ctx.fillRect(x - 1.5, y - 1.5, 3, 3);
  }

  const h = state.pose.heading_deg * Math.PI / 180;
  ctx.strokeStyle = "rgba(89,167,255,0.95)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.lineTo(cx + Math.sin(h) * 18, cy - Math.cos(h) * 18);
  ctx.stroke();

  const info = $("mapinfo");
  if (info) info.textContent = `${pts.length} points · scale ${maxR.toFixed(1)} m`;
}

// ── ground scan ───────────────────────────────────────────────────────
async function scanGround() {
  if (!state.intrinsics) { setStatus("Scan: start the camera first.", "err"); return; }
  if (!state.pose.has_gps) { setStatus("Scan needs GPS — enable location.", "err"); return; }
  setStatus("Scanning ground plane…");
  try {
    const res = await fetch("/api/scan", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        intrinsics: state.intrinsics, pose: state.pose,
        camera_height_m: parseFloat($("camh").value) || 1.5,
        room: $("room").value || "unknown",
        cols: 28, rows: 18, store: true,
      }),
    });
    const data = await res.json();
    state.mapPoints = data.points.map((p) => ({ east: p.east_m, north: p.north_m }));
    drawMap();
    setStatus(`Ground scan: ${data.count} points mapped.`, "ok");
  } catch (e) {
    setStatus("scan failed: " + e.message, "err");
  }
}

// ── fit model ─────────────────────────────────────────────────────────
async function fitModel() {
  const det = state.detection, last = state.lastResult;
  if (!det || !last) { setStatus("Fit: take a measurement first.", "err"); return; }
  const corners = [
    [det.centerX - det.width / 2, det.centerY - det.height / 2],
    [det.centerX + det.width / 2, det.centerY - det.height / 2],
    [det.centerX + det.width / 2, det.centerY + det.height / 2],
    [det.centerX - det.width / 2, det.centerY + det.height / 2],
  ];
  try {
    const res = await fetch("/api/fit", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        corners_px: corners, intrinsics: state.intrinsics,
        depth_m: last.estimate.depth_m, kind: $("shapemode").value,
      }),
    });
    const d = await res.json();
    if (!res.ok || d.error) { setStatus(d.error || "fit failed", "err"); return; }
    setStatus(`Fit ${d.kind}: ${d.w_mm} × ${d.h_mm}` +
      (d.d_mm != null ? ` × ${d.d_mm}` : "") + ` mm · rms ${d.rms_px} px`, "ok");
  } catch (e) { setStatus("fit failed: " + e.message, "err"); }
}

// ── inventory ─────────────────────────────────────────────────────────
async function refreshInventory() {
  try {
    const room = encodeURIComponent($("room").value || "");
    const res = await fetch(`/api/inventory?room=${room}`);
    renderInventory(await res.json());
  } catch (e) { /* non-fatal */ }
}

function renderInventory(data) {
  const box = $("inv");
  if (!box) return;
  if (!data.items || !data.items.length) {
    box.innerHTML = '<div class="muted">empty — seed a demo or find an item</div>';
    return;
  }
  box.innerHTML = data.items.map((it) => {
    const dims = `${fmt(it.w_mm)} × ${fmt(it.h_mm)}` + (it.d_mm != null ? ` × ${fmt(it.d_mm)}` : "") + " mm";
    return `<div class="row">
      <div><div class="t">${it.label || it.tag} ×${it.count}</div>
      <div class="g">${dims} · conf ${(it.confidence * 100).toFixed(0)}%</div></div>
      <div class="g" style="text-align:right">${Number(it.lat).toFixed(6)},<br>${Number(it.lon).toFixed(6)}</div>
    </div>`;
  }).join("");
}

function fmt(v) { return v == null ? "?" : Number(v).toFixed(1); }

// ── illusions list ────────────────────────────────────────────────────
async function loadIllusionList() {
  try {
    const res = await fetch("/api/illusions");
    const data = await res.json();
    const box = $("illusions");
    if (!box) return;
    box.innerHTML = data.illusions.map((i) => `
      <div class="ill">
        <div><div class="n">${i.name}</div><div class="k">${i.kind} · ${i.formula}</div></div>
        <div class="s ${i.severity > 0 ? "" : "zero"}">${(i.severity * 100).toFixed(0)}%</div>
      </div>`).join("")
      + `<div class="muted" style="margin-top:6px">${data.note}</div>`;
  } catch (e) { /* non-fatal */ }
}

// ── demo / premade / benchmark / reset ────────────────────────────────
async function seedDemo() {
  try {
    const res = await fetch("/api/demo", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ room: $("room").value || "basement B",
                             intrinsics: state.intrinsics || makeIntrinsics(720, 1280, 60) }),
    });
    const data = await res.json();
    if (data.inventory) {
      renderInventory({ items: data.inventory });
      state.mapPoints = data.inventory.map((it, i) => ({ east: Math.cos(i) * 2, north: Math.sin(i) * 3 }));
      drawMap();
    }
    setStatus(`Seeded ${data.seeded} demo observations into "${$("room").value}".`, "ok");
  } catch (e) { setStatus("demo failed: " + e.message, "err"); }
}

async function runPremade() {
  try {
    const res = await fetch("/api/premade", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ tag: $("preset").value, depth_m: 2.0,
                             room: $("room").value || "premade",
                             intrinsics: state.intrinsics || makeIntrinsics(720, 1280, 60) }),
    });
    const d = await res.json();
    if (!res.ok || d.error) { setStatus(d.error || "premade failed", "err"); return; }
    renderResult(d.result);
    setStatus(`Premade ${d.tag}: ${d.result.corrected_dimensions_mm.w}×${d.result.corrected_dimensions_mm.h} mm`
      + ` vs truth ${d.truth_mm.w}×${d.truth_mm.h} mm (${d.error_pct.w}% / ${d.error_pct.h}%)`, "ok");
    refreshInventory();
  } catch (e) { setStatus("premade failed: " + e.message, "err"); }
}

async function runBenchmark() {
  try {
    const res = await fetch("/api/benchmark", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ depth_m: 2.0, noise_frac: 0.02, level: 0.95 }),
    });
    const d = await res.json();
    if (!res.ok || d.error) { setStatus(d.error || "benchmark failed", "err"); return; }
    $("bench").textContent = d.mean_mape_pct.toFixed(2) + " %";
    $("coverage").textContent = `w ${d.coverage_w_pct}% / h ${d.coverage_h_pct}%`;
    $("maxerr").textContent = d.max_error_pct.toFixed(2) + " %";
    $("benchdetail").textContent = d.results
      .map((r) => `${r.tag}: ${r.error_pct.w}% / ${r.error_pct.h}%`).join(" · ");
    setStatus(`Benchmark MAPE ${d.mean_mape_pct}% — the number to minimise.`, "ok");
  } catch (e) { setStatus("benchmark failed: " + e.message, "err"); }
}

async function resetAll() {
  try {
    await fetch("/api/reset", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
    state.mapPoints = [];
    drawMap();
    renderInventory({ items: [] });
    setStatus("Map + inventory cleared.", "ok");
  } catch (e) { setStatus("reset failed: " + e.message, "err"); }
}

function setStatus(msg, kind) {
  const el = $("status");
  if (!el) return;
  el.textContent = msg;
  el.className = kind === "ok" ? "ok" : kind === "err" ? "err" : "";
}

// ── wire up ───────────────────────────────────────────────────────────
function bindButtons() {
  $("btnStart").onclick = toggleCamera;
  $("btnCal").onclick = calibrate;
  $("btnScan").onclick = scanGround;
  $("btnFit").onclick = fitModel;
  $("btnPremade").onclick = runPremade;
  $("btnDemo").onclick = seedDemo;
  $("btnBench").onclick = runBenchmark;
  $("btnReset").onclick = resetAll;
}

async function init() {
  await loadPages();
  fillPresets();
  bindButtons();
  enableOrientation();
  watchGPS();
  resizeOverlay();
  window.addEventListener("resize", resizeOverlay);
  drawMap();
  setInterval(refreshInventory, 5000);
  refreshInventory();
  loadIllusionList();
  setStatus("Ready — start the camera, then Calibrate. Or press Premade object / Benchmark.", "ok");
}

document.addEventListener("DOMContentLoaded", init);
