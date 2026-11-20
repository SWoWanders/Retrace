import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFExporter } from 'three/addons/exporters/GLTFExporter.js';

const C = window.RetraceCore;
const TF_URL = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3';
const DEPTH_MODEL = 'onnx-community/depth-anything-v2-small';
const SEG_MODEL = 'Xenova/segformer-b0-finetuned-ade-512-512';
const EPS = 0.06, WORK_LONG = 360, DEPTH_LONG = 640, SCENE_SCALE = 10, SHOT_GAP = SCENE_SCALE * 1.5;
const $ = id => document.getElementById(id);
const S = { frames: [], segs: [], path: [], pipe: null, segPipe: null, RawImage: null, device: null };
window.__retrace = S; // handy for debugging from the console

// ---------- toast ----------
let toastT;
function toast(msg, kind = '', ms = 3200) {
  const t = $('toast'); t.textContent = msg; t.className = 'show ' + kind;
  clearTimeout(toastT); toastT = setTimeout(() => t.className = kind, ms);
}
window.addEventListener('error', e => toast('Error: ' + (e.message || e), 'err', 6000));
window.addEventListener('unhandledrejection', e => toast('Error: ' + (e.reason?.message || e.reason), 'err', 6000));

// ---------- setup screen ----------
const vid = $('vid');
for (const [id, fmt] of [['fps', v => v], ['maxf', v => v], ['fov', v => v + '°']]) {
  const el = $(id), out = $(id + 'V'); const upd = () => out.textContent = fmt(el.value); el.oninput = upd; upd();
}
$('file').onchange = e => {
  const f = e.target.files[0]; if (!f) return;
  if (S.url) URL.revokeObjectURL(S.url);
  S.file = f; S.url = URL.createObjectURL(f); S.fileName = f.name.replace(/\.[^.]+$/, '');
  vid.src = S.url; vid.classList.remove('hidden'); vid.controls = true;
  $('go').disabled = true;
  toast('Loading video…');
};
vid.onloadedmetadata = () => {
  if (!isFinite(vid.duration) || !vid.videoWidth) { toast("This video's length or size can't be read. Try another file.", 'err'); return; }
  $('tstart').value = 0; $('tend').value = vid.duration.toFixed(1);
  $('go').disabled = false;
  toast(`Video ready: ${vid.videoWidth}×${vid.videoHeight}, ${vid.duration.toFixed(1)} s`);
};
vid.onerror = () => { if (vid.src) toast("This browser can't decode that video. Try an MP4 (H.264).", 'err', 6000); };
$('go').onclick = () => run().catch(err => { console.error(err); toast('Failed: ' + err.message, 'err', 8000); show('setup'); vid.controls = true; releaseWake(); });

function show(which) { for (const id of ['setup', 'proc', 'viewer']) $(id).classList.toggle('hidden', id !== which); }

// ---------- models (free, from Hugging Face, run on-device) ----------
const DEPTH_MODELS = { small: 'onnx-community/depth-anything-v2-small', base: 'onnx-community/depth-anything-v2-base' };
async function getModels(pref, wantSeg, dmodel) {
  $('pstat').textContent = 'Loading Transformers.js…';
  const T = S.T || (S.T = await import(TF_URL));
  T.env.allowLocalModels = false;
  S.RawImage = T.RawImage;
  let gpu = false, f16 = false;
  if (pref !== 'wasm' && navigator.gpu) { try { const ad = await navigator.gpu.requestAdapter(); gpu = !!ad; f16 = !!ad?.features?.has('shader-f16'); } catch { gpu = false; } }
  const files = {};
  const progress_callback = label => d => {
    if (d.status === 'progress' && d.total) {
      files[d.file] = [d.loaded, d.total];
      let l = 0, t = 0; for (const k in files) { l += files[k][0]; t += files[k][1]; }
      $('pstat').textContent = `Downloading ${label}… ${(l / 1e6).toFixed(1)} / ${(t / 1e6).toFixed(1)} MB`;
      $('pbar').style.width = (100 * l / t).toFixed(1) + '%';
    }
  };
  const load = async (task, model, label, gpuDtype) => {
    if (gpu) {
      try { return { p: await T.pipeline(task, model, { device: 'webgpu', dtype: gpuDtype || 'fp32', progress_callback: progress_callback(label) }), dev: 'GPU' }; }
      catch (e) { console.warn('WebGPU failed for', model, e); }
    }
    return { p: await T.pipeline(task, model, { device: 'wasm', dtype: 'q8', progress_callback: progress_callback(label) }), dev: 'CPU' };
  };
  const key = pref + '|' + dmodel;
  if (!S.pipe || S.pipeKey !== key) {
    S.pipe = null;
    const big = dmodel === 'base';
    const r = await load('depth-estimation', DEPTH_MODELS[dmodel] || DEPTH_MODELS.small, big ? 'high-detail depth model' : 'depth model', big ? (f16 ? 'fp16' : 'q8') : 'fp32');
    S.pipe = r.p; S.device = r.dev; S.pipeKey = key;
    if (gpu && r.dev === 'CPU') toast('GPU not usable, so using CPU instead (slower)', 'warn');
  }
  if (wantSeg && (!S.segPipe || S.segPref !== pref)) {
    try { S.segPipe = (await load('image-segmentation', SEG_MODEL, 'people/vehicle model')).p; S.segPref = pref; }
    catch (e) { console.warn(e); S.segPipe = null; toast("Couldn't load the people/vehicle model, so continuing without it", 'warn', 5000); }
  }
}

// ---------- helpers ----------
function seekTo(t) {
  return new Promise(res => {
    let done = false;
    const fin = () => { if (done) return; done = true; vid.removeEventListener('seeked', fin); res(); };
    vid.addEventListener('seeked', fin);
    vid.currentTime = t;
    setTimeout(fin, 5000);
  });
}
let wake = null;
async function holdWake() { try { wake = await navigator.wakeLock?.request('screen'); } catch { } }
function releaseWake() { try { wake?.release(); } catch { } wake = null; }
const I3 = () => C.I3();

// ---------- processing ----------
let cancelFlag = false;
$('cancel').onclick = () => { cancelFlag = true; $('cancel').disabled = true; $('pstat').textContent = 'Stopping after this frame…'; };

async function run() {
  if (!vid.src || !vid.videoWidth) { toast('Pick a video first', 'err'); return; }
  const fps = +$('fps').value, maxF = +$('maxf').value, fov = +$('fov').value;
  const wantSeg = $('rmMovers').checked, useCuts = $('cuts').checked;
  let t0 = Math.max(0, +$('tstart').value || 0), t1 = +$('tend').value || vid.duration;
  t1 = Math.min(t1, vid.duration - 0.05);
  if (t1 - t0 < 0.5) { toast('Clip is too short. Check Start/End.', 'err'); return; }
  const times = [];
  for (let t = t0; t <= t1 + 1e-6 && times.length < maxF; t += 1 / fps) times.push(+t.toFixed(4));
  if (times.length >= maxF && times[times.length - 1] < t1 - 1 / fps)
    toast(`Using the first ${(times.length / fps).toFixed(1)} s (max frames). Raise Max frames or lower fps for more.`, 'warn', 5000);

  cancelFlag = false; $('cancel').disabled = false;
  vid.pause(); vid.controls = false;
  show('proc'); holdWake();
  $('pbar').style.width = '0%'; $('pdet').textContent = '';

  const vw = vid.videoWidth, vh = vid.videoHeight;
  const sc = WORK_LONG / Math.max(vw, vh), W = Math.round(vw * sc / 2) * 2, H = Math.round(vh * sc / 2) * 2;
  const dsc = DEPTH_LONG / Math.max(vw, vh), dW = Math.round(vw * dsc), dH = Math.round(vh * dsc);
  const f = (Math.max(W, H) / 2) / Math.tan(fov * Math.PI / 360);
  const K = { fx: f, fy: f, cx: W / 2, cy: H / 2 };

  await getModels($('dq').value, wantSeg, $('dmodel').value);
  const segPipe = wantSeg ? S.segPipe : null;

  const wc = document.createElement('canvas'); wc.width = W; wc.height = H;
  const wctx = wc.getContext('2d', { willReadFrequently: true });
  const dc = document.createElement('canvas'); dc.width = dW; dc.height = dH;
  const dctx = dc.getContext('2d', { willReadFrequently: true });
  const pc = $('pcan'); pc.width = W * 2; pc.height = H;
  const pctx = pc.getContext('2d');

  const frames = [];
  let prev = null, prevHist = null, lastRel = { R: I3(), t: [0, 0, 0] }, fails = 0, shot = 0, segWarned = false, sky = null;
  const tStart = performance.now();
  for (let i = 0; i < times.length; i++) {
    if (cancelFlag) break;
    await seekTo(times[i]);
    wctx.drawImage(vid, 0, 0, W, H);
    const wimg = wctx.getImageData(0, 0, W, H), rgb = C.rgbaToRgb(wimg.data);
    dctx.drawImage(vid, 0, 0, dW, dH);
    const drgb = C.rgbaToRgb(dctx.getImageData(0, 0, dW, dH).data);
    const img = new S.RawImage(drgb, dW, dH, 3);

    const out = await S.pipe(img);
    const pd = out.predicted_depth, dims = pd.dims, ph = dims[dims.length - 2], pw = dims[dims.length - 1];
    const dn = C.normalizeDisparity(C.resample(pd.data, pw, ph, W, H));
    let mask = null;
    if (segPipe) {
      try { mask = C.maskFromSegments(await segPipe(img), W, H, 2); }
      catch (e) { if (!segWarned) { segWarned = true; console.warn(e); toast('People/vehicle detection failed on a frame, so skipping it there', 'warn'); } }
    }
    const pyr = C.buildPyramid(C.rgbToGray(rgb, W, H), W, H, 4);
    const hist = C.colorHist(rgb);

    let R, t, s, r = null, info = '', newShot = !prev;
    if (prev) {
      r = C.trackStep(prev, { pyr, dn, mask }, K, lastRel, EPS, { seed: i * 7 + 1 });
      const hd = C.histDiff(prevHist, hist);
      if (useCuts && C.isCut(hd, r)) { newShot = true; shot++; r = null; info = `✂ New shot ${shot + 1} starts here`; }
    }
    if (newShot) {
      const zs = []; for (let k = 0; k < dn.length; k += 11) if (dn[k] > 0.05 && !(mask && mask[k])) zs.push(1 / (dn[k] + EPS));
      s = SCENE_SCALE / (C.median(zs) || 1); R = I3(); t = [0, 0, 0]; lastRel = { R: I3(), t: [0, 0, 0] };
      if (!sky) { // sky colour = average of far pixels in the top half of the first frame
        let sr = 0, sg = 0, sb = 0, n = 0;
        for (let k = 0; k < W * H / 2; k++) if (dn[k] < 0.02) { sr += rgb[3 * k]; sg += rgb[3 * k + 1]; sb += rgb[3 * k + 2]; n++; }
        sky = n > 200 ? [sr / n, sg / n, sb / n] : [150, 180, 210];
      }
    } else {
      if (r.ok) lastRel = r.rel; else fails++;
      R = r.R; t = r.t; s = r.s;
      info = `${r.nTracks} tracked points · ${r.nIn} agree${r.ok ? '' : ' · ⚠ lost track, guessing motion'}`;
    }
    frames.push({ time: times[i], rgb, dn: Uint16Array.from(dn, v => Math.round(v * 65535)), s, R, t, seg: shot, mask, ok: !r || r.ok });
    prev = { pyr, dn, s, R, t, mask }; prevHist = hist;

    // preview
    if (mask) for (let k = 0; k < W * H; k++) if (mask[k]) { const d = wimg.data; d[4 * k] = (d[4 * k] + 255) >> 1; d[4 * k + 1] >>= 1; d[4 * k + 2] >>= 1; }
    pctx.putImageData(wimg, 0, 0);
    if (r) {
      pctx.lineWidth = 1;
      for (let k = 0; k < r.src.length / 2; k++) {
        pctx.strokeStyle = r.inliers[k] ? '#38e1c6' : '#ff5f6d';
        pctx.beginPath(); pctx.moveTo(r.src[2 * k], r.src[2 * k + 1]); pctx.lineTo(r.dst[2 * k], r.dst[2 * k + 1]); pctx.stroke();
      }
    }
    const dimg = pctx.createImageData(W, H);
    for (let k = 0; k < W * H; k++) { const v = Math.sqrt(dn[k]) * 255; dimg.data[4 * k] = v * 0.6; dimg.data[4 * k + 1] = v; dimg.data[4 * k + 2] = v * 0.9 + 20; dimg.data[4 * k + 3] = 255; }
    pctx.putImageData(dimg, W, 0);

    const el = (performance.now() - tStart) / 1000, per = el / (i + 1), left = per * (times.length - i - 1);
    $('pbar').style.width = (100 * (i + 1) / times.length).toFixed(1) + '%';
    $('pstat').textContent = `Frame ${i + 1} / ${times.length} · ${S.device} · ${per.toFixed(1)} s/frame · ~${Math.ceil(left)} s left`;
    $('pdet').textContent = (info || 'First frame sets the scale') + (shot ? ` · shot ${shot + 1}` : '');
  }
  releaseWake();
  if (frames.length < 2) { toast('Not enough frames were processed.', 'err'); show('setup'); vid.controls = true; return; }

  // ---- shots: split, detect still cameras, build diorama backgrounds ----
  $('pstat').textContent = 'Sorting out shots…';
  await new Promise(r => setTimeout(r, 30));
  const segs = [];
  frames.forEach((fr, i) => { if (!segs.length || segs[segs.length - 1].id !== fr.seg) segs.push({ id: fr.seg, start: i, end: i + 1 }); else segs[segs.length - 1].end = i + 1; });
  let moving = 0, movingFails = 0;
  for (const sg of segs) {
    const list = frames.slice(sg.start, sg.end), lost = list.slice(1).filter(f => !f.ok).length;
    // a shot we can't track (dancing, animation, screen recording, chaos…) becomes a 3D video "stage".
    // Check this first: lost frames fall back to "no motion", which would otherwise look like a tripod.
    sg.stage = list.length < 3 || lost > 0.5 * (list.length - 1);
    sg.still = !sg.stage && C.isStill(list, SCENE_SCALE);
    if (sg.still || sg.stage) for (const fr of list) { fr.R = I3(); fr.t = [0, 0, 0]; } // lock the camera
    if (sg.still) sg.bg = C.medianBackground(list, W, H, 41);
    if (!sg.still && !sg.stage) { moving += list.length; movingFails += lost; }
    sg.align = null; sg.offset = [0, 0, 0]; sg.live = new Map();
  }
  if (moving && movingFails > moving * 0.25) toast(`Tracking was lost on ${movingFails} frames, so the path may wobble. Try a slower walk or a different FOV.`, 'warn', 7000);

  Object.assign(S, { aspect: vw / vh, frames, segs, W, H, K, fps, t0: times[0], vfov: 2 * Math.atan((H / 2) / K.fy) * 180 / Math.PI, sky, hasMasks: frames.some(f => f.mask) });
  enterViewer();
  const still = segs.filter(s => s.still).length, stage = segs.filter(s => s.stage).length;
  if (segs.length > 1 || still || stage) toast(`Found ${segs.length} shot${segs.length > 1 ? 's' : ''}${still ? ` · ${still} still → diorama` : ''}${stage ? ` · ${stage} untrackable → 3D video stage` : ''}.${segs.length > 1 ? ' Tap the shot buttons to jump.' : ''}`, '', 6500);
}

// ---------- viewer ----------
const MUX_URL = 'https://cdn.jsdelivr.net/npm/mp4-muxer@5.2.2/+esm';
let renderer, scene, camera, controls, grid, pathGroup, marker, world, liveP, liveM, clock = performance.now();
let mode = 'ride', style = 'blocks', playing = false, idx = 0, recording = null, exporting = null, lastSeg = -1, liveKey = '', snapChase = true;
const chaseLook = new THREE.Vector3();
const STYLE_LABEL = { blocks: 'Blocks', tiles: 'Tiles', spheres: 'Spheres', points: 'Points', wire: 'Wire', sheets: 'Sheets', video3d: '3D Video' };
const STYLE_TIP = {
  blocks: 'Blocks: cubes, smaller near the camera and bigger far away',
  tiles: 'Tiles: flat squares turned to follow each surface',
  spheres: 'Spheres: low-poly gems',
  points: 'Points: the raw 3D samples',
  wire: 'Wire: joined points, a wireframe of the depth',
  sheets: 'Sheets: continuous surfaces from key frames',
  video3d: '3D Video: every frame as a moving 3D surface',
};

for (const [k, v] of Object.entries(C.PALETTES)) { const o = document.createElement('option'); o.value = k; o.textContent = v.name; $('palSel').appendChild(o); }

function initThree() {
  if (renderer) return;
  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  $('stage').appendChild(renderer.domElement);
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(60, 1, 0.05, 1000);
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true; controls.enabled = false;
  scene.add(new THREE.HemisphereLight(0xffffff, 0x445566, 1.6));
  const sun = new THREE.DirectionalLight(0xffffff, 1.6); sun.position.set(30, 60, 20); scene.add(sun);
  grid = new THREE.GridHelper(400, 400, 0x38e1c6, 0x1d2a3d); grid.material.transparent = true; grid.material.opacity = 0.35; grid.visible = false; scene.add(grid);
  pathGroup = new THREE.Group(); scene.add(pathGroup);
  marker = new THREE.Group();
  const cone = new THREE.Mesh(new THREE.ConeGeometry(0.35, 0.8, 4), new THREE.MeshBasicMaterial({ color: 0xffb347, wireframe: true }));
  cone.rotation.x = -Math.PI / 2; cone.position.z = -0.4; marker.add(cone);
  marker.add(new THREE.Mesh(new THREE.SphereGeometry(0.16, 12, 8), new THREE.MeshBasicMaterial({ color: 0xffb347 })));
  scene.add(marker);
  liveP = new THREE.Points(new THREE.BufferGeometry(), new THREE.PointsMaterial({ size: 0.12, vertexColors: true, sizeAttenuation: true }));
  liveM = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide }));
  liveP.name = 'retrace-live-points'; liveM.name = 'retrace-live-surface';
  scene.add(liveP, liveM); S.live = liveP; S.liveM = liveM;
  addEventListener('resize', resize); resize();
  renderer.setAnimationLoop(tick);
}
function resize() {
  if (!renderer || exporting) return;
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setSize(innerWidth, innerHeight, false); camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix();
}

function enterViewer() {
  show('viewer'); initThree();
  $('inset').appendChild(vid); vid.classList.remove('hidden'); vid.controls = false;
  vid.muted = true; $('bSound').textContent = '🔇';
  $('scrub').max = S.frames.length - 1; idx = 0; $('scrub').value = 0; lastSeg = -1;
  $('useMask').disabled = !S.hasMasks;
  if (S.segs.every(s => s.stage)) setStyleUI('video3d'); // nothing trackable: lead with the moving 3D surface
  buildShotChips();
  buildScene();
  setMode('ride');
  seekVideo();
  toast('Done! Press ▶ to retrace the video. Try the styles under the controls.', '', 4500);
}

function buildShotChips() {
  const box = $('shots'); box.innerHTML = '';
  box.classList.toggle('hidden', S.segs.length < 2);
  S.segs.forEach((sg, k) => {
    const b = document.createElement('button');
    const tag = sg.still ? ' ◉' : sg.stage ? ' ◎' : '';
    b.textContent = `Shot ${k + 1}${tag}`; b.title = sg.still ? 'Still camera: diorama' : sg.stage ? 'Untrackable: 3D video stage' : 'Moving camera';
    b.onclick = () => { idx = sg.start; $('scrub').value = idx; seekVideo(); if (playing) vid.play().catch(() => { }); toast(`Shot ${k + 1}${sg.still ? ': diorama (still camera)' : sg.stage ? ': 3D video stage' : ''}`); };
    box.appendChild(b);
  });
}

function readParams() {
  const p = {
    voxel: +$('vox').value, far: +$('far').value, minHits: +$('hits').value, levels: +$('pal').value, stride: +$('str').value,
    eps: EPS, sceneScale: SCENE_SCALE, useMask: $('useMask').checked, palette: $('palSel').value, adaptive: $('adaptive').checked
  };
  $('voxV').textContent = p.adaptive ? '×' + (p.voxel / 0.24).toFixed(2) : p.voxel.toFixed(2);
  $('farV').textContent = p.far.toFixed(2); $('hitsV').textContent = p.minHits;
  $('palV').textContent = p.palette !== 'original' ? 'n/a' : p.levels ? p.levels + ' levels' : 'off';
  $('pal').disabled = p.palette !== 'original';
  $('strV').textContent = p.stride; $('smV').textContent = $('sm').value;
  return p;
}

function disposeTree(o) { o.traverse?.(c => { c.geometry?.dispose(); c.material?.dispose?.(); }); }

function xRange(pos, fallback) { // robust x extent of a point set
  const n = pos.length / 3; if (n < 20) return fallback;
  const step = Math.max(1, Math.floor(n / 4000)), xs = [];
  for (let k = 0; k < n; k += step) xs.push(pos[3 * k]);
  xs.sort((a, b) => a - b);
  return [xs[Math.floor(xs.length * 0.02)], xs[Math.floor(xs.length * 0.98)]];
}
const shiftX = (arr, ox) => { if (arr && ox) for (let i = 0; i < arr.length; i += 3) arr[i] += ox; };
const litMat = (opts = {}) => $('lit').checked ? new THREE.MeshLambertMaterial(opts) : new THREE.MeshBasicMaterial(opts);

function instanced(cells, geo, mat, scaleK, orient) {
  const n = cells.count, im = new THREE.InstancedMesh(geo, mat, Math.max(1, n));
  im.count = n;
  const m = new THREE.Matrix4(), q = new THREE.Quaternion(), pp = new THREE.Vector3(), sc = new THREE.Vector3(), col = new THREE.Color(), zAxis = new THREE.Vector3(0, 0, 1), nv = new THREE.Vector3();
  for (let i = 0; i < n; i++) {
    pp.set(cells.center[3 * i], cells.center[3 * i + 1], cells.center[3 * i + 2]);
    const s = cells.size[i] * scaleK;
    if (orient && cells.normal) { nv.set(cells.normal[3 * i], cells.normal[3 * i + 1], cells.normal[3 * i + 2]); q.setFromUnitVectors(zAxis, nv); sc.set(s, s, 1); }
    else { q.set(0, 0, 0, 1); sc.set(s, s, s); }
    im.setMatrixAt(i, m.compose(pp, q, sc));
    im.setColorAt(i, col.setRGB(cells.color[3 * i], cells.color[3 * i + 1], cells.color[3 * i + 2]));
  }
  im.instanceMatrix.needsUpdate = true; if (im.instanceColor) im.instanceColor.needsUpdate = true;
  return im;
}
function sheetObject(g) {
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(g.pos, 3));
  geo.setAttribute('color', new THREE.BufferAttribute(g.col, 3));
  geo.setIndex(new THREE.BufferAttribute(g.index, 1));
  return g.wire ? new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ vertexColors: true }))
    : new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide }));
}

function buildScene() {
  const t0 = performance.now();
  const p = readParams();
  S.colorize = C.makeColorizer(p.palette, SCENE_SCALE, p.levels);
  if (world) { scene.remove(world); disposeTree(world); }
  world = new THREE.Group(); world.name = 'retrace-' + style;
  let cursorX = 0, pieces = 0, planeMissing = 0;
  S.segs.forEach((sg, k) => {
    const list = sg.still ? [sg.bg] : sg.stage ? [S.frames[sg.start]] : S.frames.slice(sg.start, sg.end);
    const w = C.buildWorld(list, S.W, S.H, S.K, { ...p, normals: style === 'tiles', blocks: style === 'blocks' }, sg.align);
    sg.align = w.align;
    if (!w.align.planeFound && !sg.stage) planeMissing++;
    // lay shots out side by side along +X so they never overlap
    const pathX = w.path.map(q => q.p[0]);
    const [lo, hi] = xRange(w.points.pos, [Math.min(...pathX) - 2, Math.max(...pathX) + 2]);
    const ox = k === 0 ? 0 : cursorX - Math.min(lo, ...pathX);
    cursorX = ox + Math.max(hi, ...pathX) + SHOT_GAP;
    sg.offset = [ox, 0, 0];
    shiftX(w.points.pos, ox); shiftX(w.cells.center, ox); shiftX(w.blocks?.position, ox);
    const raw = (sg.still || sg.stage) ? Array.from({ length: sg.end - sg.start }, () => w.path[0]) : w.path;
    sg.rawPath = raw.map(q => ({ p: [q.p[0] + ox, q.p[1], q.p[2]], q: q.q }));
    sg.live.clear();
    let obj = null;
    if (sg.stage || style === 'video3d') obj = null; // the live surface is the whole show
    else if (style === 'blocks') {
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(w.blocks.position, 3));
      g.setAttribute('normal', new THREE.BufferAttribute(w.blocks.normal, 3));
      g.setAttribute('color', new THREE.BufferAttribute(w.blocks.color, 3));
      g.setIndex(new THREE.BufferAttribute(w.blocks.index, 1));
      obj = new THREE.Mesh(g, litMat({ vertexColors: true })); pieces += w.cells.count;
    } else if (style === 'tiles') {
      obj = instanced(w.cells, new THREE.PlaneGeometry(1, 1), litMat({ side: THREE.DoubleSide }), 1.3, true); pieces += w.cells.count;
    } else if (style === 'spheres') {
      obj = instanced(w.cells, new THREE.IcosahedronGeometry(0.5, 0), $('lit').checked ? new THREE.MeshStandardMaterial({ flatShading: true, roughness: 0.75 }) : new THREE.MeshBasicMaterial(), 1.15, false); pieces += w.cells.count;
    } else if (style === 'points') {
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(w.points.pos, 3));
      g.setAttribute('color', new THREE.BufferAttribute(w.points.col, 3));
      obj = new THREE.Points(g, new THREE.PointsMaterial({ size: Math.max(0.03, p.voxel * 0.5), vertexColors: true, sizeAttenuation: true })); pieces += w.points.count;
    } else { // wire / sheets: depth surfaces from key frames
      const wire = style === 'wire';
      const g = C.sheetGeometry(list, S.W, S.H, S.K, sg.align, sg.offset, { step: wire ? 4 : 3, stride: Math.max(1, p.stride * (wire ? 2 : 3)), wire, far: p.far, eps: EPS, useMask: p.useMask, colorize: S.colorize });
      obj = sheetObject(g); pieces += g.index.length / (wire ? 2 : 3);
    }
    if (obj) { obj.name = `shot-${k + 1}${sg.still ? '-diorama' : ''}`; world.add(obj); }
  });
  scene.add(world); S.world = world;
  if (planeMissing && !S.warnedPlane) { S.warnedPlane = true; toast(`Couldn't find a clear ground plane in ${planeMissing} shot(s), so the level may be off.`, 'warn', 5000); }
  applySmoothing();
  const sky = S.colorize.sky(...S.sky);
  const skyC = new THREE.Color().setRGB(...sky.map(v => v / 255), THREE.SRGBColorSpace);
  scene.background = skyC;
  scene.fog = $('fogOn').checked ? new THREE.Fog(skyC, SCENE_SCALE * 1.5, SCENE_SCALE * 7) : null;
  grid.visible = $('showGrid').checked && mode !== 'ride';
  liveP.material.size = Math.max(0.03, p.voxel * 0.45);
  liveKey = '';
  const still = S.segs.filter(s => s.still).length, stage = S.segs.filter(s => s.stage).length;
  const unit = { blocks: 'blocks', tiles: 'tiles', spheres: 'spheres', points: 'points', wire: 'lines', sheets: 'triangles', video3d: '' }[style];
  $('stats').textContent = `${S.frames.length} frames · ${S.segs.length} shot${S.segs.length > 1 ? 's' : ''}${still ? ` (${still} diorama)` : ''}${stage ? ` (${stage} 3D video stage)` : ''}${unit ? ` · ${Math.round(pieces).toLocaleString()} ${unit}` : ''}${S.hasMasks ? ' · people/vehicles detected' : ''} · built in ${Math.round(performance.now() - t0)} ms`;
}

function applySmoothing() {
  const sm = +$('sm').value;
  S.path = [];
  pathGroup.children.slice().forEach(c => { pathGroup.remove(c); c.geometry.dispose(); });
  for (const sg of S.segs) {
    const fixed = sg.still || sg.stage;
    const seg = (fixed ? sg.rawPath : C.smoothPath(sg.rawPath, sm)).map(x => ({ p: new THREE.Vector3(...x.p), q: new THREE.Quaternion(...x.q) }));
    S.path.push(...seg);
    if (!fixed) pathGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(seg.map(x => x.p)), new THREE.LineBasicMaterial({ color: 0xffb347 })));
  }
  pathGroup.visible = $('showPath').checked && mode !== 'ride';
}

let rebuildT;
const rebuildSoon = (msg = 'Rebuilt') => { readParams(); clearTimeout(rebuildT); rebuildT = setTimeout(() => { buildScene(); toast(msg); }, 300); };
for (const id of ['vox', 'far', 'hits', 'pal', 'str']) $(id).oninput = () => rebuildSoon();
$('palSel').onchange = () => rebuildSoon('Palette: ' + C.PALETTES[$('palSel').value].name);
$('adaptive').onchange = () => rebuildSoon($('adaptive').checked ? 'Pieces sized by distance' : 'All pieces the same size');
$('sm').oninput = () => { readParams(); applySmoothing(); };
$('lit').onchange = () => buildScene(); $('fogOn').onchange = () => buildScene();
$('useMask').onchange = () => { buildScene(); toast($('useMask').checked ? 'People/vehicles cut out of the model' : 'People/vehicles kept in the model'); };
$('showPath').onchange = () => pathGroup.visible = $('showPath').checked && mode !== 'ride';
$('showGrid').onchange = () => grid.visible = $('showGrid').checked && mode !== 'ride';
$('liveOn').onchange = () => { liveKey = ''; toast($('liveOn').checked ? 'Live layer on' : 'Live layer off'); };
$('sway').onchange = () => toast($('sway').checked ? 'Diorama sway on (Ride view, still shots)' : 'Diorama sway off');

const segAt = x => S.frames[Math.max(0, Math.min(S.frames.length - 1, Math.round(x)))].seg;
const segIndex = id => S.segs.findIndex(s => s.id === id);

function frameCenter(k) { // what orbit / chase should frame for shot k
  const sg = S.segs[k], box = new THREE.Box3().setFromPoints(S.path.slice(sg.start, sg.end).map(x => x.p));
  const c = box.getCenter(new THREE.Vector3());
  if (sg.still || sg.stage) { const q = S.path[sg.start]; c.copy(q.p).addScaledVector(new THREE.Vector3(0, 0, -1).applyQuaternion(q.q), SCENE_SCALE * 0.8); }
  return { c, size: Math.max(box.getSize(new THREE.Vector3()).length(), SCENE_SCALE) };
}

function setMode(m) {
  mode = m;
  for (const b of $('modeSeg').children) b.classList.toggle('on', b.dataset.m === m);
  controls.enabled = m === 'orbit';
  marker.visible = m !== 'ride';
  pathGroup.visible = $('showPath').checked && m !== 'ride';
  grid.visible = $('showGrid').checked && m !== 'ride';
  camera.fov = m === 'ride' ? S.vfov : 60; camera.updateProjectionMatrix();
  if (m === 'orbit') {
    const { c, size } = frameCenter(segIndex(segAt(idx)));
    controls.target.copy(c);
    camera.position.set(c.x + size * 0.8, c.y + size * 0.9, c.z + size * 0.6);
    controls.update();
  }
  if (m === 'chase') snapChase = true;
}
$('modeSeg').onclick = e => { const m = e.target.dataset?.m; if (m) setMode(m); };
function setStyleUI(s) { style = s; for (const b of $('styleSeg').children) b.classList.toggle('on', b.dataset.s === s); }
$('styleSeg').onclick = e => {
  const s = e.target.closest('button')?.dataset?.s; if (!s || s === style) return;
  setStyleUI(s); buildScene(); toast(STYLE_TIP[s]);
};

function poseAt(x) {
  const n = S.path.length, i0 = Math.max(0, Math.min(n - 1, Math.floor(x))), i1 = Math.min(n - 1, i0 + 1);
  let a = Math.max(0, Math.min(1, x - i0));
  if (S.frames[i0].seg !== S.frames[i1].seg) a = 0; // never blend across a cut
  return { p: new THREE.Vector3().lerpVectors(S.path[i0].p, S.path[i1].p, a), q: new THREE.Quaternion().slerpQuaternions(S.path[i0].q, S.path[i1].q, a) };
}

// live layer. Full mode (3D Video style, or untrackable stage shots): the whole frame as a moving surface.
// Otherwise: people/vehicles (and, in dioramas, anything that moves) as 3D points.
function setGeo(obj, g) {
  obj.geometry.dispose();
  const geo = new THREE.BufferGeometry();
  if (g && g.count) {
    geo.setAttribute('position', new THREE.BufferAttribute(g.pos, 3)); geo.setAttribute('color', new THREE.BufferAttribute(g.col, 3));
    if (g.index) geo.setIndex(new THREE.BufferAttribute(g.index, 1));
  }
  obj.geometry = geo; obj.visible = !!(g && g.count);
}
function updateLive(fi) {
  const fr = S.frames[fi], k = segIndex(fr.seg), sg = S.segs[k], full = style === 'video3d' || sg.stage, on = $('liveOn').checked;
  const key = `${fi}|${style}|${full ? 'full' : on ? 'people' : 'off'}|${$('palSel').value}|${$('pal').value}|${$('far').value}`;
  if (key === liveKey || !sg.align) return;
  liveKey = key;
  if (full) {
    setGeo(liveM, C.sheetGeometry([fr], S.W, S.H, S.K, sg.align, sg.offset, { step: 2, far: +$('far').value, eps: EPS, useMask: false, colorize: S.colorize }));
    setGeo(liveP, null);
    return;
  }
  setGeo(liveM, null);
  let pts = null;
  if (on) {
    let sel = sg.live.get(fi);
    if (sel === undefined) {
      sel = sg.still ? C.orMasks(C.motionMask(fr, sg.bg, S.W, S.H), fr.mask) : fr.mask;
      if (sg.live.size > 400) sg.live.clear();
      sg.live.set(fi, sel);
    }
    if (sel) pts = C.framePoints(fr, S.W, S.H, S.K, sg.align, sg.offset, sel, { far: +$('far').value, eps: EPS, colorize: S.colorize });
  }
  setGeo(liveP, pts);
}

// playback: the original video is the clock, so 3D and video stay in sync
function seekVideo() { vid.currentTime = S.t0 + idx / S.fps; }
function setPlaying(p) {
  playing = p; $('play').textContent = p ? '❚❚' : '▶';
  if (p) {
    if (idx >= S.frames.length - 1.01) idx = 0;
    seekVideo();
    vid.play().catch(() => { /* fall back to internal clock */ });
  } else vid.pause();
}
$('play').onclick = () => { if (!exporting) setPlaying(!playing); };
$('scrub').oninput = () => { if (playing) setPlaying(false); idx = +$('scrub').value; seekVideo(); };

// draw one moment. x = frame index (fractional), tsec = clock for the sway, dt = step for chase smoothing
function renderAt(x, tsec, dt) {
  const N = S.frames.length;
  idx = Math.max(0, Math.min(N - 1, x));
  const fi = Math.round(idx), segId = S.frames[fi].seg, k = segIndex(segId), sg = S.segs[k];
  const pose = poseAt(idx);
  if (segId !== lastSeg) { // entering a new shot
    if (lastSeg !== -1 && mode === 'orbit') {
      const a = frameCenter(segIndex(lastSeg)).c, b = frameCenter(k).c, d = b.sub(a);
      camera.position.add(d); controls.target.add(d);
    }
    snapChase = true;
    [...$('shots').children].forEach((b, j) => b.classList.toggle('on', j === k));
    lastSeg = segId;
  }
  updateLive(fi);
  marker.position.copy(pose.p); marker.quaternion.copy(pose.q);
  const fwd = new THREE.Vector3(0, 0, -1).applyQuaternion(pose.q);
  if (mode === 'ride') {
    camera.position.copy(pose.p); camera.quaternion.copy(pose.q);
    if ((sg.still || sg.stage) && $('sway').checked) { // gentle parallax so a locked shot reads as 3D
      const right = new THREE.Vector3(1, 0, 0).applyQuaternion(pose.q);
      const pivot = pose.p.clone().addScaledVector(fwd, SCENE_SCALE * 0.8);
      camera.position.addScaledVector(right, Math.sin(tsec * 0.7) * SCENE_SCALE * 0.08).add(new THREE.Vector3(0, Math.sin(tsec * 0.45) * SCENE_SCALE * 0.03, 0));
      camera.lookAt(pivot);
    }
  } else if (mode === 'chase') {
    const flat = fwd.clone(); flat.y = 0;
    if (flat.lengthSq() < 1e-4) flat.set(0, 0, -1); flat.normalize();
    const want = pose.p.clone().addScaledVector(flat, -SCENE_SCALE * 0.35).add(new THREE.Vector3(0, SCENE_SCALE * 0.18, 0));
    const look = pose.p.clone().addScaledVector(flat, SCENE_SCALE * 0.3);
    const kk = snapChase ? 1 : 1 - Math.exp(-dt * 4); snapChase = false;
    camera.position.lerp(want, kk); chaseLook.lerp(look, kk);
    camera.lookAt(chaseLook);
  } else controls.update();
  renderer.render(scene, camera);
}

function tick() {
  if (exporting) return; // the exporter drives rendering itself
  const now = performance.now(), dt = Math.min(0.1, (now - clock) / 1000); clock = now;
  const N = S.frames.length;
  let x = idx;
  if (playing) {
    if (!vid.paused && vid.readyState >= 2) x = (vid.currentTime - S.t0) * S.fps;
    else x += dt * S.fps;
    if (x >= N - 1) {
      if (recording) { x = N - 1; stopRecording(); setPlaying(false); }
      else { x = 0; idx = 0; seekVideo(); }
    }
    $('scrub').value = x;
  }
  $('tlabel').textContent = (x / S.fps).toFixed(1) + ' s';
  renderAt(x, now / 1000, dt);
}

// ---------- top bar ----------
$('bInset').onclick = () => { const on = $('inset').classList.toggle('hidden'); $('bInset').classList.toggle('on', !on); };
$('bSound').onclick = () => { vid.muted = !vid.muted; $('bSound').textContent = vid.muted ? '🔇' : '🔊'; toast(vid.muted ? 'Sound off' : 'Sound on'); };
$('bSet').onclick = () => { readParams(); $('sheet').classList.toggle('hidden'); };
$('bClose').onclick = () => $('sheet').classList.add('hidden');
$('bNew').onclick = () => {
  if (recording || exporting) { toast('Finish or cancel the video export first', 'warn'); return; }
  setPlaying(false);
  $('vidHome').appendChild(vid); vid.controls = true;
  show('setup'); toast('Pick a new video, or change settings and rebuild');
};

// ---------- exports ----------
function download(blob, name) {
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = name;
  document.body.appendChild(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(a.href), 4000);
}
const baseName = () => 'retrace-' + (S.fileName || 'street').replace(/[^\w-]+/g, '_').slice(0, 40);

$('bGlb').onclick = () => {
  if (style === 'video3d' || !world.children.length) { toast('3D Video has no fixed model. Pick another style to export a .glb', 'warn', 5000); return; }
  toast('Exporting 3D model…');
  new GLTFExporter().parse(world, res => { download(new Blob([res], { type: 'model/gltf-binary' }), `${baseName()}-${style}.glb`); toast(`Saved .glb (${STYLE_LABEL[style]}, ${world.children.length} shot${world.children.length > 1 ? 's' : ''})`); },
    err => toast('GLB export failed: ' + err.message, 'err', 6000), { binary: true });
};
$('bPath').onclick = () => {
  const data = {
    app: 'Retrace', units: 'scene units (arbitrary scale; ground at y=0)', coords: 'three.js / glTF: Y up, camera looks down -Z',
    fps: S.fps, videoStart: S.t0, verticalFovDeg: +S.vfov.toFixed(3),
    shots: S.segs.map((s, k) => ({ shot: k + 1, firstFrame: s.start, lastFrame: s.end - 1, stillCamera: s.still, stage: s.stage, offset: s.offset })),
    frames: S.path.map((x, i) => ({ i, shot: segIndex(S.frames[i].seg) + 1, t: +(S.t0 + i / S.fps).toFixed(4), position: x.p.toArray().map(v => +v.toFixed(5)), quaternion: x.q.toArray().map(v => +v.toFixed(6)) }))
  };
  download(new Blob([JSON.stringify(data, null, 1)], { type: 'application/json' }), baseName() + '-path.json');
  toast('Saved camera path');
};

$('bRec').onclick = () => {
  if (recording) { stopRecording(); setPlaying(false); return; }
  if (exporting) return;
  $('sheet').classList.add('hidden');
  if ($('recFmt').value === 'mp4') renderMp4().catch(e => { console.error(e); endExport(); toast('MP4 export failed: ' + e.message, 'err', 7000); });
  else recordWebm();
};
$('busyCancel').onclick = () => { if (exporting) { exporting.cancel = true; $('busyT').textContent = 'Cancelling…'; } };

// live screen capture (real time, whatever the screen shows)
function recordWebm(note) {
  if (!window.MediaRecorder || !renderer.domElement.captureStream) { toast("This browser can't record the canvas", 'err'); return; }
  const stream = renderer.domElement.captureStream(30);
  if (!vid.muted) { try { vid.captureStream?.().getAudioTracks().forEach(t => stream.addTrack(t)); } catch { } }
  const mime = ['video/webm;codecs=vp9,opus', 'video/webm;codecs=vp8,opus', 'video/webm', 'video/mp4'].find(m => MediaRecorder.isTypeSupported(m)) || '';
  const ext = mime.startsWith('video/mp4') ? 'mp4' : 'webm';
  const chunks = [];
  const rec = new MediaRecorder(stream, mime ? { mimeType: mime, videoBitsPerSecond: 8e6 } : undefined);
  rec.ondataavailable = e => e.data.size && chunks.push(e.data);
  rec.onstop = () => { download(new Blob(chunks, { type: mime || 'video/webm' }), `${baseName()}-${mode}-${style}.${ext}`); toast(`Saved fly-through (.${ext})`); };
  recording = rec; rec.start(250);
  $('recdot').classList.remove('hidden'); $('bRec').textContent = '■ Stop recording';
  idx = 0; setPlaying(true);
  toast(note || `Recording one pass in ${mode} view…`, note ? 'warn' : '', note ? 5000 : 3200);
}
function stopRecording() {
  if (!recording) return;
  try { recording.stop(); } catch { }
  recording = null; $('recdot').classList.add('hidden'); $('bRec').textContent = '● Render fly-through video';
}

// original audio for the clip range, ready for an AudioEncoder
async function prepareAudio(dur) {
  if (!S.file || !window.AudioEncoder || !window.OfflineAudioContext) return { why: 'audio encoding not supported here' };
  let ab;
  try { ab = await new OfflineAudioContext(1, 1, 48000).decodeAudioData(await S.file.arrayBuffer()); }
  catch { return { why: 'no audio track found' }; }
  const sr = ab.sampleRate, ch = Math.min(2, ab.numberOfChannels), s0 = Math.floor(S.t0 * sr), n = Math.min(ab.length - s0, Math.ceil(dur * sr));
  if (n <= 0) return { why: 'no audio in this range' };
  for (const [codec, mux] of [['mp4a.40.2', 'aac'], ['opus', 'opus']]) {
    try { const r = await AudioEncoder.isConfigSupported({ codec, sampleRate: sr, numberOfChannels: ch, bitrate: 128000 }); if (r.supported) return { ab, sr, ch, s0, n, codec, mux }; } catch { }
  }
  return { why: 'no AAC/Opus encoder here' };
}

function endExport() {
  exporting = null;
  $('busy').classList.add('hidden');
  resize(); lastSeg = -1; snapChase = true; idx = 0; $('scrub').value = 0; seekVideo();
}

// frame-perfect MP4: renders every output frame offline (not limited by phone speed), WebCodecs H.264 + mp4-muxer
async function renderMp4() {
  if (!window.VideoEncoder || !window.VideoFrame) return recordWebm('MP4 encoding isn\'t available in this browser, so recording WebM instead');
  setPlaying(false);
  const long = +$('recSize').value, a = S.aspect || S.W / S.H, fpsOut = 30;
  let w = a >= 1 ? long : Math.round(long * a / 2) * 2, h = a >= 1 ? Math.round(long / a / 2) * 2 : long;
  let codec = null;
  for (const c of ['avc1.640028', 'avc1.4d0028', 'avc1.42e028', 'avc1.42001f']) {
    try { const r = await VideoEncoder.isConfigSupported({ codec: c, width: w, height: h, bitrate: 8e6, framerate: fpsOut }); if (r.supported) { codec = c; break; } } catch { }
  }
  if (!codec && long > 1280) { $('recSize').value = '1280'; toast('1080p isn\'t supported here, so rendering 720p', 'warn'); return renderMp4(); }
  if (!codec) return recordWebm('No MP4 encoder here, so recording WebM instead');

  exporting = { cancel: false };
  $('busy').classList.remove('hidden'); $('busyT').textContent = 'Preparing…'; $('busyBar').style.width = '0%';
  const Mux = await import(MUX_URL);
  const N = S.frames.length, dur = (N - 1) / S.fps, total = Math.max(1, Math.round(dur * fpsOut));
  let audio = null, audioNote = '';
  if ($('recAudio').checked) {
    $('busyT').textContent = 'Reading audio…';
    const r = await prepareAudio(dur);
    if (r.why) audioNote = ` (no audio: ${r.why})`; else audio = r;
  }
  const muxer = new Mux.Muxer({
    target: new Mux.ArrayBufferTarget(), fastStart: 'in-memory', firstTimestampBehavior: 'offset',
    video: { codec: 'avc', width: w, height: h, frameRate: fpsOut },
    ...(audio ? { audio: { codec: audio.mux, numberOfChannels: audio.ch, sampleRate: audio.sr } } : {})
  });
  let encErr = null;
  const venc = new VideoEncoder({ output: (c, m) => muxer.addVideoChunk(c, m), error: e => encErr = e });
  venc.configure({ codec, width: w, height: h, bitrate: long >= 1920 ? 12e6 : 8e6, framerate: fpsOut });

  renderer.setPixelRatio(1); renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix();
  lastSeg = -1; snapChase = true;
  if (mode === 'orbit') setMode('orbit'); // re-frame the first shot
  for (let k = 0; k <= total; k++) {
    if (exporting.cancel) break;
    if (encErr) throw encErr;
    const tsec = k / fpsOut;
    renderAt(tsec * S.fps, tsec, 1 / fpsOut);
    const vf = new VideoFrame(renderer.domElement, { timestamp: Math.round(k * 1e6 / fpsOut), duration: Math.round(1e6 / fpsOut) });
    venc.encode(vf, { keyFrame: k % 60 === 0 }); vf.close();
    while (venc.encodeQueueSize > 6) await new Promise(r => setTimeout(r, 4));
    if (k % 4 === 0) {
      $('busyT').textContent = `Rendering MP4 · frame ${k} / ${total}`; $('busyBar').style.width = (100 * k / total).toFixed(1) + '%';
      await new Promise(r => setTimeout(r, 0));
    }
  }
  if (exporting.cancel) { try { venc.close(); } catch { } endExport(); toast('Export cancelled', 'warn'); return; }
  $('busyT').textContent = 'Finishing video…';
  await venc.flush();
  if (encErr) throw encErr;
  if (audio) {
    $('busyT').textContent = 'Adding audio…';
    let aErr = null;
    const aenc = new AudioEncoder({ output: (c, m) => muxer.addAudioChunk(c, m), error: e => aErr = e });
    aenc.configure({ codec: audio.codec, sampleRate: audio.sr, numberOfChannels: audio.ch, bitrate: 128000 });
    const CH = 4096;
    for (let o = 0; o < audio.n; o += CH) {
      const m = Math.min(CH, audio.n - o), data = new Float32Array(m * audio.ch);
      for (let c = 0; c < audio.ch; c++) data.set(audio.ab.getChannelData(c).subarray(audio.s0 + o, audio.s0 + o + m), c * m);
      const ad = new AudioData({ format: 'f32-planar', sampleRate: audio.sr, numberOfFrames: m, numberOfChannels: audio.ch, timestamp: Math.round(o / audio.sr * 1e6), data });
      aenc.encode(ad); ad.close();
    }
    await aenc.flush();
    if (aErr) { audioNote = ' (audio failed)'; console.warn(aErr); }
  }
  muxer.finalize();
  download(new Blob([muxer.target.buffer], { type: 'video/mp4' }), `${baseName()}-${mode}-${style}.mp4`);
  endExport();
  toast(`Saved MP4 · ${w}×${h} · ${dur.toFixed(1)} s${audioNote}`, audioNote ? 'warn' : '', 6000);
}
