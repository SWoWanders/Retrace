// Node tests for core.js — synthetic data, no browser needed.
import fs from 'fs';
import vm from 'vm';
const ctx = { console, Math, Float32Array, Float64Array, Uint8Array, Array, isFinite, Infinity, NaN };
ctx.globalThis = ctx;
vm.createContext(ctx);
vm.runInContext(fs.readFileSync(new URL('../dev/core.js', import.meta.url), 'utf8'), ctx);
const C = ctx.RetraceCore;
let fails = 0;
const check = (name, cond, info = '') => { console.log(`${cond ? 'PASS' : 'FAIL'}  ${name} ${info}`); if (!cond) fails++; };

// ---------- texture + renderer ----------
function hash(x, y) { let h = (x * 374761393 + y * 668265263) | 0; h = (h ^ (h >>> 13)) * 1274126177 | 0; return ((h ^ (h >>> 16)) >>> 0) / 4294967296; }
function vnoise(u, v) { const x = Math.floor(u), y = Math.floor(v); return hash(x, y); } // blocky -> corners
function tex(u, v) { return 40 + 140 * vnoise(u * 2, v * 2) + 60 * vnoise(u * 7 + 100, v * 7); }


const W = 360, H = 202, f = 180 / Math.tan(66 * Math.PI / 360);
const K = { fx: f, fy: f, cx: W / 2, cy: H / 2 };
function render(Rwc /*world->cam*/, t) {
  const Rt = C.tr3(Rwc), Cc = C.mv3(Rt, t).map(v => -v);
  const rgb = new Uint8Array(W * H * 3), disp = new Float32Array(W * H);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const dc = [(x - K.cx) / K.fx, (y - K.cy) / K.fy, 1], d = C.mv3(Rt, dc);
    let best = Infinity, val = 210;
    const tryPlane = (axis, pos, uvf) => {
      if (Math.abs(d[axis]) < 1e-9) return;
      const tt = (pos - Cc[axis]) / d[axis];
      if (tt <= 0 || tt >= best) return;
      const p = [Cc[0] + d[0] * tt, Cc[1] + d[1] * tt, Cc[2] + d[2] * tt];
      const uv = uvf(p); if (!uv) return;
      best = tt; val = tex(uv[0], uv[1]);
    };
    tryPlane(1, 1.6, p => [p[0], p[2]]);                         // ground
    tryPlane(0, -4, p => (p[1] > -8 ? [p[2], p[1]] : null));      // left wall
    tryPlane(0, 4, p => (p[1] > -8 ? [p[2] + 50, p[1]] : null));  // right wall
    tryPlane(2, 60, p => (p[1] > -8 ? [p[0], p[1]] : null));      // far wall
    const i = y * W + x;
    rgb[3 * i] = rgb[3 * i + 1] = rgb[3 * i + 2] = Math.max(0, Math.min(255, val));
    disp[i] = isFinite(best) ? 1 / (best * 1) : 0; // depth along ray ≈ z (dc has z=1, so tt is camera z)
  }
  return { rgb, disp };
}

// ---------- 1. LK on a known sub-pixel image shift (smooth texture) ----------
{
  const smooth = (u, v) => { const x = Math.floor(u), y = Math.floor(v), fx = u - x, fy = v - y;
    const a = hash(x, y), b = hash(x + 1, y), c = hash(x, y + 1), d = hash(x + 1, y + 1);
    return (a + (b - a) * fx) * (1 - fy) + (c + (d - c) * fx) * fy; };
  const img = (ox, oy) => { const g = new Float32Array(W * H);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const u = (x - ox), v = (y - oy);
      g[y * W + x] = 40 + 120 * smooth(u / 9, v / 9) + 80 * smooth(u / 3.5 + 50, v / 3.5);
    } return g; };
  for (const [sx, sy] of [[3.4, -1.7], [17.3, 9.6], [-31.2, 4.4]]) {
    const pa = C.buildPyramid(img(0, 0), W, H, 4), pb = C.buildPyramid(img(sx, sy), W, H, 4);
    const pts = C.detectFeatures(pa[0], 300, 8);
    const tr = C.trackLK(pa, pb, pts);
    let good = 0, tot = 0;
    for (let k = 0; k < pts.length / 2; k++) {
      const ex = pts[2 * k] + sx, ey = pts[2 * k + 1] + sy;
      if (ex < 8 || ey < 8 || ex > W - 8 || ey > H - 8) continue;
      tot++; if (tr.st[k] && Math.hypot(tr.pts[2 * k] - ex, tr.pts[2 * k + 1] - ey) < 0.3) good++;
    }
    check(`LK shift (${sx},${sy})`, good / tot > 0.9, `(${good}/${tot} within 0.3px)`);
  }
}

// ---------- 2. PnP on synthetic correspondences ----------
{
  const rnd = C.makeRng(5);
  const Rt = C.expSO3(0.03, -0.05, 0.01), tt = [0.1, -0.05, -0.6];
  const X = [], uv = [];
  for (let i = 0; i < 200; i++) {
    const p = [(rnd() - 0.5) * 10, (rnd() - 0.5) * 6, 3 + rnd() * 20];
    const P = C.mv3(Rt, p); P[0] += tt[0]; P[1] += tt[1]; P[2] += tt[2];
    let u = K.fx * P[0] / P[2] + K.cx + (rnd() - 0.5) * 0.6, v = K.fy * P[1] / P[2] + K.cy + (rnd() - 0.5) * 0.6;
    if (i % 4 === 0) { u += (rnd() - 0.5) * 60; v += (rnd() - 0.5) * 60; } // 25% outliers
    X.push(...p); uv.push(u, v);
  }
  const r = C.ransacPnP(Float64Array.from(X), Float64Array.from(uv), K, C.I3(), [0, 0, 0], {});
  const terr = Math.hypot(r.t[0] - tt[0], r.t[1] - tt[1], r.t[2] - tt[2]);
  const dR = C.mul3(r.R, C.tr3(Rt)); const ang = Math.acos(Math.min(1, (dR[0] + dR[4] + dR[8] - 1) / 2)) * 180 / Math.PI;
  check('PnP translation', terr < 0.03, `(err ${terr.toFixed(4)})`);
  check('PnP rotation', ang < 0.2, `(err ${ang.toFixed(3)}°)`);
  check('PnP inliers', r.nIn >= 140 && r.nIn <= 160, `(${r.nIn})`);
}

// ---------- 3. Full chain on a rendered street walk ----------
{
  const N = 14, eps = 0.06, gt = [];
  for (let i = 0; i < N; i++) {
    const yaw = 0.02 * Math.sin(i * 0.5), bob = 0.03 * Math.sin(i * 1.3);
    const Cc = [0.3 * Math.sin(i * 0.2), bob, 0.5 * i];       // walking forward along +z (cv)
    const Rw = C.expSO3(0, yaw, 0);                           // world->cam
    const t = C.mv3(Rw, Cc).map(v => -v);
    gt.push({ R: Rw, t, C: Cc });
  }
  let prev = null, lastRel = { R: C.I3(), t: [0, 0, 0] }, est = [];
  globalThis.framesOut = []; const framesOut = globalThis.framesOut;
  const t0 = Date.now();
  for (let i = 0; i < N; i++) {
    const fr = render(gt[i].R, gt[i].t);
    const aff = fr.disp.map(d => 2.5 * d + 0.004); // simulate affine-ambiguous disparity
    const dn = C.normalizeDisparity(aff);
    const pyr = C.buildPyramid(C.rgbToGray(fr.rgb, W, H), W, H, 4);
    if (!prev) {
      const zs = []; for (let k = 0; k < dn.length; k += 13) if (dn[k] > 0.05) zs.push(1 / (dn[k] + eps));
      const s = 10 / C.median(zs);
      prev = { pyr, dn, s, R: C.I3(), t: [0, 0, 0] };
      framesOut.push({ rgb: fr.rgb, dn: Uint16Array.from(dn, v => Math.round(v * 65535)), s, R: C.I3(), t: [0, 0, 0] });
      est.push([0, 0, 0]);
      continue;
    }
    const r = C.trackStep(prev, { pyr, dn }, K, lastRel, eps);
    if (r.ok) lastRel = r.rel;
    prev = { pyr, dn, s: r.s, R: r.R, t: r.t };
    framesOut.push({ rgb: fr.rgb, dn: Uint16Array.from(dn, v => Math.round(v * 65535)), s: r.s, R: r.R, t: r.t });
    est.push(C.mv3(C.tr3(r.R), r.t).map(v => -v));
    if (!r.ok) console.log('   step', i, 'fell back to constant velocity', r.nTracks, r.nIn);
  }
  const ms = (Date.now() - t0) / N;
  // compare direction of travel & straightness (scale is arbitrary)
  const e = est[N - 1], g = gt[N - 1].C;
  const cos = (e[0] * g[0] + e[1] * g[1] + e[2] * g[2]) / (Math.hypot(...e) * Math.hypot(...g));
  const scale = Math.hypot(...e) / Math.hypot(...g);
  let maxDev = 0;
  for (let i = 1; i < N; i++) {
    const ei = est[i].map(v => v / scale), gi = gt[i].C;
    maxDev = Math.max(maxDev, Math.hypot(ei[0] - gi[0], ei[1] - gi[1], ei[2] - gi[2]));
  }
  check('trajectory heading', cos > 0.99, `(cos ${cos.toFixed(4)})`);
  check('trajectory shape', maxDev < 0.6, `(max deviation ${maxDev.toFixed(3)} m over ${(0.5 * (N - 1)).toFixed(1)} m walk)`);
  console.log(`   ~${ms.toFixed(0)} ms/frame incl. rendering`);
}


// ---------- 3b. turning walk (pan ~5°/frame) ----------
{
  const N = 10, eps = 0.06, gt = [];
  for (let i = 0; i < N; i++) {
    const yaw = 0.09 * i, Cc = [0, 0, 0.3 * i];
    const Rw = C.expSO3(0, yaw, 0), t = C.mv3(Rw, Cc).map(v => -v);
    gt.push({ R: Rw, t, yaw });
  }
  let prev = null, lastRel = { R: C.I3(), t: [0, 0, 0] }, worst = 0;
  for (let i = 0; i < N; i++) {
    const fr = render(gt[i].R, gt[i].t), dn = C.normalizeDisparity(fr.disp.map(d => 2 * d + 0.01));
    const pyr = C.buildPyramid(C.rgbToGray(fr.rgb, W, H), W, H, 4);
    if (!prev) { prev = { pyr, dn, s: 1, R: C.I3(), t: [0, 0, 0] }; continue; }
    const r = C.trackStep(prev, { pyr, dn }, K, lastRel, eps);
    if (r.ok) lastRel = r.rel;
    prev = { pyr, dn, s: r.s, R: r.R, t: r.t };
    const dR = C.mul3(r.R, C.tr3(gt[i].R)); const ang = Math.acos(Math.min(1, (dR[0] + dR[4] + dR[8] - 1) / 2)) * 180 / Math.PI;
    worst = Math.max(worst, ang);
  }
  check('turning walk rotation', worst < 1.5, `(worst heading error ${worst.toFixed(2)}° over a ${(0.09 * (N - 1) * 180 / Math.PI).toFixed(0)}° turn)`);
}

// ---------- 4. ground plane ----------
{
  const rnd = C.makeRng(9), pts = [];
  const nrm = [0.1, 0.99, 0.05], ln = Math.hypot(...nrm); nrm.forEach((v, i) => nrm[i] = v / ln);
  for (let i = 0; i < 2000; i++) {
    if (i % 3 === 0) { pts.push((rnd() - .5) * 20, rnd() * 10, (rnd() - .5) * 20); continue; }
    const x = (rnd() - .5) * 20, z = (rnd() - .5) * 20;
    const y = (-1.5 - nrm[0] * x - nrm[2] * z) / nrm[1] + (rnd() - .5) * 0.05;
    pts.push(x, y, z);
  }
  const p = C.fitGroundPlane(pts, [0, 1, 0], 0.15, 35);
  const ang = Math.acos(p.n[0] * nrm[0] + p.n[1] * nrm[1] + p.n[2] * nrm[2]) * 180 / Math.PI;
  check('ground plane normal', ang < 1, `(err ${ang.toFixed(3)}°)`);
}


// ---------- 5. scene builder: ground levelled, path heads -Z ----------
{
  const t0 = Date.now();
  const w = C.buildWorld(globalThis.framesOut, W, H, K, { stride: 1, normals: true });
  const ms = Date.now() - t0;
  const last = w.path[w.path.length - 1].p, first = w.path[0].p;
  check('alignment found ground plane', w.align.planeFound);
  check('path heads -Z', last[2] < -2 && Math.abs(last[0]) < 0.25 * Math.abs(last[2]), `(end ${last.map(v => v.toFixed(2))})`);
  check('camera above ground', first[1] > 0.5, `(height ${first[1].toFixed(2)})`);
  let low = 0, near0 = 0;
  for (let k = 0; k < w.points.count; k++) { const y = w.points.pos[3 * k + 1]; if (y < 0.5) { low++; if (Math.abs(y) < 0.35) near0++; } }
  check('ground flat at y=0', near0 / low > 0.85, `(${near0}/${low})`);
  check('cells + block faces produced', w.cells.count > 1000 && w.blocks.faces > w.cells.count, `(${w.cells.count} cells, ${w.blocks.faces} faces, ${w.points.count} pts, ${ms} ms)`);
  let okIdx = true; const nv = w.blocks.position.length / 3;
  for (const i of w.blocks.index) if (i >= nv) { okIdx = false; break; }
  check('mesh indices valid', okIdx);
  // adaptive: far cells bigger than near cells
  const near = [], far = [];
  for (let c = 0; c < w.cells.count; c++) { const d = Math.hypot(w.cells.center[3 * c] - first[0], w.cells.center[3 * c + 2] - first[2]); (d < 8 ? near : d > 25 ? far : []).push(w.cells.size[c]); }
  const mn = C.median(near), mf = C.median(far);
  check('cells grow with distance', mf > 3 * mn, `(near ${mn.toFixed(3)}, far ${mf.toFixed(3)})`);
  // ground cell normals point up
  let up = 0, gc = 0;
  for (let c = 0; c < w.cells.count; c++) if (Math.abs(w.cells.center[3 * c + 1]) < 0.3 && Math.abs(w.cells.center[3 * c]) < 2.5) { gc++; if (w.cells.normal[3 * c + 1] > 0.9) up++; }
  check('ground normals point up', up / gc > 0.85, `(${up}/${gc})`);
  // uniform mode: all sizes equal
  const u = C.buildWorld(globalThis.framesOut, W, H, K, { stride: 2, adaptive: false, voxel: 0.3, blocks: false }, w.align);
  check('uniform mode one size', u.cells.size.every(v => v === Math.fround(0.3)) && !u.blocks);
  // palette: C64 colours only
  const pal = C.buildWorld(globalThis.framesOut, W, H, K, { stride: 3, palette: 'c64', blocks: false }, w.align);
  const c64 = C.PALETTES.c64.colors.split(' ').map(h => [0, 2, 4].map(i => C.SRGB2LIN[parseInt(h.slice(i, i + 2), 16)]));
  let inPal = 0;
  for (let c = 0; c < pal.cells.count; c++) if (c64.some(q => Math.abs(q[0] - pal.cells.color[3 * c]) < 1e-6 && Math.abs(q[1] - pal.cells.color[3 * c + 1]) < 1e-6 && Math.abs(q[2] - pal.cells.color[3 * c + 2]) < 1e-6)) inPal++;
  check('C64 palette applied', inPal === pal.cells.count, `(${inPal}/${pal.cells.count})`);
  const th = C.makeColorizer('thermal', 10); const nearC = th(128, 128, 128, 0, 1).slice(), farC = th(128, 128, 128, 0, 40).slice();
  check('depth palette varies with distance', Math.abs(nearC[0] - farC[0]) + Math.abs(nearC[2] - farC[2]) > 60, `(${nearC.map(Math.round)} vs ${farC.map(Math.round)})`);
  // sheets + wire
  const sh = C.sheetGeometry(globalThis.framesOut.slice(0, 1), W, H, K, w.align, [0, 0, 0], { step: 3 });
  const wi = C.sheetGeometry(globalThis.framesOut.slice(0, 1), W, H, K, w.align, [0, 0, 0], { step: 4, wire: true });
  const maxI = a => a.index.reduce((m, v) => v > m ? v : m, 0);
  check('sheet mesh', sh.index.length / 3 > 0.7 * sh.count && maxI(sh) < sh.count, `(${sh.count} verts, ${sh.index.length / 3} tris)`);
  check('wire segments', wi.index.length / 2 > wi.count && maxI(wi) < wi.count, `(${wi.count} verts, ${wi.index.length / 2} lines)`);
  if (process.env.DUMP) {
    const u2 = C.buildWorld(globalThis.framesOut, W, H, K, { stride: 1, adaptive: false, voxel: 0.24, blocks: false }, w.align);
    const pick = c => ({ center: Array.from(c.center), size: Array.from(c.size), color: Array.from(c.color) });
    fs.writeFileSync('/tmp/cells.json', JSON.stringify({ adaptive: pick(w.cells), uniform: pick(u2.cells), pose: w.path[6], K, W, H }));
  }
  const sm = C.smoothPath(w.path, 2);
  check('smoothed quats unit', sm.every(p => Math.abs(Math.hypot(...p.q) - 1) < 1e-6));
}

console.log(fails ? `\n${fails} FAILED` : '\nALL PASSED');
process.exit(fails ? 1 : 0);
