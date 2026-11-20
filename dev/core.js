// ===== Retrace core: image pyramid, features, LK optical flow, PnP, plane fit =====
// Pure JS, no dependencies. Camera convention (OpenCV): x right, y down, z forward.
(function (G) {
  'use strict';

  function bilinear(img, w, h, x, y) {
    if (x < 0) x = 0; else if (x > w - 1.0001) x = w - 1.0001;
    if (y < 0) y = 0; else if (y > h - 1.0001) y = h - 1.0001;
    const x0 = x | 0, y0 = y | 0, fx = x - x0, fy = y - y0, i = y0 * w + x0;
    const a = img[i], b = img[i + 1], c = img[i + w], d = img[i + w + 1];
    const top = a + (b - a) * fx, bot = c + (d - c) * fx;
    return top + (bot - top) * fy;
  }

  function rgbToGray(rgb, w, h) {
    const n = w * h, g = new Float32Array(n);
    for (let i = 0, j = 0; i < n; i++, j += 3) g[i] = 0.299 * rgb[j] + 0.587 * rgb[j + 1] + 0.114 * rgb[j + 2];
    return g;
  }

  function rgbaToRgb(rgba) {
    const n = rgba.length >> 2, out = new Uint8Array(n * 3);
    for (let i = 0, j = 0, k = 0; i < n; i++, j += 4, k += 3) { out[k] = rgba[j]; out[k + 1] = rgba[j + 1]; out[k + 2] = rgba[j + 2]; }
    return out;
  }

  // separable [1 2 1]/4 blur
  function blur3(src, w, h) {
    const tmp = new Float32Array(w * h), out = new Float32Array(w * h);
    for (let y = 0; y < h; y++) {
      const r = y * w;
      for (let x = 0; x < w; x++) {
        const l = src[r + (x > 0 ? x - 1 : 0)], c = src[r + x], rr = src[r + (x < w - 1 ? x + 1 : x)];
        tmp[r + x] = (l + 2 * c + rr) * 0.25;
      }
    }
    for (let y = 0; y < h; y++) {
      const u = (y > 0 ? y - 1 : 0) * w, c = y * w, d = (y < h - 1 ? y + 1 : y) * w;
      for (let x = 0; x < w; x++) out[c + x] = (tmp[u + x] + 2 * tmp[c + x] + tmp[d + x]) * 0.25;
    }
    return out;
  }

  function halve(src, w, h) {
    const W = w >> 1, H = h >> 1, out = new Float32Array(W * H);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const i = (2 * y) * w + 2 * x;
      out[y * W + x] = (src[i] + src[i + 1] + src[i + w] + src[i + w + 1]) * 0.25;
    }
    return { d: out, w: W, h: H };
  }

  function grads(L) {
    const { d, w, h } = L, gx = new Float32Array(w * h), gy = new Float32Array(w * h);
    for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      gx[i] = (d[i + 1] - d[i - 1]) * 0.5;
      gy[i] = (d[i + w] - d[i - w]) * 0.5;
    }
    L.gx = gx; L.gy = gy;
    return L;
  }

  function buildPyramid(gray, w, h, levels) {
    const pyr = [grads({ d: blur3(gray, w, h), w, h })];
    for (let l = 1; l < levels; l++) {
      const p = pyr[l - 1];
      if ((p.w >> 1) < 40 || (p.h >> 1) < 40) break; // keep coarsest level usable
      const hv = halve(blur3(p.d, p.w, p.h), p.w, p.h);
      pyr.push(grads(hv));
    }
    return pyr;
  }

  function boxBlur(a, w, h, r) {
    const tmp = new Float32Array(w * h);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      let s = 0;
      for (let k = -r; k <= r; k++) { const xx = Math.min(w - 1, Math.max(0, x + k)); s += a[y * w + xx]; }
      tmp[y * w + x] = s;
    }
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      let s = 0;
      for (let k = -r; k <= r; k++) { const yy = Math.min(h - 1, Math.max(0, y + k)); s += tmp[yy * w + x]; }
      a[y * w + x] = s;
    }
  }

  // Shi-Tomasi corners, one per grid cell for even coverage
  function detectFeatures(L, maxN, margin) {
    const { w, h, gx, gy } = L, n = w * h;
    const a = new Float32Array(n), b = new Float32Array(n), c = new Float32Array(n);
    for (let i = 0; i < n; i++) { a[i] = gx[i] * gx[i]; b[i] = gx[i] * gy[i]; c[i] = gy[i] * gy[i]; }
    boxBlur(a, w, h, 2); boxBlur(b, w, h, 2); boxBlur(c, w, h, 2);
    const lam = new Float32Array(n);
    let maxL = 0;
    for (let y = margin; y < h - margin; y++) for (let x = margin; x < w - margin; x++) {
      const i = y * w + x, A = a[i], B = b[i], C = c[i];
      const l = (A + C) * 0.5 - Math.sqrt(((A - C) * 0.5) ** 2 + B * B);
      lam[i] = l; if (l > maxL) maxL = l;
    }
    const cell = Math.max(4, Math.floor(Math.sqrt((w - 2 * margin) * (h - 2 * margin) / maxN)));
    const thr = Math.max(maxL * 0.004, 1e-6);
    const pts = [];
    for (let cy = margin; cy < h - margin; cy += cell) for (let cx = margin; cx < w - margin; cx += cell) {
      let best = thr, bx = -1, by = -1;
      const ye = Math.min(cy + cell, h - margin), xe = Math.min(cx + cell, w - margin);
      for (let y = cy; y < ye; y++) for (let x = cx; x < xe; x++) {
        const l = lam[y * w + x];
        if (l > best) { best = l; bx = x; by = y; }
      }
      if (bx >= 0) pts.push(bx, by);
    }
    return Float32Array.from(pts);
  }

  // Brute-force global translation at a coarse level (seeds LK for big pans / fast motion)
  function globalShift(A, B, range) {
    let best = Infinity, bx = 0, by = 0;
    const m = range, step = A.w * A.h > 6000 ? 2 : 1;
    for (let dy = -range; dy <= range; dy++) for (let dx = -range; dx <= range; dx++) {
      let s = 0, c = 0;
      for (let y = m; y < A.h - m; y += step) {
        const ra = y * A.w, rb = (y + dy) * B.w;
        for (let x = m; x < A.w - m; x += step) { s += Math.abs(A.d[ra + x] - B.d[rb + x + dx]); c++; }
      }
      s = s / c + 0.02 * (dx * dx + dy * dy) ** 0.5; // tiny bias toward small motion
      if (s < best) { best = s; bx = dx; by = dy; }
    }
    return [bx, by];
  }

  // Pyramidal Lucas–Kanade. Returns tracked positions in B and a status flag per point.
  function trackLK(pA, pB, pts, win, iters, init) {
    win = win || 7; iters = iters || 15;
    const n = pts.length >> 1, out = new Float32Array(n * 2), st = new Uint8Array(n);
    const levels = Math.min(pA.length, pB.length), sz = (2 * win + 1) ** 2;
    if (!init) init = globalShift(pA[levels - 1], pB[levels - 1], Math.min(12, (pA[levels - 1].h >> 2)));
    const T = new Float32Array(sz), IX = new Float32Array(sz), IY = new Float32Array(sz);
    for (let k = 0; k < n; k++) {
      let gx = init[0], gy = init[1], ok = true;
      for (let l = levels - 1; l >= 0; l--) {
        const A = pA[l], B = pB[l], sc = 1 / (1 << l);
        const px = pts[2 * k] * sc, py = pts[2 * k + 1] * sc;
        // sampling clamps at the image edge, so only reject points that are actually outside
        if (px < 0 || py < 0 || px > A.w - 1 || py > A.h - 1) { ok = false; break; }
        let G00 = 0, G01 = 0, G11 = 0, m = 0;
        for (let dy = -win; dy <= win; dy++) for (let dx = -win; dx <= win; dx++, m++) {
          const x = px + dx, y = py + dy;
          T[m] = bilinear(A.d, A.w, A.h, x, y);
          const ix = bilinear(A.gx, A.w, A.h, x, y), iy = bilinear(A.gy, A.w, A.h, x, y);
          IX[m] = ix; IY[m] = iy;
          G00 += ix * ix; G01 += ix * iy; G11 += iy * iy;
        }
        const det = G00 * G11 - G01 * G01;
        const minEig = ((G00 + G11) - Math.sqrt((G00 - G11) ** 2 + 4 * G01 * G01)) * 0.5 / sz;
        if (det < 1e-9 || minEig < 1e-3) {
          if (l === 0) { ok = false; break; }
          gx *= 2; gy *= 2; continue;
        }
        let vx = 0, vy = 0;
        for (let it = 0; it < iters; it++) {
          let bx = 0, by = 0; m = 0;
          const ox = px + gx + vx, oy = py + gy + vy;
          if (ox < -win || oy < -win || ox > B.w + win || oy > B.h + win) { break; }
          for (let dy = -win; dy <= win; dy++) for (let dx = -win; dx <= win; dx++, m++) {
            const diff = T[m] - bilinear(B.d, B.w, B.h, ox + dx, oy + dy);
            bx += diff * IX[m]; by += diff * IY[m];
          }
          const dvx = (G11 * bx - G01 * by) / det, dvy = (G00 * by - G01 * bx) / det;
          vx += dvx; vy += dvy;
          if (dvx * dvx + dvy * dvy < 0.0009) break;
        }
        if (l > 0) { gx = 2 * (gx + vx); gy = 2 * (gy + vy); } else { gx += vx; gy += vy; }
      }
      const fx = pts[2 * k] + gx, fy = pts[2 * k + 1] + gy;
      out[2 * k] = fx; out[2 * k + 1] = fy;
      const L0 = pB[0];
      st[k] = ok && isFinite(fx) && isFinite(fy) && fx >= 1 && fy >= 1 && fx <= L0.w - 2 && fy <= L0.h - 2 ? 1 : 0;
    }
    return { pts: out, st, init };
  }

  // ---------- small linear algebra ----------
  const I3 = () => [1, 0, 0, 0, 1, 0, 0, 0, 1];
  function mul3(A, B) {
    const C = new Array(9);
    for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++)
      C[r * 3 + c] = A[r * 3] * B[c] + A[r * 3 + 1] * B[3 + c] + A[r * 3 + 2] * B[6 + c];
    return C;
  }
  const mv3 = (R, v) => [R[0] * v[0] + R[1] * v[1] + R[2] * v[2], R[3] * v[0] + R[4] * v[1] + R[5] * v[2], R[6] * v[0] + R[7] * v[1] + R[8] * v[2]];
  const tr3 = R => [R[0], R[3], R[6], R[1], R[4], R[7], R[2], R[5], R[8]];
  function expSO3(wx, wy, wz) {
    const th = Math.hypot(wx, wy, wz);
    if (th < 1e-12) return I3();
    const kx = wx / th, ky = wy / th, kz = wz / th, s = Math.sin(th), c = Math.cos(th), v = 1 - c;
    return [
      c + kx * kx * v, kx * ky * v - kz * s, kx * kz * v + ky * s,
      ky * kx * v + kz * s, c + ky * ky * v, ky * kz * v - kx * s,
      kz * kx * v - ky * s, kz * ky * v + kx * s, c + kz * kz * v];
  }
  function solve6(H, g) { // solves H x = g (H 36, g 6), in place copies
    const A = H.slice(), b = g.slice(), n = 6;
    for (let i = 0; i < n; i++) {
      let p = i, mx = Math.abs(A[i * n + i]);
      for (let r = i + 1; r < n; r++) if (Math.abs(A[r * n + i]) > mx) { mx = Math.abs(A[r * n + i]); p = r; }
      if (mx < 1e-15) return null;
      if (p !== i) { for (let c = 0; c < n; c++) { const t = A[i * n + c]; A[i * n + c] = A[p * n + c]; A[p * n + c] = t; } const t = b[i]; b[i] = b[p]; b[p] = t; }
      for (let r = i + 1; r < n; r++) {
        const f = A[r * n + i] / A[i * n + i];
        if (!f) continue;
        for (let c = i; c < n; c++) A[r * n + c] -= f * A[i * n + c];
        b[r] -= f * b[i];
      }
    }
    const x = new Array(n);
    for (let i = n - 1; i >= 0; i--) { let s = b[i]; for (let c = i + 1; c < n; c++) s -= A[i * n + c] * x[c]; x[i] = s / A[i * n + i]; }
    return x;
  }

  function reprojErr(X, uv, K, R, t, i) {
    const x = X[3 * i], y = X[3 * i + 1], z = X[3 * i + 2];
    const Px = R[0] * x + R[1] * y + R[2] * z + t[0], Py = R[3] * x + R[4] * y + R[5] * z + t[1], Pz = R[6] * x + R[7] * y + R[8] * z + t[2];
    if (Pz < 1e-4) return Infinity;
    const u = K.fx * Px / Pz + K.cx, v = K.fy * Py / Pz + K.cy;
    return Math.hypot(u - uv[2 * i], v - uv[2 * i + 1]);
  }

  // Gauss–Newton / LM pose refinement, left-perturbation on SE(3)
  function refinePose(X, uv, K, idx, R, t, iters, huber) {
    R = R.slice(); t = t.slice();
    for (let it = 0; it < iters; it++) {
      const H = new Array(36).fill(0), g = new Array(6).fill(0);
      let used = 0;
      for (let q = 0; q < idx.length; q++) {
        const i = idx[q], x = X[3 * i], y = X[3 * i + 1], z = X[3 * i + 2];
        const PX = R[0] * x + R[1] * y + R[2] * z + t[0], PY = R[3] * x + R[4] * y + R[5] * z + t[1], PZ = R[6] * x + R[7] * y + R[8] * z + t[2];
        if (PZ < 1e-4) continue;
        const iz = 1 / PZ, ru = K.fx * PX * iz + K.cx - uv[2 * i], rv = K.fy * PY * iz + K.cy - uv[2 * i + 1];
        let w = 1;
        if (huber) { const e = Math.hypot(ru, rv); if (e > huber) w = huber / e; }
        const a = K.fx * iz, b = -K.fx * PX * iz * iz, c = K.fy * iz, d = -K.fy * PY * iz * iz;
        const Ju = [b * PY, a * PZ - b * PX, -a * PY, a, 0, b];
        const Jv = [-c * PZ + d * PY, -d * PX, c * PX, 0, c, d];
        for (let r = 0; r < 6; r++) {
          g[r] += w * (Ju[r] * ru + Jv[r] * rv);
          for (let s = r; s < 6; s++) H[r * 6 + s] += w * (Ju[r] * Ju[s] + Jv[r] * Jv[s]);
        }
        used++;
      }
      if (used < 4) break;
      for (let r = 0; r < 6; r++) { for (let s = 0; s < r; s++) H[r * 6 + s] = H[s * 6 + r]; H[r * 6 + r] *= 1.0001; H[r * 6 + r] += 1e-9; }
      const dx = solve6(H, g.map(v => -v));
      if (!dx || dx.some(v => !isFinite(v))) break;
      const dR = expSO3(dx[0], dx[1], dx[2]);
      R = mul3(dR, R);
      const tt = mv3(dR, t);
      t = [tt[0] + dx[3], tt[1] + dx[4], tt[2] + dx[5]];
      if (Math.hypot(dx[0], dx[1], dx[2]) < 1e-7 && Math.hypot(dx[3], dx[4], dx[5]) < 1e-7) break;
    }
    return { R, t };
  }

  function countInliers(X, uv, K, R, t, thr) {
    const n = X.length / 3, inl = new Uint8Array(n);
    let c = 0;
    for (let i = 0; i < n; i++) if (reprojErr(X, uv, K, R, t, i) < thr) { inl[i] = 1; c++; }
    return { inl, c };
  }

  function makeRng(seed) { let s = seed >>> 0 || 1; return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296); }

  function ransacPnP(X, uv, K, R0, t0, opts) {
    opts = opts || {};
    const n = X.length / 3, thr = opts.thr || 2, iters = opts.iters || 40, m = opts.sample || 6;
    if (n < 8) return null;
    const rnd = makeRng(opts.seed || 12345), all = Array.from({ length: n }, (_, i) => i);
    let best = null;
    const consider = (P) => {
      const r = countInliers(X, uv, K, P.R, P.t, thr);
      if (!best || r.c > best.nIn) best = { R: P.R, t: P.t, inliers: r.inl, nIn: r.c };
    };
    consider({ R: R0, t: t0 });
    consider(refinePose(X, uv, K, all, R0, t0, 10, thr * 1.5));
    consider(refinePose(X, uv, K, all, I3(), [0, 0, 0], 10, thr * 1.5));
    for (let it = 0; it < iters; it++) {
      const sub = [];
      while (sub.length < m) { const j = (rnd() * n) | 0; if (!sub.includes(j)) sub.push(j); }
      consider(refinePose(X, uv, K, sub, best.R, best.t, 6, 0));
      if (best.nIn > 0.85 * n) break;
    }
    for (let k = 0; k < 2; k++) {
      const idx = []; for (let i = 0; i < n; i++) if (best.inliers[i]) idx.push(i);
      if (idx.length < 6) break;
      const P = refinePose(X, uv, K, idx, best.R, best.t, 10, thr);
      const r = countInliers(X, uv, K, P.R, P.t, thr);
      if (r.c >= best.nIn * 0.95) best = { R: P.R, t: P.t, inliers: r.inl, nIn: r.c };
    }
    return best;
  }

  function median(a) {
    if (!a.length) return NaN;
    const s = Float64Array.from(a).sort();
    const h = s.length >> 1;
    return s.length % 2 ? s[h] : 0.5 * (s[h - 1] + s[h]);
  }

  // resample a (sw x sh) float map to (W x H) with bilinear
  function resample(src, sw, sh, W, H) {
    const out = new Float32Array(W * H), sx = sw / W, sy = sh / H;
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++)
      out[y * W + x] = bilinear(src, sw, sh, (x + 0.5) * sx - 0.5, (y + 0.5) * sy - 0.5);
    return out;
  }

  // relative disparity -> 0..1 using robust percentiles
  function normalizeDisparity(d) {
    const n = d.length, step = Math.max(1, Math.floor(n / 6000)), samp = [];
    for (let i = 0; i < n; i += step) samp.push(d[i]);
    samp.sort((a, b) => a - b);
    const lo = samp[Math.floor(samp.length * 0.01)], hi = samp[Math.floor(samp.length * 0.995)];
    const inv = 1 / Math.max(1e-6, hi - lo), out = new Float32Array(n);
    for (let i = 0; i < n; i++) { const v = (d[i] - lo) * inv; out[i] = v < 0 ? 0 : v > 1 ? 1 : v; }
    return out;
  }

  // One tracking step: prev frame -> current frame.
  // prev/cur: { pyr, dn (Float32 0..1 disparity), s (scale, prev only), R, t (prev only) }
  function trackStep(prev, cur, K, lastRel, eps, opts) {
    opts = opts || {};
    const L0 = prev.pyr[0], W = L0.w, H = L0.h;
    const pts = detectFeatures(L0, opts.maxFeatures || 450, 8);
    const fw = trackLK(prev.pyr, cur.pyr, pts);
    const bw = trackLK(cur.pyr, prev.pyr, fw.pts, 0, 0, [-fw.init[0], -fw.init[1]]);
    const X = [], uv = [], src = [];
    for (let k = 0; k < pts.length / 2; k++) {
      if (!fw.st[k] || !bw.st[k]) continue;
      const ex = bw.pts[2 * k] - pts[2 * k], ey = bw.pts[2 * k + 1] - pts[2 * k + 1];
      if (ex * ex + ey * ey > 1) continue;
      const x = pts[2 * k], y = pts[2 * k + 1], d = bilinear(prev.dn, W, H, x, y);
      if (d < 0.03) continue; // sky / infinitely far: no reliable depth
      if (prev.mask && prev.mask[(y | 0) * W + (x | 0)]) continue; // person / vehicle: moves on its own
      if (cur.mask) { const ux = Math.round(fw.pts[2 * k]), uy = Math.round(fw.pts[2 * k + 1]); if (cur.mask[Math.min(H - 1, Math.max(0, uy)) * W + Math.min(W - 1, Math.max(0, ux))]) continue; }
      const z = prev.s / (d + eps);
      X.push((x - K.cx) / K.fx * z, (y - K.cy) / K.fy * z, z);
      uv.push(fw.pts[2 * k], fw.pts[2 * k + 1]);
      src.push(x, y);
    }
    const n = X.length / 3, Xa = Float64Array.from(X), uva = Float64Array.from(uv);
    let rel = null;
    if (n >= 12) {
      const r = ransacPnP(Xa, uva, K, lastRel.R, lastRel.t, { thr: opts.thr || 2, seed: opts.seed });
      if (r && r.nIn >= Math.max(12, 0.25 * n)) rel = r;
    }
    const ok = !!rel;
    if (!ok) rel = { R: lastRel.R.slice(), t: lastRel.t.slice(), inliers: new Uint8Array(n), nIn: 0 };
    // align current depth scale to predicted depths of tracked points
    const ratios = [];
    for (let i = 0; i < n; i++) {
      if (ok && !rel.inliers[i]) continue;
      const P = mv3(rel.R, [Xa[3 * i], Xa[3 * i + 1], Xa[3 * i + 2]]);
      const pz = P[2] + rel.t[2];
      const d2 = bilinear(cur.dn, W, H, uva[2 * i], uva[2 * i + 1]);
      if (d2 < 0.08 || pz <= 0) continue;
      ratios.push(pz * (d2 + eps));
    }
    let s = ratios.length >= 8 ? median(ratios) : prev.s;
    if (!isFinite(s) || s <= 0) s = prev.s;
    s = prev.s * Math.min(1.25, Math.max(0.8, s / prev.s));
    const R = mul3(rel.R, prev.R), tt = mv3(rel.R, prev.t);
    const t = [tt[0] + rel.t[0], tt[1] + rel.t[1], tt[2] + rel.t[2]];
    return { R, t, s, rel: { R: rel.R, t: rel.t }, ok, nTracks: n, nIn: rel.nIn, src, dst: uv, inliers: rel.inliers };
  }

  // symmetric 3x3 eigen decomposition (Jacobi). Returns {values, vectors(columns, row-major 9)}
  function eigSym3(M) {
    const a = M.slice(), V = I3();
    for (let sweep = 0; sweep < 30; sweep++) {
      let off = Math.abs(a[1]) + Math.abs(a[2]) + Math.abs(a[5]);
      if (off < 1e-14) break;
      for (const [p, q] of [[0, 1], [0, 2], [1, 2]]) {
        const apq = a[p * 3 + q];
        if (Math.abs(apq) < 1e-18) continue;
        const app = a[p * 3 + p], aqq = a[q * 3 + q];
        const th = (aqq - app) / (2 * apq);
        const tt = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1));
        const c = 1 / Math.sqrt(tt * tt + 1), s = tt * c;
        for (let k = 0; k < 3; k++) { // A = A J
          const akp = a[k * 3 + p], akq = a[k * 3 + q];
          a[k * 3 + p] = c * akp - s * akq; a[k * 3 + q] = s * akp + c * akq;
        }
        for (let k = 0; k < 3; k++) { // A = J^T A
          const apk = a[p * 3 + k], aqk = a[q * 3 + k];
          a[p * 3 + k] = c * apk - s * aqk; a[q * 3 + k] = s * apk + c * aqk;
        }
        for (let k = 0; k < 3; k++) {
          const vkp = V[k * 3 + p], vkq = V[k * 3 + q];
          V[k * 3 + p] = c * vkp - s * vkq; V[k * 3 + q] = s * vkp + c * vkq;
        }
      }
    }
    return { values: [a[0], a[4], a[8]], vectors: V };
  }

  // RANSAC plane with normal constrained near `up`. pts: flat [x,y,z,...]. Returns {n:[3], d, count} with n·p + d = 0
  function fitGroundPlane(pts, up, thr, maxAngleDeg, seed) {
    const n = pts.length / 3;
    if (n < 30) return null;
    const rnd = makeRng(seed || 777), cosMax = Math.cos((maxAngleDeg || 35) * Math.PI / 180);
    let best = null;
    for (let it = 0; it < 300; it++) {
      const i = (rnd() * n) | 0, j = (rnd() * n) | 0, k = (rnd() * n) | 0;
      if (i === j || j === k || i === k) continue;
      const ax = pts[3 * j] - pts[3 * i], ay = pts[3 * j + 1] - pts[3 * i + 1], az = pts[3 * j + 2] - pts[3 * i + 2];
      const bx = pts[3 * k] - pts[3 * i], by = pts[3 * k + 1] - pts[3 * i + 1], bz = pts[3 * k + 2] - pts[3 * i + 2];
      let nx = ay * bz - az * by, ny = az * bx - ax * bz, nz = ax * by - ay * bx;
      const ln = Math.hypot(nx, ny, nz); if (ln < 1e-9) continue;
      nx /= ln; ny /= ln; nz /= ln;
      let dot = nx * up[0] + ny * up[1] + nz * up[2];
      if (dot < 0) { nx = -nx; ny = -ny; nz = -nz; dot = -dot; }
      if (dot < cosMax) continue;
      const d = -(nx * pts[3 * i] + ny * pts[3 * i + 1] + nz * pts[3 * i + 2]);
      let c = 0;
      for (let q = 0; q < n; q++) if (Math.abs(nx * pts[3 * q] + ny * pts[3 * q + 1] + nz * pts[3 * q + 2] + d) < thr) c++;
      if (!best || c > best.count) best = { n: [nx, ny, nz], d, count: c };
    }
    if (!best || best.count < 30) return null;
    // PCA refine on inliers
    let cx = 0, cy = 0, cz = 0, m = 0;
    const inl = [];
    for (let q = 0; q < n; q++) {
      const x = pts[3 * q], y = pts[3 * q + 1], z = pts[3 * q + 2];
      if (Math.abs(best.n[0] * x + best.n[1] * y + best.n[2] * z + best.d) < thr) { inl.push(x, y, z); cx += x; cy += y; cz += z; m++; }
    }
    cx /= m; cy /= m; cz /= m;
    const Cv = [0, 0, 0, 0, 0, 0, 0, 0, 0];
    for (let q = 0; q < m; q++) {
      const x = inl[3 * q] - cx, y = inl[3 * q + 1] - cy, z = inl[3 * q + 2] - cz;
      Cv[0] += x * x; Cv[1] += x * y; Cv[2] += x * z; Cv[4] += y * y; Cv[5] += y * z; Cv[8] += z * z;
    }
    Cv[3] = Cv[1]; Cv[6] = Cv[2]; Cv[7] = Cv[5];
    const e = eigSym3(Cv);
    let mi = 0; for (let q = 1; q < 3; q++) if (e.values[q] < e.values[mi]) mi = q;
    let nn = [e.vectors[mi], e.vectors[3 + mi], e.vectors[6 + mi]];
    if (nn[0] * best.n[0] + nn[1] * best.n[1] + nn[2] * best.n[2] < 0) nn = nn.map(v => -v);
    if (nn[0] * up[0] + nn[1] * up[1] + nn[2] * up[2] < cosMax) nn = best.n; // refinement wandered off
    return { n: nn, d: -(nn[0] * cx + nn[1] * cy + nn[2] * cz), count: best.count };
  }


  // ================= scene building =================
  const SRGB2LIN = new Float32Array(256);
  for (let i = 0; i < 256; i++) { const c = i / 255; SRGB2LIN[i] = c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4); }

  function matToQuat(m) { // row-major 3x3 -> [x,y,z,w]
    const tr = m[0] + m[4] + m[8];
    let x, y, z, w;
    if (tr > 0) { const s = 0.5 / Math.sqrt(tr + 1); w = 0.25 / s; x = (m[7] - m[5]) * s; y = (m[2] - m[6]) * s; z = (m[3] - m[1]) * s; }
    else if (m[0] > m[4] && m[0] > m[8]) { const s = 2 * Math.sqrt(1 + m[0] - m[4] - m[8]); w = (m[7] - m[5]) / s; x = 0.25 * s; y = (m[1] + m[3]) / s; z = (m[2] + m[6]) / s; }
    else if (m[4] > m[8]) { const s = 2 * Math.sqrt(1 + m[4] - m[0] - m[8]); w = (m[2] - m[6]) / s; x = (m[1] + m[3]) / s; y = 0.25 * s; z = (m[5] + m[7]) / s; }
    else { const s = 2 * Math.sqrt(1 + m[8] - m[0] - m[4]); w = (m[3] - m[1]) / s; x = (m[2] + m[6]) / s; y = (m[5] + m[7]) / s; z = 0.25 * s; }
    const l = Math.hypot(x, y, z, w); return [x / l, y / l, z / l, w / l];
  }

  // CV pose (world->cam R,t) -> three.js camera centre + rotation (Y up, camera looks -Z)
  function poseToThree(R, t) {
    const Rt = tr3(R), c = mv3(Rt, t);
    const F = [1, -1, -1], M = new Array(9);
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) M[i * 3 + j] = F[i] * Rt[i * 3 + j] * F[j];
    return { C: [-c[0], c[1], c[2]], M };
  }

  function computeAlignment(poses, groundPts, scale) {
    const up = [0, 0, 0];
    for (const p of poses) { up[0] += p.M[1]; up[1] += p.M[4]; up[2] += p.M[7]; }
    const ul = Math.hypot(...up) || 1; up[0] /= ul; up[1] /= ul; up[2] /= ul;
    const plane = fitGroundPlane(groundPts, up, 0.03 * scale, 35);
    const n = plane ? plane.n : up;
    // rotate n onto +Y
    const axis = [-n[2], 0, n[0]], s = Math.hypot(...axis), ang = Math.atan2(s, n[1]);
    let R1 = s < 1e-9 ? I3() : expSO3(axis[0] / s * ang, 0, axis[2] / s * ang);
    // yaw: direction of travel -> -Z
    const a = poses[0].C, b = poses[poses.length - 1].C;
    let dir = mv3(R1, [b[0] - a[0], b[1] - a[1], b[2] - a[2]]);
    if (Math.hypot(dir[0], dir[2]) < 0.05 * scale) dir = mv3(R1, [-poses[0].M[2], -poses[0].M[5], -poses[0].M[8]]);
    const phi = Math.PI - Math.atan2(dir[0], dir[2]);
    const M = mul3(expSO3(0, phi, 0), R1);
    let groundY;
    if (plane) groundY = mv3(M, [-plane.d * n[0], -plane.d * n[1], -plane.d * n[2]])[1];
    else {
      const ys = []; for (let i = 0; i < groundPts.length; i += 3) ys.push(mv3(M, [groundPts[i], groundPts[i + 1], groundPts[i + 2]])[1]);
      ys.sort((p, q) => p - q); groundY = ys.length ? ys[Math.floor(ys.length * 0.1)] : mv3(M, a)[1] - 0.15 * scale;
    }
    const c0 = mv3(M, a);
    return { M, T: [-c0[0], -groundY, -c0[2]], planeFound: !!plane };
  }

  // ================= palettes =================
  const PALETTES = {
    original: { name: 'Original', mode: 'original' },
    c64: { name: 'C64', mode: 'palette', colors: '000000 ffffff 880000 aaffee cc44cc 00cc55 0000aa eeee77 dd8855 664400 ff7777 333333 777777 aaff66 0088ff bbbbbb' },
    nes: { name: 'NES', mode: 'palette', colors: '000000 7c7c7c bcbcbc f8f8f8 0000fc 0078f8 3cbcfc a4e4fc 940084 d800cc f878f8 a81000 f83800 f87858 503000 ac7c00 f8b800 f8d878 007800 00b800 58d854 b8f818 004058 008888 00e8d8' },
    pico8: { name: 'PICO-8', mode: 'palette', colors: '000000 1d2b53 7e2553 008751 ab5236 5f574f c2c3c7 fff1e8 ff004d ffa300 ffec27 00e436 29adff 83769c ff77a8 ffccaa' },
    gameboy: { name: 'Game Boy', mode: 'palette', colors: '0f380f 306230 8bac0f 9bbc0f' },
    cga: { name: 'CGA', mode: 'palette', colors: '000000 55ffff ff55ff ffffff' },
    noir: { name: 'Noir', mode: 'palette', colors: '000000 2a2a2a 555555 808080 aaaaaa d5d5d5 ffffff' },
    sepia: { name: 'Sepia', mode: 'palette', colors: '1e130a 3b2614 5a3d1e 7d5a33 a07c4f c3a271 e3caa0 f6ead0' },
    autumn: { name: 'Autumn', mode: 'palette', colors: '1c0f06 3d1c02 7a3b0c b85c1a e08e2b f2c14e 6b7d3a 3a4a24 c9b79c' },
    synthwave: { name: 'Synthwave', mode: 'palette', colors: '0d0221 261447 2e2157 541388 f706cf fd3777 ff4365 f9c80e 00f0ff 2de2e6' },
    vapor: { name: 'Vaporwave', mode: 'palette', colors: '2a2a4a ff71ce 01cdfe 05ffa1 b967ff fffb96 f4f4f4' },
    blueprint: { name: 'Blueprint', mode: 'palette', colors: '0a1f44 123a7a 1f5bbf 4f8ff7 a9c9ff eef5ff' },
    thermal: { name: 'Thermal (by depth)', mode: 'depth', colors: 'fcfdbf fe9f6d de4968 8c2981 3b0f70 000004', range: 4 },
    height: { name: 'Rainbow (by height)', mode: 'height', colors: '30123b 4686fb 1be5b5 a4fc3c fbb938 e4460a 7a0403', range: 1.2 },
    ice: { name: 'Ice (by depth)', mode: 'depth', colors: 'ffffff c6ecff 7cc8f2 2f8ccf 0d4a8a 041a3a', range: 4 },
  };
  const hexRgb = h => [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
  const lutCache = new Map();
  // returns col(r,g,b,y,z) -> [R,G,B] sRGB 0..255 (shared array, copy if you keep it)
  function makeColorizer(spec, sceneScale, levels) {
    spec = typeof spec === 'string' ? PALETTES[spec] || PALETTES.original : spec || PALETTES.original;
    sceneScale = sceneScale || 10;
    const out = [0, 0, 0];
    if (spec.mode === 'palette') {
      const cols = spec.colors.split(' ').map(hexRgb);
      let lut = lutCache.get(spec.colors);
      if (!lut) {
        lut = new Uint8Array(32768);
        for (let i = 0; i < 32768; i++) {
          const r = ((i >> 10) & 31) * 8 + 4, g = ((i >> 5) & 31) * 8 + 4, b = (i & 31) * 8 + 4;
          let best = 1e18, bi = 0;
          for (let k = 0; k < cols.length; k++) {
            const c = cols[k], rm = (r + c[0]) / 2, dr = r - c[0], dg = g - c[1], db = b - c[2];
            const d = (2 + rm / 256) * dr * dr + 4 * dg * dg + (2 + (255 - rm) / 256) * db * db; // "redmean" distance
            if (d < best) { best = d; bi = k; }
          }
          lut[i] = bi;
        }
        lutCache.set(spec.colors, lut);
      }
      const f = (r, g, b) => { const c = cols[lut[((r >> 3) << 10) | ((g >> 3) << 5) | (b >> 3)]]; out[0] = c[0]; out[1] = c[1]; out[2] = c[2]; return out; };
      f.sky = (r, g, b) => f(r, g, b).slice();
      return f;
    }
    if (spec.mode === 'depth' || spec.mode === 'height') {
      const cols = spec.colors.split(' ').map(hexRgb), n = cols.length - 1, range = sceneScale * (spec.range || 3), byDepth = spec.mode === 'depth';
      const f = (r, g, b, y, z) => {
        let t = (byDepth ? z : y) / range; t = t < 0 ? 0 : t > 1 ? 1 : t;
        const k = Math.min(n - 1, Math.floor(t * n)), a = t * n - k, c0 = cols[k], c1 = cols[k + 1];
        const lum = 0.55 + 0.45 * (0.299 * r + 0.587 * g + 0.114 * b) / 255; // keep a hint of the real shading
        out[0] = Math.min(255, (c0[0] + (c1[0] - c0[0]) * a) * lum); out[1] = Math.min(255, (c0[1] + (c1[1] - c0[1]) * a) * lum); out[2] = Math.min(255, (c0[2] + (c1[2] - c0[2]) * a) * lum);
        return out;
      };
      f.sky = () => { const c = cols[byDepth ? n : 0]; return [c[0] * 0.5, c[1] * 0.5, c[2] * 0.5]; };
      return f;
    }
    const L = levels > 1 ? levels - 1 : 0;
    const f = L ? (r, g, b) => { out[0] = Math.round(r / 255 * L) / L * 255; out[1] = Math.round(g / 255 * L) / L * 255; out[2] = Math.round(b / 255 * L) / L * 255; return out; }
      : (r, g, b) => { out[0] = r; out[1] = g; out[2] = b; return out; };
    f.sky = (r, g, b) => f(r, g, b).slice();
    return f;
  }
  const lin = v => SRGB2LIN[v < 0 ? 0 : v > 255 ? 255 : Math.round(v)];

  // depth-map point in camera (three convention) + world transform helpers
  function camPoint(x, y, z, K) { return [(x - K.cx) / K.fx * z, -(y - K.cy) / K.fy * z, -z]; }

  // frames: [{rgb, dn(Uint16), s, R, t, mask}], K, W, H.
  // params: {stride, step, far, voxel, adaptive, minHits, eps, edge, sceneScale, palette, levels, normals, blocks, useMask}
  function buildWorld(frames, W, H, K, params, align) {
    const P = Object.assign({ stride: 2, step: 3, far: 0.06, voxel: 0.24, adaptive: true, minHits: 2, levels: 0, eps: 0.06, edge: 1.12, sceneScale: 10, palette: 'original', normals: false, blocks: true }, params);
    const poses = frames.map(f => poseToThree(f.R, f.t));
    const step = P.step, eps = P.eps, inv16 = 1 / 65535;
    const nF = Math.ceil(frames.length / P.stride), perF = Math.ceil(W / step) * Math.ceil(H / step), cap = nF * perF;
    let pos = new Float32Array(cap * 3), srgb = new Uint8Array(cap * 3), zs = new Float32Array(cap), nrm = P.normals ? new Float32Array(cap * 3) : null, np = 0;
    const ground = [];
    for (let fi = 0; fi < frames.length; fi += P.stride) {
      const f = frames[fi], pose = poses[fi], M = pose.M, Cc = pose.C, dn = f.dn, rgb = f.rgb, mask = P.useMask === false ? null : f.mask;
      for (let y = 0; y < H - step; y += step) for (let x = 0; x < W - step; x += step) {
        const i = y * W + x, d = dn[i] * inv16;
        if (d < P.far || (mask && mask[i])) continue;
        const z = f.s / (d + eps), z1 = f.s / (dn[i + step] * inv16 + eps), z2 = f.s / (dn[i + step * W] * inv16 + eps);
        const r1 = z / z1, r2 = z / z2;
        if (r1 > P.edge || r1 < 1 / P.edge || r2 > P.edge || r2 < 1 / P.edge) continue; // flying pixels at depth edges
        const cx = (x - K.cx) / K.fx * z, cy = -(y - K.cy) / K.fy * z, cz = -z;
        const wx = M[0] * cx + M[1] * cy + M[2] * cz + Cc[0];
        const wy = M[3] * cx + M[4] * cy + M[5] * cz + Cc[1];
        const wz = M[6] * cx + M[7] * cy + M[8] * cz + Cc[2];
        pos[np * 3] = wx; pos[np * 3 + 1] = wy; pos[np * 3 + 2] = wz;
        srgb[np * 3] = rgb[i * 3]; srgb[np * 3 + 1] = rgb[i * 3 + 1]; srgb[np * 3 + 2] = rgb[i * 3 + 2];
        zs[np] = z;
        if (nrm) { // surface normal from neighbouring depth samples, facing the camera
          const ax = (x + step - K.cx) / K.fx * z1 - cx, ay = -(y - K.cy) / K.fy * z1 - cy, az = -z1 - cz;
          const bx = (x - K.cx) / K.fx * z2 - cx, by = -(y + step - K.cy) / K.fy * z2 - cy, bz = -z2 - cz;
          let nx = ay * bz - az * by, ny = az * bx - ax * bz, nz = ax * by - ay * bx;
          if (nx * -cx + ny * -cy + nz * -cz < 0) { nx = -nx; ny = -ny; nz = -nz; }
          const l = Math.hypot(nx, ny, nz) || 1;
          nx /= l; ny /= l; nz /= l;
          nrm[np * 3] = M[0] * nx + M[1] * ny + M[2] * nz; nrm[np * 3 + 1] = M[3] * nx + M[4] * ny + M[5] * nz; nrm[np * 3 + 2] = M[6] * nx + M[7] * ny + M[8] * nz;
        }
        np++;
        if (y > H * 0.6 && ((x + y) & 7) === 0 && ground.length < 60000) ground.push(wx, wy, wz);
      }
    }
    if (!align) align = computeAlignment(poses, ground, P.sceneScale);
    const AM = align.M, AT = align.T;
    for (let k = 0; k < np; k++) {
      const x = pos[3 * k], y = pos[3 * k + 1], z = pos[3 * k + 2];
      pos[3 * k] = AM[0] * x + AM[1] * y + AM[2] * z + AT[0];
      pos[3 * k + 1] = AM[3] * x + AM[4] * y + AM[5] * z + AT[1];
      pos[3 * k + 2] = AM[6] * x + AM[7] * y + AM[8] * z + AT[2];
      if (nrm) {
        const a = nrm[3 * k], b = nrm[3 * k + 1], c = nrm[3 * k + 2];
        nrm[3 * k] = AM[0] * a + AM[1] * b + AM[2] * c; nrm[3 * k + 1] = AM[3] * a + AM[4] * b + AM[5] * c; nrm[3 * k + 2] = AM[6] * a + AM[7] * b + AM[8] * c;
      }
    }
    pos = pos.subarray(0, np * 3); srgb = srgb.subarray(0, np * 3); zs = zs.subarray(0, np); if (nrm) nrm = nrm.subarray(0, np * 3);
    const path = poses.map(p => {
      const c = mv3(AM, p.C);
      return { p: [c[0] + AT[0], c[1] + AT[1], c[2] + AT[2]], q: matToQuat(mul3(AM, p.M)) };
    });
    const colorize = makeColorizer(P.palette, P.sceneScale, P.levels);
    const colLin = new Float32Array(np * 3);
    for (let k = 0; k < np; k++) {
      const c = colorize(srgb[3 * k], srgb[3 * k + 1], srgb[3 * k + 2], pos[3 * k + 1], zs[k]);
      colLin[3 * k] = lin(c[0]); colLin[3 * k + 1] = lin(c[1]); colLin[3 * k + 2] = lin(c[2]);
    }
    const vox = voxelize(pos, srgb, zs, nrm, np, P, K, colorize);
    return { points: { pos, col: colLin, count: np }, cells: vox.cells, blocks: vox.blocks, path, align };
  }

  // Cells: averaged points on a grid. Adaptive mode picks a cell size per point from how much ground one
  // sample covers at that distance (near = small, far = big), quantised to powers of two, so distant
  // surfaces stay solid instead of breaking into floating cubes.
  const CELL_BASE = 0.02;
  function voxelize(pos, srgb, zs, nrm, np, P, K, colorize) {
    const OFF = 32768, M = 65536, MM = M * M, MMM = MM * M;
    const map = new Map();
    let cnt = 0, cap = 1 << 16;
    let sums = new Float32Array(cap * 4), nsum = nrm ? new Float32Array(cap * 3) : null, hits = new Uint32Array(cap), keys = new Float64Array(cap);
    const mult = 2 * P.voxel / 0.24, fpk = P.step / K.fx;
    for (let k = 0; k < np; k++) {
      let L = 0, size = P.voxel;
      if (P.adaptive) {
        const want = Math.max(CELL_BASE, zs[k] * fpk * mult);
        L = Math.max(0, Math.min(9, Math.round(Math.log2(want / CELL_BASE))));
        size = CELL_BASE * (1 << L);
      }
      const inv = 1 / size;
      const ix = Math.floor(pos[3 * k] * inv), iy = Math.floor(pos[3 * k + 1] * inv), iz = Math.floor(pos[3 * k + 2] * inv);
      if (Math.abs(ix) >= OFF || Math.abs(iy) >= OFF || Math.abs(iz) >= OFF) continue;
      const key = L * MMM + (ix + OFF) * MM + (iy + OFF) * M + (iz + OFF);
      let idx = map.get(key);
      if (idx === undefined) {
        if (cnt === cap) {
          cap *= 2;
          const s2 = new Float32Array(cap * 4); s2.set(sums); sums = s2;
          if (nsum) { const n2 = new Float32Array(cap * 3); n2.set(nsum); nsum = n2; }
          const h2 = new Uint32Array(cap); h2.set(hits); hits = h2;
          const k2 = new Float64Array(cap); k2.set(keys); keys = k2;
        }
        idx = cnt++; map.set(key, idx); keys[idx] = key;
      }
      sums[idx * 4] += srgb[3 * k]; sums[idx * 4 + 1] += srgb[3 * k + 1]; sums[idx * 4 + 2] += srgb[3 * k + 2]; sums[idx * 4 + 3] += zs[k];
      if (nsum) { nsum[idx * 3] += nrm[3 * k]; nsum[idx * 3 + 1] += nrm[3 * k + 1]; nsum[idx * 3 + 2] += nrm[3 * k + 2]; }
      hits[idx]++;
    }
    const keep = new Map();
    for (let i = 0; i < cnt; i++) if (hits[i] >= P.minHits) keep.set(keys[i], i);
    const n = keep.size, center = new Float32Array(n * 3), sizeA = new Float32Array(n), color = new Float32Array(n * 3), normal = nsum ? new Float32Array(n * 3) : null;
    const decode = key => {
      const L = Math.floor(key / MMM); let r = key - L * MMM;
      const ix = Math.floor(r / MM) - OFF; r -= (ix + OFF) * MM;
      const iy = Math.floor(r / M) - OFF; const iz = r - (iy + OFF) * M - OFF;
      return [L, ix, iy, iz];
    };
    let c = 0;
    const info = new Array(n);
    for (const [key, i] of keep) {
      const [L, ix, iy, iz] = decode(key), size = P.adaptive ? CELL_BASE * (1 << L) : P.voxel;
      center[3 * c] = (ix + 0.5) * size; center[3 * c + 1] = (iy + 0.5) * size; center[3 * c + 2] = (iz + 0.5) * size; sizeA[c] = size;
      const h = hits[i], col = colorize(sums[i * 4] / h, sums[i * 4 + 1] / h, sums[i * 4 + 2] / h, center[3 * c + 1], sums[i * 4 + 3] / h);
      color[3 * c] = lin(col[0]); color[3 * c + 1] = lin(col[1]); color[3 * c + 2] = lin(col[2]);
      if (normal) { const a = nsum[i * 3], b = nsum[i * 3 + 1], d = nsum[i * 3 + 2], l = Math.hypot(a, b, d) || 1; normal[3 * c] = a / l; normal[3 * c + 1] = b / l; normal[3 * c + 2] = d / l; }
      info[c] = [key, ix, iy, iz, size];
      c++;
    }
    const cells = { center, size: sizeA, color, normal, count: n };
    if (!P.blocks) return { cells, blocks: null };
    // block faces, hiding faces shared with a same-size neighbour
    const FACES = [
      [[1, 0, 0], MM, [[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1]]],
      [[-1, 0, 0], -MM, [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]]],
      [[0, 1, 0], M, [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]]],
      [[0, -1, 0], -M, [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]]],
      [[0, 0, 1], 1, [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]],
      [[0, 0, -1], -1, [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]]]];
    let nFaces = 0;
    for (let q = 0; q < n; q++) for (const F of FACES) if (!keep.has(info[q][0] + F[1])) nFaces++;
    const vp = new Float32Array(nFaces * 12), vn = new Float32Array(nFaces * 12), vc = new Float32Array(nFaces * 12), ix = new Uint32Array(nFaces * 6);
    let f = 0;
    for (let q = 0; q < n; q++) {
      const [key, gx, gy, gz, vs] = info[q];
      for (const F of FACES) {
        if (keep.has(key + F[1])) continue;
        for (let k = 0; k < 4; k++) {
          const o = f * 12 + k * 3, cn = F[2][k];
          vp[o] = (gx + cn[0]) * vs; vp[o + 1] = (gy + cn[1]) * vs; vp[o + 2] = (gz + cn[2]) * vs;
          vn[o] = F[0][0]; vn[o + 1] = F[0][1]; vn[o + 2] = F[0][2];
          vc[o] = color[3 * q]; vc[o + 1] = color[3 * q + 1]; vc[o + 2] = color[3 * q + 2];
        }
        const b4 = f * 4;
        ix[f * 6] = b4; ix[f * 6 + 1] = b4 + 1; ix[f * 6 + 2] = b4 + 2; ix[f * 6 + 3] = b4; ix[f * 6 + 4] = b4 + 2; ix[f * 6 + 5] = b4 + 3;
        f++;
      }
    }
    return { cells, blocks: { position: vp, normal: vn, color: vc, index: ix, faces: nFaces } };
  }

  // Depth-map surfaces: each frame's depth grid as a continuous mesh (or its wireframe "joined points").
  // P: {step, stride, wire, far, eps, edge, useMask, select, palette, levels, sceneScale}
  function sheetGeometry(frames, W, H, K, align, offset, params) {
    const P = Object.assign({ step: 3, stride: 1, wire: false, far: 0.06, eps: 0.06, edge: 1.2, sceneScale: 10, palette: 'original', levels: 0 }, params);
    const colorize = P.colorize || makeColorizer(P.palette, P.sceneScale, P.levels);
    const AM = align.M, AT = align.T, off = offset || [0, 0, 0], step = P.step, inv16 = 1 / 65535;
    const gw = Math.floor((W - 1) / step) + 1, gh = Math.floor((H - 1) / step) + 1;
    const pos = [], col = [], idx = [];
    const vid = new Int32Array(gw * gh), zg = new Float32Array(gw * gh);
    let nv = 0;
    for (let fi = 0; fi < frames.length; fi += P.stride) {
      const f = frames[fi], pose = poseToThree(f.R, f.t), M = pose.M, Cc = pose.C, mask = P.useMask === false ? null : f.mask, sel = P.select;
      vid.fill(-1);
      for (let gy = 0; gy < gh; gy++) for (let gx = 0; gx < gw; gx++) {
        const x = gx * step, y = gy * step, i = y * W + x, d = f.dn[i] * inv16;
        if (d < P.far || (mask && mask[i]) || (sel && !sel[i])) continue;
        const z = f.s / (d + P.eps), cx = (x - K.cx) / K.fx * z, cy = -(y - K.cy) / K.fy * z, cz = -z;
        const wx = M[0] * cx + M[1] * cy + M[2] * cz + Cc[0], wy = M[3] * cx + M[4] * cy + M[5] * cz + Cc[1], wz = M[6] * cx + M[7] * cy + M[8] * cz + Cc[2];
        const ax = AM[0] * wx + AM[1] * wy + AM[2] * wz + AT[0] + off[0], ay = AM[3] * wx + AM[4] * wy + AM[5] * wz + AT[1] + off[1], az = AM[6] * wx + AM[7] * wy + AM[8] * wz + AT[2] + off[2];
        pos.push(ax, ay, az);
        const c = colorize(f.rgb[i * 3], f.rgb[i * 3 + 1], f.rgb[i * 3 + 2], ay, z);
        col.push(lin(c[0]), lin(c[1]), lin(c[2]));
        const g = gy * gw + gx; vid[g] = nv++; zg[g] = z;
      }
      const ok = (a, b) => { const r = zg[a] / zg[b]; return r < P.edge && r > 1 / P.edge; };
      for (let gy = 0; gy < gh; gy++) for (let gx = 0; gx < gw; gx++) {
        const a = gy * gw + gx; if (vid[a] < 0) continue;
        if (P.wire) {
          if (gx + 1 < gw && vid[a + 1] >= 0 && ok(a, a + 1)) idx.push(vid[a], vid[a + 1]);
          if (gy + 1 < gh && vid[a + gw] >= 0 && ok(a, a + gw)) idx.push(vid[a], vid[a + gw]);
        } else if (gx + 1 < gw && gy + 1 < gh) {
          const b = a + 1, c = a + gw, d = a + gw + 1;
          if (vid[b] < 0 || vid[c] < 0 || vid[d] < 0) continue;
          if (!ok(a, b) || !ok(a, c) || !ok(b, d) || !ok(c, d)) continue;
          idx.push(vid[a], vid[c], vid[b], vid[b], vid[c], vid[d]);
        }
      }
    }
    return { pos: Float32Array.from(pos), col: Float32Array.from(col), index: Uint32Array.from(idx), count: nv, wire: P.wire };
  }

  // moving-average smoothing of a path (positions + quaternions)
  function smoothPath(path, win) {
    if (!win) return path.map(p => ({ p: p.p.slice(), q: p.q.slice() }));
    return path.map((_, i) => {
      const p = [0, 0, 0], q = [0, 0, 0, 0], ref = path[i].q; let n = 0;
      for (let j = Math.max(0, i - win); j <= Math.min(path.length - 1, i + win); j++) {
        const a = path[j];
        p[0] += a.p[0]; p[1] += a.p[1]; p[2] += a.p[2];
        const sgn = (a.q[0] * ref[0] + a.q[1] * ref[1] + a.q[2] * ref[2] + a.q[3] * ref[3]) < 0 ? -1 : 1;
        for (let k = 0; k < 4; k++) q[k] += sgn * a.q[k];
        n++;
      }
      const l = Math.hypot(...q);
      return { p: p.map(v => v / n), q: q.map(v => v / l) };
    });
  }

  // ================= shots, still cameras, masks =================
  function colorHist(rgb) {
    const h = new Float32Array(48), n = rgb.length / 3;
    let c = 0;
    for (let i = 0; i < n; i += 2) { const j = i * 3; h[rgb[j] >> 4]++; h[16 + (rgb[j + 1] >> 4)]++; h[32 + (rgb[j + 2] >> 4)]++; c++; }
    for (let k = 0; k < 48; k++) h[k] /= c * 3;
    return h;
  }
  function histDiff(a, b) { let s = 0; for (let k = 0; k < a.length; k++) s += Math.abs(a[k] - b[k]); return s / 2; }
  // decide whether frame i starts a new shot. track = trackStep result (or null)
  function isCut(hd, track) {
    const lost = !track || !track.ok || track.nIn < 20;
    return hd > 0.45 || (hd > 0.2 && lost);
  }

  // still camera: centre barely moves and view barely turns (relative to scene scale)
  function cameraSpread(frames, sceneScale) {
    let maxD = 0, maxA = 0;
    const c0 = mv3(tr3(frames[0].R), frames[0].t);
    for (const f of frames) {
      const c = mv3(tr3(f.R), f.t);
      maxD = Math.max(maxD, Math.hypot(c[0] - c0[0], c[1] - c0[1], c[2] - c0[2]));
      const dR = mul3(f.R, tr3(frames[0].R));
      maxA = Math.max(maxA, Math.acos(Math.max(-1, Math.min(1, (dR[0] + dR[4] + dR[8] - 1) / 2))) * 180 / Math.PI);
    }
    return { dist: maxD / sceneScale, angle: maxA };
  }
  function isStill(frames, sceneScale) {
    if (frames.length < 3) return false;
    const s = cameraSpread(frames, sceneScale);
    return s.dist < 0.025 && s.angle < 2.5;
  }

  // per-pixel median colour + depth over (up to maxN) frames: removes anything that moves
  function medianBackground(frames, W, H, maxN) {
    const step = Math.max(1, Math.ceil(frames.length / (maxN || 41))), use = [];
    for (let i = 0; i < frames.length; i += step) use.push(frames[i]);
    const n = use.length, mid = n >> 1, rgb = new Uint8Array(W * H * 3), dn = new Uint16Array(W * H);
    const buf = new Float64Array(n);
    const med = () => { buf.sort(); return n % 2 ? buf[mid] : (buf[mid - 1] + buf[mid]) / 2; };
    for (let i = 0; i < W * H; i++) {
      for (let c = 0; c < 3; c++) { for (let k = 0; k < n; k++) buf[k] = use[k].rgb[i * 3 + c]; rgb[i * 3 + c] = Math.round(med()); }
      for (let k = 0; k < n; k++) buf[k] = use[k].dn[i]; dn[i] = Math.round(med());
    }
    const ss = new Float64Array(n); use.forEach((f, k) => ss[k] = f.s); ss.sort();
    return { rgb, dn, s: ss[mid], R: I3(), t: [0, 0, 0], mask: null };
  }

  function dilate(mask, W, H, r) {
    const out = new Uint8Array(W * H);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      if (!mask[y * W + x]) continue;
      for (let dy = -r; dy <= r; dy++) { const yy = y + dy; if (yy < 0 || yy >= H) continue;
        for (let dx = -r; dx <= r; dx++) { const xx = x + dx; if (xx >= 0 && xx < W) out[yy * W + xx] = 1; } }
    }
    return out;
  }

  // pixels that differ from the background (moving things), cleaned of speckle
  function motionMask(frame, bg, W, H, thr) {
    thr = thr || 28;
    const raw = new Uint8Array(W * H);
    for (let i = 0; i < W * H; i++) {
      const j = i * 3;
      const d = Math.max(Math.abs(frame.rgb[j] - bg.rgb[j]), Math.abs(frame.rgb[j + 1] - bg.rgb[j + 1]), Math.abs(frame.rgb[j + 2] - bg.rgb[j + 2]));
      if (d > thr) raw[i] = 1;
    }
    const clean = new Uint8Array(W * H);
    for (let y = 1; y < H - 1; y++) for (let x = 1; x < W - 1; x++) {
      const i = y * W + x; if (!raw[i]) continue;
      let c = 0; for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) c += raw[i + dy * W + dx];
      if (c >= 6) clean[i] = 1;
    }
    return dilate(clean, W, H, 1);
  }

  // semantic-segmentation output -> Uint8 mask at W x H for the listed labels
  const MOVER_LABELS = /^(person|car|bus|truck|van|bicycle|minibike|motorbike|motorcycle|animal|boat|ship|airplane|rider|train)$/i;
  function maskFromSegments(segs, W, H, dilateR) {
    const m = new Uint8Array(W * H); let any = 0;
    for (const sgm of segs || []) {
      const lab = String(sgm.label || '').split(/[;,]/)[0].trim();
      if (!MOVER_LABELS.test(lab) || !sgm.mask) continue;
      const mk = sgm.mask, mw = mk.width, mh = mk.height, ch = mk.channels || 1;
      for (let y = 0; y < H; y++) {
        const sy = Math.min(mh - 1, Math.floor((y + 0.5) * mh / H));
        for (let x = 0; x < W; x++) {
          const sx = Math.min(mw - 1, Math.floor((x + 0.5) * mw / W));
          if (mk.data[(sy * mw + sx) * ch] > 127) { m[y * W + x] = 1; any++; }
        }
      }
    }
    return any ? dilate(m, W, H, dilateR == null ? 2 : dilateR) : null;
  }
  function orMasks(a, b) { if (!a) return b; if (!b) return a; const o = new Uint8Array(a.length); for (let i = 0; i < a.length; i++) o[i] = a[i] | b[i]; return o; }

  // live layer: 3D points of selected pixels of one frame, placed with that frame's pose + shot alignment
  function framePoints(frame, W, H, K, align, offset, select, params) {
    const P = Object.assign({ step: 2, far: 0.06, eps: 0.06, edge: 1.12, sceneScale: 10 }, params);
    const colorize = P.colorize || makeColorizer(P.palette, P.sceneScale, P.levels);
    const pose = poseToThree(frame.R, frame.t), M = pose.M, Cc = pose.C, AM = align.M, AT = align.T, inv16 = 1 / 65535;
    const pos = [], col = [];
    for (let y = 0; y < H - P.step; y += P.step) for (let x = 0; x < W - P.step; x += P.step) {
      const i = y * W + x;
      if (select && !select[i]) continue;
      const d = frame.dn[i] * inv16; if (d < P.far) continue;
      const z = frame.s / (d + P.eps), z1 = frame.s / (frame.dn[i + P.step] * inv16 + P.eps), z2 = frame.s / (frame.dn[i + P.step * W] * inv16 + P.eps);
      if (z / z1 > P.edge || z1 / z > P.edge || z / z2 > P.edge || z2 / z > P.edge) continue;
      const cx = (x - K.cx) / K.fx * z, cy = -(y - K.cy) / K.fy * z, cz = -z;
      const wx = M[0] * cx + M[1] * cy + M[2] * cz + Cc[0], wy = M[3] * cx + M[4] * cy + M[5] * cz + Cc[1], wz = M[6] * cx + M[7] * cy + M[8] * cz + Cc[2];
      pos.push(AM[0] * wx + AM[1] * wy + AM[2] * wz + AT[0] + offset[0], AM[3] * wx + AM[4] * wy + AM[5] * wz + AT[1] + offset[1], AM[6] * wx + AM[7] * wy + AM[8] * wz + AT[2] + offset[2]);
      const cc = colorize(frame.rgb[i * 3], frame.rgb[i * 3 + 1], frame.rgb[i * 3 + 2], pos[pos.length - 2], z);
      col.push(lin(cc[0]), lin(cc[1]), lin(cc[2]));
    }
    return { pos: Float32Array.from(pos), col: Float32Array.from(col), count: pos.length / 3 };
  }

  G.RetraceCore = {
    bilinear, globalShift, rgbToGray, rgbaToRgb, buildPyramid, detectFeatures, trackLK, I3, mul3, mv3, tr3, expSO3,
    refinePose, ransacPnP, countInliers, median, resample, normalizeDisparity, trackStep, eigSym3, fitGroundPlane, makeRng,
    matToQuat, poseToThree, computeAlignment, buildWorld, voxelize, smoothPath, SRGB2LIN,
    PALETTES, makeColorizer, sheetGeometry,
    colorHist, histDiff, isCut, cameraSpread, isStill, medianBackground, dilate, motionMask, maskFromSegments, orMasks, framePoints
  };
})(typeof window !== 'undefined' ? window : globalThis);
