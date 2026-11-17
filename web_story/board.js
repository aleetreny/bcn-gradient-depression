/* ============================================================================
   board.js — the dressing-room tactics board
   ----------------------------------------------------------------------------
   One fixed canvas is the board. The pitch lines are printed on it (thin,
   clean); everything the coach adds is chalk (grainy, hand-wobbled): Barça as
   O, Athletic as X, the ball and the precision figure in the one yellow chalk.
   Scrolling draws each chapter's play stroke by stroke. Between chapters the
   eraser sweeps the board left to right, leaving the faint ghost every wiped
   board keeps. The coach's running score sits under the pitch: each step
   crosses out the old figure and writes the new one.

   Events (same contract the text column and the rail listen to):
     sw:layout  { vh, sections: [{ id, start, end }] }
     sw:scroll  { y, vh, activeIndex, sectionProgress, eraser: { seam, e, xDone, x1, x2, yc, down } }
   ========================================================================== */

function mountBoard(container, config) {
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const SECTIONS = config.sections || [];
  const N = SECTIONS.length;
  if (!N) return;

  // Scroll timing, in viewport heights relative to a chapter start S
  const ERASE_FROM = 0.58, ERASE_TO = 0.06;   // the eraser's passes cross the board in [S-0.58, S-0.06]
  const DRAW_LEN = 0.9;                        // a chapter's drawing spans [S-0.06, S+0.84]
  const MOBILE_READ = 0.95, MOBILE_READ0 = 0.7; // phones: notes stay up this long past S (keep in sync with copy-layer.js)

  const INK = {
    chalk: [236, 238, 230],
    yellow: [238, 204, 92],
  };
  const SCORE_STEPS = [61.2, 73.1, 80.0, 81.3, 86.9, 88.8];
  const SCORE_OF = [0, 0, 1, 2, 3, 4, 5, 5]; // chapter -> index into SCORE_STEPS

  const canvas = document.createElement('canvas');
  canvas.className = 'bd-canvas';
  canvas.setAttribute('aria-hidden', 'true');
  const track = document.createElement('div');
  track.className = 'bd-track';
  container.appendChild(canvas);
  container.appendChild(track);
  container.querySelectorAll('[data-sw-seo]').forEach(n => { n.hidden = true; });
  document.documentElement.classList.remove('sw-booting');

  const ctx = canvas.getContext('2d');
  const bg = document.createElement('canvas');      // board surface + printed pitch
  const ink = document.createElement('canvas');     // this frame's chalk, before grain
  const ictx = ink.getContext('2d');
  let grain = null;                                  // CanvasPattern of chalk grain
  const ghosts = new Map();                          // chapter -> blurred low-res wipe ghost

  let W = 0, H = 0, dpr = 1, vh = 0, totalPx = 0;
  let G = null;                                      // geometry
  let shapes = [];                                   // per chapter: prepared drawing items
  let introT = reduce ? 1 : 0, introStart = 0, introDone = reduce;
  let pending = false;

  // ---------- small helpers ----------
  const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
  function rng(seed) { let s = seed >>> 0; return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; }; }
  const col = (c, a) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;
  const isSmall = () => window.innerWidth < 860;

  // ---------- geometry ----------
  function computeGeometry() {
    const railH = 56;
    if (isSmall()) {
      const padX = 16;
      const pw = W - padX * 2;
      const s = pw / 112;                       // metres -> px, 3.5 m margin either side
      const ph = 68 * s;
      const top = Math.max(railH + 44, H * 0.5 - ph * 0.55);
      return { s, cx: W / 2, cy: top + ph / 2 + 3.5 * s, titleY: top - 10, tallyY: top + ph + 7 * s + 34, tallyX0: padX, tallyX1: W - padX, small: true };
    }
    const textW = Math.min(W * 0.36, 520);
    const left = Math.max(24, Math.min(W * 0.04, 64)) + textW + 56;
    const right = W - Math.max(32, W * 0.035);
    const availW = right - left;
    const availH = H - railH - 64 - 96;                 // title row above, tally row below
    const s = Math.min(availW / 116, availH / 80);
    const ph = 68 * s, pw = 105 * s;
    const cx = left + availW / 2;
    const cy = railH + 64 + (availH - ph) / 2 + ph / 2 + 4;
    return { s, cx, cy, titleY: cy - ph / 2 - 6 * s - 8, tallyY: cy + ph / 2 + 6 * s + 44, tallyX0: cx - pw / 2, tallyX1: cx + pw / 2, small: false };
  }
  const P = (x, y) => [G.cx + x * G.s, G.cy - y * G.s];   // pitch metres -> screen px (y up)

  // ---------- board surface ----------
  function paintBoard() {
    bg.width = W * dpr; bg.height = H * dpr;
    const b = bg.getContext('2d');
    b.setTransform(dpr, 0, 0, dpr, 0, 0);
    b.fillStyle = '#1C2722';
    b.fillRect(0, 0, W, H);
    // Old wipes: broad soft smudges of chalk dust, as many per area as on a laptop board
    // (a fixed count piles up on a phone and greys the slate)
    const r = rng(7);
    const areaK = Math.min(1, (W * H) / (1440 * 900));
    const smudges = Math.max(6, Math.round(26 * areaK));
    for (let i = 0; i < smudges; i++) {
      const x = r() * W, y = r() * H, rad = (80 + r() * 340) * Math.max(0.45, Math.sqrt(areaK));
      const g = b.createRadialGradient(x, y, 0, x, y, rad);
      g.addColorStop(0, `rgba(230,236,228,${0.012 + r() * 0.03})`);
      g.addColorStop(1, 'rgba(230,236,228,0)');
      b.fillStyle = g;
      b.beginPath(); b.arc(x, y, rad, 0, Math.PI * 2); b.fill();
    }
    // Wide, soft eraser sweeps from earlier sessions: slightly tilted, feathered at both ends
    for (let i = 0; i < Math.max(3, Math.round(9 * areaK)); i++) {
      const y = r() * H, x = r() * W * 0.7, w = (260 + r() * 520) * Math.max(0.45, Math.sqrt(areaK)), h = 30 + r() * 60;
      b.save();
      b.translate(x, y); b.rotate((r() - 0.5) * 0.12);
      const g = b.createLinearGradient(0, 0, w, 0);
      const a = 0.012 + r() * 0.016;
      g.addColorStop(0, 'rgba(230,236,228,0)');
      g.addColorStop(0.25, `rgba(230,236,228,${a})`);
      g.addColorStop(0.8, `rgba(230,236,228,${a * 0.7})`);
      g.addColorStop(1, 'rgba(230,236,228,0)');
      b.fillStyle = g;
      b.filter = 'blur(6px)';
      b.fillRect(0, 0, w, h);
      b.restore();
    }
    // Slate grain
    const n = document.createElement('canvas');
    n.width = n.height = 160;
    const nc = n.getContext('2d'), id = nc.createImageData(160, 160), rr = rng(11);
    for (let i = 0; i < id.data.length; i += 4) {
      const v = 120 + rr() * 135;
      id.data[i] = id.data[i + 1] = id.data[i + 2] = v;
      id.data[i + 3] = rr() < 0.5 ? 10 : 0;
    }
    nc.putImageData(id, 0, 0);
    b.fillStyle = b.createPattern(n, 'repeat');
    b.fillRect(0, 0, W, H);

    // Printed pitch: thin, clean, permanent
    b.strokeStyle = 'rgba(232,238,230,0.30)';
    b.lineWidth = 1.4;
    const L = (pts, close) => { b.beginPath(); pts.forEach((p, i) => { const q = P(p[0], p[1]); i ? b.lineTo(q[0], q[1]) : b.moveTo(q[0], q[1]); }); if (close) b.closePath(); b.stroke(); };
    L([[-52.5, -34], [52.5, -34], [52.5, 34], [-52.5, 34]], true);
    L([[0, -34], [0, 34]]);
    b.beginPath(); b.arc(G.cx, G.cy, 9.15 * G.s, 0, Math.PI * 2); b.stroke();
    for (const sg of [-1, 1]) {
      L([[52.5 * sg, -20.16], [36 * sg, -20.16], [36 * sg, 20.16], [52.5 * sg, 20.16]]);
      L([[52.5 * sg, -9.16], [47 * sg, -9.16], [47 * sg, 9.16], [52.5 * sg, 9.16]]);
      L([[52.5 * sg, -3.66], [54.5 * sg, -3.66], [54.5 * sg, 3.66], [52.5 * sg, 3.66]]);
      const ps = P(41.5 * sg, 0);
      b.fillStyle = 'rgba(232,238,230,0.35)';
      b.beginPath(); b.arc(ps[0], ps[1], 1.6, 0, Math.PI * 2); b.fill();
      // penalty arc outside the box
      const c = P(41.5 * sg, 0), a = Math.acos(5.5 / 9.15);
      b.beginPath();
      if (sg > 0) b.arc(c[0], c[1], 9.15 * G.s, Math.PI - a, Math.PI + a);
      else b.arc(c[0], c[1], 9.15 * G.s, -a, a);
      b.stroke();
    }
    b.fillStyle = 'rgba(232,238,230,0.35)';
    b.beginPath(); b.arc(G.cx, G.cy, 1.8, 0, Math.PI * 2); b.fill();
  }

  function makeGrain() {
    const n = document.createElement('canvas');
    n.width = n.height = 128;
    const c = n.getContext('2d'), id = c.createImageData(128, 128), r = rng(23);
    for (let i = 0; i < id.data.length; i += 4) {
      const v = r();
      id.data[i] = id.data[i + 1] = id.data[i + 2] = 255;
      id.data[i + 3] = v < 0.16 ? 25 + v * 400 : 150 + r() * 105;   // chalk skips over the slate's tooth
    }
    c.putImageData(id, 0, 0);
    grain = ictx.createPattern(n, 'repeat');
    if (!document.documentElement.style.getPropertyValue('--chalk-grain')) {
      // A gentler cut of the same grain masks the DOM lettering (.nt-title etc.),
      // so the notes are written in the same chalk as the drawing
      const t = document.createElement('canvas');
      t.width = t.height = 96;
      const tc = t.getContext('2d'), td = tc.createImageData(96, 96), tr = rng(29);
      for (let i = 0; i < td.data.length; i += 4) {
        const v = tr();
        td.data[i] = td.data[i + 1] = td.data[i + 2] = 0;
        td.data[i + 3] = v < 0.12 ? 60 + v * 500 : 185 + tr() * 70;
      }
      tc.putImageData(td, 0, 0);
      document.documentElement.style.setProperty('--chalk-grain', `url(${t.toDataURL('image/png')})`);
    }
  }

  // ---------- chalk primitives (prepared once per layout) ----------
  // A stroke is a hand-wobbled polyline in screen px with cumulative lengths.
  function prepStroke(ptsM, seed, wobble = 0.9, step = 7) {
    const r = rng(seed);
    const raw = ptsM.map(p => P(p[0], p[1]));
    const out = [];
    for (let i = 0; i < raw.length - 1; i++) {
      const [x0, y0] = raw[i], [x1, y1] = raw[i + 1];
      const len = Math.hypot(x1 - x0, y1 - y0), n = Math.max(1, Math.round(len / step));
      const nx = -(y1 - y0) / (len || 1), ny = (x1 - x0) / (len || 1);
      for (let k = 0; k < n; k++) {
        const t = k / n, j = (r() - 0.5) * 2 * wobble;
        out.push([x0 + (x1 - x0) * t + nx * j, y0 + (y1 - y0) * t + ny * j]);
      }
    }
    out.push(raw[raw.length - 1]);
    const cum = [0];
    for (let i = 1; i < out.length; i++) cum.push(cum[i - 1] + Math.hypot(out[i][0] - out[i - 1][0], out[i][1] - out[i - 1][1]));
    return { pts: out, cum, len: cum[cum.length - 1] };
  }
  function strokeTo(c, st, frac, width, color, dash) {
    if (frac <= 0) return;
    const lim = st.len * clamp(frac);
    c.lineCap = 'round'; c.lineJoin = 'round';
    c.setLineDash(dash || []);
    for (const pass of [[width, 0.95, 0], [width * 0.55, 0.55, 0.7]]) {
      c.lineWidth = pass[0];
      c.strokeStyle = col(color, pass[1]);
      c.beginPath();
      c.moveTo(st.pts[0][0] + pass[2], st.pts[0][1]);
      for (let i = 1; i < st.pts.length; i++) {
        if (st.cum[i] > lim) {
          const a = st.cum[i - 1], t = (lim - a) / ((st.cum[i] - a) || 1);
          c.lineTo(st.pts[i - 1][0] + (st.pts[i][0] - st.pts[i - 1][0]) * t + pass[2], st.pts[i - 1][1] + (st.pts[i][1] - st.pts[i - 1][1]) * t);
          break;
        }
        c.lineTo(st.pts[i][0] + pass[2], st.pts[i][1]);
      }
      c.stroke();
    }
    c.setLineDash([]);
  }
  function pointAt(st, frac) {
    const lim = st.len * clamp(frac);
    for (let i = 1; i < st.pts.length; i++) {
      if (st.cum[i] >= lim) {
        const a = st.cum[i - 1], t = (lim - a) / ((st.cum[i] - a) || 1);
        return [st.pts[i - 1][0] + (st.pts[i][0] - st.pts[i - 1][0]) * t, st.pts[i - 1][1] + (st.pts[i][1] - st.pts[i - 1][1]) * t];
      }
    }
    return st.pts[st.pts.length - 1];
  }
  const circlePts = (c, r, n = 28, a0 = -1.9, sweep = Math.PI * 2.08) =>
    Array.from({ length: n + 1 }, (_, i) => { const a = a0 + sweep * i / n; return [c[0] + r * Math.cos(a), c[1] + r * Math.sin(a)]; });
  const arcPts = (a, b, bend, n = 18) => {
    const mx = (a[0] + b[0]) / 2, my = (a[1] + b[1]) / 2, dx = b[0] - a[0], dy = b[1] - a[1];
    const cx = mx - dy * bend, cy = my + dx * bend;
    return Array.from({ length: n + 1 }, (_, i) => { const t = i / n, u = 1 - t; return [u * u * a[0] + 2 * u * t * cx + t * t * b[0], u * u * a[1] + 2 * u * t * cy + t * t * b[1]]; });
  };

  // ---------- the eight chapter diagrams (pitch metres, y up; Barça attacks right) ----------
  const HOME = [[-46, 0], [-30, -20], [-32, -7], [-32, 7], [-30, 20], [-14, -14], [-18, 0], [-12, 14], [4, -22], [8, 0], [5, 21]];
  const AWAY = [[46, 0], [31, -19], [33, -6.5], [33, 6.5], [31, 19], [15, -18], [17, -5], [17, 5], [15, 18], [-2, -8], [-4, 8]];
  function teams(a, b, shiftX = 0, shiftY = 0) {
    const items = [];
    const all = [...HOME.map(p => ['o', p]), ...AWAY.map(p => ['x', p])];
    all.forEach(([k, p], i) => items.push({ k, at: [p[0] + shiftX, p[1] + shiftY], a: a + (b - a) * (i / all.length) * 0.8, b: a + (b - a) * (i / all.length) * 0.8 + (b - a) * 0.2 }));
    return items;
  }
  // strike: true crosses out the whole note, or a leading string crosses out just that part
  const T = (s, at, a, b, o = {}) => ({ k: 'text', s, at, a, b, size: o.size || 2.6, c: o.c || 'chalk', align: o.align || 'left', strike: o.strike || false });

  const CHAPTERS = {
    ciego: () => [
      ...teams(0, 0.4),
      { k: 'ball', pts: [[-12, 14], [-6, 12], [-1, 8.5], [3, 4.5], [8, 0.6]], a: 0.45, b: 0.9 },
      { k: 'nearest', ball: 0, a: 0.5, b: 0.92 },
      T('¿quién está más cerca?', [-26, 27], 0.53, 0.64, { size: 3 }),
      { k: 'line', pts: [[-21, 25.5], [-8, 15]], a: 0.62, b: 0.68, arrow: true },
      { k: 'line', pts: [[-3.4, 12], [-1.6, 9.5], [-4.4, 8.8], [-2.4, 6.4]], a: 0.76, b: 0.82, w: 3 },
      T('corte falso', [-2, 4.2], 0.8, 0.92, { size: 2.6, align: 'center' }),
    ],
    trampa: () => [
      T('30 s de posesión del Barça', [-40, 25], 0.05, 0.22, { size: 3 }),
      { k: 'line', pts: [[-40, 18], [-0.8, 18]], a: 0.2, b: 0.36, w: 7 },
      { k: 'line', pts: [[0.8, 18], [40, 18]], a: 0.36, b: 0.5, w: 7 },
      { k: 'line', pts: [[0, 22], [0, 14]], a: 0.5, b: 0.54, c: 'yellow', w: 2.5 },
      T('0.2 s de duda', [0, 9.5], 0.52, 0.62, { size: 2.6, align: 'center', c: 'yellow' }),
      T('49%', [-20, 11], 0.62, 0.7, { size: 4, align: 'center' }),
      T('49%', [20, 11], 0.66, 0.74, { size: 4, align: 'center' }),
      { k: 'line', pts: [[-24, 15], [-16, 7]], a: 0.74, b: 0.78, w: 3 },
      { k: 'line', pts: [[-16, 15], [-24, 7]], a: 0.78, b: 0.81, w: 3 },
      { k: 'line', pts: [[16, 15], [24, 7]], a: 0.81, b: 0.84, w: 3 },
      { k: 'line', pts: [[24, 15], [16, 7]], a: 0.84, b: 0.87, w: 3 },
      T('menos del 70% seguido = no cuenta', [-40, -6], 0.87, 0.97, { size: 3.2 }),
      { k: 'line', pts: [[-40, -9], [8, -9.6]], a: 0.95, b: 1, w: 2.5 },
    ],
    fantasma: () => [
      ...teams(0, 0.4, 4, 0),
      { k: 'ball', pts: [[6, -10], [11, -6], [15, -1.5], [19, 3.5], [24, 9]], a: 0.42, b: 0.62 },
      { k: 'line', pts: [[17, 1.5], [58, 39]], a: 0.55, b: 0.66, dash: [2, 9], arrow: true },
      T('150 km/h a la grada', [36, 30], 0.62, 0.72, { size: 2.6, align: 'right' }),
      { k: 'line', pts: [[22, 31], [40, 27]], a: 0.72, b: 0.76, w: 3 },
      T('imposible', [38, 23.5], 0.74, 0.8, { size: 2.6, align: 'right' }),
      { k: 'line', pts: [[11, -6], [13, -2.6], [15, -1.5]], a: 0.8, b: 0.86, c: 'yellow', dash: [2, 6] },
      T('curva reconstruida', [0, -16], 0.82, 0.9, { size: 2.6 }),
      { k: 'tally', at: [-46, -27], n: 62, a: 0.86, b: 1 },
      T('62 pistas por fotograma', [-46, -31.5], 0.92, 1, { size: 2.4 }),
    ],
    espejo: () => [
      ...teams(0, 0.36),
      { k: 'line', pts: [[0, 36], [0, -36]], a: 0.38, b: 0.5, w: 3.4, dash: [10, 8] },
      T('espejo', [1.6, 30], 0.48, 0.56, { size: 3 }),
      { k: 'line', pts: arcPts([-14, -14], [15, 18], 0.28), a: 0.55, b: 0.66, arrow: true, arrowBoth: true },
      { k: 'line', pts: arcPts([-12, 14], [15, -18], -0.28), a: 0.62, b: 0.72, arrow: true, arrowBoth: true },
      T('3 de foto fija', [-50, -26], 0.72, 0.8, { size: 2.6 }),
      T('3 que ven 30 s antes y después', [6, -26], 0.76, 0.86, { size: 2.6 }),
      T('cambio de camisetas → 0.000000', [-50, -31], 0.86, 0.98, { size: 2.6, c: 'yellow' }),
    ],
    tribunal: () => {
      const rows = [
        ['estadios limpios', '7 de 7 suben'],
        ['cámaras con fallos', '80.5%'],
        ['otro estadio', '+7.1 pts'],
        ['Barça – Athletic', '88.8%'],
      ];
      const items = [T('4 exámenes · 11 partidos', [-42, 27], 0.04, 0.18, { size: 3.4 })];
      rows.forEach((r, i) => {
        const y = 17 - i * 9, a = 0.18 + i * 0.12;
        items.push(T(r[0], [-42, y], a, a + 0.06, { size: 2.8 }));
        items.push({ k: 'line', pts: [[2, y + 1.2], [3.6, y - 0.4], [6.8, y + 3]], a: a + 0.05, b: a + 0.08, c: 'yellow', w: 3 });
        items.push(T(r[1], [10, y], a + 0.07, a + 0.12, { size: 2.8 }));
      });
      items.push(T('descartado:', [-42, -17], 0.72, 0.77, { size: 2.6 }));
      items.push(T('afinar inicios con 1 especialista  −1.5', [-42, -22.5], 0.76, 0.84, { size: 2.4, strike: 'afinar inicios con 1 especialista' }));
      items.push(T('adivinar robos sin cámara  −0.7', [-42, -28], 0.86, 0.93, { size: 2.4, strike: 'adivinar robos sin cámara' }));
      return items;
    },
    banda: () => [
      ...teams(0, 0.34, 2, 4),
      { k: 'circle', at: [12, 33.4], r: 2.4, a: 0.36, b: 0.42 },
      { k: 'ballAt', at: [12, 33.4], a: 0.4, b: 0.42 },
      { k: 'clock', at: [12, 33.4], r: 4.6, a: 0.42, b: 0.62 },
      T('4 s con el balón en las manos', [52.5, 36.4], 0.48, 0.6, { size: 2.5, align: 'right' }),
      T('= tiempo muerto', [52.5, 30.6], 0.56, 0.64, { size: 2.5, align: 'right' }),
      { k: 'ball', pts: [[12, 33.4], [13.4, 30], [15, 25.8], [16.2, 22.2]], a: 0.66, b: 0.78 },
      { k: 'burst', at: [16.2, 22.2], a: 0.78, b: 0.84 },
      T('aquí empieza la posesión', [20, 19.5], 0.82, 0.94, { size: 2.6, c: 'yellow' }),
    ],
    cazador: () => [
      ...teams(0, 0.34),
      { k: 'line', pts: [[-14, -14], [-7, -9], [-1, -4.6]], a: 0.38, b: 0.48, arrow: false, c: 'yellow', dash: [2, 7] },
      { k: 'line', pts: [[-1, -4.6], [4, -1.4], [8, 0.6]], a: 0.46, b: 0.52, dash: [2, 7] },
      { k: 'ball', pts: [[-14, -14], [-7, -9], [-1, -4.6], [-1.8, -6.6]], a: 0.5, b: 0.66 },
      { k: 'burst', at: [-1, -4.6], a: 0.64, b: 0.7 },
      T('0.88 s', [2, -9], 0.68, 0.76, { size: 3, c: 'yellow' }),
      T('frenazo de 12 m/s²', [2, -13], 0.72, 0.82, { size: 2.6 }),
      T('rival a menos de 1.5 m', [2, -17], 0.78, 0.88, { size: 2.6 }),
      { k: 'strip', at: [-40, -27], segs: [['o', 18], ['x', 4], ['o', 22]], a: 0.86, b: 1 },
    ],
    finale: () => [
      T('197 posesiones · 13.7 s de media', [-40, 25], 0.04, 0.2, { size: 3.2 }),
      { k: 'o', at: [-44, 13], a: 0.2, b: 0.24 },
      { k: 'line', pts: [[-40, 13], [-40 + 80 * 0.615, 13]], a: 0.24, b: 0.44, w: 9 },
      T('61.5%', [-40 + 80 * 0.615 + 3, 11.6], 0.44, 0.52, { size: 3.4 }),
      T('Barça', [-40, 6.6], 0.46, 0.52, { size: 2.6 }),
      { k: 'x', at: [-44, -3], a: 0.5, b: 0.54 },
      { k: 'line', pts: [[-40, -3], [-40 + 80 * 0.385, -3]], a: 0.54, b: 0.68, w: 9 },
      T('38.5%', [-40 + 80 * 0.385 + 3, -4.4], 0.68, 0.76, { size: 3.4 }),
      T('Athletic', [-40, -9.4], 0.7, 0.76, { size: 2.6 }),
      T('7 de 7 partidos mejorados', [-40, -22], 0.8, 0.92, { size: 3, c: 'yellow' }),
    ],
  };

  function prepChapter(i) {
    const id = SECTIONS[i].id;
    const def = (CHAPTERS[id] || (() => []))();
    const items = def.map((it, j) => {
      const seed = i * 1000 + j * 17 + 3;
      const o = Object.assign({ c: 'chalk', w: 2.6 }, it);
      if (o.k === 'line' || o.k === 'ball') o.st = prepStroke(o.pts, seed, o.k === 'ball' ? 0.4 : 0.9);
      if (o.k === 'o') o.st = prepStroke(circlePts(o.at, 1.55), seed, 0.5, 4);
      if (o.k === 'circle') o.st = prepStroke(circlePts(o.at, o.r), seed, 0.6, 5);
      if (o.k === 'x') { o.st = prepStroke([[o.at[0] - 1.4, o.at[1] + 1.4], [o.at[0] + 1.4, o.at[1] - 1.4]], seed, 0.4, 4); o.st2 = prepStroke([[o.at[0] + 1.4, o.at[1] + 1.4], [o.at[0] - 1.4, o.at[1] - 1.4]], seed + 1, 0.4, 4); }
      if (o.k === 'arrow' || o.arrow) o.arrow = true;
      return o;
    });
    // the main ball path (for 'nearest' lines)
    items.ball = items.find(o => o.k === 'ball');
    // title above the pitch, written first
    items.title = { k: 'text', s: SECTIONS[i].title || '', a: 0, b: 0.2, size: 2.5, c: 'chalk', align: 'left', screen: true };
    return items;
  }

  // ---------- drawing ----------
  function drawText(c, it, t, strikeT = 0) {
    if (t <= 0 || !it.s) return;
    const px = Math.max(12, it.size * G.s);
    c.font = `${px}px Kalam, "Comic Sans MS", cursive`;
    c.textBaseline = 'alphabetic';
    const w = c.measureText(it.s).width;
    let x, y;
    if (it.screen) { x = G.small ? G.tallyX0 : G.tallyX0; y = G.titleY; }
    else { const q = P(it.at[0], it.at[1]); x = q[0]; y = q[1]; }
    if (it.align === 'center') x -= w / 2; else if (it.align === 'right') x -= w;
    c.save();
    c.beginPath(); c.rect(x - 4, y - px * 1.4, (w + 8) * clamp(t), px * 2); c.clip();
    c.fillStyle = col(INK[it.c] || INK.chalk, 0.94);
    c.fillText(it.s, x, y);
    c.restore();
    if (strikeT > 0) {
      // crossed out across its measured width, whatever size the screen gave it;
      // the figure after a partial strike stays readable
      const sw = typeof it.strike === 'string' ? c.measureText(it.strike).width : w;
      c.strokeStyle = col(INK.chalk, 0.9); c.lineWidth = 2.4; c.lineCap = 'round';
      c.beginPath(); c.moveTo(x - 3, y - px * 0.3); c.lineTo(x - 3 + (sw + 6) * clamp(strikeT), y - px * 0.36); c.stroke();
    }
  }
  function arrowHead(c, st, color, atStart) {
    const n = st.pts.length;
    const [x1, y1] = atStart ? st.pts[0] : st.pts[n - 1];
    const [x0, y0] = atStart ? st.pts[Math.min(3, n - 1)] : st.pts[Math.max(0, n - 4)];
    const a = Math.atan2(y1 - y0, x1 - x0), L = Math.max(8, 1.8 * G.s);
    c.strokeStyle = col(color, 0.95); c.lineWidth = 2.4; c.lineCap = 'round';
    c.beginPath();
    c.moveTo(x1 - L * Math.cos(a - 0.45), y1 - L * Math.sin(a - 0.45)); c.lineTo(x1, y1);
    c.lineTo(x1 - L * Math.cos(a + 0.45), y1 - L * Math.sin(a + 0.45));
    c.stroke();
  }
  function drawItems(c, items, f) {
    const local = it => clamp((f - it.a) / Math.max(0.001, it.b - it.a));
    drawText(c, items.title, clamp(f / 0.2));
    let ballPos = null;
    for (const it of items) {
      const t = local(it);
      if (t <= 0 && it.k !== 'nearest') continue;
      const color = INK[it.c] || INK.chalk;
      const w = it.w * (G.small ? 0.8 : 1);
      switch (it.k) {
        case 'o': case 'circle': strokeTo(c, it.st, t, 3, color); break;
        case 'x': strokeTo(c, it.st, t * 2, 3.2, color); strokeTo(c, it.st2, t * 2 - 1, 3.2, color); break;
        case 'line':
          strokeTo(c, it.st, t, w, color, it.dash);
          if (it.arrow && t >= 1) { arrowHead(c, it.st, color, false); if (it.arrowBoth) arrowHead(c, it.st, color, true); }
          break;
        case 'text': drawText(c, it, t, it.strike ? clamp((f - it.b) / 0.04) : 0); break;
        case 'ball': {
          strokeTo(c, it.st, t, 2, INK.yellow, [1.5, 7]);
          const p = pointAt(it.st, t);
          if (it === items.ball) ballPos = p;
          c.fillStyle = col(INK.yellow, 0.98);
          c.beginPath(); c.arc(p[0], p[1], Math.max(4.5, 0.85 * G.s), 0, Math.PI * 2); c.fill();
          break;
        }
        case 'ballAt': {
          const p = P(it.at[0], it.at[1]);
          c.fillStyle = col(INK.yellow, 0.98 * t);
          c.beginPath(); c.arc(p[0], p[1], Math.max(4.5, 0.85 * G.s), 0, Math.PI * 2); c.fill();
          break;
        }
        case 'clock': {
          const p = P(it.at[0], it.at[1]), R = it.r * G.s;
          for (let k = 0; k < 4; k++) {
            const tk = clamp(t * 4 - k); if (tk <= 0) break;
            const a = -Math.PI / 2 + k * (Math.PI / 2) * 0.55 - 0.9;
            c.strokeStyle = col(INK.chalk, 0.92 * tk); c.lineWidth = 2.6; c.lineCap = 'round';
            c.beginPath(); c.moveTo(p[0] + Math.cos(a) * R, p[1] + Math.sin(a) * R); c.lineTo(p[0] + Math.cos(a) * (R + 1.6 * G.s), p[1] + Math.sin(a) * (R + 1.6 * G.s)); c.stroke();
          }
          break;
        }
        case 'burst': {
          const p = P(it.at[0], it.at[1]);
          for (let k = 0; k < 8; k++) {
            const a = k * Math.PI / 4 + 0.3, r0 = 1.4 * G.s, r1 = r0 + 1.6 * G.s * t;
            c.strokeStyle = col(INK.yellow, 0.95); c.lineWidth = 2.2; c.lineCap = 'round';
            c.beginPath(); c.moveTo(p[0] + Math.cos(a) * r0, p[1] + Math.sin(a) * r0); c.lineTo(p[0] + Math.cos(a) * r1, p[1] + Math.sin(a) * r1); c.stroke();
          }
          break;
        }
        case 'tally': {
          const shown = Math.floor(it.n * t), p0 = P(it.at[0], it.at[1]), gap = 1.15 * G.s, hgt = 2.6 * G.s;
          c.strokeStyle = col(INK.chalk, 0.9); c.lineWidth = 2; c.lineCap = 'round';
          for (let k = 0; k < shown; k++) {
            const g = Math.floor(k / 5), m = k % 5, gx = p0[0] + g * (gap * 5.2);
            c.beginPath();
            if (m < 4) { c.moveTo(gx + m * gap, p0[1]); c.lineTo(gx + m * gap + 1, p0[1] - hgt); }
            else { c.moveTo(gx - gap * 0.5, p0[1] - hgt * 0.2); c.lineTo(gx + gap * 3.6, p0[1] - hgt * 0.8); }
            c.stroke();
          }
          break;
        }
        case 'strip': {
          let x = it.at[0];
          it.segs.forEach(([k, len], j) => {
            const tj = clamp(t * it.segs.length - j); if (tj <= 0) return;
            const a = P(x, it.at[1]), b = P(x + len * tj, it.at[1]);
            c.strokeStyle = col(k === 'o' ? INK.chalk : INK.yellow, 0.92); c.lineWidth = 7; c.lineCap = 'butt';
            c.beginPath(); c.moveTo(a[0], a[1]); c.lineTo(b[0], b[1]); c.stroke();
            c.font = `${Math.max(12, 2.2 * G.s)}px Kalam, cursive`; c.fillStyle = col(INK.chalk, 0.9 * tj);
            c.fillText(k === 'o' ? 'Barça' : 'robo', a[0] + 2, a[1] - 1.6 * G.s);
            x += len + 1.4;
          });
          break;
        }
      }
    }
    // The naive rule, live: the three players nearest to wherever the ball is now
    const near = items.find(o => o.k === 'nearest');
    if (near && ballPos) {
      const t = local(near);
      if (t > 0) {
        const ppl = [...HOME, ...AWAY].map(p => P(p[0], p[1]));
        ppl.sort((a, b) => Math.hypot(a[0] - ballPos[0], a[1] - ballPos[1]) - Math.hypot(b[0] - ballPos[0], b[1] - ballPos[1]));
        c.setLineDash([3, 6]);
        c.strokeStyle = col(INK.chalk, 0.75 * Math.min(1, t * 3)); c.lineWidth = 1.6;
        for (let k = 0; k < 3; k++) { c.beginPath(); c.moveTo(ballPos[0], ballPos[1]); c.lineTo(ppl[k][0], ppl[k][1]); c.stroke(); }
        c.setLineDash([]);
      }
    }
  }

  // The coach's running score under the pitch
  function drawTally(c, chapterIdx, t) {
    const stepIdx = SCORE_OF[chapterIdx];
    const n = SCORE_STEPS.length;
    const x0 = G.tallyX0, x1 = G.tallyX1, y = G.tallyY;
    const slot = (x1 - x0) / (n + 0.5);   // room left over for the final double circle
    const px = Math.max(16, Math.min(34, slot * 0.33));
    c.font = `${px}px Kalam, cursive`;
    c.textBaseline = 'alphabetic';
    for (let k = 0; k <= stepIdx; k++) {
      const s = SCORE_STEPS[k].toFixed(1);
      const x = x0 + k * slot;
      const newest = k === stepIdx;
      const wt = newest && stepIdx > 0 ? clamp((t - 0.25) / 0.5) : 1;
      if (wt <= 0) continue;
      const w = c.measureText(s).width;
      c.save(); c.beginPath(); c.rect(x - 4, y - px * 1.2, (w + 8) * wt, px * 1.7); c.clip();
      c.fillStyle = col(newest ? INK.yellow : INK.chalk, newest ? 0.98 : 0.6);
      c.fillText(s, x, y);
      c.restore();
      if (!newest) {
        // crossed out when the next figure is written
        const st = k === stepIdx - 1 ? clamp(t / 0.25) : 1;
        c.strokeStyle = col(INK.chalk, 0.8); c.lineWidth = 2.4; c.lineCap = 'round';
        c.beginPath(); c.moveTo(x - 3, y - px * 0.28); c.lineTo(x - 3 + (w + 6) * st, y - px * 0.38); c.stroke();
        // arrow to the next figure
        if (k < stepIdx || st >= 1) {
          const ax = x + w + slot * 0.08, bx = x + slot - slot * 0.12, ay = y - px * 0.32;
          if (bx > ax + 6) {
            c.beginPath(); c.moveTo(ax, ay); c.lineTo(bx, ay); c.lineTo(bx - 7, ay - 5); c.moveTo(bx, ay); c.lineTo(bx - 7, ay + 5); c.stroke();
          }
        }
      } else if (chapterIdx === N - 1) {
        // the final figure gets circled twice, then the total gain
        const ct = clamp((t - 0.5) / 0.4);
        c.strokeStyle = col(INK.yellow, 0.95); c.lineWidth = 2.6;
        for (let r = 0; r < 2; r++) {
          const tt = clamp(ct * 2 - r); if (tt <= 0) break;
          c.beginPath();
          c.ellipse(x + w / 2, y - px * 0.35, w * 0.72 + r * 4, px * 0.8 + r * 3, -0.06 + r * 0.05, -2.2, -2.2 + Math.PI * 2.1 * tt);
          c.stroke();
        }
        if (ct >= 1) {
          c.font = `${px * 0.7}px Kalam, cursive`;
          c.fillStyle = col(INK.yellow, 0.95);
          c.fillText('+27.6 pts', x + w * 0.1, y + px * 0.95);
        }
      }
    }
    // label
    c.font = `${Math.max(12, px * 0.5)}px Kalam, cursive`;
    c.fillStyle = col(INK.chalk, 0.55);
    c.fillText('precisión', x0, y - px * 1.25);
  }

  // The coach wipes the board the way anyone does: the felt held flat, passes up and
  // down, each pass a felt-width further right. Only what the felt has covered is gone.
  // Everything here (canvas rub-out, ghost, dust, the eraser itself, the notes' mask in
  // copy-layer.js) reads the same wipe state.
  function wipeAt(e) {
    const L = G.small ? clamp(W * 0.34, 110, 160) : clamp(W * 0.13, 130, 200);  // felt length = band width
    const step = L * 0.86;                                                        // passes overlap a little
    const P = Math.ceil((W + 40) / step);
    const s = clamp(e) * P;
    const p = Math.min(P - 1, Math.floor(s)), t = clamp(s - p);
    const down = p % 2 === 0;
    const x1 = -20 + p * step;
    return {
      L, h: L * 0.36, down,
      x1,                                  // left of here: wiped by earlier passes (full height)
      xDone: x1 + (L - step) * 0.5,        // their overlap with this band
      x2: x1 + L,                          // this pass's band is [x1, x2]
      yc: down ? t * H : H - t * H,        // the felt's leading edge in this pass
      e,
    };
  }

  // What the felt leaves: earlier passes fully gone behind a ragged edge, the current
  // pass gone up to its leading edge, which is feathered, with a few streaks where the
  // felt was not pressed flat. Seeded per seam so the edges hold still while they travel.
  function rubOut(c, w, seam) {
    const r = rng(seam * 97 + 5);
    c.save();
    c.globalCompositeOperation = 'destination-out';
    c.fillStyle = '#000';
    c.beginPath();
    c.moveTo(-20, -20);
    for (let y = -20; y <= H + 20; y += 18) c.lineTo(w.xDone + (r() - 0.5) * 14, y);
    c.lineTo(-20, H + 20);
    c.closePath();
    c.fill();
    const lead = 30;
    const y0 = w.down ? -20 : w.yc + lead * 0.4, y1 = w.down ? w.yc - lead * 0.4 : H + 20;
    if (y1 > y0) c.fillRect(w.x1, y0, w.L, y1 - y0);
    const g = w.down
      ? c.createLinearGradient(0, w.yc - lead * 0.4, 0, w.yc + lead * 0.6)
      : c.createLinearGradient(0, w.yc + lead * 0.4, 0, w.yc - lead * 0.6);
    g.addColorStop(0, 'rgba(0,0,0,1)');
    g.addColorStop(1, 'rgba(0,0,0,0)');
    c.fillStyle = g;
    c.fillRect(w.x1, Math.min(w.yc - lead * 0.6, w.yc - lead * 0.4), w.L, lead);
    for (let i = 0; i < 14; i++) {
      const x = w.x1 + r() * w.L, len = 10 + r() * 40, wd = 2 + r() * 5;
      c.fillStyle = `rgba(0,0,0,${0.3 + r() * 0.6})`;
      c.fillRect(x, w.down ? w.yc : w.yc - len, wd, len);
    }
    c.restore();
  }

  function wipedPath(c, w) {
    c.beginPath();
    c.rect(0, 0, Math.max(0, w.xDone), H);
    if (w.down) c.rect(w.x1, 0, w.L, Math.max(0, w.yc));
    else c.rect(w.x1, w.yc, w.L, Math.max(0, H - w.yc));
  }

  // The felt eraser, seen from the front: a chalk-caked felt pad under a ridged plastic back
  function drawEraser(c, w) {
    const bw = w.L, bh = w.h;
    c.save();
    c.translate(w.x1 + bw / 2, clamp(w.yc, 70, H - 20) + (w.down ? -bh * 0.35 : bh * 0.35));
    c.rotate((w.down ? 1 : -1) * 0.035);
    const rr = (x, y, ww, hh, rad) => { c.beginPath(); c.roundRect ? c.roundRect(x, y, ww, hh, rad) : c.rect(x, y, ww, hh); };
    c.shadowColor = 'rgba(0,0,0,0.5)';
    c.shadowOffsetX = 5; c.shadowOffsetY = 12; c.shadowBlur = 18;
    // felt pad, a little proud of the back on every side
    rr(-bw / 2, -bh / 2, bw, bh, 6);
    c.fillStyle = '#8F958F';
    c.fill();
    c.shadowColor = 'transparent';
    const r = rng(41);
    for (let i = 0; i < 160; i++) {
      c.fillStyle = `rgba(236,238,230,${0.2 + r() * 0.55})`;
      c.fillRect(-bw / 2 + r() * bw, -bh / 2 + r() * bh, 1.5, 1.5);
    }
    // plastic back
    rr(-bw / 2 + 6, -bh / 2 + 5, bw - 12, bh - 10, 8);
    c.fillStyle = '#2E3531';
    c.fill();
    // grip ridges
    c.strokeStyle = 'rgba(255,255,255,0.07)'; c.lineWidth = 2;
    for (let k = -2; k <= 2; k++) {
      c.beginPath(); c.moveTo(-bw * 0.32, k * bh * 0.11); c.lineTo(bw * 0.32, k * bh * 0.11); c.stroke();
    }
    c.strokeStyle = 'rgba(0,0,0,0.35)';
    for (let k = -2; k <= 2; k++) {
      c.beginPath(); c.moveTo(-bw * 0.32, k * bh * 0.11 + 2); c.lineTo(bw * 0.32, k * bh * 0.11 + 2); c.stroke();
    }
    // chalk dust settled on the back
    for (let i = 0; i < 40; i++) {
      c.fillStyle = `rgba(236,238,230,${0.06 + r() * 0.14})`;
      c.fillRect(-bw / 2 + 6 + r() * (bw - 12), -bh / 2 + 5 + r() * (bh - 10), 1.6, 1.6);
    }
    c.restore();
  }

  function ghost(i) {
    if (ghosts.has(i)) return ghosts.get(i);
    const sc = 0.5;
    const a = document.createElement('canvas');
    a.width = Math.ceil(W * sc); a.height = Math.ceil(H * sc);
    const ac = a.getContext('2d');
    ac.setTransform(sc, 0, 0, sc, 0, 0);
    drawItems(ac, shapes[i], 1);
    const g = document.createElement('canvas');
    g.width = a.width; g.height = a.height;
    const gc = g.getContext('2d');
    gc.filter = 'blur(2.5px)';
    gc.drawImage(a, 3, 1);
    ghosts.set(i, g);
    return g;
  }

  // ---------- layout & scroll ----------
  const starts = [];
  function layout() {
    W = window.innerWidth; H = window.innerHeight; vh = H;
    dpr = Math.min(2, window.devicePixelRatio || 1);
    canvas.width = W * dpr; canvas.height = H * dpr;
    canvas.style.width = W + 'px'; canvas.style.height = H + 'px';
    ink.width = W * dpr; ink.height = H * dpr;
    G = computeGeometry();
    paintBoard();
    makeGrain();
    ghosts.clear();
    shapes = SECTIONS.map((_, i) => prepChapter(i));
    const wf = isSmall() ? 1.2 : 1;
    let off = 0;
    SECTIONS.forEach((s, i) => { starts[i] = off * vh; off += (s.scroll || 1.7) * wf; });
    totalPx = off * vh;
    track.style.height = (totalPx + vh) + 'px';
    container.dispatchEvent(new CustomEvent('sw:layout', {
      bubbles: true,
      detail: { vh, totalPx, sections: SECTIONS.map((s, i) => ({ id: s.id, start: starts[i], end: i < N - 1 ? starts[i + 1] : totalPx })) }
    }));
    render();
  }

  function stateAt(y) {
    let k = 0;
    for (let i = 0; i < N; i++) if (y >= starts[i] - ERASE_TO * vh) k = i;
    // eraser crossing into chapter k+1?
    let seam = -1, e = 0;
    if (k < N - 1) {
      const S = starts[k + 1];
      if (y >= S - ERASE_FROM * vh) { seam = k + 1; e = (y - (S - ERASE_FROM * vh)) / ((ERASE_FROM - ERASE_TO) * vh); }
    }
    let f, tallyT;
    if (G.small) {
      // Phones: the notes cover the pitch while they are read, then step aside (copy-layer.js)
      // and the coach draws the play on the open board
      const from = starts[k] + (k === 0 ? MOBILE_READ0 : MOBILE_READ) * vh;
      f = clamp((y - from) / (0.5 * vh));     // done before the eraser's first pass
      tallyT = k === 0 ? 1 : clamp((y - from) / (0.4 * vh));
    } else {
      f = k === 0
        ? Math.min(1, 0.5 * introT + 0.5 * clamp(y / (DRAW_LEN * 0.6 * vh)))
        : clamp((y - (starts[k] - ERASE_TO * vh)) / (DRAW_LEN * vh));
      tallyT = k === 0 ? 1 : clamp((y - (starts[k] - ERASE_TO * vh)) / (0.5 * vh));
    }
    return { k, f: reduce ? 1 : f, seam, e: reduce ? (e > 0.5 ? 1 : 0) : clamp(e), tallyT: reduce ? 1 : tallyT };
  }

  function render() {
    pending = false;
    if (!G) return;
    const y = window.scrollY || window.pageYOffset;
    const st = stateAt(y);
    const w = st.seam > 0 ? wipeAt(st.e) : null;

    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.drawImage(bg, 0, 0);

    // ghosts of chapters already wiped
    ctx.save();
    for (let i = 0; i < st.k; i++) { ctx.globalAlpha = 0.07; ctx.drawImage(ghost(i), 0, 0, canvas.width, canvas.height); }
    if (st.seam > 0) {
      // the chapter being wiped leaves its ghost behind the eraser
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      wipedPath(ctx, w); ctx.clip();
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.globalAlpha = 0.11; ctx.drawImage(ghost(st.k), 0, 0, canvas.width, canvas.height);
    }
    ctx.restore();

    // this frame's chalk
    ictx.setTransform(1, 0, 0, 1, 0, 0);
    ictx.globalCompositeOperation = 'source-over';
    ictx.clearRect(0, 0, ink.width, ink.height);
    ictx.setTransform(dpr, 0, 0, dpr, 0, 0);
    drawItems(ictx, shapes[st.k], st.f);
    if (w) rubOut(ictx, w, st.seam);
    drawTally(ictx, st.k, st.tallyT);
    ictx.setTransform(1, 0, 0, 1, 0, 0);
    ictx.globalCompositeOperation = 'destination-in';
    ictx.fillStyle = grain;
    ictx.fillRect(0, 0, ink.width, ink.height);
    ictx.globalCompositeOperation = 'source-over';
    ctx.drawImage(ink, 0, 0);

    // dust dragged along the band behind the felt, then the eraser itself
    if (w && st.e > 0 && st.e < 1) {
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const trail = H * 0.5;
      const g = w.down
        ? ctx.createLinearGradient(0, w.yc - trail, 0, w.yc)
        : ctx.createLinearGradient(0, w.yc + trail, 0, w.yc);
      g.addColorStop(0, 'rgba(226,232,224,0)');
      g.addColorStop(1, 'rgba(226,232,224,0.07)');
      ctx.fillStyle = g;
      ctx.filter = 'blur(3px)';
      ctx.fillRect(w.x1 + 4, w.down ? w.yc - trail : w.yc, w.L - 8, trail);
      ctx.filter = 'none';
      drawEraser(ctx, w);
      ctx.setTransform(1, 0, 0, 1, 0, 0);
    }

    container.dispatchEvent(new CustomEvent('sw:scroll', {
      bubbles: true,
      detail: {
        y, vh, activeIndex: st.k,
        sectionProgress: st.f,
        eraser: w ? { seam: st.seam, e: st.e, xDone: w.xDone, x1: w.x1, x2: w.x2, yc: w.yc, down: w.down } : { seam: -1, e: 0 },
      }
    }));
  }
  const requestRender = () => { if (!pending) { pending = true; requestAnimationFrame(render); } };

  function intro(now) {
    if (introDone) return;
    if (!introStart) introStart = now;
    introT = clamp((now - introStart) / 2200);
    render();
    if (introT < 1 && window.scrollY < 40) requestAnimationFrame(intro);
    else { introT = 1; introDone = true; requestRender(); }
  }

  function jumpTo(i) {
    const top = starts[i] + (i === 0 ? 0 : 0.5 * vh);
    if (window.bcnLenis && !reduce) { window.bcnLenis.scrollTo(top, { duration: 1.8 }); return; }
    window.scrollTo({ top, behavior: reduce ? 'auto' : 'smooth' });
  }

  window.addEventListener('scroll', requestRender, { passive: true });
  let lastW = window.innerWidth;
  window.addEventListener('resize', () => {
    const coarse = window.matchMedia('(hover: none) and (pointer: coarse)').matches;
    if (coarse && window.innerWidth === lastW) return;   // ignore mobile URL-bar resizes
    lastW = window.innerWidth;
    layout();
  });
  layout();
  const start = () => { layout(); if (!introDone) requestAnimationFrame(intro); };
  if (document.fonts && document.fonts.load) {
    Promise.all([document.fonts.load('24px Kalam'), document.fonts.ready]).then(start, start);
  } else start();

  return { jumpTo };
}

if (typeof window !== 'undefined') window.mountBoard = mountBoard;
