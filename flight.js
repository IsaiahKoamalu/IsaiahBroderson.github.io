/* ============================================================
   Hero: draws the fleet, runs training in small slices between
   frames, and wires up the controls. Sim lives in flight-sim.js.
   ============================================================ */
(function () {
  'use strict';
  const S = FlightSim;
  const REDUCED = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const TOUCH = window.matchMedia('(hover: none)').matches;
  const $ = (id) => document.getElementById(id);

  const MAX_GENERATIONS = 300;   // stop training here to spare the visitor's CPU
  const MAX_IF_STRUGGLING = 900; // landing gets longer if this run has not cracked it yet
  const TRAIN_MS = 9;            // training budget per animation frame
  const TRAINED_AT = 50;         // generation where the fleet is good enough to play with
  const TRAIL = 16;
  const COL = { lead: '#c6ff4a', fleet: '#8fd3ff', hot: '#ffb25a', red: '#ff5b4a', ink: '#f2f0ea' };
  const rgba = (hex, a) => {
    const n = parseInt(hex.slice(1), 16);
    return `rgba(${n >> 16}, ${(n >> 8) & 255}, ${n & 255}, ${a})`;
  };
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const rand = (a, b) => a + Math.random() * (b - a);

  const hero = $('hero'), wrap = $('skywrap'), cv = $('sky'), ctx = cv.getContext('2d');
  const chart = $('chart'), cctx = chart.getContext('2d');
  const net = $('net'), nctx = net.getContext('2d');

  // ---------- State ----------
  let task = 'hunt', mode = 'live', running = !REDUCED, onscreen = true, scrubGen = null;
  const trainers = {};
  const trainer = () => trainers[task] || (trainers[task] = new S.Trainer(task));

  // ---------- Brains trained offline in Python (github.com/IsaiahKoamalu/fleet-rl) ----------
  const python = {};   // task → { status: 'loading' | 'ready' | 'failed', src }
  function fromPolicyFile(doc) {
    const cps = doc.checkpoints;
    return {
      python: true, algo: doc.algo, unit: doc.unit, layers: doc.layers,
      generation: cps.length - 1,
      at: cps.map((c) => c.at),
      snapshots: cps.map((c) => Float64Array.from(c.params)),
      history: cps.slice(1).map((c) => c.metrics)   // history[g - 1] scores snapshot g, as in the live trainer
    };
  }
  function loadPython(t) {
    if (python[t] && python[t].status !== 'failed') return;
    python[t] = { status: 'loading' };
    fetch(`policies/${t}.json`)
      .then((r) => { if (!r.ok) throw new Error(r.status); return r.json(); })
      .then((doc) => {
        python[t] = { status: 'ready', src: fromPolicyFile(doc) };
        if (t === task && mode === 'python') { scrubGen = null; startWave(); updateReadouts(); }
      })
      .catch(() => { python[t] = { status: 'failed' }; if (t === task) updateReadouts(); });
  }
  // What the fleet flies right now: the live trainer, or a loaded Python policy file.
  function source() {
    const b = mode === 'python' && python[task];
    return b && b.src ? b.src : trainer();
  }
  const gauss = () => Math.sqrt(-2 * Math.log(1 - Math.random())) * Math.cos(2 * Math.PI * Math.random());

  let sc, fleet = [], particles = [], rings = [], rocks = [], userRocks = [];
  let beacon = null, beaconIdx = 0, waveT = 0, doneT = 0;
  let trace = S.makeActs(S.LAYERS);   // lead ship's activations, layer by layer
  const pointer = { x: 0, y: 0, at: -1e9 };
  const pointerActive = () => performance.now() - pointer.at < 2500;

  // ---------- Viewport: the 1600×900 world is fitted inside the canvas ----------
  const view = { w: 1, h: 1, s: 1, ox: 0, oy: 0 };
  let stars = null;

  function sizeCanvas(c, g) {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const r = c.getBoundingClientRect();
    c.width = Math.max(1, r.width * dpr); c.height = Math.max(1, r.height * dpr);
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    return r;
  }
  function resize() {
    const r = sizeCanvas(cv, ctx);
    view.w = r.width; view.h = r.height;
    view.s = Math.min(r.width / S.W, r.height / S.H);
    view.ox = (r.width - S.W * view.s) / 2;
    view.oy = (r.height - S.H * view.s) / 2;
    sizeCanvas(chart, cctx);
    sizeCanvas(net, nctx);
    paintStars();
  }
  function paintStars() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    stars = document.createElement('canvas');
    stars.width = view.w * dpr; stars.height = view.h * dpr;
    const g = stars.getContext('2d');
    g.scale(dpr, dpr);
    const n = Math.floor(view.w * view.h / 3800);
    for (let i = 0; i < n; i++) {
      const b = Math.random();
      g.fillStyle = `rgba(${b > 0.85 ? '255,226,190' : b > 0.7 ? '190,216,255' : '235,238,245'}, ${rand(0.12, 0.75)})`;
      g.beginPath();
      g.arc(Math.random() * view.w, Math.random() * view.h, Math.pow(Math.random(), 3) * 1.3 + 0.35, 0, Math.PI * 2);
      g.fill();
    }
  }
  const X = (x) => view.ox + x * view.s;
  const Y = (y) => view.oy + y * view.s;

  // ---------- Waves: each one flies the newest brains ----------
  function buildRock(c) {
    const pts = [], n = 11, seed = Math.floor(c.x * 7 + c.y * 13);
    for (let i = 0; i < n; i++) {
      const wob = 0.82 + (Math.abs(Math.sin(seed + i * 2.4)) * 0.22);
      pts.push([Math.cos((i / n) * Math.PI * 2) * c.r * wob, Math.sin((i / n) * Math.PI * 2) * c.r * wob]);
    }
    return { c, pts };
  }

  function fleetSize() { return view.w < 700 ? 14 : 28; }

  function startWave() {
    const tr = source();
    const gen = scrubGen === null ? tr.generation : scrubGen;
    const mean = tr.snapshots[gen], layers = tr.layers || S.LAYERS;
    trace = S.makeActs(layers);
    sc = S.makeScenario(task, (Math.random() * 1e9) | 0);
    sc.asteroids = sc.asteroids.concat(task === 'field' ? userRocks : []);
    rocks = sc.asteroids.filter((a) => !a.planet).map(buildRock);
    if (task === 'land' && pointerActive()) movePad(pointer.x);
    if (task === 'orbit') { if (orbitR) sc.orbitR = orbitR; if (pointerActive()) setOrbit(pointer.x, pointer.y); }

    beaconIdx = 0;
    beacon = sc.targets[0];
    sc.override = task === 'land' || task === 'orbit' ? null : beacon;
    waveT = 0; doneT = 0;

    fleet = [];
    const n = fleetSize();
    for (let i = 0; i < n; i++) {
      const p = new Float64Array(mean.length);
      if (tr.python) {
        // one trained policy; a little weight noise so the fleet does not fly in lockstep
        for (let k = 0; k < p.length; k++) p[k] = mean[k] + (i === 0 || gen === 0 ? 0 : gauss() * 0.03);
      } else {
        const spread = gen === 0 ? 0.8 : scrubGen === null ? 0.5 : 0.12;
        for (let k = 0; k < p.length; k++) {
          const sd = scrubGen === null ? tr.sigma[k] : 1;
          p[k] = mean[k] + (i === 0 ? 0 : tr.gauss() * sd * spread);
        }
      }
      const ship = S.newShip(sc, task === 'land' ? 0 : 90);
      if (task === 'land') {
        ship.x = clamp(ship.x + rand(-320, 320), 60, S.W - 60);
        ship.y += rand(-40, 40);
      }
      fleet.push({ ship, brain: { p, layers }, lead: i === 0, trail: [], px: ship.x, py: ship.y, pa: ship.a, was: true });
    }
  }

  function movePad(x) {
    sc.padX = clamp(x, 160, S.W - 160);
    sc.targets[0].x = sc.padX;
  }
  // Orbit: the cursor's distance from the planet sets the target orbit, and it sticks between waves.
  let orbitR = 0;
  function setOrbit(x, y) {
    orbitR = sc.orbitR = clamp(Math.hypot(x - S.PLANET.x, y - S.PLANET.y), S.ORBIT_MIN, S.ORBIT_MAX);
  }

  function burst(x, y, color, n, speed) {
    for (let i = 0; i < n && particles.length < 400; i++) {
      const a = Math.random() * Math.PI * 2, v = rand(0.2, 1) * speed;
      particles.push({ x, y, vx: Math.cos(a) * v, vy: Math.sin(a) * v, life: 1, decay: rand(0.02, 0.05), color });
    }
  }

  // ---------- Simulation tick (30 Hz) ----------
  function tick() {
    if (task === 'land') {
      if (pointerActive()) movePad(pointer.x);
    } else if (task === 'orbit') {
      if (pointerActive()) setOrbit(pointer.x, pointer.y);
    } else if (pointerActive()) {
      sc.override = { x: clamp(pointer.x, 30, S.W - 30), y: clamp(pointer.y, 30, S.H - 30) };
    } else {
      sc.override = beacon;
    }

    let alive = 0;
    for (const f of fleet) {
      const s = f.ship;
      f.px = s.x; f.py = s.y; f.pa = s.a;
      if (s.alive) {
        S.stepShip(s, f.brain, sc, f.lead ? trace : undefined);
        f.trail.push(s.x, s.y);
        if (f.trail.length > TRAIL * 2) f.trail.splice(0, 2);
        alive++;
      } else if (f.trail.length) {
        f.trail.splice(0, 2);
      }
      if (f.was && !s.alive) {
        if (s.crashed) burst(s.x, s.y, COL.hot, f.lead ? 34 : 18, 7);
        if (s.landed) rings.push({ x: s.x, y: s.y, r: 8, life: 1, color: COL.lead });
      }
      f.was = s.alive;
    }

    // shared beacon: first ship to touch it moves it along
    if (task !== 'land' && task !== 'orbit' && sc.override === beacon) {
      const hit = fleet.some((f) => f.ship.alive && Math.hypot(f.ship.x - beacon.x, f.ship.y - beacon.y) < S.REACH_R);
      if (hit) {
        rings.push({ x: beacon.x, y: beacon.y, r: S.REACH_R, life: 1, color: COL.lead });
        burst(beacon.x, beacon.y, COL.lead, 14, 4);
        beaconIdx = (beaconIdx + 1) % sc.targets.length;
        beacon = sc.targets[beaconIdx];
      }
    }

    waveT += S.DT;
    if (!alive) doneT += S.DT;
    const tr = source();
    const training = !tr.python && scrubGen === null && tr.generation < generationCap(tr);
    const waveLen = task === 'land' ? 11.5 : training ? 9 : 24;
    if (doneT > 1.1 || waveT > waveLen) startWave();
  }

  // ---------- Drawing ----------
  function drawShip(f, alpha) {
    const s = f.ship;
    const x = X(f.px + (s.x - f.px) * alpha), y = Y(f.py + (s.y - f.py) * alpha);
    const a = f.pa + (s.a - f.pa) * alpha;
    const size = (f.lead ? 15 : 10.5) * Math.max(0.72, view.s);
    const color = f.lead ? COL.lead : COL.fleet;
    ctx.save();
    ctx.translate(x, y); ctx.rotate(a);
    if (s.alive && s.thrust > 0.04) {
      const len = size * (0.7 + s.thrust * 1.7) * rand(0.82, 1.12);
      const g = ctx.createLinearGradient(-size * 0.5, 0, -size * 0.5 - len, 0);
      g.addColorStop(0, rgba(COL.hot, 0.95)); g.addColorStop(1, rgba(COL.red, 0));
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.moveTo(-size * 0.45, -size * 0.3); ctx.lineTo(-size * 0.5 - len, 0); ctx.lineTo(-size * 0.45, size * 0.3);
      ctx.fill();
    }
    if (f.lead) { ctx.shadowColor = color; ctx.shadowBlur = 16; }
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.moveTo(size, 0); ctx.lineTo(-size * 0.72, -size * 0.62); ctx.lineTo(-size * 0.36, 0); ctx.lineTo(-size * 0.72, size * 0.62);
    ctx.closePath(); ctx.fill();
    ctx.restore();
  }

  function drawPlanet() {
    const P = S.PLANET, cx = X(P.x), cy = Y(P.y), r = P.r * view.s;
    // target orbit
    ctx.save();
    ctx.setLineDash([3, 9]);
    ctx.strokeStyle = rgba(COL.lead, 0.45); ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(cx, cy, sc.orbitR * view.s, 0, Math.PI * 2); ctx.stroke();
    ctx.restore();
    // atmosphere glow, then the planet lit from the upper left
    const glow = ctx.createRadialGradient(cx, cy, r * 0.9, cx, cy, r * 1.9);
    glow.addColorStop(0, rgba(COL.fleet, 0.22)); glow.addColorStop(1, rgba(COL.fleet, 0));
    ctx.fillStyle = glow;
    ctx.beginPath(); ctx.arc(cx, cy, r * 1.9, 0, Math.PI * 2); ctx.fill();
    const body = ctx.createRadialGradient(cx - r * 0.4, cy - r * 0.45, r * 0.1, cx, cy, r);
    body.addColorStop(0, '#5d7fa8'); body.addColorStop(0.55, '#26385a'); body.addColorStop(1, '#0d1424');
    ctx.fillStyle = body;
    ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.fill();
  }

  function draw(alpha, now) {
    ctx.clearRect(0, 0, view.w, view.h);
    if (stars) ctx.drawImage(stars, 0, 0, view.w, view.h);

    // asteroids
    for (const r of rocks) {
      ctx.save();
      ctx.translate(X(r.c.x), Y(r.c.y)); ctx.scale(view.s, view.s);
      ctx.beginPath();
      r.pts.forEach((p, i) => ctx[i ? 'lineTo' : 'moveTo'](p[0], p[1]));
      ctx.closePath();
      ctx.fillStyle = '#1b1f2a'; ctx.fill();
      ctx.lineWidth = 2 / view.s; ctx.strokeStyle = '#4a5163'; ctx.stroke();
      ctx.restore();
    }

    // ground + landing pad
    if (task === 'land') {
      const gy = Y(S.GROUND);
      const g = ctx.createLinearGradient(0, gy, 0, view.h);
      g.addColorStop(0, '#171b25'); g.addColorStop(1, '#0a0c12');
      ctx.fillStyle = g; ctx.fillRect(0, gy, view.w, view.h - gy);
      ctx.strokeStyle = 'rgba(255,255,255,0.22)'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(view.w, gy); ctx.stroke();
      const x0 = X(sc.padX - S.PAD_HALF), x1 = X(sc.padX + S.PAD_HALF);
      ctx.save();
      ctx.shadowColor = COL.lead; ctx.shadowBlur = 18;
      ctx.fillStyle = COL.lead; ctx.fillRect(x0, gy - 2, x1 - x0, 4);
      ctx.restore();
    } else if (task === 'orbit') {
      drawPlanet();
    } else {
      // beacon or cursor target
      const t = sc.override, pulse = REDUCED ? 0.5 : (Math.sin(now / 240) + 1) / 2;
      const tx = X(t.x), ty = Y(t.y), R = S.REACH_R * view.s;
      ctx.save();
      ctx.shadowColor = COL.lead; ctx.shadowBlur = 20;
      ctx.strokeStyle = rgba(COL.lead, 0.9); ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(tx, ty, R * (0.55 + pulse * 0.14), 0, Math.PI * 2); ctx.stroke();
      ctx.fillStyle = COL.lead;
      ctx.beginPath(); ctx.arc(tx, ty, 3, 0, Math.PI * 2); ctx.fill();
      ctx.restore();
    }

    // trails
    ctx.globalCompositeOperation = 'lighter';
    ctx.lineCap = 'round';
    for (const f of fleet) {
      const tr = f.trail, n = tr.length / 2;
      for (let i = 1; i < n; i++) {
        ctx.strokeStyle = rgba(f.lead ? COL.lead : COL.fleet, (i / n) * (f.lead ? 0.55 : 0.26));
        ctx.lineWidth = (i / n) * (f.lead ? 3 : 2);
        ctx.beginPath();
        ctx.moveTo(X(tr[(i - 1) * 2]), Y(tr[(i - 1) * 2 + 1]));
        ctx.lineTo(X(tr[i * 2]), Y(tr[i * 2 + 1]));
        ctx.stroke();
      }
    }

    // sensors on the lead ship
    const lead = fleet[0];
    if (lead && lead.ship.alive && task === 'field') {
      S.RAYS.forEach((off, i) => {
        const a = lead.ship.a + off, len = lead.ship.rays[i] * S.RAY_LEN;
        ctx.strokeStyle = rgba(lead.ship.rays[i] < 0.5 ? COL.red : COL.lead, 0.28);
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(X(lead.ship.x), Y(lead.ship.y));
        ctx.lineTo(X(lead.ship.x + Math.cos(a) * len), Y(lead.ship.y + Math.sin(a) * len));
        ctx.stroke();
      });
    }

    // sparks and rings
    for (let i = particles.length - 1; i >= 0; i--) {
      const p = particles[i];
      p.x += p.vx; p.y += p.vy; p.vx *= 0.95; p.vy *= 0.95; p.life -= p.decay;
      if (p.life <= 0) { particles.splice(i, 1); continue; }
      ctx.fillStyle = rgba(p.color, p.life);
      ctx.fillRect(X(p.x) - 1.2, Y(p.y) - 1.2, 2.4, 2.4);
    }
    for (let i = rings.length - 1; i >= 0; i--) {
      const r = rings[i];
      r.r += 3; r.life -= 0.04;
      if (r.life <= 0) { rings.splice(i, 1); continue; }
      ctx.strokeStyle = rgba(r.color, r.life); ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(X(r.x), Y(r.y), r.r * view.s, 0, Math.PI * 2); ctx.stroke();
    }
    ctx.globalCompositeOperation = 'source-over';

    // ships, lead on top
    for (let i = fleet.length - 1; i >= 0; i--) {
      const f = fleet[i];
      if (f.ship.alive || f.ship.landed) drawShip(f, alpha);
    }
  }

  // ---------- Training readouts ----------
  const METRIC = {
    hunt:  { label: 'beacons / run', get: (h) => h.reached.toFixed(1) },
    field: { label: 'survival',      get: (h) => Math.round((1 - h.crashed) * 100) + '%' },
    land:  { label: 'safe landings', get: (h) => Math.round(h.landed * 100) + '%' },
    orbit: { label: 'orbits / run',  get: (h) => (h.orbits || 0).toFixed(2) }
  };

  function generationCap(tr) {
    const struggling = task === 'land' && (!tr.bestHoldout || tr.bestHoldout.landed < 0.75);
    return struggling ? MAX_IF_STRUGGLING : MAX_GENERATIONS;
  }

  function isReady(tr) {
    if (tr.python) return true;
    const h = tr.history[tr.history.length - 1];
    if (!h) return false;
    return task === 'land' ? h.landed >= 0.6 : tr.generation >= TRAINED_AT;
  }

  function updateReadouts() {
    const tr = source(), gen = scrubGen === null ? tr.generation : scrubGen;
    const unit = tr.python ? tr.unit : 'generation';
    $('gen-label').textContent = unit;
    $('gen').textContent = tr.python ? tr.at[gen] : gen;
    $('metric-label').textContent = METRIC[task].label;
    const h = tr.history[gen - 1];
    $('metric').textContent = h ? METRIC[task].get(h) : '—';

    const range = $('scrub');
    range.max = tr.generation;
    if (scrubGen === null) range.value = tr.generation;
    $('live').hidden = scrubGen === null;
    $('live').textContent = tr.python ? 'final' : 'live';
    $('restart').disabled = !!tr.python;

    const status = mode === 'python' && python[task] ? python[task].status : null;
    let msg;
    if (status === 'loading') msg = 'Loading the brain trained in Python…';
    else if (status === 'failed') msg = 'Could not load the Python-trained brain. Showing live training instead.';
    else if (scrubGen !== null) msg = `Rewound to ${unit} ${tr.python ? tr.at[gen] : gen}. This is how they flew back then.`;
    else if (!isReady(tr)) msg = 'Still learning the controls. Give them a few seconds.';
    else if (task === 'hunt') msg = TOUCH ? 'Tap the sky. They will chase it.' : 'Move your cursor. They will chase it.';
    else if (task === 'field') msg = (TOUCH ? 'Tap' : 'Click') + ' to drop an asteroid in their way.';
    else if (task === 'orbit') msg = (TOUCH ? 'Tap' : 'Move your cursor') + ' to raise or lower their orbit.';
    else msg = TOUCH ? 'Tap the sky to move the landing pad.' : 'Move your cursor to slide the landing pad.';
    if (tr.python && scrubGen === null) msg = `${tr.algo.toUpperCase()}, trained in Python. ${msg}`;
    if ($('hint').textContent !== msg) $('hint').textContent = msg;

    drawChart(tr, gen);
  }

  function drawChart(tr, gen) {
    const w = chart.clientWidth, h = chart.clientHeight, H_ = tr.history;
    cctx.clearRect(0, 0, w, h);
    if (H_.length < 2) return;
    const vals = H_.map((v) => v.fitness);
    const lo = Math.min(...vals), hi = Math.max(...vals), span = Math.max(1e-6, hi - lo);
    const px = (i) => (i / Math.max(tr.python ? 1 : MAX_GENERATIONS - 1, H_.length - 1)) * w;
    const py = (v) => h - 3 - ((v - lo) / span) * (h - 6);
    cctx.beginPath();
    vals.forEach((v, i) => cctx[i ? 'lineTo' : 'moveTo'](px(i), py(v)));
    cctx.strokeStyle = COL.lead; cctx.lineWidth = 1.5; cctx.stroke();
    cctx.lineTo(px(vals.length - 1), h); cctx.lineTo(0, h); cctx.closePath();
    cctx.fillStyle = rgba(COL.lead, 0.1); cctx.fill();
    if (scrubGen !== null) {
      cctx.fillStyle = COL.ink;
      cctx.fillRect(px(Math.max(0, gen - 1)) - 0.5, 0, 1, h);
    }
  }

  // Live view of the lead ship's network: nodes glow with activation.
  function drawNet() {
    const w = net.clientWidth, h = net.clientHeight;
    if (!w || !fleet[0]) return;
    nctx.clearRect(0, 0, w, h);
    netMax = Math.max(netPeak, 1e-6); netPeak = 0;
    const { p, layers: cols } = fleet[0].brain, L = cols.length;
    if (trace.length !== L) return;
    const xs = cols.map((_, c) => 8 + ((w - 16) * c) / (L - 1));
    const nodeY = (c, i) => (h * (i + 0.5)) / cols[c];
    let k = 0;
    nctx.lineWidth = 1;
    for (let l = 1; l < L; l++) {
      for (let o = 0; o < cols[l]; o++) {
        for (let i = 0; i < cols[l - 1]; i++) link(xs[l - 1], nodeY(l - 1, i), xs[l], nodeY(l, o), p[k++] * trace[l - 1][i]);
        k++;
      }
    }
    trace.forEach((vals, c) => {
      for (let i = 0; i < cols[c]; i++) {
        const v = Math.min(1, Math.abs(vals[i]));
        nctx.fillStyle = rgba(vals[i] >= 0 ? COL.lead : COL.fleet, 0.2 + 0.8 * v);
        nctx.beginPath(); nctx.arc(xs[c], nodeY(c, i), 2.6, 0, Math.PI * 2); nctx.fill();
      }
    });
  }
  let netMax = 1, netPeak = 0;   // links are scaled to the strongest signal of the last frame
  function link(x0, y0, x1, y1, signal) {
    netPeak = Math.max(netPeak, Math.abs(signal));
    const a = Math.min(0.6, (Math.abs(signal) / netMax) * 0.6);
    if (a < 0.02) return;
    nctx.strokeStyle = rgba(signal >= 0 ? COL.lead : COL.fleet, a);
    nctx.beginPath(); nctx.moveTo(x0, y0); nctx.lineTo(x1, y1); nctx.stroke();
  }

  // ---------- Controls ----------
  function setRunning(r) {
    running = r;
    $('pause').textContent = r ? 'Pause' : 'Resume';
  }
  document.querySelectorAll('#tasks button').forEach((b) => b.addEventListener('click', () => {
    if (task === b.dataset.task) return;
    task = b.dataset.task;
    scrubGen = null;
    document.querySelectorAll('#tasks button').forEach((x) => x.classList.toggle('on', x === b));
    if (mode === 'python') loadPython(task);
    startWave(); updateReadouts();
  }));
  document.querySelectorAll('#modes button').forEach((b) => b.addEventListener('click', () => {
    if (mode === b.dataset.mode) return;
    mode = b.dataset.mode;
    scrubGen = null;
    document.querySelectorAll('#modes button').forEach((x) => x.classList.toggle('on', x === b));
    if (mode === 'python') loadPython(task);
    startWave(); updateReadouts();
  }));
  $('pause').addEventListener('click', () => setRunning(!running));
  $('restart').addEventListener('click', () => {
    if (source().python) return;
    delete trainers[task];
    scrubGen = null; userRocks = [];
    startWave(); updateReadouts();
  });
  $('scrub').addEventListener('input', (e) => {
    const v = +e.target.value;
    scrubGen = v >= source().generation ? null : v;
    startWave(); updateReadouts();
  });
  $('live').addEventListener('click', () => { scrubGen = null; startWave(); updateReadouts(); });

  function toWorld(e) {
    const r = cv.getBoundingClientRect();
    return [(e.clientX - r.left - view.ox) / view.s, (e.clientY - r.top - view.oy) / view.s];
  }
  const overUi = (e) => e.target.closest && e.target.closest('.panel, a, button, input');
  function aim(e) {
    if (overUi(e)) { pointer.at = -1e9; return; }
    [pointer.x, pointer.y] = toWorld(e);
    pointer.at = performance.now();
  }
  hero.addEventListener('pointermove', aim);
  hero.addEventListener('pointerdown', aim);
  hero.addEventListener('pointerleave', () => { pointer.at = -1e9; });
  hero.addEventListener('click', (e) => {
    if (task !== 'field' || overUi(e)) return;
    const [x, y] = toWorld(e);
    if (x < 0 || y < 0 || x > S.W || y > S.H) return;
    const c = { x, y, r: rand(38, 62) };
    userRocks.push(c);
    if (userRocks.length > 10) {
      const old = userRocks.shift();
      sc.asteroids = sc.asteroids.filter((a) => a !== old);
      rocks = rocks.filter((r) => r.c !== old);
    }
    sc.asteroids.push(c);
    rocks.push(buildRock(c));
  });

  // ---------- Main loop ----------
  new IntersectionObserver(([e]) => { onscreen = e.isIntersecting; }).observe(hero);

  let last = performance.now(), acc = 0, frame = 0;
  function loop(now) {
    requestAnimationFrame(loop);
    const elapsed = Math.min(now - last, 60);
    last = now;
    if (!onscreen || document.hidden) return;

    if (running) {
      const tr = source();
      if (!tr.python && tr.generation < generationCap(tr)) tr.run(TRAIN_MS);
      acc += elapsed / 1000;
      while (acc >= S.DT) { tick(); acc -= S.DT; }
    }
    draw(running ? acc / S.DT : 1, now);
    if (frame % 3 === 0) drawNet();
    if (frame % 12 === 0) updateReadouts();
    frame++;
  }

  resize();
  startWave();
  setRunning(running);
  updateReadouts();
  window.addEventListener('resize', resize);
  requestAnimationFrame(loop);
})();

/* ---------- Nav + section reveals ---------- */
(function () {
  const nav = document.querySelector('.topnav');
  const onScroll = () => nav.classList.toggle('scrolled', window.scrollY > 20);
  window.addEventListener('scroll', onScroll, { passive: true });
  onScroll();
  const links = new Map([...nav.querySelectorAll('.links a')].map((a) => [a.getAttribute('href').slice(1), a]));
  const spy = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (!e.isIntersecting) continue;
      links.forEach((a) => a.classList.remove('active'));
      links.get(e.target.id)?.classList.add('active');
    }
  }, { rootMargin: '-45% 0px -50% 0px' });
  document.querySelectorAll('section[id]').forEach((s) => spy.observe(s));

  const reveal = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (e.isIntersecting) { e.target.classList.add('in'); reveal.unobserve(e.target); }
    }
  }, { threshold: 0.12 });
  document.querySelectorAll('.reveal').forEach((el) => reveal.observe(el));
})();
