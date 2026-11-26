/* ============================================================
   Flight sim + trainer. No DOM in this file.
   Ships are flown by a small neural network, trained with the
   cross-entropy method (CEM): sample a population of networks,
   keep the best, refit the sampling distribution, repeat.
   ============================================================ */
var FlightSim = (function () {
  'use strict';

  const W = 1600, H = 900, DT = 1 / 30;
  const THRUST = 560, TORQUE = 16, ANG_DAMP = 0.9, CRASH_PENALTY = 4;
  const SHIP_R = 12, REACH_R = 42, RAY_LEN = 280;
  const GROUND = 830, PAD_HALF = 80, GRAVITY = 130;
  const RAYS = [-1.05, -0.5, 0, 0.5, 1.05];
  const RESTART_EVERY = 70;
  const WALL_R = 7;          // half the thickness of a wall the visitor draws
  const ADAPT_GENS = 160;    // extra training a change to the world buys

  const TASKS = {
    hunt:  { label: 'Hunt',  steps: 300, asteroids: 0, gravity: 0, drag: 0.97 },
    field: { label: 'Asteroid field', steps: 300, asteroids: 10, gravity: 0, drag: 0.94 },   // slow enough to dodge
    land:  { label: 'Land',  steps: 300, asteroids: 0, gravity: GRAVITY, drag: 0.97 }
  };

  // ---------- Network: inputs → hidden (softsign) → outputs ----------
  const N_IN = 8 + RAYS.length, N_HID = 10, N_OUT = 2;
  const N_PARAMS = N_IN * N_HID + N_HID + N_HID * N_OUT + N_OUT;
  const soft = (x) => x / (1 + Math.abs(x));

  const obs = new Float64Array(N_IN), hid = new Float64Array(N_HID), out = new Float64Array(N_OUT);

  function forward(p, input, hiddenOut, output) {
    let k = 0;
    for (let h = 0; h < N_HID; h++) {
      let s = 0;
      for (let i = 0; i < N_IN; i++) s += p[k++] * input[i];
      hiddenOut[h] = soft(s + p[k++]);
    }
    for (let o = 0; o < N_OUT; o++) {
      let s = 0;
      for (let h = 0; h < N_HID; h++) s += p[k++] * hiddenOut[h];
      output[o] = soft(s + p[k++]);
    }
  }

  // ---------- Seeded random numbers, so every network faces the same tests ----------
  function rng(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // ---------- Scenarios ----------
  // Distance from a point to a wall segment.
  function segDist(px, py, w) {
    const ex = w.x2 - w.x1, ey = w.y2 - w.y1, len2 = ex * ex + ey * ey || 1;
    const u = Math.max(0, Math.min(1, ((px - w.x1) * ex + (py - w.y1) * ey) / len2));
    return Math.hypot(px - (w.x1 + u * ex), py - (w.y1 + u * ey));
  }
  // How far a point is from the nearest thing it could hit.
  function clearance(x, y, sc) {
    let d = Infinity;
    for (const a of sc.asteroids) d = Math.min(d, Math.hypot(x - a.x, y - a.y) - a.r);
    for (const w of sc.walls) d = Math.min(d, segDist(x, y, w) - WALL_R);
    return d;
  }

  // `world` holds what the visitor has changed: their rocks, walls and gravity.
  function makeScenario(task, seed, world) {
    const r = rng(seed), cfg = TASKS[task], wd = world || {};
    const sc = { task, gravity: wd.gravity == null ? cfg.gravity : wd.gravity, drag: cfg.drag, steps: cfg.steps,
                 asteroids: (wd.rocks || []).slice(), walls: (wd.walls || []).slice(), targets: [], override: null, padX: 0 };
    const between = (a, b) => a + r() * (b - a);

    if (task === 'land') {
      sc.padX = between(220, W - 220);
      for (let tries = 0; tries < 30; tries++) {
        sc.start = { x: between(200, W - 200), y: between(90, 260), a: -Math.PI / 2 + between(-0.5, 0.5),
                     vx: between(-70, 70), vy: between(-20, 40) };
        if (clearance(sc.start.x, sc.start.y, sc) > 70) break;
      }
      sc.targets = [{ x: sc.padX, y: GROUND }];
      return sc;
    }

    for (let tries = 0; tries < 30; tries++) {
      sc.start = { x: between(200, W - 200), y: between(150, H - 150), a: between(-Math.PI, Math.PI), vx: 0, vy: 0 };
      if (clearance(sc.start.x, sc.start.y, sc) > 90) break;
    }
    for (let i = 0; i < cfg.asteroids; i++) {
      for (let tries = 0; tries < 20; tries++) {
        const c = { x: between(60, W - 60), y: between(60, H - 60), r: between(32, 66) };
        if (Math.hypot(c.x - sc.start.x, c.y - sc.start.y) < c.r + 120) continue;
        sc.asteroids.push(c);
        break;
      }
    }
    for (let i = 0; i < 12; i++) {
      let t = null;
      for (let tries = 0; tries < 40 && !t; tries++) {
        const c = { x: between(90, W - 90), y: between(90, H - 90) };
        const prev = i ? sc.targets[i - 1] : sc.start;
        if (Math.hypot(c.x - prev.x, c.y - prev.y) < 300) continue;
        if (clearance(c.x, c.y, sc) < 70) continue;
        t = c;
      }
      sc.targets.push(t || { x: W / 2, y: H / 2 });
    }
    return sc;
  }

  function newShip(sc, jitter) {
    const s = sc.start, j = jitter || 0;
    return {
      x: s.x + (Math.random() - 0.5) * j, y: s.y + (Math.random() - 0.5) * j,
      a: s.a, vx: s.vx, vy: s.vy, w: 0,
      thrust: 0, turn: 0, alive: true, landed: false, crashed: false,
      idx: 0, reached: 0, t: 0, fitness: 0,
      prevD: null, rays: new Float64Array(RAYS.length)
    };
  }

  const currentTarget = (ship, sc) => sc.override || sc.targets[Math.min(ship.idx, sc.targets.length - 1)];

  // ---------- Sensors ----------
  function castRay(x, y, dx, dy, asteroids, walls) {
    let best = RAY_LEN;
    if (dx > 1e-6) best = Math.min(best, (W - x) / dx);
    else if (dx < -1e-6) best = Math.min(best, -x / dx);
    if (dy > 1e-6) best = Math.min(best, (H - y) / dy);
    else if (dy < -1e-6) best = Math.min(best, -y / dy);
    for (let i = 0; i < asteroids.length; i++) {
      const c = asteroids[i], fx = c.x - x, fy = c.y - y, along = fx * dx + fy * dy;
      if (along < 0) continue;
      const off2 = fx * fx + fy * fy - along * along, r2 = c.r * c.r;
      if (off2 > r2) continue;
      const hit = along - Math.sqrt(r2 - off2);
      if (hit < best) best = Math.max(0, hit);
    }
    for (let i = 0; i < walls.length; i++) {
      const w = walls[i], ex = w.x2 - w.x1, ey = w.y2 - w.y1, den = dx * ey - dy * ex;
      if (Math.abs(den) < 1e-9) continue;
      const qx = w.x1 - x, qy = w.y1 - y;
      const t = (qx * ey - qy * ex) / den, u = (qx * dy - qy * dx) / den;
      if (t >= 0 && u >= 0 && u <= 1 && t - WALL_R < best) best = Math.max(0, t - WALL_R);
    }
    return best;
  }
  function hitsObstacle(ship, sc) {
    for (let i = 0; i < sc.asteroids.length; i++) {
      const c = sc.asteroids[i];
      if (Math.hypot(ship.x - c.x, ship.y - c.y) < c.r + SHIP_R) return true;
    }
    for (let i = 0; i < sc.walls.length; i++) if (segDist(ship.x, ship.y, sc.walls[i]) < WALL_R + SHIP_R) return true;
    return false;
  }

  function observe(ship, sc, input) {
    const ca = Math.cos(ship.a), sa = Math.sin(ship.a), t = currentTarget(ship, sc);
    const dx = t.x - ship.x, dy = t.y - ship.y, d = Math.hypot(dx, dy);
    input[0] = soft((dx * ca + dy * sa) / 300);
    input[1] = soft((-dx * sa + dy * ca) / 300);
    input[2] = Math.min(1, d / 900);
    input[3] = soft((ship.vx * ca + ship.vy * sa) / 250);
    input[4] = soft((-ship.vx * sa + ship.vy * ca) / 250);
    input[5] = soft(ship.w / 3);
    input[6] = sa;
    input[7] = ca;
    for (let i = 0; i < RAYS.length; i++) {
      const ang = ship.a + RAYS[i];
      const hit = castRay(ship.x, ship.y, Math.cos(ang), Math.sin(ang), sc.asteroids, sc.walls);
      ship.rays[i] = hit / RAY_LEN;
      input[8 + i] = 1 - ship.rays[i];
    }
    return d;
  }

  // ---------- One physics + control step ----------
  function stepShip(ship, params, sc, trace) {
    if (!ship.alive) return;
    const hiddenBuf = trace ? trace.hidden : hid, outBuf = trace ? trace.output : out, inBuf = trace ? trace.input : obs;
    const d0 = observe(ship, sc, inBuf);
    if (ship.prevD === null) ship.prevD = d0;
    forward(params, inBuf, hiddenBuf, outBuf);

    ship.thrust = Math.max(0, outBuf[0]);
    ship.turn = outBuf[1];
    ship.w = (ship.w + ship.turn * TORQUE * DT) * ANG_DAMP;
    ship.a += ship.w * DT;
    ship.vx += Math.cos(ship.a) * ship.thrust * THRUST * DT;
    ship.vy += (Math.sin(ship.a) * ship.thrust * THRUST + sc.gravity) * DT;
    ship.vx *= sc.drag; ship.vy *= sc.drag;
    ship.x += ship.vx * DT; ship.y += ship.vy * DT;
    ship.t++;

    if (sc.task === 'land') return stepLand(ship, sc);

    // walls cost momentum and a small penalty
    if (ship.x < SHIP_R || ship.x > W - SHIP_R) {
      ship.x = Math.min(W - SHIP_R, Math.max(SHIP_R, ship.x)); ship.vx *= -0.3; ship.fitness -= 0.04;
    }
    if (ship.y < SHIP_R || ship.y > H - SHIP_R) {
      ship.y = Math.min(H - SHIP_R, Math.max(SHIP_R, ship.y)); ship.vy *= -0.3; ship.fitness -= 0.04;
    }
    if (hitsObstacle(ship, sc)) {
      ship.alive = false; ship.crashed = true; ship.fitness -= CRASH_PENALTY;
      return;
    }

    const t =currentTarget(ship, sc), d = Math.hypot(t.x - ship.x, t.y - ship.y);
    ship.fitness += (ship.prevD - d) / 500;
    ship.prevD = d;
    if (d < REACH_R) {
      ship.reached++;
      ship.fitness += 1.5;
      if (!sc.override) ship.idx++;
      ship.prevD = null;
    }
  }

  function stepLand(ship, sc) {
    const off = Math.abs(ship.x - sc.padX);
    const tilt = Math.abs(Math.atan2(Math.sin(ship.a + Math.PI / 2), Math.cos(ship.a + Math.PI / 2)));
    const speed = Math.hypot(ship.vx, ship.vy);

    if (ship.y >= GROUND - SHIP_R) {
      ship.y = GROUND - SHIP_R;
      ship.alive = false;
      const soft = off < PAD_HALF && speed < 110 && tilt < 0.45;
      ship.landed = soft; ship.crashed = !soft;
      ship.fitness = landScore(off, speed, tilt) + (soft ? 6 : 0);
      return;
    }
    if (ship.x < 0 || ship.x > W || ship.y < -200 || hitsObstacle(ship, sc)) {
      ship.alive = false; ship.crashed = true;
      ship.fitness = landScore(off, speed, tilt) - 2;
      return;
    }
    ship.fitness = landScore(off, speed, tilt) - 1 - (GROUND - ship.y) / H;
  }
  const landScore = (off, speed, tilt) => 3 - off / 400 - speed / 220 - tilt;

  // ---------- Evaluate one network over a set of scenarios ----------
  function evaluate(params, scenarios) {
    let total = 0, reached = 0, landed = 0, crashed = 0;
    for (let i = 0; i < scenarios.length; i++) {
      const sc = scenarios[i], ship = newShip(sc);
      for (let t = 0; t < sc.steps && ship.alive; t++) stepShip(ship, params, sc);
      total += ship.fitness;
      reached += ship.reached;
      landed += ship.landed ? 1 : 0;
      crashed += ship.crashed ? 1 : 0;
    }
    const n = scenarios.length;
    return { fitness: total / n, reached: reached / n, landed: landed / n, crashed: crashed / n };
  }

  // ---------- Trainer (cross-entropy method), runs in small time slices ----------
  function Trainer(task, opts) {
    const o = opts || {};
    this.task = task;
    this.pop = o.pop || 72;
    this.nElite = o.elite || 10;
    this.nScen = o.scenarios || 4;
    this.mean = new Float64Array(N_PARAMS);
    this.sigma = new Float64Array(N_PARAMS).fill(o.sigma || 0.8);
    this.rand = rng(o.seed === undefined ? (Math.random() * 1e9) | 0 : o.seed);
    this.generation = 0;
    this.sinceRestart = 0;
    this.best = this.mean.slice();
    this.bestVal = null;
    this.history = [];
    this.snapshots = [this.mean.slice()];
    this.world = { rocks: [], walls: [], gravity: null };
    this.adaptUntil = 0;
    this.change = null;
    this._tests();
    this.bestHoldout = null;
    this._begin();
  }
  Trainer.prototype._tests = function () {
    this.validation = [];
    for (let i = 0; i < 16; i++) this.validation.push(makeScenario(this.task, 900001 + i * 17, this.world));
    // Maps never used for training or for picking the best network: the score shown to visitors.
    this.holdout = [];
    for (let i = 0; i < 40; i++) this.holdout.push(makeScenario(this.task, 700001 + i * 23, this.world));
  };
  // The visitor changed the world. Re-test the current best there, then keep training so it adapts.
  Trainer.prototype.setWorld = function (world) {
    this.world = { rocks: world.rocks.slice(), walls: world.walls.slice(), gravity: world.gravity };
    this._tests();
    // Several changes in a row count as one: compare against the fleet before the first of them.
    const c = this.change, pending = c && !c.done && this.generation < this.adaptUntil;
    const before = pending ? c.before : this.bestHoldout;
    this.bestVal = evaluate(this.best, this.validation);
    this.bestHoldout = evaluate(this.best, this.holdout);
    if (this.history.length) this.history[this.history.length - 1] = this.bestHoldout;
    this.mean.set(this.best);
    for (let i = 0; i < N_PARAMS; i++) this.sigma[i] = Math.max(this.sigma[i], 0.25);
    this.sinceRestart = 0;
    this.adaptUntil = this.generation + ADAPT_GENS;
    this.change = { gen: pending ? c.gen : this.generation, before, after: this.bestHoldout, done: 0, easy: false };
    if (before) this.change.easy = score(this.task, this.bestHoldout) >= recovered(this.task, before);
    this._begin();
  };
  // The one number each task is judged by, and what counts as having recovered it.
  const score = (task, h) => (task === 'hunt' ? h.reached : task === 'field' ? 1 - h.crashed : h.landed);
  const recovered = (task, before) => score(task, before) * 0.95 - 1e-9;
  Trainer.prototype.gauss = function () {
    const u = 1 - this.rand(), v = this.rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
  Trainer.prototype.sample = function () {
    const p = new Float64Array(N_PARAMS);
    for (let i = 0; i < N_PARAMS; i++) p[i] = this.mean[i] + this.sigma[i] * this.gauss();
    return p;
  };
  Trainer.prototype._begin = function () {
    this.cursor = 0;
    this.members = [];
    this.scenarios = [];
    for (let i = 0; i < this.nScen; i++) {
      this.scenarios.push(makeScenario(this.task, (this.rand() * 1e9) | 0, this.world));
    }
  };
  Trainer.prototype._finish = function () {
    this.members.sort((a, b) => b.fitness - a.fitness);
    const elite = this.members.slice(0, this.nElite);
    const extra = 0.25 * Math.max(0, 1 - this.sinceRestart / 60);
    for (let i = 0; i < N_PARAMS; i++) {
      let m = 0, v = 0;
      for (const e of elite) m += e.params[i];
      m /= elite.length;
      for (const e of elite) v += (e.params[i] - m) * (e.params[i] - m);
      this.mean[i] = m;
      this.sigma[i] = Math.sqrt(v / elite.length + extra * extra) + 0.01;
    }
    this.generation++;
    this.sinceRestart++;

    // Keep the best network seen so far: that is what the fleet flies, so it never gets worse.
    const val = evaluate(this.mean, this.validation);
    if (!this.bestVal || val.fitness >= this.bestVal.fitness) {
      this.bestVal = val;
      this.best = this.mean.slice();
      this.bestHoldout = evaluate(this.best, this.holdout);
    }
    this.history.push(this.bestHoldout);
    this.snapshots.push(this.best.slice());
    const c = this.change;
    if (c && c.before && !c.easy && !c.done && score(this.task, this.bestHoldout) >= recovered(this.task, c.before)) {
      c.done = this.generation - c.gen;
    }

    // Landing can get stuck in a dead end (e.g. "just fall"). If it has, start the search over.
    if (this.task === 'land' && this.sinceRestart >= RESTART_EVERY && this.bestVal.landed < 0.75) {
      this.mean.fill(0);
      this.sigma.fill(0.8);
      this.sinceRestart = 0;
    }
    this._begin();
  };
  // Work for up to `ms` milliseconds. Returns how many generations finished.
  Trainer.prototype.run = function (ms) {
    const end = Date.now() + ms;
    let done = 0;
    do {
      const params = this.sample();
      this.members.push({ params, fitness: evaluate(params, this.scenarios).fitness });
      if (++this.cursor >= this.pop) { this._finish(); done++; }
    } while (Date.now() < end);
    return done;
  };

  return {
    W, H, DT, GROUND, PAD_HALF, SHIP_R, REACH_R, RAY_LEN, RAYS, TASKS, WALL_R, GRAVITY,
    N_IN, N_HID, N_OUT, N_PARAMS,
    makeScenario, newShip, stepShip, evaluate, Trainer, currentTarget
  };
})();
