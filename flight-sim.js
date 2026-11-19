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

  // Orbit: a planet with inverse-square gravity; ships score for circling it near a target radius.
  const PLANET = { x: W / 2, y: H / 2, r: 70 };
  const GM = 1.1e7;         // circular speed at r = 240 is ~214 px/s: one orbit takes ~7 s
  const ORBIT_LEAD = 0.5;   // the ship's "target" is a guide point this far ahead (radians) on the target orbit
  const ORBIT_BAND = 150;   // how far off the target radius still earns credit
  const ORBIT_PULL = 200;   // fitness for closing radial distance to the target orbit: 1 per this many px
  const ORBIT_SCALE = 10;   // fitness per full orbit
  const FUEL = 0.1;         // fitness cost per tick at full thrust: high enough that coasting pays
  const ORBIT_MIN = 150, ORBIT_MAX = 340;

  // Dock: match orbits with a station circling the planet and touch it gently.
  const DOCK_R = 36;        // touching distance
  const DOCK_V = 70;        // docking succeeds below this relative speed; faster is a crash
  const DOCK_BONUS = 8;     // fitness for docking
  const DOCK_PULL = 150;    // shaping: 1 fitness per this many px closer to the station
  const DOCK_NEAR = 200;    // shaping: within this distance, relative speed starts to cost
  const DOCK_SLOW = 60;     // shaping: 1 fitness per this much relative speed shed right at the station
  const DOCK_FUEL = 0.02;   // fitness cost per tick at full thrust
  const DOCK_GAP = [0.2, 1.0], DOCK_DR = 50, DOCK_SPEED = [0.9, 1.1];   // starts: behind (rad), off-orbit (px), speed
  const planetary = (task) => task === 'orbit' || task === 'dock';

  const TASKS = {
    hunt:  { label: 'Hunt',  steps: 300, asteroids: 0, gravity: 0, drag: 0.97 },
    field: { label: 'Asteroid field', steps: 300, asteroids: 10, gravity: 0, drag: 0.94 },   // slow enough to dodge
    land:  { label: 'Land',  steps: 300, asteroids: 0, gravity: GRAVITY, drag: 0.97 },
    orbit: { label: 'Orbit', steps: 300, asteroids: 0, gravity: 0, drag: 1 },                // no drag, or orbits decay
    dock:  { label: 'Dock',  steps: 300, asteroids: 0, gravity: 0, drag: 1 }
  };

  // ---------- Network: inputs → hidden layers (softsign) → outputs ----------
  // Params are flat: for each layer, for each unit, its input weights then its bias.
  // The live trainer uses LAYERS; networks trained in Python (fleet-rl) may be wider or deeper.
  const N_IN = 8 + RAYS.length, N_HID = 10, N_OUT = 2;
  const LAYERS = [N_IN, N_HID, N_OUT];
  const nParams = (layers) => layers.slice(1).reduce((s, o, i) => s + o * (layers[i] + 1), 0);
  const N_PARAMS = nParams(LAYERS);
  const soft = (x) => x / (1 + Math.abs(x));

  // One activation buffer per layer; acts[0] is the observation.
  const makeActs = (layers) => layers.map((n) => new Float64Array(n));
  const scratch = makeActs(LAYERS);

  function forward(p, layers, acts) {
    let k = 0;
    for (let l = 1; l < layers.length; l++) {
      const input = acts[l - 1], output = acts[l], nIn = layers[l - 1];
      for (let o = 0; o < layers[l]; o++) {
        let s = 0;
        for (let i = 0; i < nIn; i++) s += p[k++] * input[i];
        output[o] = soft(s + p[k++]);
      }
    }
    return acts[acts.length - 1];
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
  function makeScenario(task, seed) {
    const r = rng(seed), cfg = TASKS[task];
    const sc = { task, gravity: cfg.gravity, drag: cfg.drag, steps: cfg.steps, asteroids: [], targets: [], override: null, padX: 0, orbitR: 0, stationA: 0 };
    const between = (a, b) => a + r() * (b - a);

    if (task === 'dock') {
      // A station on a circular orbit (radius orbitR); the ship starts behind it on a nearby,
      // roughly circular orbit.
      sc.orbitR = between(200, 320);
      sc.stationA = between(-Math.PI, Math.PI);
      const ang = sc.stationA - between(DOCK_GAP[0], DOCK_GAP[1]);
      const rad = sc.orbitR + between(-DOCK_DR, DOCK_DR);
      const a = between(-Math.PI, Math.PI), v = Math.sqrt(GM / rad) * between(DOCK_SPEED[0], DOCK_SPEED[1]);
      sc.start = { x: PLANET.x + Math.cos(ang) * rad, y: PLANET.y + Math.sin(ang) * rad, a,
                   vx: -Math.sin(ang) * v, vy: Math.cos(ang) * v };
      sc.asteroids = [{ x: PLANET.x, y: PLANET.y, r: PLANET.r, planet: true }];
      return sc;
    }

    if (task === 'orbit') {
      // Start on a rough orbit (60-115% of circular speed); the job is to transfer to the
      // target radius and circularize, not to escape an immediate fall.
      sc.orbitR = between(170, 330);
      const ang = between(-Math.PI, Math.PI), rad = between(190, 400);
      const a = between(-Math.PI, Math.PI), v = Math.sqrt(GM / rad) * between(0.6, 1.15);
      sc.start = { x: PLANET.x + Math.cos(ang) * rad, y: PLANET.y + Math.sin(ang) * rad, a,
                   vx: -Math.sin(ang) * v, vy: Math.cos(ang) * v };
      // The planet doubles as an asteroid: the rays see it and touching it is a crash.
      sc.asteroids = [{ x: PLANET.x, y: PLANET.y, r: PLANET.r, planet: true }];
      return sc;
    }

    if (task === 'land') {
      sc.padX = between(220, W - 220);
      sc.start = { x: between(200, W - 200), y: between(90, 260), a: -Math.PI / 2 + between(-0.5, 0.5),
                   vx: between(-70, 70), vy: between(-20, 40) };
      sc.targets = [{ x: sc.padX, y: GROUND }];
      return sc;
    }

    sc.start = { x: between(200, W - 200), y: between(150, H - 150), a: between(-Math.PI, Math.PI), vx: 0, vy: 0 };
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
        if (sc.asteroids.some((a) => Math.hypot(c.x - a.x, c.y - a.y) < a.r + 70)) continue;
        t = c;
      }
      sc.targets.push(t || { x: W / 2, y: H / 2 });
    }
    return sc;
  }

  function newShip(sc, jitter) {
    const s = sc.start, j = jitter || 0;
    const x = s.x + (Math.random() - 0.5) * j, y = s.y + (Math.random() - 0.5) * j;
    let pot = 0;   // Dock: shaping potential last tick
    if (sc.task === 'dock') {
      const st = stationAt(sc, 0);
      pot = dockPotential(Math.hypot(st.x - x, st.y - y), Math.hypot(s.vx - st.vx, s.vy - st.vy));
    }
    return {
      x, y, a: s.a, vx: s.vx, vy: s.vy, w: 0,
      thrust: 0, turn: 0, alive: true, landed: false, crashed: false,   // landed: touched down or docked
      idx: 0, reached: 0, t: 0, fitness: 0,
      phi: Math.atan2(y - PLANET.y, x - PLANET.x), orbits: 0,   // angle around the planet; orbits credited
      rerr: Math.abs(Math.hypot(x - PLANET.x, y - PLANET.y) - sc.orbitR),   // distance from the target orbit
      pot, prevD: null, rays: new Float64Array(RAYS.length)
    };
  }

  // Dock: the station's position and velocity after t ticks. It circles at circular speed.
  function stationAt(sc, t) {
    const r = sc.orbitR, w = Math.sqrt(GM / (r * r * r)), ang = sc.stationA + w * t * DT;
    const c = Math.cos(ang), s = Math.sin(ang);
    return { x: PLANET.x + c * r, y: PLANET.y + s * r, vx: -s * w * r, vy: c * w * r, a: ang };
  }
  // Shaping for Dock: closer is better, and near the station, slower relative to it is better.
  const dockPotential = (d, rv) => -d / DOCK_PULL - Math.max(0, 1 - d / DOCK_NEAR) * rv / DOCK_SLOW;

  function currentTarget(ship, sc) {
    if (sc.task === 'dock') return stationAt(sc, ship.t);
    if (sc.task === 'orbit') {   // a guide point on the target orbit, a little ahead of the ship
      const ang = Math.atan2(ship.y - PLANET.y, ship.x - PLANET.x) + ORBIT_LEAD;
      return { x: PLANET.x + Math.cos(ang) * sc.orbitR, y: PLANET.y + Math.sin(ang) * sc.orbitR };
    }
    return sc.override || sc.targets[Math.min(ship.idx, sc.targets.length - 1)];
  }

  // ---------- Sensors ----------
  function castRay(x, y, dx, dy, asteroids) {
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
    return best;
  }

  function observe(ship, sc, input) {
    const ca = Math.cos(ship.a), sa = Math.sin(ship.a), t = currentTarget(ship, sc);
    const dx = t.x - ship.x, dy = t.y - ship.y, d = Math.hypot(dx, dy);
    input[0] = soft((dx * ca + dy * sa) / 300);
    input[1] = soft((-dx * sa + dy * ca) / 300);
    input[2] = Math.min(1, d / 900);
    // Velocity relative to the target (only the Dock station moves).
    let vx = ship.vx, vy = ship.vy;
    if (sc.task === 'dock') { vx -= t.vx; vy -= t.vy; }
    input[3] = soft((vx * ca + vy * sa) / 250);
    input[4] = soft((-vx * sa + vy * ca) / 250);
    input[5] = soft(ship.w / 3);
    // Heading relative to "down": the screen's down, or toward the planet in Orbit and Dock.
    // With down = (0, 1) these are just sin(a) and cos(a).
    let ux = 0, uy = 1;
    if (planetary(sc.task)) {
      const gx = PLANET.x - ship.x, gy = PLANET.y - ship.y, g = Math.hypot(gx, gy);
      ux = gx / g; uy = gy / g;
    }
    input[6] = ca * ux + sa * uy;
    input[7] = ca * uy - sa * ux;
    for (let i = 0; i < RAYS.length; i++) {
      const ang = ship.a + RAYS[i];
      const hit = castRay(ship.x, ship.y, Math.cos(ang), Math.sin(ang), sc.asteroids);
      ship.rays[i] = hit / RAY_LEN;
      input[8 + i] = 1 - ship.rays[i];
    }
    return d;
  }

  // ---------- One physics + control step ----------
  // brain: flat params for the default LAYERS, or { p, layers } for any other network.
  // trace: optional activation buffers (makeActs(layers)) to read the network's state afterwards.
  function stepShip(ship, brain, sc, trace) {
    if (!ship.alive) return;
    const p = brain.layers ? brain.p : brain, layers = brain.layers || LAYERS;
    const acts = trace || (brain.layers ? brain.acts || (brain.acts = makeActs(layers)) : scratch);
    const d0 = observe(ship, sc, acts[0]);
    if (ship.prevD === null) ship.prevD = d0;
    const outBuf = forward(p, layers, acts);

    ship.thrust = Math.max(0, outBuf[0]);
    ship.turn = outBuf[1];
    ship.w = (ship.w + ship.turn * TORQUE * DT) * ANG_DAMP;
    ship.a += ship.w * DT;
    ship.vx += Math.cos(ship.a) * ship.thrust * THRUST * DT;
    ship.vy += (Math.sin(ship.a) * ship.thrust * THRUST + sc.gravity) * DT;
    if (planetary(sc.task)) {   // inverse-square pull toward the planet, from the pre-move position
      const gx = PLANET.x - ship.x, gy = PLANET.y - ship.y, r2 = gx * gx + gy * gy;
      const f = GM / (r2 * Math.sqrt(r2)) * DT;
      ship.vx += gx * f; ship.vy += gy * f;
    }
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
    for (let i = 0; i < sc.asteroids.length; i++) {
      const c = sc.asteroids[i];
      if (Math.hypot(ship.x - c.x, ship.y - c.y) < c.r + SHIP_R) {
        ship.alive = false; ship.crashed = true; ship.fitness -= CRASH_PENALTY;
        return;
      }
    }
    if (sc.task === 'orbit') return orbitReward(ship, sc);
    if (sc.task === 'dock') return dockReward(ship, sc);

    const t = currentTarget(ship, sc), d = Math.hypot(t.x - ship.x, t.y - ship.y);
    ship.fitness += (ship.prevD - d) / 500;
    ship.prevD = d;
    if (d < REACH_R) {
      ship.reached++;
      ship.fitness += 1.5;
      if (!sc.override) ship.idx++;
      ship.prevD = null;
    }
  }

  // Credit angle travelled around the planet, scaled by how close the ship is to the target radius
  // and to circular speed there, plus a small reward for closing in on the target radius.
  // Going the wrong way costs credit; thrust costs fuel.
  function orbitReward(ship, sc) {
    const dx = ship.x - PLANET.x, dy = ship.y - PLANET.y, r = Math.hypot(dx, dy);
    const phi = Math.atan2(dy, dx);
    let dphi = phi - ship.phi;
    dphi = Math.atan2(Math.sin(dphi), Math.cos(dphi));
    const vc = Math.sqrt(GM / r), speed = Math.hypot(ship.vx, ship.vy);
    const band = Math.max(0, 1 - Math.abs(r - sc.orbitR) / ORBIT_BAND);
    const match = Math.max(0, 1 - Math.abs(speed - vc) / vc);
    const gain = dphi / (2 * Math.PI) * band * match;
    const err = Math.abs(r - sc.orbitR);
    ship.fitness += gain * ORBIT_SCALE + (ship.rerr - err) / ORBIT_PULL - ship.thrust * FUEL;
    ship.phi = phi;
    ship.rerr = err;
    ship.orbits += gain;
  }

  // Touching the station slowly docks (landed); touching it fast is a crash.
  // Otherwise potential-based shaping toward it, and a fuel cost.
  function dockReward(ship, sc) {
    const st = stationAt(sc, ship.t);
    const d = Math.hypot(st.x - ship.x, st.y - ship.y), rv = Math.hypot(ship.vx - st.vx, ship.vy - st.vy);
    if (d < DOCK_R) {
      ship.alive = false;
      if (rv < DOCK_V) { ship.landed = true; ship.fitness += DOCK_BONUS; }
      else { ship.crashed = true; ship.fitness -= CRASH_PENALTY; }
      return;
    }
    const pot = dockPotential(d, rv);
    ship.fitness += pot - ship.pot - ship.thrust * DOCK_FUEL;
    ship.pot = pot;
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
    if (ship.x < 0 || ship.x > W || ship.y < -200) {
      ship.alive = false; ship.crashed = true;
      ship.fitness = landScore(off, speed, tilt) - 2;
      return;
    }
    ship.fitness = landScore(off, speed, tilt) - 1 - (GROUND - ship.y) / H;
  }
  const landScore = (off, speed, tilt) => 3 - off / 400 - speed / 220 - tilt;

  // ---------- Evaluate one network over a set of scenarios ----------
  function evaluate(params, scenarios) {
    let total = 0, reached = 0, landed = 0, crashed = 0, orbits = 0;
    for (let i = 0; i < scenarios.length; i++) {
      const sc = scenarios[i], ship = newShip(sc);
      for (let t = 0; t < sc.steps && ship.alive; t++) stepShip(ship, params, sc);
      total += ship.fitness;
      reached += ship.reached;
      landed += ship.landed ? 1 : 0;
      crashed += ship.crashed ? 1 : 0;
      orbits += ship.orbits;
    }
    const n = scenarios.length;
    return { fitness: total / n, reached: reached / n, landed: landed / n, crashed: crashed / n, orbits: orbits / n };
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
    this.validation = [];
    for (let i = 0; i < 16; i++) this.validation.push(makeScenario(task, 900001 + i * 17));
    // Maps never used for training or for picking the best network: the score shown to visitors.
    this.holdout = [];
    for (let i = 0; i < 40; i++) this.holdout.push(makeScenario(task, 700001 + i * 23));
    this.bestHoldout = null;
    this._begin();
  }
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
      this.scenarios.push(makeScenario(this.task, (this.rand() * 1e9) | 0));
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
    W, H, DT, GROUND, PAD_HALF, SHIP_R, REACH_R, RAY_LEN, RAYS, TASKS, PLANET, GM, ORBIT_MIN, ORBIT_MAX, DOCK_R,
    N_IN, N_HID, N_OUT, N_PARAMS, LAYERS, nParams, makeActs,
    makeScenario, newShip, stepShip, evaluate, Trainer, currentTarget, stationAt
  };
})();
