// sokoban-solver-01-run-20261121 entry: push-optimal room-by-room Sokoban solver.
// Written fresh for this entry. Node 22, no dependencies, no network.
//
// Idea: every level the generator builds is one hallway plus 1-4 rooms (6x6 slots,
// one door each, 3-4 boxes per room). In an optimal solution no box ever leaves
// its room, so each room is solved independently with a push-optimal A* and the
// room solutions are joined with hallway walks. Total pushes then equal the sum
// of the rooms' optima, which is the level's push-optimal count P*.
//
// Room search: A* over pushes. A state is (sorted box cells, canonical player
// cell) where the canonical cell is the smallest cell of the player's reachable
// region, so states that differ only by where the player stands inside one
// region collapse. Heuristic: minimum assignment of boxes to goals by BFS
// distance (admissible: one push moves one box one step). Pruning: dead squares
// (reverse-pull reachability from the goals), 2x2 wall/box blocks with no goal,
// and states with no finite box-goal assignment. Work is bounded by expansion
// caps, never by the wall clock, so two runs are byte-identical.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const IN_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

const DX = [0, 0, -1, 1];
const DY = [-1, 1, 0, 0];
const LETTER = ['u', 'd', 'l', 'r'];
const OPP = [1, 0, 3, 2];

// ---------------------------------------------------------------- parsing

interface Level {
  w: number;
  h: number;
  wall: Uint8Array;
  goal: Uint8Array;
  boxes: number[];
  player: number;
}

function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goal = new Uint8Array(w * h);
  const boxes: number[] = [];
  let player = 0;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      if (ch === '#') continue;
      const i = y * w + x;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goal[i] = 1;
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  boxes.sort((a, b) => a - b);
  return { w, h, wall, goal, boxes, player };
}

// Apply a move string to a grid state; null on an illegal move.
function applyMoves(
  L: Level,
  moves: string,
  player: number,
  boxes: number[],
): { player: number; boxes: number[] } | null {
  let p = player;
  const bs = boxes.slice();
  for (let k = 0; k < moves.length; k++) {
    const d = LETTER.indexOf(moves[k]);
    if (d < 0) return null;
    const nx = (p % L.w) + DX[d];
    const ny = ((p / L.w) | 0) + DY[d];
    if (nx < 0 || ny < 0 || nx >= L.w || ny >= L.h) return null;
    const n = ny * L.w + nx;
    if (L.wall[n]) return null;
    const at = bs.indexOf(n);
    if (at >= 0) {
      const bx = nx + DX[d];
      const by = ny + DY[d];
      if (bx < 0 || by < 0 || bx >= L.w || by >= L.h) return null;
      const b = by * L.w + bx;
      if (L.wall[b] || bs.includes(b)) return null;
      bs[at] = b;
      bs.sort((a, b2) => a - b2);
    }
    p = n;
  }
  return { player: p, boxes: bs };
}

// Replay exactly the way the checker does: legal moves and every box on a goal.
function verifySolution(L: Level, moves: string): boolean {
  const r = applyMoves(L, moves, L.player, L.boxes);
  if (r === null) return false;
  return r.boxes.every((b) => L.goal[b] === 1);
}

// Shortest walk (letters) from `from` to `to` over non-wall cells avoiding
// `blocked`; null when unreachable. Fixed direction order: deterministic.
function walkPath(
  L: Level,
  from: number,
  to: number,
  blocked: ReadonlySet<number>,
): string | null {
  if (from === to) return '';
  const prev = new Int32Array(L.w * L.h).fill(-2);
  const prevDir = new Int8Array(L.w * L.h).fill(-1);
  const q: number[] = [from];
  prev[from] = -1;
  for (let head = 0; head < q.length; head++) {
    const c = q[head];
    const cx = c % L.w;
    const cy = (c / L.w) | 0;
    for (let d = 0; d < 4; d++) {
      const nx = cx + DX[d];
      const ny = cy + DY[d];
      if (nx < 0 || ny < 0 || nx >= L.w || ny >= L.h) continue;
      const n = ny * L.w + nx;
      if (L.wall[n] || blocked.has(n) || prev[n] !== -2) continue;
      prev[n] = c;
      prevDir[n] = d;
      if (n === to) {
        let s = '';
        let cur = n;
        while (cur !== from) {
          s = LETTER[prevDir[cur]] + s;
          cur = prev[cur];
        }
        return s;
      }
      q.push(n);
    }
  }
  return null;
}

// ---------------------------------------------------------------- rooms

interface Comp {
  cells: number[];
  boxes: number[];
  goals: number[];
  doors: number[];
}

// Split the level into the hallway row and the room components off it.
function decompose(L: Level): Comp[] | null {
  const hw = (L.player / L.w) | 0;
  const seen = new Uint8Array(L.w * L.h);
  const comps: Comp[] = [];
  for (let y = 0; y < L.h; y++) {
    if (y === hw) continue;
    for (let x = 0; x < L.w; x++) {
      const i = y * L.w + x;
      if (L.wall[i] || seen[i]) continue;
      const cells: number[] = [];
      const q = [i];
      seen[i] = 1;
      for (let head = 0; head < q.length; head++) {
        const c = q[head];
        cells.push(c);
        const cx = c % L.w;
        const cy = (c / L.w) | 0;
        for (let d = 0; d < 4; d++) {
          const nx = cx + DX[d];
          const ny = cy + DY[d];
          if (nx < 0 || ny < 0 || nx >= L.w || ny >= L.h) continue;
          const n = ny * L.w + nx;
          if (ny === hw || L.wall[n] || seen[n]) continue;
          seen[n] = 1;
          q.push(n);
        }
      }
      const boxes = cells.filter((c) => L.boxes.includes(c));
      const goals = cells.filter((c) => L.goal[c] === 1);
      const doorSet = new Set<number>();
      for (const c of cells) {
        const cx = c % L.w;
        const cy = (c / L.w) | 0;
        for (let d = 0; d < 4; d++) {
          const nx = cx + DX[d];
          const ny = cy + DY[d];
          if (ny === hw && nx >= 0 && nx < L.w && !L.wall[ny * L.w + nx]) {
            doorSet.add(ny * L.w + nx);
          }
        }
      }
      comps.push({ cells, boxes, goals, doors: [...doorSet].sort((a, b) => a - b) });
    }
  }
  const boxed = comps.flatMap((c) => c.boxes);
  if (boxed.length !== L.boxes.length) return null;
  for (const c of comps) {
    if (c.boxes.length === 0) continue;
    if (c.boxes.length !== c.goals.length) return null;
    if (c.doors.length === 0) return null;
  }
  return comps.filter((c) => c.boxes.length > 0);
}

interface Room {
  w: number;
  h: number;
  cells: number[]; // grid indices: component cells + door cells
  loc: Int32Array; // grid index -> local index or -1
  xs: number[];
  ys: number[];
  nbr: number[][]; // nbr[local][dir] = local index or -1
  goals: number[]; // local indices of goal cells
  isGoal: Uint8Array;
  dist: Int32Array[]; // dist[g][local] = BFS distance from goal g
  dead: Uint8Array; // dead square for a box
  boxes0: number[]; // sorted local indices of starting boxes
  doors: number[]; // local indices of door cells
}

function buildRoom(L: Level, comp: Comp): Room {
  const cells = comp.cells.concat(comp.doors);
  const loc = new Int32Array(L.w * L.h).fill(-1);
  cells.forEach((g, i) => {
    loc[g] = i;
  });
  const n = cells.length;
  const xs: number[] = [];
  const ys: number[] = [];
  const nbr: number[][] = [];
  for (let i = 0; i < n; i++) {
    const g = cells[i];
    const x = g % L.w;
    const y = (g / L.w) | 0;
    xs.push(x);
    ys.push(y);
    const row: number[] = [];
    for (let d = 0; d < 4; d++) {
      const nx = x + DX[d];
      const ny = y + DY[d];
      row.push(nx < 0 || ny < 0 || nx >= L.w || ny >= L.h ? -1 : loc[ny * L.w + nx]);
    }
    nbr.push(row);
  }
  const isGoal = new Uint8Array(n);
  const goals: number[] = [];
  cells.forEach((g, i) => {
    if (L.goal[g]) {
      isGoal[i] = 1;
      goals.push(i);
    }
  });
  const boxes0: number[] = [];
  cells.forEach((g, i) => {
    if (L.boxes.includes(g)) boxes0.push(i);
  });
  boxes0.sort((a, b) => a - b);
  // BFS distance from each goal over the room's cells.
  const dist: Int32Array[] = goals.map((g0) => {
    const dd = new Int32Array(n).fill(-1);
    const q = [g0];
    dd[g0] = 0;
    for (let head = 0; head < q.length; head++) {
      const c = q[head];
      for (let d = 0; d < 4; d++) {
        const nb = nbr[c][d];
        if (nb >= 0 && dd[nb] === -1) {
          dd[nb] = dd[c] + 1;
          q.push(nb);
        }
      }
    }
    return dd;
  });
  // Dead squares: reverse-pull reachability from the goals. A box at b can be
  // pulled to a = b - d when a and b - 2d are both room cells (the player ends
  // at b - 2d). Cells never reached hold a box that can never reach a goal.
  const live = new Uint8Array(n);
  const lq: number[] = [];
  goals.forEach((g) => {
    live[g] = 1;
    lq.push(g);
  });
  for (let head = 0; head < lq.length; head++) {
    const b = lq[head];
    for (let d = 0; d < 4; d++) {
      const ax = xs[b] - DX[d];
      const ay = ys[b] - DY[d];
      const px = xs[b] - 2 * DX[d];
      const py = ys[b] - 2 * DY[d];
      if (ax < 0 || ay < 0 || ax >= L.w || ay >= L.h) continue;
      if (px < 0 || py < 0 || px >= L.w || py >= L.h) continue;
      const a = loc[ay * L.w + ax];
      const pe = loc[py * L.w + px];
      if (a >= 0 && pe >= 0 && !live[a]) {
        live[a] = 1;
        lq.push(a);
      }
    }
  }
  const dead = new Uint8Array(n);
  for (let i = 0; i < n; i++) if (!live[i]) dead[i] = 1;
  return {
    w: L.w,
    h: L.h,
    cells,
    loc,
    xs,
    ys,
    nbr,
    goals,
    isGoal,
    dist,
    dead,
    boxes0,
    doors: comp.doors.map((g) => loc[g]),
  };
}

// Flood fill over room cells avoiding boxes; returns the reached mask.
function roomFlood(room: Room, from: number, boxSet: ReadonlySet<number>): Uint8Array {
  const seen = new Uint8Array(room.cells.length);
  const q = [from];
  seen[from] = 1;
  for (let head = 0; head < q.length; head++) {
    const c = q[head];
    for (let d = 0; d < 4; d++) {
      const nb = room.nbr[c][d];
      if (nb >= 0 && !seen[nb] && !boxSet.has(nb)) {
        seen[nb] = 1;
        q.push(nb);
      }
    }
  }
  return seen;
}

// Canonical player cell: the smallest cell of the player's reachable region.
function canonical(room: Room, from: number, boxes: number[]): number {
  const seen = roomFlood(room, from, new Set(boxes));
  for (let i = 0; i < seen.length; i++) if (seen[i]) return i;
  return from;
}

// Minimum assignment of boxes to goals by BFS distance; Infinity if impossible.
function heuristic(room: Room, boxes: number[]): number {
  const G = room.goals.length;
  if (boxes.length !== G) return Infinity;
  let best = Infinity;
  const used = new Uint8Array(G);
  const rec = (i: number, acc: number): void => {
    if (acc >= best) return;
    if (i === boxes.length) {
      best = acc;
      return;
    }
    for (let g = 0; g < G; g++) {
      if (used[g]) continue;
      const dd = room.dist[g][boxes[i]];
      if (dd < 0) continue;
      used[g] = 1;
      rec(i + 1, acc + dd);
      used[g] = 0;
    }
  };
  rec(0, 0);
  return best;
}

// 2x2 block of walls/boxes holding a box and no goal: a deadlock.
function is2x2Dead(room: Room, boxSet: ReadonlySet<number>, t: number): boolean {
  const x = room.xs[t];
  const y = room.ys[t];
  const corners: [number, number][] = [
    [x - 1, y - 1],
    [x, y - 1],
    [x - 1, y],
    [x, y],
  ];
  for (const [cx, cy] of corners) {
    let blocked = 0;
    let hasBox = false;
    let hasGoal = false;
    for (const [ox, oy] of [
      [0, 0],
      [1, 0],
      [0, 1],
      [1, 1],
    ]) {
      const gx = cx + ox;
      const gy = cy + oy;
      if (gx < 0 || gy < 0 || gx >= room.w || gy >= room.h) {
        blocked++;
        continue;
      }
      const li = room.loc[gy * room.w + gx];
      if (li < 0) {
        blocked++;
        continue;
      }
      if (boxSet.has(li)) {
        blocked++;
        hasBox = true;
      }
      if (room.isGoal[li]) hasGoal = true;
    }
    if (blocked === 4 && hasBox && !hasGoal) return true;
  }
  return false;
}

// Walk (letters) between two room cells avoiding boxes; null if unreachable.
function roomWalk(
  room: Room,
  from: number,
  to: number,
  boxSet: ReadonlySet<number>,
): string | null {
  if (from === to) return '';
  const prev = new Int32Array(room.cells.length).fill(-2);
  const prevDir = new Int8Array(room.cells.length).fill(-1);
  const q = [from];
  prev[from] = -1;
  for (let head = 0; head < q.length; head++) {
    const c = q[head];
    for (let d = 0; d < 4; d++) {
      const nb = room.nbr[c][d];
      if (nb < 0 || boxSet.has(nb) || prev[nb] !== -2) continue;
      prev[nb] = c;
      prevDir[nb] = d;
      if (nb === to) {
        let s = '';
        let cur = nb;
        while (cur !== from) {
          s = LETTER[prevDir[cur]] + s;
          cur = prev[cur];
        }
        return s;
      }
      q.push(nb);
    }
  }
  return null;
}

interface RoomResult {
  moves: string;
  pushes: number;
  expansions: number;
}

// Push-optimal A* for one room. Player starts on `startPlayer` (a door cell).
function solveRoom(room: Room, startPlayer: number, maxExp: number): RoomResult | null {
  const B = room.boxes0.length;
  if (B === 0) return { moves: '', pushes: 0, expansions: 0 };
  if (room.boxes0.every((b) => room.isGoal[b])) return { moves: '', pushes: 0, expansions: 0 };
  const h0 = heuristic(room, room.boxes0);
  if (h0 === Infinity) return null;
  const startCanon = canonical(room, startPlayer, room.boxes0);
  const sBoxes: number[][] = [room.boxes0.slice()];
  const sCanon: number[] = [startCanon];
  const sG: number[] = [0];
  const sF: number[] = [h0];
  const sParent: number[] = [-1];
  const sPushFrom: number[] = [-1];
  const sPushDir: number[] = [-1];
  const closed = new Map<string, number>();
  closed.set(room.boxes0.join(',') + '|' + startCanon, 0);
  const heap: number[] = [0];
  const less = (a: number, b: number): boolean =>
    sF[a] !== sF[b] ? sF[a] < sF[b] : sG[a] !== sG[b] ? sG[a] > sG[b] : a < b;
  const heapPush = (v: number): void => {
    let i = heap.length;
    heap.push(v);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i], heap[p])) break;
      const tmp = heap[i];
      heap[i] = heap[p];
      heap[p] = tmp;
      i = p;
    }
  };
  const heapPop = (): number => {
    const top = heap[0];
    const last = heap.pop() as number;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1], heap[c])) c++;
        if (!less(heap[c], heap[i])) break;
        const tmp = heap[i];
        heap[i] = heap[c];
        heap[c] = tmp;
        i = c;
      }
    }
    return top;
  };
  let expansions = 0;
  while (heap.length > 0) {
    const s = heapPop();
    const key = sBoxes[s].join(',') + '|' + sCanon[s];
    if (closed.get(key) !== s) continue;
    const boxes = sBoxes[s];
    if (boxes.every((b) => room.isGoal[b])) {
      const pushes: [number, number][] = [];
      for (let t = s; t > 0; t = sParent[t]) pushes.push([sPushFrom[t], sPushDir[t]]);
      pushes.reverse();
      let player = startPlayer;
      let cur = room.boxes0.slice();
      let moves = '';
      for (const [from, dir] of pushes) {
        const behind = room.nbr[from][OPP[dir]];
        const walk = roomWalk(room, player, behind, new Set(cur));
        if (walk === null) return null;
        moves += walk + LETTER[dir];
        player = from;
        const nt = room.nbr[from][dir];
        cur = cur.slice();
        cur[cur.indexOf(from)] = nt;
        cur.sort((a, b) => a - b);
      }
      return { moves, pushes: pushes.length, expansions };
    }
    if (expansions >= maxExp) return null;
    expansions++;
    const boxSet = new Set(boxes);
    const reach = roomFlood(room, sCanon[s], boxSet);
    for (let i = 0; i < boxes.length; i++) {
      const b = boxes[i];
      for (let d = 0; d < 4; d++) {
        const t = room.nbr[b][d];
        if (t < 0 || boxSet.has(t)) continue;
        const behind = room.nbr[b][OPP[d]];
        if (behind < 0 || !reach[behind]) continue;
        const nb = boxes.slice();
        nb[i] = t;
        nb.sort((a, b2) => a - b2);
        if (nb.some((x) => room.dead[x])) continue;
        const nbSet = new Set(nb);
        if (is2x2Dead(room, nbSet, t)) continue;
        const h = heuristic(room, nb);
        if (h === Infinity) continue;
        const g2 = sG[s] + 1;
        const canon2 = canonical(room, b, nb);
        const key2 = nb.join(',') + '|' + canon2;
        const prev = closed.get(key2);
        if (prev !== undefined && sG[prev] <= g2) continue;
        const idx = sBoxes.length;
        sBoxes.push(nb);
        sCanon.push(canon2);
        sG.push(g2);
        sF.push(g2 + h);
        sParent.push(s);
        sPushFrom.push(b);
        sPushDir.push(d);
        closed.set(key2, idx);
        heapPush(idx);
      }
    }
  }
  return null;
}

// ---------------------------------------------------------------- whole level

function solveByRooms(L: Level, comps: Comp[]): string | null {
  let moves = '';
  let player = L.player;
  let boxes = L.boxes.slice();
  const ordered = comps.slice().sort((a, b) => {
    const da = Math.min(...a.doors);
    const db = Math.min(...b.doors);
    const ca = da % L.w;
    const cb = db % L.w;
    return ca !== cb ? ca - cb : da - db;
  });
  for (const comp of ordered) {
    const room = buildRoom(L, comp);
    const blocked = new Set(boxes);
    let solved: RoomResult | null = null;
    let walkToDoor = '';
    for (const door of comp.doors) {
      const w = walkPath(L, player, door, blocked);
      if (w === null) continue;
      const r = solveRoom(room, room.loc[door], 1_000_000);
      if (r !== null) {
        solved = r;
        walkToDoor = w;
        break;
      }
    }
    if (solved === null) return null;
    const seg = walkToDoor + solved.moves;
    const applied = applyMoves(L, seg, player, boxes);
    if (applied === null) return null;
    moves += seg;
    player = applied.player;
    boxes = applied.boxes;
  }
  return moves;
}

// Fallback: plain A* over single steps (minimises moves, not pushes), with a
// work cap. Only used when the room decomposition fails.
function solveMoveAStar(L: Level, maxExp: number): string | null {
  const minMan = new Int32Array(L.w * L.h);
  for (let i = 0; i < L.w * L.h; i++) {
    if (L.wall[i]) continue;
    let m = Infinity;
    for (let g = 0; g < L.w * L.h; g++) {
      if (!L.goal[g]) continue;
      const d =
        Math.abs((g % L.w) - (i % L.w)) + Math.abs(((g / L.w) | 0) - ((i / L.w) | 0));
      if (d < m) m = d;
    }
    minMan[i] = m === Infinity ? 0 : m;
  }
  const hOf = (boxes: number[]): number => boxes.reduce((a, b) => a + minMan[b], 0);
  const sP: number[] = [L.player];
  const sB: number[][] = [L.boxes.slice()];
  const sG: number[] = [0];
  const sF: number[] = [hOf(L.boxes)];
  const sParent: number[] = [-1];
  const sStep: string[] = [''];
  const closed = new Map<string, number>();
  closed.set(L.player + '|' + L.boxes.join(','), 0);
  const heap: number[] = [0];
  const less = (a: number, b: number): boolean =>
    sF[a] !== sF[b] ? sF[a] < sF[b] : sG[a] !== sG[b] ? sG[a] > sG[b] : a < b;
  const heapPush = (v: number): void => {
    let i = heap.length;
    heap.push(v);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i], heap[p])) break;
      const tmp = heap[i];
      heap[i] = heap[p];
      heap[p] = tmp;
      i = p;
    }
  };
  const heapPop = (): number => {
    const top = heap[0];
    const last = heap.pop() as number;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1], heap[c])) c++;
        if (!less(heap[c], heap[i])) break;
        const tmp = heap[i];
        heap[i] = heap[c];
        heap[c] = tmp;
        i = c;
      }
    }
    return top;
  };
  let expansions = 0;
  while (heap.length > 0) {
    const s = heapPop();
    const key = sP[s] + '|' + sB[s].join(',');
    if (closed.get(key) !== s) continue;
    const boxes = sB[s];
    if (boxes.every((b) => L.goal[b] === 1)) {
      let moves = '';
      for (let t = s; t > 0; t = sParent[t]) moves = sStep[t] + moves;
      return moves;
    }
    if (expansions >= maxExp) return null;
    expansions++;
    const p = sP[s];
    const px = p % L.w;
    const py = (p / L.w) | 0;
    for (let d = 0; d < 4; d++) {
      const nx = px + DX[d];
      const ny = py + DY[d];
      if (nx < 0 || ny < 0 || nx >= L.w || ny >= L.h) continue;
      const n = ny * L.w + nx;
      if (L.wall[n]) continue;
      let nb = boxes;
      const at = boxes.indexOf(n);
      if (at >= 0) {
        const bx = nx + DX[d];
        const by = ny + DY[d];
        if (bx < 0 || by < 0 || bx >= L.w || by >= L.h) continue;
        const b2 = by * L.w + bx;
        if (L.wall[b2] || boxes.includes(b2)) continue;
        nb = boxes.slice();
        nb[at] = b2;
        nb.sort((a, b3) => a - b3);
      }
      const g2 = sG[s] + 1;
      const key2 = n + '|' + nb.join(',');
      const prev = closed.get(key2);
      if (prev !== undefined && sG[prev] <= g2) continue;
      const idx = sP.length;
      sP.push(n);
      sB.push(nb);
      sG.push(g2);
      sF.push(g2 + hOf(nb));
      sParent.push(s);
      sStep.push(LETTER[d]);
      closed.set(key2, idx);
      heapPush(idx);
    }
  }
  return null;
}

function solveLevel(text: string): string {
  const L = parseLevel(text);
  const comps = decompose(L);
  if (comps !== null) {
    const byRooms = solveByRooms(L, comps);
    if (byRooms !== null && verifySolution(L, byRooms)) return byRooms;
  }
  const fb = solveMoveAStar(L, 200_000);
  if (fb !== null && verifySolution(L, fb)) return fb;
  return '';
}

for (const id of readdirSync(IN_DIR).sort()) {
  const moves = solveLevel(readFileSync(join(IN_DIR, id, 'level.txt'), 'utf8'));
  mkdirSync(join(OUT_DIR, id), { recursive: true });
  writeFileSync(join(OUT_DIR, id, 'solution.txt'), `${moves}\n`);
}
