# Sokoban Solver — make.fun sokoban-solver-01-run-20261121

Push-optimal room-by-room Sokoban solver, written fresh for this competition.
Node 22, no dependencies, no network.

## Approach

Every level the generator builds is one hallway plus 1–4 rooms (6×6 slots, one
door each, 3–4 boxes per room). In an optimal solution no box ever leaves its
room, so each room is solved independently with a push-optimal A* and the room
solutions are joined with hallway walks. Total pushes then equal the sum of
the rooms' optima — the level's push-optimal count P*.

Room search: A* over pushes. A state is (sorted box cells, canonical player
cell), where the canonical cell is the smallest cell of the player's reachable
region. Heuristic: minimum assignment of boxes to goals by BFS distance.
Pruning: dead squares (reverse-pull reachability from the goals), 2×2
wall/box blocks with no goal, and states with no finite box–goal assignment.
Work is bounded by expansion caps, never by the wall clock, so two runs are
byte-identical.

If the room decomposition ever fails, the entry falls back to a plain A* over
single steps (work-capped), and to an empty answer if that fails too.

## Run locally

```sh
mkdir -p out scores
MAKEFUN_INPUT_DIR=practice/input MAKEFUN_OUTPUT_DIR=out node main.ts
MAKEFUN_OUTPUT_DIR=out MAKEFUN_REFERENCE_DIR=practice/reference \
  MAKEFUN_SCORES_DIR=scores node program/scorer/score.ts
```

## Files

- `main.ts` — the whole solver (entry point).
- `.makefun/owner.json` — make.fun owner record (seeded at repo creation).

## License

MIT — see `LICENSE`.
