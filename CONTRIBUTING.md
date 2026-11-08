# Contributing to LiDar

Thanks for helping. This project is MIT-licensed and runs on phones, so two
rules shape everything else: **pure Python stdlib** and **honest numbers**.

## Ground rules

1. **No new runtime dependencies.** No numpy, Pillow, FastAPI, requests,
   WebSockets, no compiled extensions. `pytest` is the only dev dependency and
   CI installs it. If an idea needs a library, implement the small part you
   need with `math`, `statistics`, `sqlite3`, `mmap`, `http.server`.
2. **Small diffs, one change per commit.** A feature, a fix or a docs change —
   not all three in one commit.
3. **Measured claims only.** If you change accuracy, report the median **and**
   p90 over at least 16 noise seeds, and say `n`. A single seed is labelled
   "seed 0" and never quoted as the result. See the v2.3 → v2.4 note in the
   README for why.
4. **Abstain rather than guess.** Every estimator has a confidence interval and
   every matcher a refusal path. A confident wrong answer is a bug.

## Quick start

```bash
git clone https://github.com/chrisalunlloyd2-sudo/LiDar.git
cd LiDar
python3 --version                 # 3.9+ (developed on 3.14)
python3 -m pytest tests -q        # 210 tests, ~15 s
python3 server.py --memory        # http://127.0.0.1:8790
python3 flash_lidar.py --demo     # torch-pulse depth demo + report
```

Open the server URL in Chrome, allow camera + location. The GUI is a shell plus
one fragment per page, all memory-mapped from `gui_pages/`.

## Before you open a PR

```bash
python3 -m pytest tests -q
node --check gui_pages/app.js     # if you touched the frontend
python3 scripts/scrub_secrets.py --fail-on high
```

CI runs the test suite and the secret scrub on every push and pull request. It
fails on `critical`/`high` findings; emails in commit metadata do not fail it.

## Where things go

| You want to add… | Touch |
|---|---|
| an API route | `server.py` (`do_GET`/`do_POST` + `_handle_*`) and `tests/test_server.py` |
| a GUI card | `gui_pages/<name>.html`, `PAGES` in `gui_pages/app.js`, a `slot-<name>` in `shell.html` |
| a depth cue | `depth_engine.py` (return a `CueEstimate`), then `uncertainty.fuse_cues` picks it up |
| a fuser / size estimator / triangulator | the matching `arena/` module + `tests/test_arena.py` |
| a depth source | model it on `flash_lidar.py`: calibration, per-sample sigma, abstention |
| a fixture object | `fixtures/premade_objects.py` + `inventory.py` catalog |

New modules must import cleanly on 32-bit ARM and keep memory bounded; the
target device has 3.6 GB RAM and 8 cores with a GIL.

## Measurement standards

- Distances are **Euclidean range** along the ray, not z-depth. The forward
  renderer in `flash_lidar.py` had exactly that bug once; do not reintroduce it.
- Uncertainties compose: a per-cell sigma, a between-cue variance and a
  quantisation term are separate contributions — report the total.
- A label is not a measurement. The flash demo shows a 2×8 match whose range is
  35 % too far under mixed albedo; keep that distinction in docs and UI.

## Security

- Never commit tokens, `.env` files, keystores or real device dumps.
- Run `scripts/scrub_secrets.py` before pushing. It scans the tree, every
  reachable blob and the commit metadata, and redacts matches.
- Do not push tags without scanning them: local-only refs are where old
  mistakes hide (`git push --tags` publishes them).
- If you find a leaked credential, say so in the PR instead of quietly
  rewriting history; revoking the credential comes first.

## Commit messages

Imperative subject, then a body explaining **why** and the evidence (test
count, measured effect). `Co-authored-by:` lines are welcome. Example:

```
feat(flash): cluster the reticle cue before fusing

Background wall inside the 80 px radius dragged a 1.5 m board to 1.67 m.
Cells are now gated around the depth of the cell nearest the reticle.
Flash tests: 23 pass.
```

## Review checklist

- [ ] tests pass, no new deps, no network calls in tests
- [ ] numbers carry `n` and a spread, not one lucky draw
- [ ] refusal paths exist for weak evidence
- [ ] README/CHANGELOG-style notes updated if behaviour changed
- [ ] scrub clean at `--fail-on high`

## License

By contributing you agree your work is released under the [MIT License](LICENSE).
