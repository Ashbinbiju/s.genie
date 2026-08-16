# CLAUDE.md

## What this is

A Fantasy Premier League AI engine. Pipeline: **fetch → featurize → predict → optimize → present**.
LightGBM two-stage model (minutes → points) feeding a PuLP integer program that picks the optimal
15-man squad, transfers, captain and chips. Streamlit dashboard on top.

**📖 Read [ARCHITECTURE.md](ARCHITECTURE.md) first** — every module, the data flow, the magic values,
and the known traps. Don't re-derive the pipeline from scratch; it's all there.

**🐛 [AUDIT.md](AUDIT.md)** records the bugs that were found (all now fixed) and **[FIXES.md](FIXES.md)**
what changed and how it was verified. Read FIXES.md before retraining — the model feature vocabulary
changed, so old pickles are not compatible.

## Entrypoints

```bash
streamlit run src/interface/dashboard.py      # primary product (4 tabs)
python src/main.py --gw 18 --team_id 5989967  # CLI, thinner/older path
python src/model/predictor.py                 # offline training + full audit report
```

## Rules that will bite you

- **Always run from the project root.** No `__init__.py` files exist; modules use `sys.path` hacks and
  relative `data/` paths.
- **`data/` is entirely gitignored** — a fresh clone has no data and no trained model. See
  ARCHITECTURE.md §6/§7 for the bootstrap order.
- **`src/utils/season.py` is the single source of truth** for anything season-dependent: the current
  gameweek, team names, team shirt codes, and the canonical team/position vocabularies. Never
  hardcode a club list, a shirt code or a gameweek number — that is exactly what broke every August.
- **Integer team ids are NOT stable across seasons.** FPL reassigns them. Model features use
  `team_name` and `opponent_name`; the integer `team` is for app logic (shirts, the max-3-per-club
  constraint) only, and is explicitly dropped from the feature set.
- **The element-summary cache is season-stamped** (`element_summary_{season}_gw_{N}.json`) and
  validated on load. Do not loosen that check — an unstamped cache from a prior season joins cleanly
  onto the current player list while describing entirely different people.
- **Feature engineering is duplicated** between `features/history_builder.py` (training, with
  `.shift(1)` anti-leakage) and `model/predictor.py::_build_rolling_features()` (inference, rebuilt
  from cache). Change a feature in one, change it in the other — including `days_rest`, which must
  measure *upcoming kickoff − last played* in both.
- **`ODDS_API_KEY`** env var is optional; without it odds fall back to league averages and
  `odds_confidence` reports `LOW`.

## Conventions

- `element_type`: 1=GK, 2=DEF, 3=MID, 4=FWD; canonical position strings are `GK/DEF/MID/FWD`
  (FPL's own API says `GKP` — always pass it through `canon_position`)
- FPL prices arrive in tenths — always `/10.0`
- FDR 1–5 (lower = easier); this project uses the 5-fixture mean
- Squad: 15 = 2/5/5/3, max 3 per club; hit cost 4 pts per transfer above your FT count
- Chip restoration threshold: GW20
- The model predicts **expected** points (~1-6, best player ≈6), not actual points. Any threshold
  compared against it must be relative, not absolute — see the header of `optimization/chips.py`.

## Tests

```bash
pytest            # 176 tests, ~11s
```

`tests/test_train_serve_parity.py` is the one to keep green: it runs the same player history
through **both** rolling-feature implementations (training's shifted groupby and inference's JSON
loop) and asserts they agree value-for-value. Any new feature must be added to both and covered
there, or the model will be scored on something it was not trained on.

`tests/test_pipeline_integrity.py` asserts against the real built artifacts and is skipped if they
are absent. Run `history_builder.py` → `processor.py` → `predictor.py` first for full coverage.

Watch for `NaN` reaching any categorical: `astype(str)` turns it into the literal category `'nan'`,
which trains and predicts without raising. Always `fillna("UNKNOWN")` first — there are asserts
guarding this in `history_builder`.

## Before changing prediction logic

Run `python src/model/predictor.py` and read the audit report — the time-series **CV RMSE** is the
model-quality number (the per-slice RMSEs below it are in-sample and optimistic), plus feature gain
ranking and the drift table.
