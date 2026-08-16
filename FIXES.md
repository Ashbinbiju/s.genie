# Fixes applied

Every item in [AUDIT.md](AUDIT.md), fixed and verified by running the code. Measurements below are
real output, not estimates.

**One-time steps you must run** — see [§ What you need to do](#what-you-need-to-do) at the bottom.

---

## New shared module

`src/utils/season.py` is the single source of truth for everything that changes between seasons:
season label, current/next gameweek, pre-season detection, team id→name/code maps, shirt URLs, and
the canonical team-name and position vocabularies. Season-dependent facts were previously hardcoded
in three or four places each and went stale every August.

---

## P0 — was broken

| # | Fix | Verification |
|---|---|---|
| **P0-1** | Cache files are now stamped with the season (`element_summary_{season}_gw_{N}.json`), so a previous season's file can never be *found*. On top of that, `load_summary_cache()` rejects any unstamped legacy file and any cache overlapping the live player list by <80%. | Confirmed: the stale `element_summary_gw_37.json` is now **refused** with an explicit reason, instead of silently mapping Setford's history onto Gabriel. |
| **P0-2** | `get_team_picks` returns `None` cleanly when `gw <= 1`, and the dashboard now **drafts an optimal squad from scratch** instead of erroring out. | Dashboard runs end-to-end pre-season with **no exception**; renders a valid 15 and a captain. |
| **P0-3** | `LEAGUE_ID = 1019782`; the Rival Spy input derives from it rather than holding a second hardcoded copy. League id is now a sidebar input. | Old id verified **404**; new id returns HTTP 200. |
| **P0-4** | `get_current_gw()` reads `is_current`, then derives from `is_next`, then from finished events — never from cache filenames. | Returns **0** pre-season (was **37**, i.e. last season's final GW). |
| **P0-5** | All three hardcoded `TEAM_SHIRT_MAP` tables deleted. Shirt URLs are built from `bootstrap_static`'s `teams[].code`. | Verified `shirt_{code}` resolves for **20/20** clubs, including Coventry, Hull and Sunderland. The workaround was never needed. |

## P1 — silently wrong model

**P1-1 — odds inflation.** Removed the `* 1.8` multiplier and unified both the historical and live
paths behind one `implied_goals_from_odds()` helper. The goal split is now damped toward an even
share (win probability is a more extreme signal than goal share); the damping constant was fitted
against 1140 real matches rather than guessed. `LEAGUE_DEFAULTS` was recalibrated onto the same scale.

| | before | after | actual |
|---|---|---|---|
| implied goals / match | 5.360 | **2.83 – 2.98** | 2.85 – 3.28 |
| clean-sheet probability | 0.150 | **0.256 – 0.272** | 0.207 – 0.272 |
| defaults vs computed | 1.35 vs 2.68 (different scales) | **1.465 vs 1.491** | — |

**P1-2 / P1-4 — categorical vocabularies.** Club **names** are now the canonical key everywhere
(`team_name`), and the integer `team` id is explicitly dropped from the feature set. `GKP` is
normalised to `GK`. While verifying the retrained model I found I had only half-fixed this:
`opponent_team` was still a raw per-season integer id with exactly the same instability, so that is
now resolved to `opponent_name` too — derived for historical seasons from the fixture pairings, and
from bootstrap at inference.

```
before   team      2022-23:['Arsenal',...]  2024-25:['1','10','11',...]   position: GK / GK / GKP
after    team_name every season: 20 club names            opponent_name: 20 club names
                   position     every season: ['DEF','FWD','GK','MID']
```

`opponent_name` is now the **3rd most important feature by gain**, so this was carrying real signal
through a broken encoding.

**P1-3 — odds join alignment.** Replaced the match-ordinal (`gw_rank`) join with a join on
`(season, team_name, match date)`. Match ordinal assumed match N == GW N, so one postponement shifted
every later fixture's odds.

```
before   16 of 20 clubs misaligned in 2023-24 alone
after    Matched 53699/53699 rows with historical odds (100.0%)
```

**P1-5 — stacking leakage.** `projected_minutes` fed into points-model *training* is now generated
**out-of-fold** across 5 chronological blocks; the minutes model is separately fitted on everything
and persisted for serving. The A/B test and CV RMSE are no longer optimistically biased.

```
--- Generating out-of-fold minutes predictions ---
  fold 1/5: trained on 44541 rows, predicted 9158
  ... (5 folds)
Time-Series CV RMSE (out-of-sample): 1.9039
```

**P1-6 — `days_rest` phase.** At inference this measured the gap between the last two *completed*
matches; it now measures **upcoming kickoff − last played**, matching the training definition.
`processor` carries `next_kickoff_time` for the purpose (populated 587/587).

**P1-7 — Understat.** The silent zero-fill is now a loud warning naming the fix command, and
`main.py --fetch` writes the CSV. Also fixed the mojibake decode (`utf-8`→`unicode_escape` mangled
every accented name; now latin-1 based) and added a request timeout. **xG/xA remain 0 until you run
`python src/api/understat.py`** — the data genuinely is not on disk.

## P2 — logic and UX

- **P2-1 chip thresholds** — rewritten as **ratios to your own average starter**, so they survive
  retraining. Verified firing correctly: Bench Boost "Recommended" at 77% of a starter, Wildcard at
  +29% XI gain, Free Hit crisis at 7 active players. Previously Triple Captain needed 11.0 XP when
  the best player in the game projected 6.32 — **zero players could clear even the 8.0 "Consider" bar**.
- **P2-2 chip advice** now evaluates your **current** squad, not the post-transfer one.
- **P2-3 `SHIRT_MAP` NameError** — gone; all image resolution goes through `resolve_player_image()`.
- **P2-4 free transfers** now skip Wildcard/Free Hit gameweeks. Verified: 11 wildcard transfers in
  GW5 preserve the bank (2→5 FT) instead of resetting it to 1.
- **P2-5 XI-based gains** — `squad_expected_points()` sums the XI with the captain doubled.
  Bench points are no longer counted as gains.
- **P2-6 captaincy** — the dashboard now ranks captains by `captaincy_score`, which was computed and
  discarded before. *Correction to the audit:* the `minutes_prob` multiplication is **not** a genuine
  double-count — it carries injury news the minutes model cannot see, and is 1.0 for unflagged
  players. Left in place, now documented.
- **P2-7 triplicated shirt maps** — deleted, see P0-5.
- **P2-8 recomputation** — every pipeline stage is behind `@st.cache_data(ttl=900)` keyed on scalars.
  Interacting with a tab-4 widget no longer re-runs two CBC solves.
- **P2-9 dependencies** — added `joblib`, `aiohttp`, `pyarrow`; removed the duplicate `streamlit` and
  the unused `fpl`.
- **P2-10 timeouts** — every outbound request is bounded (FPL 20s, Understat 20s, vaastav 60s, async
  session 300s total / 15s connect).
- **P2-11 silent async failures** — `get_all_summaries` now **raises** past a 5% failure rate rather
  than writing a partial cache, and warns below it.
- **P2-12 CLI** — `reports/` is created; the report is written **UTF-8** (accented names no longer
  raise on Windows); `name_map.get()` replaces the bare index; `--fetch` refreshes the summary cache;
  the CLI passes `freehit_gws`; `--gw` defaults to the next deadline; added `--bank`.
- **P2-13 stale parquet** — the cache-validity check covers every required column including odds.
- **P2-14 bank** — added a sidebar **Bank (£m)** input; spending power is squad value + bank.
- **P2-15 minor** — dead `IMAGE_CACHE`, dead `.pitch-line`/`.pitch-circle` markup, the duplicated
  `age_hours` computation, the unused `prob` in the solver, the `importlib.reload` hack, duplicate
  imports and the unused `full_squad_value` parameter are all removed. `MinutesPredictor.predict`
  fills missing columns *before* casting categoricals. Bare `except:` replaced with typed handlers.
  Solver constraint building uses dict lookups instead of ~78k `df.loc` scalar calls.

## Found while fixing (not in the original audit)

- **Duplicate player rows.** The Understat merge could duplicate an FPL row when two Understat names
  normalise to the same key — letting the optimizer select the same player twice and double-count
  squad value. Understat rows are now de-duplicated before the merge, and a `drop_duplicates` guard
  runs after it.
- **Goalkeeper captaincy.** `pick_captain` was recommending a keeper pre-season, when flat
  projections leave a GK on top. Keepers are now excluded from captaincy unless the XI has nothing
  else — a keeper's scoring ceiling makes doubling one never correct.
- **Transfer churn on ties.** k=2 and k=3 transfers scored within 0.01 net, and the optimizer took
  the 3-transfer plan. An extra transfer must now beat the incumbent by `NET_GAIN_MARGIN = 0.5` XP —
  roughly the model's own noise — so ties keep the free transfer. Verified: 3 transfers → 1.
- **Chip "used" in a future gameweek.** `_is_chip_available` now ignores chip events later than the
  gameweek being analysed.
- **Fragile XI selection.** `select_starting_xi` raised `IndexError` on a squad missing a position.
  It now degrades gracefully, and orders the bench with the reserve keeper first (matching auto-subs).

---

---

## Second pass — adversarial review + test suite

A fresh hostile read of the code *including the fixes above*, backed by a **176-test
pytest suite**. Five further bugs found, three of them in code written during the first pass.

| Bug | Impact | Test that locks it |
|---|---|---|
| **Double gameweeks not collapsed at inference.** Training groups by `(player_id, GW)` and sums; `element-summary` returns one entry **per match**, so a DGW made the last-3 window span 3 *matches* instead of 3 *gameweeks*. | Train/serve skew on every rolling feature, worst exactly when DGWs matter most | `test_parity_holds_across_a_double_gameweek` |
| **Accented names never matched Understat.** Normalisation stripped anything outside `[a-z]` instead of folding it: `Ødegaard`→`degaard` vs FPL `Odegaard`→`odegaard`. | Every player with `Ø/Ł/Đ/Æ/ß` silently dropped from the join with zero xG/xA | `test_names.py` (12 name pairs) |
| **`NaN` becoming the literal category `'nan'`.** Found in two places: unresolvable opponent ids, and missing player names. `str(NaN)` is `'nan'`, which becomes a real category / a match key every nameless row shares. | Silently corrupted opponent vocabulary; unrelated players joined to each other | `test_unmapped_opponents_are_never_silently_stringified_as_nan`, `test_missing_names_never_become_the_literal_key_nan` |
| **Empty or duplicated cache crashed / desynced `predict()`.** An empty summaries dict gave a column-less frame (`KeyError` on merge); duplicate ids expanded the row count and desynced the positional assignments after it. | Crash, or predictions attached to the wrong players | `test_empty_cache_falls_back`, `test_duplicate_ids_in_cache_trigger_the_fallback_not_a_desync` |
| **Fragile `processor` inputs.** A missing Understat `time` column raised `KeyError`; an all-`None` `chance_of_playing_next_round` (i.e. every player at season start) produced an object-dtype column. | Crash on a new season | covered by the pipeline-integrity tests |

Also: pinned **`pulp<4`** — PuLP 4.0 removes `LpVariable.dicts` and `PULP_CBC_CMD`, both of
which `solver.py` uses, so an unpinned upgrade would break the optimizer outright.

### Test suite

`pytest` — **176 tests, all passing**, in ~11s.

| File | Covers |
|---|---|
| `test_season.py` | season label, current/next GW, pre-season, canonical vocabularies, shirt URLs |
| `test_fpl_client.py` | free-transfer accounting incl. chip weeks, chip parsing, the GW1 picks guard, Free Hit skipping |
| `test_optimization.py` | squad legality, XI formation rules, captaincy, chip thresholds + restoration, solver constraints |
| `test_model.py` | cache integrity guards, rolling-feature semantics, feature hygiene, odds calibration |
| `test_train_serve_parity.py` | **numerical** parity between the two rolling-feature implementations, incl. DGWs |
| `test_pipeline_integrity.py` | anti-leakage proofs, train/serve feature parity, vocabulary agreement against the real artifacts |
| `test_ml_path.py` | the real two-stage ML prediction path via a synthesised valid cache |
| `test_analysis_and_reporting.py` | rival comparison, report encoding, fixture difficulty, fallback behaviour |
| `test_names.py` | cross-source name matching |
| `test_async_cache.py` | partial-failure threshold, cache reuse, pre-season no-op |

The two strongest are worth calling out. `test_train_serve_parity.py` runs an identical
player history through **both** feature builders — the shifted pandas groupby used in
training and the JSON loop used at inference — and asserts they agree value-for-value on
12 features. `test_pipeline_integrity.py` proves anti-leakage structurally: on a player's
first gameweek every shifted feature must be 0 no matter how many points they scored.

---

## Third pass — the deployment gap

The deployed app reported:

```
[FALLBACK] ML Points Model file not found at data/models/lgb_ts_points.pkl
```

**Cause:** `data/models/` was gitignored, so the trained model never reached Streamlit Cloud. A
deployed instance can refetch everything else it needs (bootstrap, fixtures, the element-summary
cache, the processed feature frame) but it *cannot* retrain — training needs multi-season history
that is not deployed. So the app was permanently stuck on the heuristic.

**Fixes**

1. **Models now ship with the repo.** `.gitignore` re-includes exactly the two live artifacts and
   keeps everything else (per-GW snapshots, raw data, caches) local. Note the rule must be
   `data/models/*`, not `data/models/` — git cannot re-include a file whose parent directory is
   excluded.
2. **Switched from pickle to LightGBM's native text format** plus a JSON metadata sidecar. Models
   are trained locally on Python 3.14 but loaded on Streamlit Cloud under Python 3.11
   (`runtime.txt`), and a pickle carries its writer's Python and library versions. The text format
   is version-independent and diffable. Verified it round-trips `pandas_categorical` — the category
   ordering the team/opponent/position features are encoded against — with byte-identical
   predictions. Legacy `.pkl` bundles still load.
3. **Metadata now records provenance**: `trained_at`, `season`, `cv_rmse`, `train_seasons`,
   `n_train_rows`.

**Verified by simulating a fresh checkout** — copying only git-tracked files into a clean directory
and running the pipeline there:

| scenario | before | after |
|---|---|---|
| fresh checkout, pre-season | `fallback` — *model not found* | `fallback` — *no element-summary cache* (correct: no matches played yet) |
| fresh checkout, GW1-3 played | `fallback` — *model not found* | **`ml`** — valid squad, 2/5/5/3, max 3/club, outfield captain |

`tests/test_deployment.py` locks this in: the artifacts must not be gitignored, must be stageable,
must be the portable format, must carry a feature list and provenance, and both predictors must
resolve them. A path rename can no longer silently turn the ML path back into a fallback — which is
how the ML-path tests came to be skipping rather than running when I first changed the format.

## Verification performed

- All modules compile; all import cleanly.
- Season helpers checked against the **live** FPL API (`2026-27`, next GW 1, pre-season `True`).
- Cache guard confirmed to reject the stale file.
- Full pipeline run end-to-end: processor (587 players, 0 duplicates, canonical vocab) → predictor →
  solver (15 players, exact 2/5/5/3 quota, max 3 per club, £100.0) → XI + captain.
- `history_builder` rebuilt: 53,699 rows, 100% odds match.
- Models retrained: out-of-fold minutes across 5 folds, A/B test, CV RMSE 1.9039.
- Dashboard driven headlessly via `streamlit.testing.AppTest` — **no exception** on the pre-season
  path that previously dead-ended.
- Chip thresholds, chip restoration, and chip-aware free-transfer maths tested against constructed
  scenarios.
- Transfer optimizer verified for constraint validity, hit accounting and tie-breaking.

**Not verified:** the live mid-season path (a squad with real picks). No FPL team has a squad yet in
`2026-27`, so `get_team_picks` cannot return data. The pre-season branch and the solver mechanics are
both tested; the picks-driven branch is exercised only with synthetic squads.

---

## What you need to do

```bash
pip install -r requirements.txt          # joblib / aiohttp / pyarrow were missing

# Optional but recommended — xG/xA are currently ZERO without it
python src/api/understat.py

# Once GW1 has been played, build the season-stamped summary cache:
python src/api/async_fpl.py

# Then rebuild and retrain (models trained on the old vocabulary are not compatible):
python src/features/history_builder.py
python src/model/predictor.py

streamlit run src/interface/dashboard.py
```

Two things worth knowing:

1. **Retraining is mandatory, not optional.** The feature vocabulary changed (`team_name`,
   `opponent_name`, `GK`), so an old pickle cannot be scored against the new inference frame. I have
   already retrained against 2022-23 + 2023-24; re-run after GW1 to fold in current-season data.
2. **Until GW1 is played the app runs on the heuristic fallback** — correctly, and now loudly. There
   is no match history in a new season, so no rolling features exist. Expect flat predictions; the
   red banner is telling the truth.
