# Code Audit — bugs & logical defects

> **Status: all findings below are FIXED.** See [FIXES.md](FIXES.md) for what changed, the
> before/after measurements, and the one-time steps you must run. This document is kept as the
> record of what was wrong and why it mattered.


Deep read of all 16 modules plus empirical verification against the live FPL API, the on-disk
parquet/cache/model artifacts, and the historical odds CSVs. Every claim below marked **[verified]**
was reproduced by running code, not inferred from reading.

**Audit date context:** the live API reports `is_current = None`, `is_next = 1`, `0/38` events
finished — i.e. a **brand-new season before GW1**. That state is what triggers most of the P0 items.

---

## P0 — Broken right now

### P0-1. Stale `element_summary` cache is silently merged onto the wrong players ⚠️ worst bug in the repo

`PointsPredictor.predict()` picks the newest `data/cache/element_summary_gw_*.json` and merges it onto
the current player list **on `id`** ([predictor.py:296](src/model/predictor.py#L296),
[:350](src/model/predictor.py#L350)). FPL **reassigns element ids every season**, so last season's
cache keys collide perfectly with this season's ids while pointing at different people.

**[verified]** Cached file has 840 ids; the live game has 587; **587/587 (100%) collide**:

```
id 3    cached history = Hein          → merged onto Meslier
id 4    cached history = Setford       → merged onto Gabriel
id 5    cached history = Gabriel       → merged onto J.Timber
id 7    cached history = Calafiori     → merged onto Lewis-Skelly
id 8    cached history = J.Timber      → merged onto Calafiori
```

Every rolling feature — form, minutes, xG, bps, benchings — is attributed to the wrong player. No
exception, no warning, no fallback: the app looks healthy and every prediction is garbage.

**Fix:** stamp the season into the cache filename (`element_summary_{season}_gw_{N}.json`) and refuse
to load a cache whose season ≠ the live season. A cheap guard in the meantime: compare the cache's id
set against `bootstrap_static` and bail if the overlap is suspicious.

### P0-2. GW1 is a hard dead-end

`get_team_picks(team_id, gw)` starts at `gw - 1` ([fpl.py:118](src/api/fpl.py#L118)).

**[verified]** At `gw=1`: `start_gw=0` → `range(0, 0, -1)` → **empty** → the loop never executes →
returns `None` → the dashboard hits `st.error("Could not fetch team picks")` and stops. Since
`get_current_gameweek()` returns `is_next` (= 1 right now), **the app dead-ends on every run until
GW2**. The whole product is unusable for the entire first gameweek of a season.

**Fix:** when `gw <= 1`, skip the picks lookup and fall through to `solve_team()` (build-from-scratch),
which is the correct pre-season behaviour anyway.

### P0-3. Dead league ID *(fixed in this pass)*

**[verified]** League `1311994` → **HTTP 404**. Leagues are per-season. `1019782` ("RCFC League",
created 2026-08-16) → HTTP 200, currently 0 entries. Updated `LEAGUE_ID`, and the Rival Spy text input
now derives from it instead of carrying a second hardcoded copy.

Note `get_league_members` swallows everything in a bare `except:` and returns `{}`, so the 404 was
invisible — it silently degraded to the manual Team ID box.

### P0-4. `_get_current_gw()` resolves to last season's GW

With no `is_current` event, it falls back to parsing cache filenames
([predictor.py:26](src/model/predictor.py#L26)).
**[verified]** it returns **37** during a pre-GW1 new season. That number then names the audit CSV and
the versioned model pickles (`points_model_gw37.pkl`), overwriting last season's artifacts.

### P0-5. Hardcoded 2024/25 shirt maps are wrong for this season

`TEAM_SHIRT_MAP` is pinned to the 2024/25 club list. **[verified]** the live league now contains
**Coventry City, Hull City, Sunderland, Leeds**, and no longer contains Ipswich / Leicester /
Southampton / Wolves / West Ham. Every shirt image resolves wrong or blank. The map should be built
from `bootstrap_static`'s `teams[].code` at runtime, not hardcoded (see also P2-7).

---

## P1 — Silently wrong model output

### P1-1. Bookmaker implied goals are inflated ~80%

[odds.py:152](src/api/odds.py#L152) splits total goals between teams by win-probability ratio and then
multiplies by `1.8`. The ratio `win_h / (win_h + win_a)` already sums to 1 across both teams, so the
`* 1.8` is pure inflation.

**[verified]** on real 2023-24 odds:

| Quantity | Computed | Reality / stated default |
|---|---|---|
| mean `team_implied_goals` | **2.680** | `LEAGUE_DEFAULTS` = 1.35 |
| implied goals **per match** | **5.360** | actual 2023-24 = **3.279** |
| mean `clean_sheet_prob` | **0.150** | `LEAGUE_DEFAULTS` = 0.30 |
| max `team_implied_goals` | **5.998** | (one team, one match) |

Two compounding problems: the values are wrong in absolute terms, **and** they sit on a completely
different scale from `LEAGUE_DEFAULTS`. Since fallback rows get 1.35/0.30 and computed rows get
2.68/0.15, the feature is bimodal and mostly encodes *"did the odds join succeed"* rather than
anything about football. `compute_anytime_scorer_prob` inherits the inflation directly.

**Fix:** drop the `* 1.8`. Then `home_goals + away_goals ≈ total_goals`, and recalibrate
`LEAGUE_DEFAULTS` to match (~1.35 per team, CS ≈ 0.28).

### P1-2. The odds join never matches current-season rows

`history_builder` keys the lookup on `(season, team, GW)` — but `team` has two incompatible types.
**[verified]** in `historical_features.parquet`:

```
season 2022-23  team = ['Arsenal', 'Aston Villa', 'Bournemouth', ...]   ← names
season 2023-24  team = ['Arsenal', 'Aston Villa', 'Bournemouth', ...]   ← names
season 2024-25  team = ['1', '10', '11', '12', '13', '14', ...]         ← integer FPL ids
```

`_load_vaastav_season` carries vaastav's team **name**; `_load_current_season` writes
`p['team']`, an **int id** ([history_builder.py:90](src/features/history_builder.py#L90)). Current-season
rows therefore never hit `odds_gw_lookup` and silently take `LEAGUE_DEFAULTS`.

### P1-3. The odds join is misaligned even when it does match

`gw_rank` is assigned by sorting a team's matches by date and numbering them 1..38
([history_builder.py:251](src/features/history_builder.py#L251)) — that is *match order*, not
*gameweek*. Any postponement, blank or double GW shifts every subsequent match by one, cumulatively.

**[verified]** in 2023-24, **16 of 20 teams** have ≠38 distinct GWs. So for the large majority of
teams the odds attached to a row belong to a **different fixture**.

**Fix:** join on `(season, team, date)` against `kickoff_time` — the date is already in both frames and
is postponement-proof. (`kickoff_time` is currently dropped at
[history_builder.py:310](src/features/history_builder.py#L310); keep it until after the join.)

### P1-4. `team` and `position` categorical vocabularies differ between train and inference

Consequence of P1-2, plus a second split:

- **`team`** — model trained on a mixed vocabulary of names *and* ids. At inference `processor` emits
  int ids → `"14"`. So two thirds of training rows used a vocabulary the inference path never
  produces, and the categorical is close to meaningless.
- **`position`** — **[verified]** vaastav seasons use `GK`; the current-season loader uses
  `element_types[...]['singular_name_short']`, which the live API confirms is **`GKP`**; and
  `processor` maps `{1: 'GK'}`. So goalkeepers are `GK` in two training seasons, `GKP` in the third,
  and `GK` at inference. (The tell that someone half-noticed: `POSITION_GOAL_SHARE` carries *both*
  `'GK'` and `'GKP'` keys.)

LightGBM stores `pandas_categorical` and remaps at predict time, so this does not crash — unseen
categories just become missing. It fails quietly, which is worse.

**Fix:** normalise both to a single canonical vocabulary in both loaders (ids→names or names→ids, and
`GKP`→`GK`) before the categorical cast.

### P1-5. Minutes-model stacking leakage

[predictor.py:153](src/model/predictor.py#L153): `MinutesPredictor` is trained on `df_train`, then
immediately used to predict `projected_minutes` **for that same `df_train`**, which becomes a feature
of the points model.

Those are **in-sample** predictions — far more accurate than anything achievable at serving time. The
points model therefore learns to over-trust `projected_minutes`, and both the A/B comparison and the
CV RMSE are optimistically biased. This also partly invalidates the A/B result that selects Set A.

**Fix:** generate `projected_minutes` out-of-fold (fit the minutes model on the CV train split only,
predict the val split), as you would for any stacked ensemble.

### P1-6. `days_rest` means different things at train and inference time

- **Train** ([history_builder.py:186](src/features/history_builder.py#L186)):
  `this match kickoff − previous match kickoff` = rest *before the match being predicted*. ✅
- **Infer** ([predictor.py:336](src/model/predictor.py#L336)): `history[-1] − history[-2]` = gap
  between the **two most recent completed** matches — the rest before *last* week's game, not the
  upcoming one. ❌

The feature is shifted one match out of phase at serving time. It matters precisely when it should
matter most: congested fixture periods, where it drives rotation prediction.

**Fix:** use `next_fixture_kickoff − history[-1].kickoff_time`; the next kickoff is available from
`fixtures.json`, which is already loaded.

### P1-7. Understat is never fetched — xG/xA are all zero

`main.py --fetch` fetches it, but the dashboard never does, and nothing else calls `understat.py`.

**[verified]** `data/raw/understat_players.csv` does not exist, and in the live
`player_features.parquet` **`xG`, `xA`, `xG_per_90`, `xA_per_90` are 0 for all 840 rows.**
`processor` takes the `else` branch and fills zeros with only a `print` warning.

Two of the four are in the display/feature set but carry no signal at all. Either wire the fetch into
the dashboard or delete the columns — right now they are dead weight that looks alive.

Related: `understat.py` decodes with `.encode('utf-8').decode('unicode_escape')`
([understat.py:36](src/api/understat.py#L36)), the classic mojibake pattern — it decodes as latin-1
and mangles every accented name. The `[^a-z]` normalisation in `processor` hides some of the damage.

---

## P2 — Logic and UX defects

### P2-1. Chip thresholds are mathematically unreachable

`chips.py` thresholds were calibrated for *actual* FPL points, but the model emits *expected* points —
a conditional mean, which regresses hard toward ~2-6.

**[verified]** against `predictions_gw37.csv` (n=840): `max = 6.32`, `mean = 1.03`, `p99 = 4.50`.

| Chip | Threshold | Players in the **entire game** that clear it |
|---|---|---|
| Triple Captain — *Recommended* | top XP ≥ 11.0 | **0** |
| Triple Captain — *Consider* | top XP ≥ 8.0 | **0** |
| Bench Boost — *Recommended* | bench XP > 18.0 | best-4-in-the-league sum = 21.5, a real bench ≈ 2-6 |

Triple Captain can **never** fire, not even "Consider". Bench Boost effectively never fires. The chip
advisor is decorative for two of its four chips.

**Fix:** rescale to the model's actual output distribution (e.g. TC on percentile of `captaincy_score`,
BB on bench XP relative to the league's bench distribution), or calibrate predictions to true points.

### P2-2. Chip advice is computed on a squad you don't own

`dashboard.py` passes `starters`/`bench` from the **post-transfer optimized** team into
`ChipStrategy.analyze()` ([dashboard.py:190](src/interface/dashboard.py#L190)). So "your bench is
strong enough to Bench Boost" describes a hypothetical bench. Chip decisions must be evaluated against
`current_team_df`.

### P2-3. `SHIRT_MAP` NameError in the Rival Spy tab

Defined inside tab 2's `if not transfers_out.empty:` loop
([dashboard.py:292](src/interface/dashboard.py#L292)), referenced in tab 4
([:532](src/interface/dashboard.py#L532), [:550](src/interface/dashboard.py#L550)). When the optimizer
recommends **no transfers** — the common case for a good squad — Rival Spy raises `NameError`.

### P2-4. Free-transfer count ignores chips

`calculate_free_transfers` ([fpl.py:71](src/api/fpl.py#L71)) subtracts every transfer in the history.
But transfers made on a **Wildcard** or **Free Hit** are free and do **not** consume FTs — and your
banked FTs are preserved across them.

A manager who wildcards and makes 10 transfers gets `available_ft` driven to 0 and reset to 1, when
FPL would have kept their real balance (up to 5). The resulting FT count feeds the hit-cost penalty in
`recommend_transfers`, so a wrong FT directly produces wrong transfer advice.

**Fix:** fetch `history['chips']` (already fetched in the dashboard) and skip transfer counts for GWs
where a `wildcard` or `freehit` was played.

### P2-5. "Projected Gain" compares 15 players against 15

[dashboard.py:265](src/interface/dashboard.py#L265) and the Wildcard delta
([:176](src/interface/dashboard.py#L176)) both sum `predicted_points` over the **whole 15-man squad**.
You only score your **XI** (plus the captain's double). Bench upgrades inflate the reported gain, and
`wc_diff` — the number the Wildcard recommendation is thresholded on — is inflated the same way.

**Fix:** run `select_starting_xi()` on both squads and compare XI totals, adding the captain's XP again.

### P2-6. `minutes_prob` double-counts availability

[predictor.py:379](src/model/predictor.py#L379) multiplies `predicted_points` by `minutes_prob`. But
the model already takes `projected_minutes` / `start_probability` as features, and its target (actual
points) already reflects injuries. Rotation and injury risk are charged twice.

Related: `captaincy_score` — which *does* combine XP, minutes confidence and win probability, and is
the better-designed metric — is computed and then **never used**; the dashboard still ranks captains by
raw `predicted_points` ([dashboard.py:156](src/interface/dashboard.py#L156)). The good metric only
surfaces in the audit report.

### P2-7. `TEAM_SHIRT_MAP` is defined three times, inconsistently

Twice in `pitch_view.py` ([:118](src/interface/pitch_view.py#L118), [:153](src/interface/pitch_view.py#L153)
— the second silently wins and disagrees with the first on Southampton/Spurs/West Ham/Wolves) and once
inlined as `SHIRT_MAP` in `dashboard.py`. Combined with P0-5, all three are now wrong.

### P2-8. The whole pipeline re-runs on every widget interaction

Streamlit re-executes the script top-to-bottom on any interaction, and `st.session_state['has_run']`
stays `True`. So typing in the Rival Spy League ID box, or clicking "Fetch Standings", re-runs:
2 API fetches → `process(force_refresh=True)` (re-downloads odds, rewrites the parquet) → `predict()`
→ **two full CBC integer-program solves**. Nothing between `if st.session_state.get('has_run')` and the
end of tab 4 is cached.

**Fix:** wrap fetch/process/predict/optimize in `@st.cache_data` keyed on `(team_id, gw, budget)`.

### P2-9. Missing runtime dependencies

**[verified]** `requirements.txt` omits three modules that are imported at module load:
`joblib` (predictor), `aiohttp` (async_fpl), `pyarrow`/`fastparquet` (every parquet read/write).
It currently only works because `scikit-learn` drags in `joblib` transitively. `streamlit` is listed
twice, and `fpl` is listed but never imported.

### P2-10. No timeouts on outbound requests

`FPLClient._get`, `UnderstatClient.get_player_stats` and `VaastavClient.download_season` all call
`requests.get(...)` with no `timeout=`. A hung upstream socket hangs the Streamlit worker
indefinitely. `odds.py` gets this right (`timeout=30`/`15`) — apply the same everywhere.

### P2-11. Silent partial failure in the async fetch

`async_fpl.py:51` builds `{str(pid): data for pid, data in results if data is not None}` — failed
players are dropped without a count. They then left-merge as NaN across every rolling feature and are
scored anyway. A 100-player failure is indistinguishable from success. Log the drop count and fail
loudly past a threshold.

### P2-12. CLI-only defects

- `reporter.py:46` writes to `reports/` which is gitignored and **never created** → `FileNotFoundError`
  on the first `main.py` run. Same line uses `open(..., "w")` with **no `encoding`** → on Windows
  (cp1252) any accented name (`Højlund`, `Sánchez`) raises `UnicodeEncodeError`. It also prints all 15
  players under a "Starting XI:" heading, and assigns an unused `starters` variable.
- `main.py:93` uses `name_map[params_out[i]]` — a bare dict index. Any player who left the league
  between the picks fetch and the feature build raises `KeyError`. Use `.get(pid, "?")`.
- `main.py:60` calls `get_team_picks` **without** `freehit_gws`, so the CLI still has the Free Hit
  squad-leak bug that the dashboard path fixed.
- `main.py` `--fetch` does not refresh the element-summary cache, so the CLI reliably runs on stale
  rolling features (and now, per P0-1, corrupt ones).

### P2-13. `processor` cache can return an odds-less parquet

The cache validity check only requires `['next_opponent','news','fixture_difficulty','photo']`
([processor.py:48](src/features/processor.py#L48)). A parquet written before the odds feature existed
passes, and `predictor.predict()` then fills the absent odds columns with **`0.0`** — not
`LEAGUE_DEFAULTS` ([predictor.py:364](src/model/predictor.py#L364)). The dashboard is safe because it
forces a refresh; `main.py` is not.

### P2-14. Budget model ignores the bank and selling prices

`TransferOptimizer` constrains `Σ price ≤ budget` where the dashboard sets
`budget = max(sidebar_budget, current_squad_value)`. Two errors: there is no **bank** input (so
spending power is systematically understated by whatever is in the bank), and FPL sells at
*purchase price + half the rise*, not current price, so the true liquidation value of the squad is
lower than `current_value`. Recommendations can be quietly unaffordable in the real game.

### P2-15. Minor

- `pitch_view.py:7` — module-level `IMAGE_CACHE` dict is dead code (`st.session_state` is used instead).
- `pitch_view.py:268` — emits `.pitch-line` / `.pitch-circle` divs; neither class exists in
  `get_pitch_style()`.
- `pitch_view.py` — `check_image_exists` does a blocking network HEAD (2s timeout) per player on first
  render; ~15 sequential calls before the pitch paints.
- `odds.py:200` — `age_hours` is computed, then immediately overwritten by a second computation. The
  first line is dead.
- `solver.py:69` — `prob` is constructed and never used; `x` is shared across four distinct
  `LpProblem`s (works, but fragile).
- `solver.py` — `df.loc[i, col]` scalar lookups inside comprehensions run ~78k times per
  `recommend_transfers` call. Vectorise to dicts before building constraints.
- `dashboard.py:183` — leftover `importlib.reload(src.optimization.chips)` hot-reload hack.
- `dashboard.py:9-20` — `FPLClient`, `FeatureProcessor` and `RivalSpy` are each imported twice.
- `chips.py:11` — `full_squad_value` parameter is never used.
- `processor.py:93` — `ppm` is computed but not in the `features` list; dead.
- `predictor.py:99` — `MinutesPredictor.predict` casts `position`/`team` **before** filling missing
  columns, so a missing `team` raises `KeyError` instead of defaulting.
- `rivals.py:50` — `main_gap_pos` uses `min()` over a dict; ties resolve by insertion order, silently
  always preferring `GK`.
- Bare `except:` in `dashboard.py` (×4) and `pitch_view.py` swallow every error including
  `KeyboardInterrupt`.
- No `__init__.py`, no test suite; every module depends on `sys.path` mutation and CWD-relative paths.

---

## Suggested fix order

1. **P0-1** (cache/season id collision) — everything downstream is meaningless until this is right.
2. **P0-2** (GW1 dead-end) — the app is unusable this week without it.
3. **P1-2 / P1-4** (team & position vocabulary) — one normalisation fixes the odds join *and* the
   categorical features.
4. **P1-1** (drop the `* 1.8`, recalibrate defaults) — one-line change, large accuracy effect.
5. **P1-3** (join odds on date, not match rank).
6. **P2-1 / P2-2** (chip thresholds and current-vs-optimized squad) — makes half the advisor functional.
7. **P2-3, P2-4, P2-9** — cheap, high-annoyance-reduction.
8. **P1-5 / P1-6** (stacking leakage, `days_rest` phase) — real modelling wins, more involved.
