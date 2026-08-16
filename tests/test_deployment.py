"""
The deployment contract.

A deployed instance (Streamlit Cloud) starts from a fresh git checkout with no data
directory. It can refetch everything it needs EXCEPT the trained model, which requires
multi-season history that is not deployed. If the model is not committed, the app
permanently runs the emergency heuristic — which is exactly what happened in production.
"""
import json
import os
import subprocess

import lightgbm as lgb
import pytest

from src.model.predictor import (
    load_booster, save_booster, model_bundle_exists, PointsPredictor, MinutesPredictor,
)

POINTS_BASE = 'data/models/lgb_ts_points'
MINUTES_BASE = 'data/models/lgb_ts_minutes'

SHIPPED = [f'{POINTS_BASE}.txt', f'{POINTS_BASE}.meta.json',
           f'{MINUTES_BASE}.txt', f'{MINUTES_BASE}.meta.json']

requires_models = pytest.mark.skipif(
    not (model_bundle_exists(POINTS_BASE) and model_bundle_exists(MINUTES_BASE)),
    reason="trained models required; run python src/model/predictor.py",
)


def is_gitignored(path):
    """
    True if git would exclude `path`.

    Must be read from the EXIT CODE, not stdout: `check-ignore -v` also prints the
    matching rule when that rule is a negation (`!pattern`), in which case the file is
    explicitly *included* and the command exits non-zero.
    """
    return subprocess.run(['git', 'check-ignore', '-q', path],
                          capture_output=True).returncode == 0


def git(*args):
    return subprocess.run(['git', *args], capture_output=True, text=True).stdout.strip()


# ---------------------------------------------------------------- shipping
@requires_models
@pytest.mark.parametrize('path', SHIPPED)
def test_model_artifact_is_not_gitignored(path):
    """The bug this catches: `data/models/` ignored, so the model never deployed."""
    assert not is_gitignored(path), (
        f"{path} is gitignored and would never reach the deployed app")


@requires_models
@pytest.mark.parametrize('path', SHIPPED)
def test_model_artifact_is_actually_tracked_or_stageable(path):
    """Stronger than the ignore check: git must really be willing to add the file."""
    tracked = git('ls-files', '--', path)
    if not tracked:
        dry = subprocess.run(['git', 'add', '--dry-run', '--', path],
                             capture_output=True, text=True)
        assert dry.returncode == 0 and 'add' in dry.stdout, (
            f"git refuses to stage {path}: {dry.stdout}{dry.stderr}")


@requires_models
@pytest.mark.parametrize('path', SHIPPED)
def test_model_artifact_exists_and_is_reasonably_sized(path):
    assert os.path.exists(path), f"{path} missing — run python src/model/predictor.py"
    size = os.path.getsize(path)
    assert size > 200, f"{path} is suspiciously small ({size} bytes)"
    assert size < 20 * 1024 * 1024, f"{path} is too large for git ({size} bytes)"


def test_local_only_artifacts_stay_ignored():
    """Per-GW snapshots and raw data must NOT be committed."""
    for path in ['data/models/points_model_gw5.txt', 'data/processed/x.parquet',
                 'data/cache/element_summary_2026-27_gw_1.json', 'data/raw/bootstrap_static.json']:
        assert is_gitignored(path), f"{path} should be gitignored but is not"


# ---------------------------------------------------------------- portability
@requires_models
def test_models_are_stored_in_the_portable_text_format():
    """
    Not a pickle: these are trained locally and loaded on Streamlit Cloud under a
    different Python (runtime.txt pins 3.11). A pickle carries its writer's versions.
    """
    for base in (POINTS_BASE, MINUTES_BASE):
        assert os.path.exists(f'{base}.txt')
        with open(f'{base}.txt', 'r', encoding='utf-8') as f:
            head = f.read(200)
        assert 'tree' in head, "not a LightGBM native model file"


@requires_models
def test_shipped_models_load_and_carry_their_feature_list():
    for base in (POINTS_BASE, MINUTES_BASE):
        booster, meta = load_booster(base)
        assert booster is not None
        assert meta['features'], f"{base} has no feature list"
        assert booster.num_feature() == len(meta['features'])
        assert booster.num_trees() > 0


@requires_models
def test_points_model_metadata_records_provenance():
    _, meta = load_booster(POINTS_BASE)
    for key in ['trained_at', 'season', 'cv_rmse', 'train_seasons', 'n_train_rows']:
        assert key in meta, f"missing provenance field: {key}"
    assert meta['n_train_rows'] > 1000
    assert meta['cv_rmse'] and meta['cv_rmse'] < 10


@requires_models
def test_categorical_ordering_survives_the_round_trip():
    """
    pandas_categorical is the category ordering the categorical features are encoded
    against. If it were lost, team/opponent/position would decode to the wrong values.
    """
    booster, _ = load_booster(POINTS_BASE)
    assert booster.pandas_categorical, "categorical ordering not preserved"
    sizes = [len(g) for g in booster.pandas_categorical]
    assert all(s > 1 for s in sizes), sizes


def test_save_load_round_trip_is_lossless(tmp_path):
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    X = pd.DataFrame({'a': rng.normal(size=200), 'b': rng.normal(size=200)})
    y = X['a'] * 2 + rng.normal(scale=0.1, size=200)
    booster = lgb.train({'objective': 'regression', 'verbose': -1},
                        lgb.Dataset(X, label=y), num_boost_round=10)

    base = str(tmp_path / 'm')
    save_booster(booster, ['a', 'b'], {'trained_at': 'now', 'cv_rmse': 1.0}, base)

    loaded, meta = load_booster(base)
    assert meta['features'] == ['a', 'b']
    assert meta['cv_rmse'] == 1.0
    assert np.allclose(booster.predict(X), loaded.predict(X))


def test_missing_bundle_reports_absence_rather_than_raising(tmp_path):
    booster, meta = load_booster(str(tmp_path / 'nothing'))
    assert booster is None and meta is None
    assert not model_bundle_exists(str(tmp_path / 'nothing'))


def test_legacy_pickle_bundle_is_still_readable(tmp_path):
    """An existing checkout with .pkl artifacts must keep working."""
    import joblib
    import numpy as np
    import pandas as pd

    X = pd.DataFrame({'a': np.arange(100.0), 'b': np.arange(100.0)})
    booster = lgb.train({'objective': 'regression', 'verbose': -1},
                        lgb.Dataset(X, label=X['a']), num_boost_round=5)
    base = str(tmp_path / 'legacy')
    joblib.dump({'model': booster, 'features': ['a', 'b'], 'cv_rmse': 2.0}, f'{base}.pkl')

    assert model_bundle_exists(base)
    loaded, meta = load_booster(base)
    assert loaded is not None and meta['features'] == ['a', 'b']


# ---------------------------------------------------------------- wiring
@requires_models
def test_predictors_resolve_the_shipped_bundles():
    """Guards against a path rename silently turning the ML path into a fallback."""
    p = PointsPredictor()
    assert p.load_model(), f"PointsPredictor cannot load {p.model_base}"
    assert p.features_list

    m = MinutesPredictor()
    assert m.load_model(), f"MinutesPredictor cannot load {m.model_base}"
    assert m.features_list
