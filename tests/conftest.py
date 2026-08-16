import os
import sys

import pandas as pd
import pytest

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


@pytest.fixture
def bootstrap():
    """Minimal bootstrap_static payload: 3 clubs, 4 element types, 3 gameweeks."""
    return {
        'events': [
            {'id': 1, 'deadline_time': '2026-08-21T17:30:00Z', 'is_current': False,
             'is_next': False, 'finished': True},
            {'id': 2, 'deadline_time': '2026-08-28T17:30:00Z', 'is_current': True,
             'is_next': False, 'finished': False},
            {'id': 3, 'deadline_time': '2026-09-04T17:30:00Z', 'is_current': False,
             'is_next': True, 'finished': False},
        ],
        'teams': [
            {'id': 1, 'name': 'Arsenal', 'short_name': 'ARS', 'code': 3},
            {'id': 2, 'name': 'Man Utd', 'short_name': 'MUN', 'code': 1},
            {'id': 3, 'name': 'Coventry City', 'short_name': 'COV', 'code': 9},
        ],
        'element_types': [
            {'id': 1, 'singular_name_short': 'GKP'},
            {'id': 2, 'singular_name_short': 'DEF'},
            {'id': 3, 'singular_name_short': 'MID'},
            {'id': 4, 'singular_name_short': 'FWD'},
        ],
        'elements': [
            {'id': 10, 'web_name': 'Raya', 'team': 1, 'element_type': 1},
            {'id': 11, 'web_name': 'Gabriel', 'team': 1, 'element_type': 2},
            {'id': 12, 'web_name': 'Saka', 'team': 1, 'element_type': 3},
        ],
    }


@pytest.fixture
def preseason_bootstrap(bootstrap):
    """Same shape, but nothing has been played yet."""
    payload = {k: (list(v) if isinstance(v, list) else v) for k, v in bootstrap.items()}
    payload['events'] = [
        {'id': 1, 'deadline_time': '2026-08-21T17:30:00Z', 'is_current': False,
         'is_next': True, 'finished': False},
        {'id': 2, 'deadline_time': '2026-08-28T17:30:00Z', 'is_current': False,
         'is_next': False, 'finished': False},
    ]
    return payload


def make_squad(n_gk=2, n_def=5, n_mid=5, n_fwd=3, points=None, teams=None):
    """A squad DataFrame shaped like the optimizer's output."""
    rows = []
    pid = 1
    for etype, count in ((1, n_gk), (2, n_def), (3, n_mid), (4, n_fwd)):
        for _ in range(count):
            rows.append({
                'id': pid,
                'web_name': f'P{pid}',
                'element_type': etype,
                'team': (pid % 6) + 1,
                'team_name': f'Club{(pid % 6) + 1}',
                'team_code': 3,
                'position': {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}[etype],
                'price': 4.0 + (pid % 8),
                'predicted_points': float(points[pid - 1]) if points else float(20 - pid),
                'minutes_prob': 1.0,
                'photo': '',
                'next_opponent': 'XYZ (H)',
                'fixture_difficulty': 3.0,
            })
            pid += 1
    df = pd.DataFrame(rows)
    df['captaincy_score'] = df['predicted_points']
    if teams is not None:
        df['team'] = teams
    return df


@pytest.fixture
def squad():
    return make_squad()
