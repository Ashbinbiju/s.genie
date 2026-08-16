"""Squad legality, XI selection, captaincy and chip advice."""
import pandas as pd
import pytest

from conftest import make_squad
from src.optimization.solver import TransferOptimizer, SQUAD_QUOTA, MAX_PER_CLUB, SQUAD_SIZE
from src.optimization.team_selection import (
    select_starting_xi, squad_expected_points, pick_captain,
    FORMATION_MIN, FORMATION_MAX, XI_SIZE,
)
from src.optimization.chips import ChipStrategy


# ---------------------------------------------------------------- starting XI
def test_xi_is_formation_legal(squad):
    starters, bench = select_starting_xi(squad)
    assert len(starters) == XI_SIZE
    assert len(bench) == 4
    counts = starters['element_type'].value_counts().to_dict()
    for pos in (1, 2, 3, 4):
        assert FORMATION_MIN[pos] <= counts.get(pos, 0) <= FORMATION_MAX[pos]


def test_xi_and_bench_partition_the_squad(squad):
    starters, bench = select_starting_xi(squad)
    assert set(starters['id']) | set(bench['id']) == set(squad['id'])
    assert not set(starters['id']) & set(bench['id'])


def test_xi_prefers_higher_scorers():
    # One monster forward must start even though only 1 FWD is required.
    pts = [1] * 15
    pts[12] = 99  # a forward
    squad = make_squad(points=pts)
    starters, _ = select_starting_xi(squad)
    assert 99 in starters['predicted_points'].values


def test_reserve_keeper_is_first_on_the_bench(squad):
    _, bench = select_starting_xi(squad)
    assert bench.iloc[0]['element_type'] == 1, "auto-subs can only use the GK for the GK"


def test_xi_tolerates_incomplete_squad():
    """Regression: a squad missing a position raised IndexError."""
    partial = make_squad(n_gk=1, n_def=2, n_mid=3, n_fwd=1)
    starters, bench = select_starting_xi(partial)
    assert len(starters) + len(bench) == len(partial)


def test_xi_handles_empty_frame():
    starters, bench = select_starting_xi(pd.DataFrame())
    assert starters.empty and bench.empty


# ---------------------------------------------------------------- captaincy
def test_captain_is_never_a_goalkeeper():
    """A keeper's ceiling makes doubling one wrong even when a flat model ranks it top."""
    pts = [1] * 15
    pts[0] = 99  # a goalkeeper
    squad = make_squad(points=pts)
    starters, _ = select_starting_xi(squad)
    captain, _ = pick_captain(starters)
    assert captain['element_type'] != 1


def test_captain_uses_captaincy_score_when_present(squad):
    squad = squad.copy()
    squad['captaincy_score'] = 0.0
    target = squad.index[squad['element_type'] != 1][3]
    squad.loc[target, 'captaincy_score'] = 999.0
    starters, _ = select_starting_xi(squad)
    if squad.loc[target, 'id'] in set(starters['id']):
        captain, _ = pick_captain(starters)
        assert captain['id'] == squad.loc[target, 'id']


def test_captain_and_vice_are_different(squad):
    starters, _ = select_starting_xi(squad)
    captain, vice = pick_captain(starters)
    assert captain['id'] != vice['id']


# ---------------------------------------------------------------- expected points
def test_squad_expected_points_doubles_the_captain(squad):
    starters, _ = select_starting_xi(squad)
    captain, _ = pick_captain(starters)
    plain = starters['predicted_points'].sum()
    with_cap = squad_expected_points(starters, captain['id'])
    assert with_cap == pytest.approx(plain + captain['predicted_points'])


def test_squad_expected_points_ignores_the_bench(squad):
    """Regression: gains were computed over all 15, but only the XI scores."""
    starters, bench = select_starting_xi(squad)
    total = squad_expected_points(starters)
    assert total == pytest.approx(starters['predicted_points'].sum())
    assert total < squad['predicted_points'].sum()


# ---------------------------------------------------------------- solver
@pytest.fixture(scope='module')
def player_pool():
    import numpy as np
    rng = np.random.default_rng(0)
    rows = []
    pid = 1
    for team in range(1, 21):
        for etype, n in ((1, 3), (2, 6), (3, 6), (4, 4)):
            for _ in range(n):
                rows.append({
                    'id': pid, 'web_name': f'P{pid}', 'element_type': etype,
                    'team': team, 'price': float(rng.integers(40, 130)) / 10,
                    'predicted_points': float(rng.uniform(0, 8)),
                })
                pid += 1
    return pd.DataFrame(rows)


def assert_legal_squad(df):
    assert len(df) == SQUAD_SIZE
    assert df['element_type'].value_counts().to_dict() == SQUAD_QUOTA
    assert df['team'].value_counts().max() <= MAX_PER_CLUB
    assert df['id'].nunique() == SQUAD_SIZE, "the same player must not be picked twice"


def test_solve_team_respects_every_constraint(player_pool):
    squad = TransferOptimizer(budget=100.0).solve_team(player_pool)
    assert squad is not None
    assert_legal_squad(squad)
    assert squad['price'].sum() <= 100.0 + 1e-6


def test_solve_team_returns_none_when_no_players(player_pool):
    assert TransferOptimizer(budget=100.0).solve_team(player_pool.iloc[0:0]) is None


def test_transfers_respect_constraints_and_k_cap(player_pool):
    opt = TransferOptimizer(budget=100.0)
    base = opt.solve_team(player_pool)
    current_ids = base['id'].tolist()
    result = opt.recommend_transfers(player_pool, current_ids, free_transfers=1)
    assert result is not None
    assert_legal_squad(result)
    n_new = len(set(result['id']) - set(current_ids))
    assert n_new <= 3, "k is capped to avoid churning the whole squad"


def test_transfers_prefer_holding_when_gain_is_marginal(player_pool):
    """An already-optimal squad should not be churned for noise."""
    opt = TransferOptimizer(budget=100.0)
    optimal = opt.solve_team(player_pool)
    result = opt.recommend_transfers(player_pool, optimal['id'].tolist(), free_transfers=1)
    assert len(set(result['id']) - set(optimal['id'])) == 0


def test_transfers_survive_a_player_missing_from_the_pool(player_pool):
    """A squad member who left the league must not make the problem infeasible."""
    opt = TransferOptimizer(budget=100.0)
    base = opt.solve_team(player_pool)
    current_ids = base['id'].tolist() + [999999]  # id that does not exist
    result = opt.recommend_transfers(player_pool, current_ids, free_transfers=1)
    assert result is not None
    assert_legal_squad(result)


# ---------------------------------------------------------------- chips
@pytest.fixture
def chip_inputs():
    starters = pd.DataFrame({
        'web_name': [f'S{i}' for i in range(11)],
        'element_type': [1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4],
        'predicted_points': [6.3, 5.1, 4.4, 4.0, 3.8, 3.6, 3.4, 3.1, 2.9, 2.6, 2.2],
    })
    starters['captaincy_score'] = starters['predicted_points']
    return starters, starters['predicted_points'].sum()


def chip(recs, name):
    return next(r for r in recs if r['chip'] == name)


def test_triple_captain_can_actually_fire(chip_inputs):
    """Regression: the old 11.0 absolute threshold was unreachable — 0 players in the
    entire game cleared even the 8.0 'Consider' bar."""
    starters, xi = chip_inputs
    star = starters.copy()
    star.loc[0, 'predicted_points'] = star['predicted_points'].mean() * 2.5
    star['captaincy_score'] = star['predicted_points']
    bench = pd.DataFrame({'web_name': list('ABCD'), 'predicted_points': [1.0] * 4})
    recs = ChipStrategy(1, {}).analyze(star, bench, 5, current_xi_xp=xi)
    assert chip(recs, 'Triple Captain')['recommendation'] == 'Recommended'


def test_bench_boost_fires_on_a_strong_bench(chip_inputs):
    starters, xi = chip_inputs
    strong = pd.DataFrame({'web_name': list('ABCD'), 'predicted_points': [3.2, 3.0, 2.8, 2.6]})
    recs = ChipStrategy(1, {}).analyze(starters, strong, 5, current_xi_xp=xi)
    assert chip(recs, 'Bench Boost')['recommendation'] == 'Recommended'


def test_bench_boost_saved_on_a_weak_bench(chip_inputs):
    starters, xi = chip_inputs
    weak = pd.DataFrame({'web_name': list('ABCD'), 'predicted_points': [0.9, 0.6, 0.4, 0.2]})
    recs = ChipStrategy(1, {}).analyze(starters, weak, 5, current_xi_xp=xi)
    assert chip(recs, 'Bench Boost')['recommendation'] == 'Save'


def test_free_hit_fires_on_a_blank_gameweek_crisis(chip_inputs):
    starters, xi = chip_inputs
    bench = pd.DataFrame({'web_name': list('ABCD'), 'predicted_points': [0.1] * 4})
    recs = ChipStrategy(1, {}).analyze(starters, bench, 5, active_players=7, current_xi_xp=xi)
    assert chip(recs, 'Free Hit')['recommendation'] == 'Recommended'


def test_chip_used_before_gw20_is_restored_at_gw20(chip_inputs):
    starters, xi = chip_inputs
    bench = pd.DataFrame({'web_name': list('ABCD'), 'predicted_points': [1.0] * 4})
    history = {'chips': [{'name': 'wildcard', 'event': 3}]}

    before = ChipStrategy(1, history).analyze(starters, bench, 19, wildcard_diff=99, current_xi_xp=xi)
    after = ChipStrategy(1, history).analyze(starters, bench, 20, wildcard_diff=99, current_xi_xp=xi)

    assert chip(before, 'Wildcard')['recommendation'] == 'Used'
    assert chip(after, 'Wildcard')['recommendation'] == 'Recommended'
    assert 'RESTORED' in chip(after, 'Wildcard')['reason']


def test_chip_used_after_gw20_stays_used(chip_inputs):
    starters, xi = chip_inputs
    bench = pd.DataFrame({'web_name': list('ABCD'), 'predicted_points': [1.0] * 4})
    history = {'chips': [{'name': 'wildcard', 'event': 3}, {'name': 'wildcard', 'event': 25}]}
    recs = ChipStrategy(1, history).analyze(starters, bench, 30, wildcard_diff=99, current_xi_xp=xi)
    assert chip(recs, 'Wildcard')['recommendation'] == 'Used'


def test_future_chip_event_does_not_block_now(chip_inputs):
    """A chip logged for a later GW cannot make it unavailable today."""
    starters, xi = chip_inputs
    bench = pd.DataFrame({'web_name': list('ABCD'), 'predicted_points': [1.0] * 4})
    history = {'chips': [{'name': 'freehit', 'event': 25}]}
    recs = ChipStrategy(1, history).analyze(starters, bench, 10, active_players=5, current_xi_xp=xi)
    assert chip(recs, 'Free Hit')['recommendation'] == 'Recommended'


def test_chip_advice_survives_an_all_zero_squad():
    """Pre-season every projection collapses; nothing may divide by zero."""
    starters = pd.DataFrame({
        'web_name': [f'S{i}' for i in range(11)],
        'element_type': [1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4],
        'predicted_points': [0.0] * 11,
        'captaincy_score': [0.0] * 11,
    })
    bench = pd.DataFrame({'web_name': list('ABCD'), 'predicted_points': [0.0] * 4})
    recs = ChipStrategy(1, {}).analyze(starters, bench, 1, current_xi_xp=0.0)
    assert len(recs) == 4
    assert all(r['recommendation'] in {'Recommended', 'Consider', 'Save', 'Used'} for r in recs)


# ---------------------------------------------------------------- objective shape
def premium_pool():
    """A cheap-but-decent field plus one expensive standout."""
    rows = []
    pid = 1
    for team in range(1, 21):
        for etype, n in ((1, 3), (2, 6), (3, 6), (4, 4)):
            for _ in range(n):
                rows.append({'id': pid, 'web_name': f'P{pid}', 'element_type': etype,
                             'team': team, 'price': 4.5, 'predicted_points': 3.0})
                pid += 1
    df = pd.DataFrame(rows)
    # One clear premium: double the points, but expensive.
    df.loc[df.index[-1], ['web_name', 'price', 'predicted_points']] = ['Star', 13.0, 8.0]
    return df


def test_optimizer_annotates_starters_and_captain(player_pool):
    squad = TransferOptimizer(budget=100.0).solve_team(player_pool)
    assert 'is_starter' in squad.columns and 'is_captain' in squad.columns
    assert squad['is_starter'].sum() == XI_SIZE
    assert squad['is_captain'].sum() == 1
    captain = squad[squad['is_captain']].iloc[0]
    assert captain['is_starter'], "the captain must be in the starting XI"


def test_starting_xi_chosen_by_the_optimizer_is_formation_legal(player_pool):
    squad = TransferOptimizer(budget=100.0).solve_team(player_pool)
    counts = squad[squad['is_starter']]['element_type'].value_counts().to_dict()
    for pos in (1, 2, 3, 4):
        assert FORMATION_MIN[pos] <= counts.get(pos, 0) <= FORMATION_MAX[pos]


def test_captain_is_the_highest_scoring_starter(player_pool):
    """The armband doubles, so it belongs on the best starter."""
    squad = TransferOptimizer(budget=100.0).solve_team(player_pool)
    starters = squad[squad['is_starter']]
    captain = squad[squad['is_captain']].iloc[0]
    assert captain['predicted_points'] == pytest.approx(starters['predicted_points'].max())


def test_bench_is_cheaper_than_the_starting_xi(player_pool):
    """
    Regression: the objective summed all 15 equally, so bench points counted as if
    they were scored and the optimizer bought fifteen mid-price players instead of a
    strong XI with cheap cover.
    """
    squad = TransferOptimizer(budget=100.0).solve_team(player_pool)
    bench_cost = squad[~squad['is_starter']]['price'].mean()
    xi_cost = squad[squad['is_starter']]['price'].mean()
    assert bench_cost < xi_cost


def test_premium_is_bought_when_the_armband_justifies_it():
    """
    Regression: with the captain's doubling missing from the objective, a premium was
    valued at face and priced out. Here the standout is worth 8.0 -> 16.0 captained,
    against a field of 3.0, so it must be selected and given the armband.
    """
    squad = TransferOptimizer(budget=100.0).solve_team(premium_pool())
    assert 'Star' in set(squad['web_name']), "premium priced out despite doubling"
    star = squad[squad['web_name'] == 'Star'].iloc[0]
    assert star['is_starter'] and star['is_captain']


def test_select_starting_xi_honours_the_optimizer_decision(player_pool):
    """The UI must display the XI the objective was actually maximised over."""
    squad = TransferOptimizer(budget=100.0).solve_team(player_pool)
    starters, bench = select_starting_xi(squad)
    assert set(starters['id']) == set(squad[squad['is_starter']]['id'])
    assert set(bench['id']) == set(squad[~squad['is_starter']]['id'])


def test_pick_captain_honours_the_optimizer_decision(player_pool):
    squad = TransferOptimizer(budget=100.0).solve_team(player_pool)
    starters, _ = select_starting_xi(squad)
    captain, vice = pick_captain(starters)
    assert captain['id'] == squad[squad['is_captain']].iloc[0]['id']
    assert vice is not None and vice['id'] != captain['id']
