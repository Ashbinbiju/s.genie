"""
Stale-module eviction.

Regression: Streamlit Cloud hot-reloaded dashboard.py against the NEW source while
leaving src.api.fpl pinned at the OLD version in sys.modules, so a method added in the
same commit raised:

    AttributeError: 'FPLClient' object has no attribute 'get_league_members'
"""
import os
import sys
import time

import pytest

from src.utils.hotreload import (
    drop_stale_modules, newest_source_mtime, reset_marker, MTIME_ATTR,
)


@pytest.fixture(autouse=True)
def clean_marker():
    reset_marker()
    yield
    reset_marker()


@pytest.fixture
def fake_pkg(tmp_path):
    """A throwaway package on disk plus a fake sys.modules holding it."""
    pkg = tmp_path / 'pkg'
    (pkg / 'sub').mkdir(parents=True)
    (pkg / '__init__.py').write_text('', encoding='utf-8')
    (pkg / 'a.py').write_text('VALUE = 1\n', encoding='utf-8')
    (pkg / 'sub' / 'b.py').write_text('VALUE = 1\n', encoding='utf-8')
    modules = {'pkg': object(), 'pkg.a': object(), 'pkg.sub.b': object(),
               'os': object(), 'pkgutil': object(), 'other.pkg': object()}
    return tmp_path, modules


def test_evicts_package_modules_when_source_changes(fake_pkg):
    root, modules = fake_pkg
    evicted = drop_stale_modules(str(root), package='pkg', modules=modules)
    assert set(evicted) == {'pkg', 'pkg.a', 'pkg.sub.b'}
    assert 'pkg.a' not in modules


def test_leaves_unrelated_modules_alone(fake_pkg):
    root, modules = fake_pkg
    drop_stale_modules(str(root), package='pkg', modules=modules)
    assert 'os' in modules
    assert 'pkgutil' in modules, "prefix match must not catch 'pkgutil'"
    assert 'other.pkg' in modules


def test_second_call_is_a_noop_when_nothing_changed(fake_pkg):
    root, modules = fake_pkg
    drop_stale_modules(str(root), package='pkg', modules=modules)
    modules['pkg.a'] = object()
    assert drop_stale_modules(str(root), package='pkg', modules=modules) == []
    assert 'pkg.a' in modules, "unchanged source must not force a re-import"


def test_evicts_again_after_a_later_edit(fake_pkg):
    """The actual deploy scenario: a new commit lands and modules must refresh."""
    root, modules = fake_pkg
    drop_stale_modules(str(root), package='pkg', modules=modules)
    modules['pkg.a'] = object()

    future = time.time() + 10
    src = root / 'pkg' / 'a.py'
    src.write_text('VALUE = 2\n', encoding='utf-8')
    os.utime(src, (future, future))

    evicted = drop_stale_modules(str(root), package='pkg', modules=modules)
    assert 'pkg.a' in evicted
    assert 'pkg.a' not in modules


def test_protected_modules_are_never_evicted(fake_pkg):
    """The entry module must survive; evicting it mid-execution is unsafe."""
    root, modules = fake_pkg
    evicted = drop_stale_modules(str(root), package='pkg', modules=modules,
                                 protect={'pkg.a'})
    assert 'pkg.a' not in evicted
    assert 'pkg.a' in modules


def test_missing_package_is_a_safe_noop(tmp_path):
    modules = {'pkg': object()}
    assert drop_stale_modules(str(tmp_path), package='pkg', modules=modules) == []
    assert 'pkg' in modules, "no sources found -> do not evict anything"


def test_newest_source_mtime_reflects_the_latest_edit(fake_pkg):
    root, _ = fake_pkg
    before = newest_source_mtime(str(root), 'pkg')
    future = time.time() + 100
    target = root / 'pkg' / 'sub' / 'b.py'
    os.utime(target, (future, future))
    assert newest_source_mtime(str(root), 'pkg') > before


def test_marker_is_recorded_on_the_sys_module(fake_pkg):
    root, modules = fake_pkg
    assert not hasattr(sys, MTIME_ATTR)
    drop_stale_modules(str(root), package='pkg', modules=modules)
    assert getattr(sys, MTIME_ATTR) > 0


def test_real_src_package_round_trip():
    """Against the real tree: evicting src.* must leave it importable."""
    import src.api.fpl  # noqa: F401
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    future = time.time() + 5
    marker = os.path.join(root, 'src', 'utils', 'season.py')
    original = os.path.getmtime(marker)
    try:
        os.utime(marker, (future, future))
        evicted = drop_stale_modules(root, package='src')
        assert any(n.startswith('src.') for n in evicted)

        from src.api.fpl import FPLClient
        assert hasattr(FPLClient, 'get_league_members'), (
            "re-import must expose methods added in the same commit")
        assert hasattr(FPLClient, 'get_entry')
    finally:
        os.utime(marker, (original, original))
        reset_marker()
