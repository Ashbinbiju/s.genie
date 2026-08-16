"""
Evict cached project modules when their source changes.

Streamlit re-executes the ENTRY SCRIPT on every rerun, but `import` returns whatever is
already in `sys.modules`. After a hot code update the platform reports "Updated app!"
and reruns the new `dashboard.py` while its dependencies stay pinned at the OLD
version. A function added to a class in the same commit then raises AttributeError even
though the code on disk is correct:

    AttributeError: 'FPLClient' object has no attribute 'get_league_members'

Purging by source mtime fixes it precisely: no work when nothing changed (a few dozen
stat calls), a clean re-import when it did.

This must run BEFORE any `from src... import name` statement, because those bind the
class and function OBJECTS — reloading afterwards leaves the caller holding the old
references. It replaces the scattered `importlib.reload(...)` calls and "force
redeploy" commits this repo accumulated while fighting the same behaviour.
"""

import glob
import os
import sys

MTIME_ATTR = '_fpl_src_mtime'


def newest_source_mtime(project_root, package='src'):
    """Most recent mtime across the package's .py files, or 0.0 if there are none."""
    pattern = os.path.join(project_root, package, '**', '*.py')
    mtimes = []
    for path in glob.glob(pattern, recursive=True):
        try:
            mtimes.append(os.path.getmtime(path))
        except OSError:
            continue
    return max(mtimes) if mtimes else 0.0


def drop_stale_modules(project_root, package='src', modules=None, protect=()):
    """
    Remove cached `package.*` modules if any source file is newer than the last check.

    Returns the list of module names evicted. `protect` names modules to keep (used to
    avoid evicting a module that is mid-execution).
    """
    modules = sys.modules if modules is None else modules

    newest = newest_source_mtime(project_root, package)
    if newest <= 0.0:
        return []
    if getattr(sys, MTIME_ATTR, 0.0) >= newest:
        return []

    protected = set(protect)
    stale = [name for name in list(modules)
             if (name == package or name.startswith(package + '.')) and name not in protected]

    for name in stale:
        modules.pop(name, None)

    setattr(sys, MTIME_ATTR, newest)
    return stale


def reset_marker():
    """Forget the last-seen mtime, so the next call re-evaluates. For tests."""
    if hasattr(sys, MTIME_ATTR):
        delattr(sys, MTIME_ATTR)
