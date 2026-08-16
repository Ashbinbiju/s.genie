"""
Season / team / position canonicalisation helpers.

Single source of truth for anything that changes between seasons. Before this module
existed, season-dependent facts (the current GW, team shirt codes, team names, the
goalkeeper position label) were hardcoded in three or four places each and silently
went stale every August.

Nothing here does network I/O — callers pass in an already-loaded bootstrap_static dict.
"""

import json
import os

BOOTSTRAP_PATH = os.path.join("data", "raw", "bootstrap_static.json")


# ---------------------------------------------------------------------------
# Bootstrap loading
# ---------------------------------------------------------------------------
def load_bootstrap(path=None):
    """Load bootstrap_static.json from disk. Returns None if absent."""
    path = path or BOOTSTRAP_PATH
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Season identity
# ---------------------------------------------------------------------------
def get_season_label(static):
    """
    Derive the season label ('2026-27') from the earliest GW deadline.

    FPL does not expose the season string anywhere, so we infer it: a season that
    kicks off in Aug 2026 is '2026-27'. Anything deadlined before June belongs to
    the season that started the previous calendar year.
    """
    if not static:
        return "unknown"
    deadlines = [e["deadline_time"] for e in static.get("events", []) if e.get("deadline_time")]
    if not deadlines:
        return "unknown"
    first = min(deadlines)
    year, month = int(first[:4]), int(first[5:7])
    start = year if month >= 6 else year - 1
    return f"{start}-{str(start + 1)[-2:]}"


def get_current_gw(static):
    """
    The most recently *started* gameweek, or 0 before the season begins.

    Use this for naming artifacts (model versions, audit reports). Do NOT use it to
    decide which gameweek to plan for — that is get_next_gw().
    """
    if not static:
        return 0
    events = static.get("events", [])
    for ev in events:
        if ev.get("is_current"):
            return ev["id"]
    # Pre-season: no GW is current yet. Derive from is_next rather than guessing.
    for ev in events:
        if ev.get("is_next"):
            return max(0, ev["id"] - 1)
    # Season over: every event finished.
    finished = [ev["id"] for ev in events if ev.get("finished")]
    return max(finished) if finished else 0


def get_next_gw(static):
    """The gameweek we are planning for. Falls back to is_current, then 1."""
    if not static:
        return 1
    events = static.get("events", [])
    for ev in events:
        if ev.get("is_next"):
            return ev["id"]
    for ev in events:
        if ev.get("is_current"):
            return ev["id"]
    return 1


def is_preseason(static):
    """True when no gameweek has started yet — squad picks do not exist."""
    if not static:
        return False
    return not any(ev.get("is_current") or ev.get("finished") for ev in static.get("events", []))


# ---------------------------------------------------------------------------
# Teams
# ---------------------------------------------------------------------------
def team_id_to_name(static):
    """{1: 'Arsenal', ...} — FPL team id → full name. Ids are NOT stable across seasons."""
    if not static:
        return {}
    return {t["id"]: t["name"] for t in static.get("teams", [])}


def team_id_to_code(static):
    """
    {1: 3, ...} — FPL team id → shirt/badge code.

    `code` is the correct and complete source for shirt image URLs
    (https://fantasy.premierleague.com/dist/img/shirts/standard/shirt_{code}-110.webp);
    verified to resolve for all 20 clubs. The hardcoded TEAM_SHIRT_MAP tables this
    replaces were an unnecessary workaround that broke on promotion/relegation.
    """
    if not static:
        return {}
    return {t["id"]: t["code"] for t in static.get("teams", [])}


def shirt_url(team_code):
    """Club shirt image URL, or the generic blank shirt when the code is unknown."""
    if not team_code:
        return "https://fantasy.premierleague.com/img/shirts/standard/shirt_0.png"
    return f"https://fantasy.premierleague.com/dist/img/shirts/standard/shirt_{int(team_code)}-110.webp"


def player_photo_url(photo):
    """Player headshot URL from bootstrap's `photo` field ('123456.jpg'), or None."""
    pid = str(photo or "").replace(".jpg", "").replace(".png", "").lstrip("p")
    return (
        f"https://resources.premierleague.com/premierleague/photos/players/110x140/p{pid}.png"
        if pid.isdigit()
        else None
    )


# ---------------------------------------------------------------------------
# Canonical vocabularies for model categorical features
# ---------------------------------------------------------------------------
# Team NAMES are the only cross-season-stable team key: FPL reassigns team ids every
# season, so a model trained on ids learns nothing transferable. Every training and
# inference path must agree on these strings.
TEAM_NAME_CANON = {
    # vaastav / football-data.co.uk spellings → FPL bootstrap `name`
    "Man United": "Man Utd",
    "Manchester United": "Man Utd",
    "Manchester City": "Man City",
    "Tottenham": "Spurs",
    "Tottenham Hotspur": "Spurs",
    "Sheffield United": "Sheffield Utd",
    "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nott'm Forest",
    "Nott'm Forest": "Nott'm Forest",
    "Wolverhampton": "Wolves",
    "West Ham United": "West Ham",
    "Brighton and Hove Albion": "Brighton",
    "Leeds United": "Leeds",
    "Leicester City": "Leicester",
    "Ipswich": "Ipswich Town",
    "Hull": "Hull City",
    "Coventry": "Coventry City",
    "Luton": "Luton",
}

# FPL's element_types use 'GKP'; vaastav's historical CSVs use 'GK'. Collapse to 'GK'.
POSITION_CANON = {
    "GK": "GK",
    "GKP": "GK",
    "GOALKEEPER": "GK",
    "DEF": "DEF",
    "MID": "MID",
    "FWD": "FWD",
}

ELEMENT_TYPE_TO_POSITION = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


def canon_team(name):
    """Normalise any team spelling to the FPL bootstrap name."""
    s = str(name).strip()
    return TEAM_NAME_CANON.get(s, s)


def canon_position(pos):
    """Normalise a position label; 'GKP' → 'GK'."""
    s = str(pos).strip().upper()
    return POSITION_CANON.get(s, s)
