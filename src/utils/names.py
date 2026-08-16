"""
Player-name normalisation for cross-source matching.

FPL and Understat spell the same player differently: FPL's `web_name` is largely
ASCII-folded ("Odegaard"), Understat keeps the native form ("Ødegaard").

The naive approach — lowercase then strip everything outside [a-z] — looks like it
handles this but does not: it DELETES the special character rather than folding it, so
"Ødegaard" becomes "degaard" while "Odegaard" becomes "odegaard", and the two never
match. Every player with Ø, Ł, Đ, Æ, ß or Ħ in their name is silently dropped from the
join and ends up with zero xG/xA.
"""

import unicodedata

# Letters that Unicode NFKD does NOT decompose — the stroke/bar is part of the glyph,
# not a combining mark — so they need an explicit fold.
_SPECIAL_FOLD = {
    'ø': 'o', 'Ø': 'o',
    'đ': 'd', 'Đ': 'd',
    'ð': 'd', 'Ð': 'd',
    'ł': 'l', 'Ł': 'l',
    'æ': 'ae', 'Æ': 'ae',
    'œ': 'oe', 'Œ': 'oe',
    'ß': 'ss',
    'þ': 'th', 'Þ': 'th',
    'ħ': 'h', 'Ħ': 'h',
    'ı': 'i', 'İ': 'i',
    'ŋ': 'n', 'Ŋ': 'n',
}


def _is_missing(value):
    """True for None and for float NaN, without requiring pandas here.

    NaN must not fall through to str(): it stringifies to 'nan', which becomes a real
    match key that every nameless row shares — silently joining unrelated players to
    each other and duplicating rows.
    """
    return value is None or value != value


def fold_accents(text):
    """'Ødegaard' -> 'odegaard', 'Guimarães' -> 'guimaraes', 'Højlund' -> 'hojlund'."""
    if _is_missing(text):
        return ""
    s = str(text)
    # Explicit folds first: NFKD leaves these glyphs intact.
    s = "".join(_SPECIAL_FOLD.get(ch, ch) for ch in s)
    # Decompose the rest and drop the combining marks (é -> e, ã -> a).
    s = unicodedata.normalize('NFKD', s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.lower()


def normalize_player_name(text):
    """Match key for joining player names across sources: folded, letters only."""
    folded = fold_accents(text)
    return "".join(ch for ch in folded if 'a' <= ch <= 'z')


def normalize_name_series(series):
    """Vectorised `normalize_player_name` for a pandas Series."""
    return series.map(normalize_player_name)
