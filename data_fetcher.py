# data_fetcher.py
"""
Robust helpers to fetch scoreboard + roster data from ESPN's public endpoints.
This module normalizes roster JSON into a list of players:
  [{'id': <id>, 'name': <displayName>, 'position': <abbrev>}...]
It attempts several common JSON shapes returned by ESPN and falls back gracefully.
"""

import requests
import datetime
from typing import Dict, Any, List, Union, Iterable, Optional

BASE = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"


def get_scoreboard_for_date(date: datetime.date) -> Dict[str, Any]:
    """Return scoreboard JSON for a given date (YYYYMMDD)."""
    datestr = date.strftime("%Y%m%d")
    url = f"{BASE}/scoreboard?dates={datestr}"
    r = requests.get(url, timeout=12)
    r.raise_for_status()
    return r.json()


def _safe_get(d: Dict[str, Any], *keys, default=None):
    """Helper to safely descend dicts."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
        if cur is default:
            return default
    return cur


def _extract_player_from_obj(obj: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """
    Given a dict representing a player or a wrapper, attempt to extract
    {'id','name','position'}.
    Return None if can't find reasonable fields.
    """
    if not isinstance(obj, dict):
        return None

    # Common: object might be { 'athlete': {...}, 'position': {...} }
    if 'athlete' in obj and isinstance(obj['athlete'], dict):
        p = obj['athlete']
        pos = obj.get('position') or p.get('position') or {}
        return {
            'id': str(p.get('id') or p.get('guid') or p.get('uid') or ""),
            'name': p.get('displayName') or p.get('fullName') or p.get('name') or "",
            'position': (pos.get('abbreviation') if isinstance(pos, dict) else pos) or ""
        }

    # Common: the object itself is the athlete
    if obj.get('id') and (obj.get('displayName') or obj.get('shortName') or obj.get('fullName') or obj.get('name')):
        pos = obj.get('position') or {}
        return {
            'id': str(obj.get('id') or obj.get('guid') or obj.get('uid') or ""),
            'name': obj.get('displayName') or obj.get('fullName') or obj.get('name') or "",
            'position': (pos.get('abbreviation') if isinstance(pos, dict) else pos) or ""
        }

    # Some shapes: item contains nested fields under 'player' or 'person'
    for alt in ('player', 'person'):
        if alt in obj and isinstance(obj[alt], dict):
            p = obj[alt]
            pos = obj.get('position') or p.get('position') or {}
            return {
                'id': str(p.get('id') or p.get('guid') or p.get('uid') or ""),
                'name': p.get('displayName') or p.get('fullName') or p.get('name') or "",
                'position': (pos.get('abbreviation') if isinstance(pos, dict) else pos) or ""
            }

    # Not recognized
    return None


def _normalize_roster_json(j: Union[Dict[str, Any], List[Any]]) -> List[Dict[str, str]]:
    """
    Try many reasonable parsing strategies for varying ESPN roster JSON shapes.
    Return list of players: {'id','name','position'}
    """
    players: List[Dict[str, str]] = []

    # If top-level is a list of player dicts
    if isinstance(j, list):
        for item in j:
            p = _extract_player_from_obj(item)
            if p:
                players.append(p)
        if players:
            return players

    # If top-level is a dict
    if isinstance(j, dict):
        # Strategy A: j['athletes'] -> list of groups. Each group may contain 'items'
        athletes = j.get('athletes')
        if isinstance(athletes, list) and athletes:
            for group in athletes:
                # group may be dict with 'items'
                if isinstance(group, dict) and 'items' in group and isinstance(group['items'], list):
                    for item in group['items']:
                        # item could be dict with 'athlete' or a player dict itself
                        p = _extract_player_from_obj(item)
                        if p:
                            players.append(p)
                # or group might be a player dict itself
                else:
                    p = _extract_player_from_obj(group)
                    if p:
                        players.append(p)
            if players:
                return players

        # Strategy B: search any list-valued keys for player-lists
        for k, v in j.items():
            if isinstance(v, list) and v:
                # check first element for 'athlete' or 'displayName'
                first = v[0]
                if isinstance(first, dict):
                    # try extracting players from this list
                    temp = []
                    for item in v:
                        p = _extract_player_from_obj(item)
                        if p:
                            temp.append(p)
                    if temp:
                        players.extend(temp)
                        # continue searching other lists but keep results
        if players:
            return players

        # Strategy C: some endpoints embed under nested dicts. Walk values and recursively attempt to find lists
        for v in j.values():
            if isinstance(v, dict):
                # try to find 'athletes' deeper
                deeper = v.get('athletes') or v.get('items') or v.get('players')
                if isinstance(deeper, list):
                    for el in deeper:
                        if isinstance(el, dict):
                            # if el has 'items' list, go deeper
                            if 'items' in el and isinstance(el['items'], list):
                                for item in el['items']:
                                    p = _extract_player_from_obj(item)
                                    if p:
                                        players.append(p)
                            else:
                                p = _extract_player_from_obj(el)
                                if p:
                                    players.append(p)
            elif isinstance(v, list):
                for el in v:
                    p = _extract_player_from_obj(el) if isinstance(el, dict) else None
                    if p:
                        players.append(p)
        if players:
            return players

    # nothing found
    return []


def get_team_roster(team_id: int) -> List[Dict[str, str]]:
    """
    Fetch roster JSON and return a normalized list of player dicts:
      [{'id':..., 'name':..., 'position':...}, ...]
    On failure returns empty list (caller should fallback to demo player).
    """
    try:
        url = f"{BASE}/teams/{team_id}/roster"
        r = requests.get(url, timeout=12)
        r.raise_for_status()
        j = r.json()
        players = _normalize_roster_json(j)
        return players
    except Exception:
        # caller handles fallback
        return []
