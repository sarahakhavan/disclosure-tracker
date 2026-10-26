"""Follow list: a small JSON file of people to watch (config/follow_list.json)."""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config" / "follow_list.json"


def load_follow_list(path: Path | str = DEFAULT_PATH) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    data = json.loads(p.read_text())
    entries = data.get("follow", []) if isinstance(data, dict) else data
    return [e for e in entries if isinstance(e, dict) and (e.get("match") or e.get("id"))]


def save_follow_list(entries: list[dict], path: Path | str = DEFAULT_PATH) -> None:
    clean = []
    for e in entries:
        item = {k: str(e[k]).strip() for k in ("label", "match", "id") if e.get(k)}
        if item.get("match") or item.get("id"):
            clean.append(item)
    p = Path(path)
    existing = json.loads(p.read_text()) if p.exists() else {}
    comment = existing.get("_comment") if isinstance(existing, dict) else None
    out = {"_comment": comment, "follow": clean} if comment else {"follow": clean}
    p.write_text(json.dumps(out, indent=2) + "\n")


def resolve_followed(entries: list[dict], actors: dict[str, str]) -> set[str]:
    """Map follow entries to actor_ids. ``actors`` is {actor_id: display name}."""
    ids: set[str] = set()
    for e in entries:
        if e.get("id") and e["id"] in actors:
            ids.add(e["id"])
        m = (e.get("match") or "").lower()
        if m:
            ids.update(a for a, name in actors.items() if m in name.lower())
    return ids
