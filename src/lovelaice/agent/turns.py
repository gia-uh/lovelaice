"""Turn numbers for stored history.

A turn is one ``prompt()`` call. The loop records its number on every message
it appends. Messages stored before that field existed get one inferred: a user
message opens a turn unless it is the loop's own empty-answer nudge, which
belongs to the turn it was sent in.
"""


def assign_turns(items: list[tuple[str, str, int | None]], nudge: str) -> list[int]:
    """Turn number for each (role, text content, stored turn) history item."""
    out: list[int] = []
    current = 0
    for role, content, stored in items:
        if stored is not None:
            current = stored
        elif role == "user" and content != nudge:
            current += 1
        out.append(current)
    return out
