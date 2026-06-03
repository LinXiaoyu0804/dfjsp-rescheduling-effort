from __future__ import annotations

from pathlib import Path
from typing import Any

from .parser_fjsp import parse_fjsp_file
from .parser_jsp import parse_jsp_file
from .schema import ProblemInstance


def parse_instance(path: str | Path, family: str = "auto", due_date_factor: float = 1.5, **_: Any) -> ProblemInstance:
    path = Path(path)
    if family == "auto":
        suffix = path.suffix.lower()
        if suffix in {".fjs", ".fjsp"}:
            family = "fjsp"
        elif suffix == ".jsp":
            family = "jsp"
        else:
            raise ValueError(f"Cannot infer family from suffix '{suffix}'.")
    if family == "fjsp":
        return parse_fjsp_file(path=path, due_date_factor=due_date_factor)
    if family == "jsp":
        return parse_jsp_file(path=path, due_date_factor=due_date_factor)
    raise ValueError(f"Unsupported family: {family}")
