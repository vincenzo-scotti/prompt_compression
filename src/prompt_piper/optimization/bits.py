"""Bit-vector layout and lossless (de)serialisation of compression strategies.

A strategy is a binary vector laid out as ``[instruction ops | shot ops | data
ops]`` (section-aware mode), where each block lists the operators of the
problem's operator pool in execution order.  The pool differs between the
search problem (e.g. O1–O22) and the full-operator problem used by some
baselines (O1–O28), so bit vectors are only meaningful together with their
layout.  The layout is therefore stored in every result and log.

Serialised form: operator ids with a section suffix, e.g. ``"O3@instruction"``.
"""

from __future__ import annotations

from typing import Sequence

SECTIONS = ("instruction", "shots", "data")
_PREFIX = {"instruction": "I", "shots": "S", "data": "D", "all": ""}


def make_layout(
    instruction_ops: Sequence,
    shot_ops: Sequence,
    data_ops: Sequence,
) -> list[dict]:
    """Return one entry per bit: ``{index, section, op_id, op_name}``.

    *\\*_ops* may be transformation classes or instances (both expose ``id``
    and ``name``).
    """
    layout: list[dict] = []
    for section, ops in zip(SECTIONS, (instruction_ops, shot_ops, data_ops)):
        for op in ops:
            layout.append({
                "index": len(layout),
                "section": section,
                "op_id": op.id,
                "op_name": op.name,
            })
    return layout


def make_flat_layout(ops: Sequence) -> list[dict]:
    """Layout for legacy flat mode: every bit applies to the whole prompt."""
    return [
        {"index": i, "section": "all", "op_id": op.id, "op_name": op.name}
        for i, op in enumerate(ops)
    ]


def _check(layout: list[dict], bits: Sequence) -> None:
    if len(bits) != len(layout):
        raise ValueError(f"bit vector has {len(bits)} bits, layout has {len(layout)}")


def bits_to_ids(layout: list[dict], bits: Sequence) -> list[str]:
    """``[0, 1, …]`` → ``["O2@instruction", …]`` (active bits only, in order)."""
    _check(layout, bits)
    return [
        f"{e['op_id']}@{e['section']}" if e["section"] != "all" else e["op_id"]
        for e, b in zip(layout, bits) if int(b)
    ]


def bits_to_names(layout: list[dict], bits: Sequence) -> list[str]:
    """``[0, 1, …]`` → ``["I:TrimSpaces", …]`` (same format as ``names_from_bits``)."""
    _check(layout, bits)
    out = []
    for e, b in zip(layout, bits):
        if int(b):
            prefix = _PREFIX[e["section"]]
            out.append(f"{prefix}:{e['op_name']}" if prefix else e["op_name"])
    return out


def ids_to_bits(layout: list[dict], ids: Sequence[str]) -> list[int]:
    """Inverse of :func:`bits_to_ids`; raises on unknown or duplicate ids."""
    position = {
        (f"{e['op_id']}@{e['section']}" if e["section"] != "all" else e["op_id"]): e["index"]
        for e in layout
    }
    bits = [0] * len(layout)
    for op in ids:
        if op not in position:
            raise ValueError(f"operator {op!r} is not in the layout")
        if bits[position[op]]:
            raise ValueError(f"operator {op!r} appears twice")
        bits[position[op]] = 1
    return bits
