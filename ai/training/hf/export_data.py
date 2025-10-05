#!/usr/bin/env python3
"""Export training data from SQLite into Hugging Face friendly artifacts."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv

from ..utils import normalize_text

load_dotenv()

DEFAULT_DB_PATH = os.getenv("DB_PATH", "total.db")
CLASSIFICATION_FILENAME = "classification_data.csv"
NER_FILENAME = "ner_data.jsonl"

# Mapping that collapses binary + direction labels into a unified 4-class schema.
CLASS_LABEL_MAPPING: Dict[Tuple[int, Optional[str]], str] = {
    (0, None): "NON_SIGNAL",
    (1, "LONG"): "SIGNAL_LONG",
    (1, "SHORT"): "SIGNAL_SHORT",
    (1, "NONE"): "SIGNAL_NONE",
    # Handle numeric direction values
    (1, "1"): "SIGNAL_LONG",  # numeric 1 -> LONG
    (1, "0"): "SIGNAL_SHORT",  # numeric 0 -> SHORT
}


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _iterate_labeled_rows(conn: sqlite3.Connection) -> Iterable[sqlite3.Row]:
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT
            id,
            message,
            is_signal,
            direction,
            pair_start, pair_end,
            leverage_start, leverage_end,
            entry_start, entry_end,
            stop_loss_start, stop_loss_end,
            target_1_start, target_1_end,
            target_2_start, target_2_end,
            target_3_start, target_3_end,
            target_4_start, target_4_end,
            target_5_start, target_5_end
        FROM labeled
        ORDER BY id
        """
    )
    yield from cursor.fetchall()


def _build_class_label(is_signal: int, direction: Optional[str]) -> str:
    # Convert direction to string if it's provided
    direction_str = str(direction) if direction is not None else None
    key = (int(is_signal), direction_str)
    if key not in CLASS_LABEL_MAPPING:
        raise ValueError(f"Unexpected label combination: {key}")
    return CLASS_LABEL_MAPPING[key]


def _maybe_add_entity(
    entities: List[Dict[str, Any]],
    label: str,
    start: Optional[Any],
    end: Optional[Any],
) -> None:
    if start is None or end is None:
        return
    start_int = int(start)
    end_int = int(end)
    if end_int <= start_int:
        return
    entities.append({"start": start_int, "end": end_int, "label": label})


def _extract_entities(row: sqlite3.Row) -> List[Dict[str, Any]]:
    entities: List[Dict[str, Any]] = []

    _maybe_add_entity(entities, "PAIR", row["pair_start"], row["pair_end"])
    _maybe_add_entity(entities, "LEVERAGE", row["leverage_start"], row["leverage_end"])
    _maybe_add_entity(entities, "ENTRY", row["entry_start"], row["entry_end"])
    _maybe_add_entity(
        entities, "STOP_LOSS", row["stop_loss_start"], row["stop_loss_end"]
    )

    for idx in range(1, 6):
        start_col = row[f"target_{idx}_start"]
        end_col = row[f"target_{idx}_end"]
        _maybe_add_entity(entities, "TARGET", start_col, end_col)

    return entities


def export_data(db_path: str, output_dir: Path) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    conn = _connect(db_path)
    rows = list(_iterate_labeled_rows(conn))
    conn.close()

    classification_rows: List[Tuple[str, str]] = []
    ner_rows: List[Dict[str, Any]] = []

    for row in rows:
        normalized_text = normalize_text(row["message"], collapse_digit_spaces=True)
        if not normalized_text:
            continue

        label = _build_class_label(row["is_signal"], row["direction"])
        classification_rows.append((normalized_text, label))

        if row["is_signal"] == 1:
            entities = _extract_entities(row)
            if entities:
                ner_rows.append({"text": normalized_text, "entities": entities})

    class_path = output_dir / CLASSIFICATION_FILENAME
    ner_path = output_dir / NER_FILENAME

    if classification_rows:
        df = pd.DataFrame(classification_rows, columns=["text", "label"])
        df.to_csv(class_path, index=False)
    else:
        class_path.touch()

    if ner_rows:
        with ner_path.open("w", encoding="utf-8") as file:
            for record in ner_rows:
                json.dump(record, file, ensure_ascii=False)
                file.write("\n")
    else:
        ner_path.touch()

    return class_path, ner_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        default=DEFAULT_DB_PATH,
        help="Path to the SQLite database (defaults to DB_PATH env or total.db)",
    )
    parser.add_argument(
        "--out",
        default="data_exports",
        help="Directory where the exported files will be stored",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.out)
    class_path, ner_path = export_data(args.db, output_dir)
    print(f"[classification] wrote data to {class_path}")
    print(f"[ner] wrote data to {ner_path}")


if __name__ == "__main__":
    main()
