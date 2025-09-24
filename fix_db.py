from random import choice
import sqlite3
import re
import os


def format_number_with_commas(num: float) -> str:
    """Format a number with commas as thousands separators."""
    return f"{num:,}"


def normalize_text(text: str) -> str:
    """Normalize text by replacing common Cyrillic lookalikes with Latin equivalents."""
    if text is None:
        return None

    # Common Cyrillic -> Latin character mappings
    cyrillic_to_latin = {
        "а": "a",  # Cyrillic а -> Latin a
        "о": "o",  # Cyrillic о -> Latin o
        "р": "p",  # Cyrillic р -> Latin p
        "с": "c",  # Cyrillic с -> Latin c
        "е": "e",  # Cyrillic е -> Latin e
        "х": "x",  # Cyrillic х -> Latin x
        "у": "y",  # Cyrillic у -> Latin y
        "В": "B",  # Cyrillic В -> Latin B
        "Н": "H",  # Cyrillic Н -> Latin H
        "К": "K",  # Cyrillic К -> Latin K
        "М": "M",  # Cyrillic М -> Latin M
        "Р": "P",  # Cyrillic Р -> Latin P
        "С": "C",  # Cyrillic С -> Latin C
        "Т": "T",  # Cyrillic Т -> Latin T
        "Х": "X",  # Cyrillic Х -> Latin X
    }

    normalized = text
    for cyrillic, latin in cyrillic_to_latin.items():
        normalized = normalized.replace(cyrillic, latin)

    normalized = normalized.replace(",", ".")

    return normalized


DB_PATH = "./total.db"
if not os.path.exists(DB_PATH):
    raise FileNotFoundError(f"Database file {DB_PATH} not found.")

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

all_rows = cursor.execute(
    "SELECT * FROM labeled WHERE is_signal = 1 ORDER BY labeled_at"
).fetchall()


direction_lose = 0
direction_win = 0
for row in all_rows:
    # id, message_id, channel_id, message, is_signal, labeled_at, direction, pair, stop_loss, take_profit(unused), leverage, targets, entry

    id: int = int(row[0])
    message_id: int = int(row[1])
    channel_id: int = int(row[2])
    message: str = row[3].lower()
    is_signal: int = int(row[4])
    labeled_at: str = row[5]
    direction: str | None = row[6].lower() if row[6] is not None else None
    pair: str | None = row[7].lower() if row[7] is not None else None
    stop_loss: float | None = row[8]
    leverage: float | None = row[10]
    targets_raw = row[11]
    entry: float | None = row[12]

    # Parse targets from string representation if needed
    targets: list[float] | None = None
    if targets_raw is not None:
        try:
            if isinstance(targets_raw, str):
                # Handle string representation of list
                import ast

                targets = ast.literal_eval(targets_raw)
            else:
                targets = targets_raw

            # Ensure all targets are floats
            if targets and isinstance(targets, list):
                targets = [float(t) for t in targets]
        except (ValueError, SyntaxError, TypeError) as e:
            print(f"Warning: Could not parse targets '{targets_raw}' for row {id}: {e}")
            targets = None

    # Normalize message text to handle Cyrillic lookalikes
    normalized_message = normalize_text(message)
    if direction is None:
        continue

    if pair is not None:
        # find pair in message with multiple pattern attempts
        pair_patterns = [
            re.escape(pair),  # exact match
            re.escape(pair.replace("/", "")),  # without slash: storj/usdt -> storjusdt
            re.escape(pair.replace("/", " ")),  # with space: storj/usdt -> storj usdt
            re.escape(pair.split("/")[0])
            if "/" in pair
            else re.escape(pair),  # just coin name
        ]

        pair_match = None
        for pattern in pair_patterns:
            pair_match = re.search(pattern, normalized_message, re.IGNORECASE)
            if pair_match:
                break

        if pair_match is None:
            print(f"PAIR MISMATCH: Row ID {id}: Pair '{pair}' not found in message.")
            print(f"Searched patterns: {pair_patterns}")
            print(f"Normalized message: {normalized_message}")
            choice = input(
                "1. Change pair in DB, 2. Make is_signal=0, 3. Make pair=None, any other key to skip: "
            )
            if choice == "1":
                new_pair = input("Enter new pair value: ")
                cursor.execute(
                    "UPDATE labeled SET pair = ? WHERE id = ?",
                    (new_pair, id),
                )
                conn.commit()
                print(f"Updated pair to {new_pair} for Row ID {id}.")
            elif choice == "2":
                cursor.execute("UPDATE labeled SET is_signal = 0 WHERE id = ?", (id,))
                conn.commit()
                print(f"Set is_signal=0 for Row ID {id}.")
            elif choice == "3":
                cursor.execute("UPDATE labeled SET pair = NULL WHERE id = ?", (id,))
                conn.commit()
                print(f"Set pair=NULL for Row ID {id}.")
            continue

        # get start pos and end pos of pair in message
        pair_start, pair_end = pair_match.span()
        print(
            f"Row ID {id}: Pair '{pair}' found at positions {pair_start}-{pair_end} in message."
        )
        # add to db pair_start and pair_end
        # also add such columns to db if not exist
        try:
            cursor.execute("ALTER TABLE labeled ADD COLUMN pair_start INTEGER")
        except sqlite3.OperationalError:
            pass  # Column already exists

        try:
            cursor.execute("ALTER TABLE labeled ADD COLUMN pair_end INTEGER")
        except sqlite3.OperationalError:
            pass  # Column already exists

        cursor.execute(
            "UPDATE labeled SET pair_start = ?, pair_end = ? WHERE id = ?",
            (pair_start, pair_end, id),
        )
        conn.commit()

    if stop_loss is not None:
        sl_str = f"{stop_loss:.10f}".rstrip("0").rstrip(
            "."
        )  # Format to avoid scientific notation for smaller numbers
        sl_int_str = str(int(stop_loss)) if stop_loss == int(stop_loss) else None

        # ['1\\.6', '1\\.6', '1\\.6\\.0', '1\\.6\\.00', '1\\.6\\.000', '1\\.6\\.0000', '1\\.6\\.00000', '1\\.6\\.000000', '1\\.6\\.0000000', '1\\.6\\.00000000', '1\\.6\\.000000000', '2e\\+00', '1\\.6e\\+00', '1\\.60e\\+00', '1\\.600e\\+00', '1\\.6000e\\+00', '1\\.60000e\\+00', '1\\.600000e\\+00']
        patterns = [
            re.escape(sl_str),
            re.escape(f"{stop_loss:.1e}"),
            re.escape(f"{stop_loss:.2e}"),
            re.escape(f"{stop_loss:.3e}"),
            re.escape(f"{stop_loss:.4e}"),
            re.escape(f"{stop_loss:.5e}"),
            re.escape(f"{stop_loss:.6e}"),
            re.escape(f"{stop_loss:.7e}"),
            re.escape(f"{stop_loss:.8e}"),
            re.escape(f"{stop_loss:.9e}"),
            re.escape(f"{stop_loss:.10e}"),
            re.escape(f"{stop_loss}0"),
            re.escape(f"{stop_loss}00"),
            re.escape(f"{stop_loss}000"),
            re.escape(f"{stop_loss}0000"),
            re.escape(f"{stop_loss}00000"),
            re.escape(f"{stop_loss}000000"),
            re.escape(f"{stop_loss}0000000"),
            re.escape(f"{stop_loss}00000000"),
            re.escape(f"{stop_loss}000000000"),
        ]

        # Add the original formatted stop_loss value as it appears in calculations
        if stop_loss == int(stop_loss):
            # For integer floats like 122.0, also add the .0 version
            patterns.append(re.escape(f"{int(stop_loss)}.0"))

        # Add pattern for grouped number with dots
        dot_sl = f"{stop_loss:,.1f}".replace(",", ".")
        patterns.append(re.escape(dot_sl))

        # Add patterns for malformed numbers with extra dots/zeros (e.g., "0.4.6541" for 4.6541)
        patterns.append(re.escape(f"0.{sl_str}"))
        patterns.append(re.escape(f"0{sl_str}"))
        if "." in sl_str:
            # Handle cases like "0.4.6541" where there's an extra "0." prefix
            parts = sl_str.split(".")
            if len(parts) >= 2:
                malformed = f"0.{parts[0]}.{'.'.join(parts[1:])}"
                patterns.append(re.escape(malformed))

        # Add pattern for the number with trailing zeros, as it might appear in the message
        # For a stop_loss of 1.6, this will add patterns for 1.6, 1.60, 1.600, etc.
        # For integer floats like 43.0, add 43.0, 43.00, etc.
        if "." in sl_str:
            base_sl_str = sl_str.rstrip("0").rstrip(".")
            patterns.append(re.escape(base_sl_str))
            for i in range(1, 10):
                pattern_to_add = base_sl_str + "0" * i
                # check if it exceeds 10 decimal places
                if len(pattern_to_add.split(".")[-1]) > 10:
                    break
                patterns.append(re.escape(pattern_to_add))
        else:
            # For integer floats, add decimal versions
            base_sl_str = sl_str
            patterns.append(re.escape(base_sl_str))
            for i in range(1, 10):
                pattern_to_add = base_sl_str + "." + "0" * i
                patterns.append(re.escape(pattern_to_add))

        # Add pattern for grouped number with spaces (like "16 650" or "101 735.5")
        if stop_loss >= 1000:
            # For numbers >= 1000, format with thousands separator and preserve original decimal places
            # Use the same precision as sl_str to maintain original decimal places
            decimal_places = len(sl_str.split(".")[-1]) if "." in sl_str else 0
            space_sl = f"{stop_loss:,.{decimal_places}f}".replace(",", " ")
            # Remove trailing .0 if it's an integer
            if space_sl.endswith(" .0"):
                space_sl = space_sl[:-3]
            elif space_sl.endswith(".0"):
                space_sl = space_sl[:-2]
            patterns.append(re.escape(space_sl))
        else:
            space_sl = f"{stop_loss:,.1f}".replace(",", " ")
            patterns.append(re.escape(space_sl))

        # For space-separated format, also try with decimal versions
        if " " in space_sl:
            # Add patterns like "16 650.0", "16 650.00", etc.
            for i in range(1, 6):
                pattern_to_add = space_sl + "." + "0" * i
                patterns.append(re.escape(pattern_to_add))

            # Also try replacing space with dot (like "16.650")
            dot_version = space_sl.replace(" ", ".")
            patterns.append(re.escape(dot_version))
            for i in range(1, 6):
                pattern_to_add = dot_version + "0" * i
                patterns.append(re.escape(pattern_to_add))

        print(patterns)

        # Build a robust regex to find the stop loss value
        # It should not be preceded or followed by digits or a decimal point.
        # Allow punctuation after the number (period, comma, etc.)
        # Modified to handle range formats like "86.2-84.8"
        pattern_group = "|".join(patterns)
        sl_pattern = re.compile(rf"(?<![\d\w\.])({pattern_group})(?![0-9])")

        sl_match = sl_pattern.search(normalized_message)

        if sl_match is None:
            direction_lose += 1
            os.system("clear")
            print(f"Could not find stop_loss: {stop_loss}")
            print(f"Searched with patterns: {patterns}")
            print("Normalized message:")
            print(normalized_message)
            choice = input(
                "1. Change stop_loss in DB, 2. Make is_signal=0, 3. Make stop_loss=None, any other key to skip: "
            )
            if choice == "1":
                new_sl = input("Enter new stop_loss value: ")
                try:
                    new_sl_value = float(new_sl)
                    cursor.execute(
                        "UPDATE labeled SET stop_loss = ? WHERE id = ?",
                        (new_sl_value, id),
                    )
                    conn.commit()
                    print(f"Updated stop_loss to {new_sl_value} for Row ID {id}.")
                except ValueError:
                    print("Invalid float value. Skipping update.")
            elif choice == "2":
                cursor.execute("UPDATE labeled SET is_signal = 0 WHERE id = ?", (id,))
                conn.commit()
                print(f"Set is_signal=0 for Row ID {id}.")
            elif choice == "3":
                cursor.execute(
                    "UPDATE labeled SET stop_loss = NULL WHERE id = ?", (id,)
                )
                conn.commit()
                print(f"Set stop_loss=NULL for Row ID {id}.")
            continue

        else:
            # create and update stop_loss_start and stop_loss_end in db
            sl_start, sl_end = sl_match.span()

            # Check if columns exist before adding them
            try:
                cursor.execute("ALTER TABLE labeled ADD COLUMN stop_loss_start INTEGER")
            except sqlite3.OperationalError:
                pass  # Column already exists

            try:
                cursor.execute("ALTER TABLE labeled ADD COLUMN stop_loss_end INTEGER")
            except sqlite3.OperationalError:
                pass  # Column already exists

            cursor.execute(
                "UPDATE labeled SET stop_loss_start = ?, stop_loss_end = ? WHERE id = ?",
                (sl_start, sl_end, id),
            )
            conn.commit()
            direction_win += 1
            print(f"Found stop_loss: {stop_loss} at positions {sl_start}-{sl_end}")

    if leverage is not None:
        # Convert leverage to float first in case it's stored as string
        try:
            leverage = float(leverage)
        except (ValueError, TypeError):
            print(
                f"Warning: Invalid leverage value '{leverage}' for row {id}, skipping..."
            )
            continue

        lev_str = f"{leverage:.10f}".rstrip("0").rstrip(
            "."
        )  # Format to avoid scientific notation for smaller numbers
        lev_int_str = str(int(leverage)) if leverage == int(leverage) else None

        patterns = [
            re.escape(lev_str),
            re.escape(f"{leverage:.1e}"),
            re.escape(f"{leverage:.2e}"),
            re.escape(f"{leverage:.3e}"),
            re.escape(f"{leverage:.4e}"),
            re.escape(f"{leverage:.5e}"),
            re.escape(f"{leverage:.6e}"),
            re.escape(f"{leverage:.7e}"),
            re.escape(f"{leverage:.8e}"),
            re.escape(f"{leverage:.9e}"),
            re.escape(f"{leverage:.10e}"),
            re.escape(f"{leverage}0"),
            re.escape(f"{leverage}00"),
            re.escape(f"{leverage}000"),
            re.escape(f"{leverage}0000"),
            re.escape(f"{leverage}00000"),
            re.escape(f"{leverage}000000"),
            re.escape(f"{leverage}0000000"),
            re.escape(f"{leverage}00000000"),
            re.escape(f"{leverage}000000000"),
        ]

        if leverage == int(leverage):
            patterns.append(re.escape(f"{int(leverage)}.0"))

        # Add patterns with "x" prefix (e.g., "x25", "x20")
        patterns.append(re.escape(f"x{lev_str}"))
        if leverage == int(leverage):
            patterns.append(re.escape(f"x{int(leverage)}.0"))

        # Add patterns for leverage within words (e.g., "cross25x", "isolated25x")
        patterns.append(re.escape(f"{lev_str}x"))
        if leverage == int(leverage):
            patterns.append(re.escape(f"{int(leverage)}x"))

        if "." in lev_str:
            base_lev_str = lev_str.rstrip("0").rstrip(".")
            patterns.append(re.escape(base_lev_str))
            patterns.append(re.escape(f"x{base_lev_str}"))
            patterns.append(re.escape(f"{base_lev_str}x"))  # Add suffix pattern
            for i in range(1, 10):
                pattern_to_add = base_lev_str + "0" * i
                if len(pattern_to_add.split(".")[-1]) > 10:
                    break
                patterns.append(re.escape(pattern_to_add))
                patterns.append(re.escape(f"x{pattern_to_add}"))
                patterns.append(re.escape(f"{pattern_to_add}x"))  # Add suffix pattern
        else:
            base_lev_str = lev_str
            patterns.append(re.escape(base_lev_str))
            patterns.append(re.escape(f"x{base_lev_str}"))
            patterns.append(re.escape(f"{base_lev_str}x"))  # Add suffix pattern
            for i in range(1, 10):
                pattern_to_add = base_lev_str + "." + "0" * i
                patterns.append(re.escape(pattern_to_add))
                patterns.append(re.escape(f"x{pattern_to_add}"))
                patterns.append(re.escape(f"{pattern_to_add}x"))  # Add suffix pattern

        print(patterns)

        pattern_group = "|".join(patterns)
        # Fixed regex to allow leverage within words but prevent false digit matches
        # Only prevent matching when preceded/followed by digits
        lev_pattern = re.compile(rf"(?<!\d)({pattern_group})(?!\d)")
        lev_match = lev_pattern.search(normalized_message)
        # lev_match = re.search(re.escape(lev_str), normalized_message)
        if lev_match:
            print(f"Found leverage: {lev_match.group(0)}")
            try:
                cursor.execute("ALTER TABLE labeled ADD COLUMN leverage_start INTEGER")
            except sqlite3.OperationalError:
                pass  # Column already exists

            try:
                cursor.execute("ALTER TABLE labeled ADD COLUMN leverage_end INTEGER")
            except sqlite3.OperationalError:
                pass  # Column already exists

            cursor.execute(
                "UPDATE labeled SET leverage_start = ?, leverage_end = ? WHERE id = ?",
                (lev_match.start(), lev_match.end(), id),
            )
            conn.commit()
        else:
            os.system("clear")
            print(f"Could not find leverage: {leverage}")
            print(f"Searched with patterns: {patterns}")
            print("Normalized message:")
            print(normalized_message)
            choice = input(
                "1. Change leverage in DB, 2. Make is_signal=0, 3. Make leverage=None, any other key to skip: "
            )
            if choice == "1":
                new_lev = input("Enter new leverage value: ")
                try:
                    new_lev_value = float(new_lev)
                    cursor.execute(
                        "UPDATE labeled SET leverage = ? WHERE id = ?",
                        (new_lev_value, id),
                    )
                    conn.commit()
                    print(f"Updated leverage to {new_lev_value} for Row ID {id}.")
                except ValueError:
                    print("Invalid float value. Skipping update.")
            elif choice == "2":
                cursor.execute("UPDATE labeled SET is_signal = 0 WHERE id = ?", (id,))
                conn.commit()
                print(f"Set is_signal=0 for Row ID {id}.")
            elif choice == "3":
                cursor.execute("UPDATE labeled SET leverage = NULL WHERE id = ?", (id,))
                conn.commit()
                print(f"Set leverage=NULL for Row ID {id}.")
            continue

    if targets is not None and len(targets) > 0:
        missing_targets = []
        found_targets = []

        for idx, target in enumerate(targets):
            try:
                target = float(target)
            except (ValueError, TypeError):
                print(
                    f"Warning: Invalid target value '{target}' for row {id}, skipping..."
                )
                continue

            target_str = f"{target:.10f}".rstrip("0").rstrip(".")

            patterns = [
                re.escape(target_str),
                re.escape(f"{target:.1e}"),
                re.escape(f"{target:.2e}"),
                re.escape(f"{target:.3e}"),
                re.escape(f"{target:.4e}"),
                re.escape(f"{target:.5e}"),
                re.escape(f"{target:.6e}"),
                re.escape(f"{target:.7e}"),
                re.escape(f"{target:.8e}"),
                re.escape(f"{target:.9e}"),
                re.escape(f"{target:.10e}"),
            ]

            # Add the original formatted target value as it appears in calculations
            if target == int(target):
                patterns.append(re.escape(f"{int(target)}.0"))

            # Add pattern for grouped number with dots
            dot_target = f"{target:,.1f}".replace(",", ".")
            patterns.append(re.escape(dot_target))

            # Add patterns for malformed numbers with extra dots/zeros
            patterns.append(re.escape(f"0.{target_str}"))
            patterns.append(re.escape(f"0{target_str}"))
            if "." in target_str:
                parts = target_str.split(".")
                if len(parts) >= 2:
                    malformed = f"0.{parts[0]}.{'.'.join(parts[1:])}"
                    patterns.append(re.escape(malformed))

            # Add pattern for the number with trailing zeros
            if "." in target_str:
                base_target_str = target_str.rstrip("0").rstrip(".")
                patterns.append(re.escape(base_target_str))
                for i in range(1, 10):
                    pattern_to_add = base_target_str + "0" * i
                    if len(pattern_to_add.split(".")[-1]) > 10:
                        break
                    patterns.append(re.escape(pattern_to_add))
            else:
                base_target_str = target_str
                patterns.append(re.escape(base_target_str))
                for i in range(1, 10):
                    pattern_to_add = base_target_str + "." + "0" * i
                    patterns.append(re.escape(pattern_to_add))

            # Add pattern for grouped number with spaces
            if target >= 1000:
                decimal_places = (
                    len(target_str.split(".")[-1]) if "." in target_str else 0
                )
                space_target = f"{target:,.{decimal_places}f}".replace(",", " ")
                if space_target.endswith(" .0"):
                    space_target = space_target[:-3]
                elif space_target.endswith(".0"):
                    space_target = space_target[:-2]
                patterns.append(re.escape(space_target))
            else:
                space_target = f"{target:,.1f}".replace(",", " ")
                patterns.append(re.escape(space_target))

            # For space-separated format, also try with decimal versions
            if " " in space_target:
                for i in range(1, 6):
                    pattern_to_add = space_target + "." + "0" * i
                    patterns.append(re.escape(pattern_to_add))

                dot_version = space_target.replace(" ", ".")
                patterns.append(re.escape(dot_version))
                for i in range(1, 6):
                    pattern_to_add = dot_version + "0" * i
                    patterns.append(re.escape(pattern_to_add))

            # Build regex pattern
            pattern_group = "|".join(patterns)
            # Updated regex to allow matching numbers separated by spaces and dashes
            target_pattern = re.compile(rf"(?<![\d])({pattern_group})(?![\d])")

            target_match = target_pattern.search(normalized_message)

            if target_match is None:
                missing_targets.append((idx + 1, target))
            else:
                target_start, target_end = target_match.span()
                found_targets.append((idx + 1, target, target_start, target_end))

        # If any targets are missing, handle them all at once
        if missing_targets:
            print("Original message:")
            print(message)
            print(f"Could not find targets from list {targets}:")
            for target_num, target_val in missing_targets:
                print(f"  Target {target_num}: {target_val}")
            print("Normalized message:")
            print(normalized_message)
            print("1. Replace entire targets list")
            print("2. Make is_signal=0")
            print("3. Set targets=None")
            choice = input("Enter choice (any other key to skip): ")

            if choice == "1":
                print(f"Current targets: {targets}")
                new_targets_str = input(
                    "Enter new targets list (e.g., [7.0, 8.1, 9.1]): "
                )
                try:
                    import ast

                    new_targets = ast.literal_eval(new_targets_str)
                    if isinstance(new_targets, list):
                        new_targets = [float(t) for t in new_targets]
                        cursor.execute(
                            "UPDATE labeled SET targets = ? WHERE id = ?",
                            (str(new_targets), id),
                        )
                        conn.commit()
                        print(f"Updated targets to {new_targets} for Row ID {id}.")
                    else:
                        print("Invalid format. Expected a list.")
                except (ValueError, SyntaxError, TypeError):
                    print("Invalid format. Skipping update.")
            elif choice == "2":
                cursor.execute("UPDATE labeled SET is_signal = 0 WHERE id = ?", (id,))
                conn.commit()
                print(f"Set is_signal=0 for Row ID {id}.")
            elif choice == "3":
                cursor.execute("UPDATE labeled SET targets = NULL WHERE id = ?", (id,))
                conn.commit()
                print(f"Set targets=NULL for Row ID {id}.")
        else:
            # All targets found, update their positions
            for target_num, target_val, target_start, target_end in found_targets:
                # Check if columns exist before adding them
                try:
                    cursor.execute(
                        f"ALTER TABLE labeled ADD COLUMN target_{target_num}_start INTEGER"
                    )
                except sqlite3.OperationalError:
                    pass  # Column already exists

                try:
                    cursor.execute(
                        f"ALTER TABLE labeled ADD COLUMN target_{target_num}_end INTEGER"
                    )
                except sqlite3.OperationalError:
                    pass  # Column already exists

                cursor.execute(
                    f"UPDATE labeled SET target_{target_num}_start = ?, target_{target_num}_end = ? WHERE id = ?",
                    (target_start, target_end, id),
                )
                conn.commit()
                print(
                    f"Found target {target_num}: {target_val} at positions {target_start}-{target_end}"
                )

    if entry is not None:
        # Convert entry to float first in case it's stored as string
        try:
            entry = float(entry)
        except (ValueError, TypeError):
            print(f"Warning: Invalid entry value '{entry}' for row {id}, skipping...")
            continue

        entry_str = f"{entry:.10f}".rstrip("0").rstrip(
            "."
        )  # Format to avoid scientific notation for smaller numbers

        patterns = [
            re.escape(entry_str),
            re.escape(f"{entry:.1e}"),
            re.escape(f"{entry:.2e}"),
            re.escape(f"{entry:.3e}"),
            re.escape(f"{entry:.4e}"),
            re.escape(f"{entry:.5e}"),
            re.escape(f"{entry:.6e}"),
            re.escape(f"{entry:.7e}"),
            re.escape(f"{entry:.8e}"),
            re.escape(f"{entry:.9e}"),
            re.escape(f"{entry:.10e}"),
            re.escape(f"{entry}0"),
            re.escape(f"{entry}00"),
            re.escape(f"{entry}000"),
            re.escape(f"{entry}0000"),
            re.escape(f"{entry}00000"),
            re.escape(f"{entry}000000"),
            re.escape(f"{entry}0000000"),
            re.escape(f"{entry}00000000"),
            re.escape(f"{entry}000000000"),
        ]

        # Add high-precision decimal format for very small numbers
        # This handles cases like 1.3375e-08 = 0.000000013375
        for precision in [12, 15, 18, 20]:
            high_precision_str = f"{entry:.{precision}f}".rstrip("0").rstrip(".")
            if high_precision_str != entry_str and high_precision_str != "0":
                patterns.append(re.escape(high_precision_str))

        # Add the original formatted entry value as it appears in calculations
        if entry == int(entry):
            # For integer floats like 122.0, also add the .0 version
            patterns.append(re.escape(f"{int(entry)}.0"))

        # Add pattern for grouped number with dots
        dot_entry = f"{entry:,.1f}".replace(",", ".")
        patterns.append(re.escape(dot_entry))

        # Add patterns for malformed numbers with extra dots/zeros (e.g., "0.4.6541" for 4.6541)
        patterns.append(re.escape(f"0.{entry_str}"))
        patterns.append(re.escape(f"0{entry_str}"))
        if "." in entry_str:
            # Handle cases like "0.4.6541" where there's an extra "0." prefix
            parts = entry_str.split(".")
            if len(parts) >= 2:
                malformed = f"0.{parts[0]}.{'.'.join(parts[1:])}"
                patterns.append(re.escape(malformed))

        # Add pattern for the number with trailing zeros, as it might appear in the message
        # For an entry of 1.6, this will add patterns for 1.6, 1.60, 1.600, etc.
        # For integer floats like 43.0, add 43.0, 43.00, etc.
        if "." in entry_str:
            base_entry_str = entry_str.rstrip("0").rstrip(".")
            patterns.append(re.escape(base_entry_str))
            for i in range(1, 10):
                pattern_to_add = base_entry_str + "0" * i
                # check if it exceeds 10 decimal places
                if len(pattern_to_add.split(".")[-1]) > 10:
                    break
                patterns.append(re.escape(pattern_to_add))
        else:
            # For integer floats, add decimal versions
            base_entry_str = entry_str
            patterns.append(re.escape(base_entry_str))
            for i in range(1, 10):
                pattern_to_add = base_entry_str + "." + "0" * i
                patterns.append(re.escape(pattern_to_add))

        # Add pattern for grouped number with spaces (like "16 650" or "101 735.5")
        if entry >= 1000:
            # For numbers >= 1000, format with thousands separator and preserve original decimal places
            # Use the same precision as entry_str to maintain original decimal places
            decimal_places = len(entry_str.split(".")[-1]) if "." in entry_str else 0
            space_entry = f"{entry:,.{decimal_places}f}".replace(",", " ")
            # Remove trailing .0 if it's an integer
            if space_entry.endswith(" .0"):
                space_entry = space_entry[:-3]
            elif space_entry.endswith(".0"):
                space_entry = space_entry[:-2]
            patterns.append(re.escape(space_entry))
        else:
            space_entry = f"{entry:,.1f}".replace(",", " ")
            patterns.append(re.escape(space_entry))

        # For space-separated format, also try with decimal versions
        if " " in space_entry:
            # Add patterns like "16 650.0", "16 650.00", etc.
            for i in range(1, 6):
                pattern_to_add = space_entry + "." + "0" * i
                patterns.append(re.escape(pattern_to_add))

            # Also try replacing space with dot (like "16.650")
            dot_version = space_entry.replace(" ", ".")
            patterns.append(re.escape(dot_version))
            for i in range(1, 6):
                pattern_to_add = dot_version + "0" * i
                patterns.append(re.escape(pattern_to_add))

        print(patterns)

        # Build a robust regex to find the entry value
        # It should not be preceded by digits or decimal point, but allow letters (like "entry0.0551")
        # Allow punctuation after the number (period, comma, etc.)
        pattern_group = "|".join(patterns)
        entry_pattern = re.compile(rf"(?<![\d\.-])({pattern_group})(?![0-9])")

        entry_match = entry_pattern.search(normalized_message)

        if entry_match is None:
            direction_lose += 1
            os.system("clear")
            print(f"Could not find entry: {entry}")
            print(f"Searched with patterns: {patterns}")
            print("Normalized message:")
            print(normalized_message)

            choice = input(
                "1. Change entry in DB, 2. Make is_signal=0, 3. Make entry=None, any other key to skip: "
            )
            if choice == "1":
                new_entry = input("Enter new entry value: ")
                try:
                    new_entry_value = float(new_entry)
                    cursor.execute(
                        "UPDATE labeled SET entry = ? WHERE id = ?",
                        (new_entry_value, id),
                    )
                    conn.commit()
                    print(f"Updated entry to {new_entry_value} for Row ID {id}.")
                except ValueError:
                    print("Invalid float value. Skipping update.")
            elif choice == "2":
                cursor.execute("UPDATE labeled SET is_signal = 0 WHERE id = ?", (id,))
                conn.commit()
                print(f"Set is_signal=0 for Row ID {id}.")
            elif choice == "3":
                cursor.execute("UPDATE labeled SET entry = NULL WHERE id = ?", (id,))
                conn.commit()
                print(f"Set entry=NULL for Row ID {id}.")
            continue

        else:
            # create and update entry_start and entry_end in db
            entry_start, entry_end = entry_match.span()

            # Check if columns exist before adding them
            try:
                cursor.execute("ALTER TABLE labeled ADD COLUMN entry_start INTEGER")
            except sqlite3.OperationalError:
                pass  # Column already exists

            try:
                cursor.execute("ALTER TABLE labeled ADD COLUMN entry_end INTEGER")
            except sqlite3.OperationalError:
                pass  # Column already exists

            cursor.execute(
                "UPDATE labeled SET entry_start = ?, entry_end = ? WHERE id = ?",
                (entry_start, entry_end, id),
            )
            conn.commit()
            direction_win += 1
            print(f"Found entry: {entry} at positions {entry_start}-{entry_end}")

print(len(all_rows))
print(f"Total mismatched directions: {direction_lose}")
print(f"Total matched directions: {direction_win}")
conn.close()
