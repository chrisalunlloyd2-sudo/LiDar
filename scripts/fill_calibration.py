#!/usr/bin/env python3
"""fill_calibration — write copied ping rows into CALIBRATION.md's device table.

The flash card copies Markdown rows shaped like:

    | 1.00 | 57.1 | 57.1 | 1.004 | +0.38 | ok | locked k 2026-11-08 |

This tool validates them and rewrites the table under "### Device results",
leaving the rest of the document byte-for-byte intact.

```bash
python3 scripts/fill_calibration.py --rows rows.md            # preview
python3 scripts/fill_calibration.py --rows rows.md --write    # apply
```
"""

import argparse
import os
import sys

STATUSES = {"ok", "marginal", "saturated", "dark"}
HEADER = ("| target range | centre DN | k = S0·d0² | cue range | error % "
          "| status | notes |")
SEPARATOR = "|---|---|---|---|---|---|---|"


class RowError(ValueError):
    pass


def _number(cell, line, what):
    text = cell.strip().lower().replace("dn", "").replace("m", "").strip()
    try:
        return float(text)
    except ValueError:
        raise RowError("line %d: %s is not a number: %r" % (line, what, cell))


def _optional(cell, line, what):
    text = cell.strip()
    if text in ("", "—", "-", "n/a"):
        return None
    try:
        return float(text.replace("+", "").replace("%", ""))
    except ValueError:
        raise RowError("line %d: %s is not a number: %r" % (line, what, cell))


def parse_rows(text):
    rows = []
    for number, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if len(cells) < 6:
            raise RowError("line %d: expected at least 6 columns, got %d"
                           % (number, len(cells)))
        if set(cells[0]) <= set("-: "):
            continue                                   # separator
        if cells[0].lower().startswith("target"):
            continue                                   # header
        status = cells[5].lower()
        if status not in STATUSES:
            raise RowError("line %d: unknown status %r (expected one of %s)"
                           % (number, cells[5], ", ".join(sorted(STATUSES))))
        rows.append({
            "range": _number(cells[0], number, "range"),
            "dn": _number(cells[1], number, "centre DN"),
            "k": _number(cells[2], number, "k"),
            "cue": _optional(cells[3], number, "cue range"),
            "error": _optional(cells[4], number, "error"),
            "status": status,
            "note": cells[6] if len(cells) > 6 else "",
        })
    if not rows:
        raise RowError("no table rows found in the input")
    rows.sort(key=lambda r: r["range"])
    return rows


def render(rows):
    lines = [HEADER, SEPARATOR]
    for r in rows:
        lines.append("| %.2f m | %.1f | %.1f | %s | %s | %s | %s |" % (
            r["range"], r["dn"], r["k"],
            "—" if r["cue"] is None else "%.3f" % r["cue"],
            "—" if r["error"] is None else "%+.2f" % r["error"],
            r["status"], r["note"]))
    return lines


def replace_table(text, rows):
    lines = text.splitlines()
    try:
        heading = lines.index("### Device results")
    except ValueError:
        raise RowError("no '### Device results' section in the document")
    start = None
    for i in range(heading + 1, len(lines)):
        if lines[i].startswith("|"):
            start = i
            break
        if lines[i].startswith("#"):
            break
    if start is None:
        raise RowError("no table under '### Device results'")
    end = start
    while end < len(lines) and lines[end].startswith("|"):
        end += 1
    out = lines[:start] + render(rows) + lines[end:]
    return "\n".join(out) + ("\n" if text.endswith("\n") else "")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", required=True, help="file holding the copied rows")
    parser.add_argument("--doc", default="CALIBRATION.md")
    parser.add_argument("--write", action="store_true",
                        help="rewrite --doc (default: print the result)")
    args = parser.parse_args(argv)

    try:
        with open(args.rows, "r", encoding="utf-8") as fh:
            rows = parse_rows(fh.read())
        with open(args.doc, "r", encoding="utf-8") as fh:
            text = fh.read()
        updated = replace_table(text, rows)
    except (OSError, RowError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2

    if args.write:
        with open(args.doc, "w", encoding="utf-8") as fh:
            fh.write(updated)
        print("wrote %d row(s) to %s" % (len(rows), args.doc))
    else:
        print(updated)
        print("(preview only — pass --write to update %s)" % args.doc,
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
