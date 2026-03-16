#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
REPORTS = ROOT / "reports"

START = "<!-- WEEKLY_UPDATES_START -->"
END = "<!-- WEEKLY_UPDATES_END -->"


def main() -> None:
    today = dt.date.today().isoformat()
    report_path = REPORTS / f"{today}.md"
    report_rel = f"reports/{today}.md"

    if not report_path.exists():
        raise FileNotFoundError(f"Missing weekly report: {report_path}")

    if not README.exists():
        README.write_text(
            "# WOSAC / Autonomous Driving Simulation Research Tracker\n\n"
            "## Weekly updates\n\n"
            f"{START}\n{END}\n",
            encoding="utf-8",
        )

    text = README.read_text(encoding="utf-8")
    new_line = f"- {today}: [weekly report]({report_rel})"

    if START not in text or END not in text:
        text += f"\n## Weekly updates\n\n{START}\n{new_line}\n{END}\n"
    else:
        before, rest = text.split(START, 1)
        middle, after = rest.split(END, 1)

        existing_lines = [line.rstrip() for line in middle.strip().splitlines() if line.strip()]
        if new_line not in existing_lines:
            existing_lines.insert(0, new_line)

        existing_lines = existing_lines[:12]
        replacement = "\n" + "\n".join(existing_lines) + "\n"
        text = before + START + replacement + END + after

    README.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
