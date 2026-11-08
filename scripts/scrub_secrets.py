#!/usr/bin/env python3
"""scrub_secrets — three-pass secret/PII scan for a git repo (pure stdlib).

Passes
------
  tree      every tracked file in the working tree
  history   every blob reachable from any ref (deduped by object id)
  metadata  commit messages, author/committer identities, ref names

Nothing is printed in full: matches are redacted to a short prefix so the scan
report itself is safe to paste into an issue.

Usage
-----
  python3 scripts/scrub_secrets.py            # all three passes
  python3 scripts/scrub_secrets.py --tree
  python3 scripts/scrub_secrets.py --history --json report.json
  python3 scripts/scrub_secrets.py --max-bytes 2000000

Exit code 1 when anything is found, 0 when clean.
"""

import argparse
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (rule, severity, regex). Severity drives the exit summary; everything found
# is reported regardless.
RULES = [
    ("github_pat", "critical", re.compile(r"github_pat_[A-Za-z0-9_]{20,}")),
    ("github_token", "critical", re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}")),
    ("aws_access_key", "critical", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("private_key", "critical",
     re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----")),
    ("slack_token", "critical", re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}")),
    ("google_api_key", "critical", re.compile(r"AIza[0-9A-Za-z_-]{35}")),
    ("bearer_token", "high",
     re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/-]{20,}")),
    ("assigned_secret", "high", re.compile(
        r"(?i)\b(?:password|passwd|secret|token|api[_-]?key|access[_-]?key|"
        r"client[_-]?secret)\b\s*[:=]\s*[\"']([^\"'\s]{8,})[\"']")),
    ("git_credentials", "high", re.compile(r"https://[^/\s:@]+:[^/\s@]+@github\.com")),
    ("email", "medium",
     re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("phone", "medium", re.compile(r"(?<![\d.])(?:\+?\d[\d ()-]{8,}\d)(?![\d.])")),
    ("home_path", "low", re.compile(r"/data/data/com\.termux/files/home/[\w./-]+")),
]

# Values that are obviously not secrets: placeholders, doc examples, test data.
ALLOWLIST = re.compile(
    r"(?i)(example|placeholder|your[_-]?token|xxx+|<[^>]+>|\{|\}|"
    r"noreply@|@localhost|users\.noreply\.github\.com|chrisalunlloyd2-sudo"
    r"|0\.0\.0\.0|127\.0\.0\.1|localhost)"
)

SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", "node_modules", ".venv", "venv"}
TEXT_LIMIT = 90  # print at most this many characters of a redacted match


def redact(value):
    value = value.strip()
    if len(value) <= 8:
        return value[:2] + "*" * max(0, len(value) - 2)
    return value[:6] + "…" + ("(%d chars)" % len(value))


def scan_text(text, where, findings, line_prefix=""):
    for rule, severity, rx in RULES:
        for match in rx.finditer(text):
            hit = match.group(0)
            if ALLOWLIST.search(hit):
                continue
            line_no = text.count("\n", 0, match.start()) + 1
            findings.append({
                "pass": line_prefix or "tree",
                "where": where,
                "rule": rule,
                "severity": severity,
                "line": line_no,
                "match": redact(hit),
            })


def git(*args, cwd=HERE):
    return subprocess.run(["git"] + list(args), cwd=cwd, check=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout


def walk_tree(findings):
    for root, dirs, files in os.walk(HERE):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in files:
            path = os.path.join(root, name)
            rel = os.path.relpath(path, HERE)
            try:
                if os.path.getsize(path) > 2_000_000:
                    continue
                with open(path, "r", encoding="utf-8", errors="replace") as fh:
                    scan_text(fh.read(), rel, findings, "tree")
            except OSError:
                continue


def scan_history(findings, max_bytes, revs=None):
    revs = revs or ["--all"]
    listing = git("rev-list", "--objects", *revs).decode("utf-8", "replace")
    seen = set()
    for line in listing.splitlines():
        parts = line.split(" ", 1)
        sha = parts[0]
        path = parts[1] if len(parts) > 1 else ""
        if sha in seen:
            continue
        seen.add(sha)
        try:
            kind = git("cat-file", "-t", sha).decode().strip()
            if kind != "blob":
                continue
            if int(git("cat-file", "-s", sha).decode().strip()) > max_bytes:
                continue
            blob = git("cat-file", "-p", sha)
        except subprocess.CalledProcessError:
            continue
        if b"\x00" in blob[:4096]:
            continue
        text = blob.decode("utf-8", "replace")
        label = "%s:%s" % (sha[:10], path or "(unnamed)")
        scan_text(text, label, findings, "history")


def scan_metadata(findings):
    log = git("log", "--all", "--format=%H%x00%an%x00%ae%x00%cn%x00%ce%x00%B%x00%x01")
    for record in log.decode("utf-8", "replace").split("\x01"):
        record = record.strip("\n\x00 ")
        if not record:
            continue
        fields = record.split("\x00")
        if len(fields) < 6:
            continue
        sha, an, ae, cn, ce, body = fields[:6]
        scan_text(body, "commit %s message" % sha[:10], findings, "metadata")
        for who in (an, ae, cn, ce):
            if who and not ALLOWLIST.search(who) and RULES[9][2].search(who):
                findings.append({
                    "pass": "metadata",
                    "where": "commit %s identity" % sha[:10],
                    "rule": "email",
                    "severity": "medium",
                    "line": 0,
                    "match": redact(who),
                })
    refs = git("for-each-ref", "--format=%(refname)").decode("utf-8", "replace")
    scan_text(refs, "refs", findings, "metadata")


def main(argv=None):
    parser = argparse.ArgumentParser(description="three-pass secret/PII scan")
    parser.add_argument("--tree", action="store_true")
    parser.add_argument("--history", action="store_true")
    parser.add_argument("--metadata", action="store_true")
    parser.add_argument("--json", default=None, help="write findings as JSON")
    parser.add_argument("--rev", action="append", default=None,
                        help="limit history to this rev (repeatable, default --all)")
    parser.add_argument("--max-bytes", type=int, default=1_000_000)
    args = parser.parse_args(argv)

    everything = not (args.tree or args.history or args.metadata)
    findings = []
    if everything or args.tree:
        walk_tree(findings)
    if everything or args.history:
        scan_history(findings, args.max_bytes, args.rev)
    if everything or args.metadata:
        scan_metadata(findings)

    order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    findings.sort(key=lambda f: (order[f["severity"]], f["pass"], f["where"]))

    for f in findings:
        print("[%s] %-14s %-16s %s:%s  %s"
              % (f["severity"].upper(), f["rule"], f["pass"], f["where"],
                 f["line"], f["match"]))
    counts = {}
    for f in findings:
        counts[f["severity"]] = counts.get(f["severity"], 0) + 1
    print("\n%d finding(s): %s" % (
        len(findings),
        ", ".join("%s=%d" % (k, counts[k]) for k in
                  ("critical", "high", "medium", "low") if k in counts) or "none"))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"findings": findings, "counts": counts}, fh, indent=2)
        print("wrote", args.json)

    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
