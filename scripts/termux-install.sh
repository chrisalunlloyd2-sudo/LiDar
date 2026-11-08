#!/data/data/com.termux/files/usr/bin/bash
# One-line Termux installer for LiDar.
#
#   curl -fsSL https://raw.githubusercontent.com/chrisalunlloyd2-sudo/LiDar/main/scripts/termux-install.sh | bash
#   bash scripts/termux-install.sh dist/lidar_2.5.1_all.deb     # local package
#
# Downloads the latest release .deb, checks it against the published
# SHA256SUMS, installs it with pkg, and verifies the launcher.
set -euo pipefail

REPO="chrisalunlloyd2-sudo/LiDar"
PREFIX="${PREFIX:-/data/data/com.termux/files/usr}"
LOCAL_DEB="${1:-}"

say() { printf '  %s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

echo "== LiDar installer =="
command -v python3 >/dev/null || die "python3 missing — run: pkg install -y python"
command -v curl >/dev/null || die "curl missing — run: pkg install -y curl"

TMP="$(mktemp -d "${TMPDIR:-${PREFIX}/tmp}/lidar.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT

if [ -n "$LOCAL_DEB" ]; then
    [ -f "$LOCAL_DEB" ] || die "no such file: $LOCAL_DEB"
    DEB="$LOCAL_DEB"
    say "using local package $DEB"
else
    command -v sha256sum >/dev/null || die "sha256sum missing — pkg install -y coreutils"
    say "querying latest release…"
    JSON="$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest")" \
        || die "cannot reach GitHub (offline?)"
    DEB_URL="$(printf '%s' "$JSON" | python3 -c 'import json,sys;d=json.load(sys.stdin);print(next(a["browser_download_url"] for a in d["assets"] if a["name"].endswith(".deb")))')" \
        || die "release has no .deb asset"
    SUM_URL="$(printf '%s' "$JSON" | python3 -c 'import json,sys;d=json.load(sys.stdin);print(next(a["browser_download_url"] for a in d["assets"] if a["name"]=="SHA256SUMS"))')" \
        || die "release has no SHA256SUMS asset"

    say "downloading $(basename "$DEB_URL")"
    curl -fsSL -o "$TMP/lidar.deb" "$DEB_URL" || die "download failed"
    curl -fsSL -o "$TMP/SHA256SUMS" "$SUM_URL" || die "checksum download failed"

    ( cd "$TMP" && sha256sum -c SHA256SUMS --ignore-missing ) \
        || die "checksum mismatch — refusing to install"
    DEB="$TMP/lidar.deb"
fi

say "installing $(basename "$DEB")"
if command -v pkg >/dev/null; then
    pkg install -y "$DEB"
else
    dpkg -i "$DEB"
fi

if command -v lidar >/dev/null; then
    say "installed: $(lidar --version 2>&1)"
else
    say "package installed; launcher not on PATH (check $PREFIX/bin)"
fi
cat <<'EOF'

Run it:
  lidar                 # GUI + API on http://127.0.0.1:8790
  lidar --memory        # throwaway database
  python3 $PREFIX/share/lidar/flash_lidar.py --demo
  python3 $PREFIX/share/lidar/scripts/flash_calibrate.py --sweep

Uninstall:
  pkg uninstall lidar
EOF
