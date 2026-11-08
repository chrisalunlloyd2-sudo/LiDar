#!/data/data/com.termux/files/usr/bin/bash
# Build the Termux install package: .deb + tarball + SHA256SUMS in dist/.
# Pure Termux tooling (dpkg-deb, tar, sha256sum); no network needed.
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"

VERSION="$(python3 -c 'import re;print(re.search(r"VERSION = \"([^\"]+)\"", open("server.py").read()).group(1))')"
PKG="lidar"
DIST="$HERE/dist"
STAGE="$DIST/stage"
PREFIX_TREE="$STAGE/data/data/com.termux/files/usr"
SHARE="$PREFIX_TREE/share/$PKG"
DOC="$PREFIX_TREE/share/doc/$PKG"
TAR_DIR="$DIST/tar/$PKG-$VERSION"

rm -rf "$STAGE" "$DIST/tar" "$DIST/$PKG"_*.deb "$DIST/$PKG-"*.tar.gz "$DIST/SHA256SUMS"
mkdir -p "$PREFIX_TREE/bin" "$SHARE" "$DOC" "$TAR_DIR"

echo "== staging $PKG $VERSION =="
# every tracked python module except tests, keeping the arena/ layout
git ls-files '*.py' | grep -vE '(^|/)tests?/' | while read -r f; do
    mkdir -p "$SHARE/$(dirname "$f")"
    cp "$f" "$SHARE/$f"
done
cp -r gui_pages "$SHARE/gui_pages"
cp README.md CALIBRATION.md CHANGELOG.md CONTRIBUTING.md LICENSE "$DOC/"

install -m 755 packaging/termux/lidar "$PREFIX_TREE/bin/lidar"

# tarball mirrors the share/ payload, for environments without dpkg
cp -r "$SHARE/." "$TAR_DIR/"
tar -czf "$DIST/$PKG-$VERSION.tar.gz" -C "$DIST/tar" "$PKG-$VERSION"

SIZE_KB="$(du -sk "$STAGE" | cut -f1)"
mkdir -p "$STAGE/DEBIAN"
cat > "$STAGE/DEBIAN/control" <<EOF
Package: $PKG
Version: $VERSION
Architecture: all
Maintainer: Chris <chrisalunlloyd2-sudo@users.noreply.github.com>
Installed-Size: $SIZE_KB
Depends: python
Section: science
Priority: optional
Homepage: https://github.com/chrisalunlloyd2-sudo/LiDar
Description: Emulative LiDAR and spatial inventory from a phone camera
 Monocular depth cues with confidence intervals, illusion correction,
 geotagged inventory and torch-pulse ranging. Pure Python stdlib; serves an
 mmap GUI and a JSON API on port 8790.
EOF

echo "== building .deb =="
# the shell umask is 077 here; dpkg-deb insists on 0755 control dirs
find "$STAGE" -type d -exec chmod 755 {} +
find "$STAGE" -type f -exec chmod 644 {} +
chmod 755 "$PREFIX_TREE/bin/lidar"
dpkg-deb --build --root-owner-group "$STAGE" "$DIST/${PKG}_${VERSION}_all.deb" >/dev/null

cd "$DIST"
sha256sum "${PKG}_${VERSION}_all.deb" "$PKG-$VERSION.tar.gz" > SHA256SUMS
chmod 644 SHA256SUMS "${PKG}_${VERSION}_all.deb" "$PKG-$VERSION.tar.gz"
echo
ls -lh "${PKG}_${VERSION}_all.deb" "$PKG-$VERSION.tar.gz" SHA256SUMS
cat SHA256SUMS
