#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
VERSION="$($PYTHON -c 'from gpuq import __version__; print(__version__)')"
ARCH="${GPUQ_BINARY_ARCH:-linux-x86_64}"

cd "$ROOT"
$PYTHON -m PyInstaller \
  --clean \
  --noconfirm \
  --onefile \
  --name gpuq \
  --distpath dist \
  --workpath build/pyinstaller \
  --specpath build/pyinstaller-spec \
  packaging/gpuq_entry.py

./dist/gpuq --version

BUNDLE="dist/gpuq-v${VERSION}-${ARCH}"
rm -rf "$BUNDLE"
mkdir -p "$BUNDLE"
install -m 0755 dist/gpuq "$BUNDLE/gpuq"
install -m 0644 LICENSE README.md README.zh-CN.md config.example.toml "$BUNDLE/"
install -m 0644 systemd/gpuqd.service "$BUNDLE/"
tar -C dist -czf "${BUNDLE}.tar.gz" "$(basename "$BUNDLE")"
(cd dist && sha256sum "$(basename "$BUNDLE").tar.gz" > "$(basename "$BUNDLE").tar.gz.sha256")

echo "built ${BUNDLE}.tar.gz"
