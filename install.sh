#!/bin/sh
# MVM installer (phase 14.6).
#
#   curl -fsSL https://raw.githubusercontent.com/howadX/Machine-Vision-Module/master/install.sh | sh
#
# Downloads the latest `mvm` release binary for this machine, verifies its
# SHA-256, and installs it into ${MVM_INSTALL_DIR:-~/.local/bin}. Linux only
# for now (x86_64 / aarch64); macOS and Windows arrive with their camera
# backends (roadmap "later", seam `CameraSource`).
#
# While the repository is private, anonymous download URLs return 404 — the
# installer then falls back to an authenticated `gh release download`.
set -eu

REPO="howadX/Machine-Vision-Module"
INSTALL_DIR="${MVM_INSTALL_DIR:-$HOME/.local/bin}"

fail() { echo "install.sh: $*" >&2; exit 1; }

os=$(uname -s)
[ "$os" = "Linux" ] || fail "unsupported OS '$os' (Linux only for now)"

case "$(uname -m)" in
    x86_64)          target="x86_64-unknown-linux-gnu" ;;
    aarch64 | arm64) target="aarch64-unknown-linux-gnu" ;;
    *)               fail "unsupported architecture '$(uname -m)' (x86_64/aarch64)" ;;
esac

command -v curl >/dev/null 2>&1 || fail "curl is required"
command -v sha256sum >/dev/null 2>&1 || fail "sha256sum is required"

asset="mvm-$target.tar.gz"
base="https://github.com/$REPO/releases/latest/download"

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

echo "▶ downloading $asset (latest release)…"
if curl -fsSL "$base/$asset" -o "$tmp/$asset" \
    && curl -fsSL "$base/$asset.sha256" -o "$tmp/$asset.sha256"; then
    :
elif command -v gh >/dev/null 2>&1; then
    echo "  anonymous download failed (private repo?) — retrying via gh"
    gh release download -R "$REPO" --pattern "$asset*" --dir "$tmp" \
        || fail "gh release download failed"
else
    fail "download failed (private repo needs an authenticated \`gh\`)"
fi

echo "▶ verifying SHA-256…"
(cd "$tmp" && sha256sum -c "$asset.sha256" >/dev/null) || fail "checksum mismatch — aborting"

tar -xzf "$tmp/$asset" -C "$tmp"
mkdir -p "$INSTALL_DIR"
install -m 755 "$tmp/mvm" "$INSTALL_DIR/mvm"

echo "✅ installed: $INSTALL_DIR/mvm ($("$INSTALL_DIR/mvm" --version 2>/dev/null || echo 'mvm'))"

case ":$PATH:" in
    *":$INSTALL_DIR:"*) ;;
    *) echo "⚠️  $INSTALL_DIR is not on your PATH — add:  export PATH=\"$INSTALL_DIR:\$PATH\"" ;;
esac

cat <<'EOF'

Next steps:
  mvm setup     # fetch the ONNX models (~110 MB, sha256-verified, resumable)
  mvm doctor    # camera / models / data-dir health check
  mvm run       # live vision → ~/.local/share/mvm
EOF
