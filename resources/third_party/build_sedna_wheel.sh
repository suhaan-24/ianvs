#!/usr/bin/env sh
# Rebuild sedna-0.6.0.3 from the vendored sedna-0.6.0.1 wheel and the patches
# in patches/, applied in order.
#
#   build_sedna_wheel.sh OUTPUT_DIR   write sedna-0.6.0.3-py3-none-any.whl
#   build_sedna_wheel.sh --check      compare the rebuilt package with the
#                                     vendored wheel, file by file
#
# Needs: python with the "wheel" package, and patch.
set -eu

HERE=$(cd "$(dirname "$0")" && pwd)
PYTHON=${PYTHON:-python}
BASE=0.6.0.1
VERSION=0.6.0.3
PATCHES="sedna-0.6.0.2.patch sedna-0.6.0.3.patch"

if [ "$#" -ne 1 ]; then
    echo "usage: $0 OUTPUT_DIR | --check" >&2
    exit 2
fi

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

"$PYTHON" -m wheel unpack -d "$WORK/src" "$HERE/sedna-$BASE-py3-none-any.whl" >/dev/null
SRC="$WORK/src/sedna-$BASE"
# The 0.6.0.1 wheel ships some compiled __pycache__ files; leave them out.
find "$SRC" -name __pycache__ -type d -prune -exec rm -rf {} +
for name in $PATCHES; do
    patch -s -d "$SRC" -p1 < "$HERE/patches/$name"
done
mv "$SRC/sedna-$BASE.dist-info" "$SRC/sedna-$VERSION.dist-info"
sed "s/^Version: $BASE\$/Version: $VERSION/" "$SRC/sedna-$VERSION.dist-info/METADATA" \
    > "$WORK/METADATA"
mv "$WORK/METADATA" "$SRC/sedna-$VERSION.dist-info/METADATA"
mv "$SRC" "$WORK/src/sedna-$VERSION"
SRC="$WORK/src/sedna-$VERSION"

if [ "$1" != "--check" ]; then
    "$PYTHON" -m wheel pack -d "$1" "$SRC"
    exit 0
fi

"$PYTHON" -m wheel unpack -d "$WORK/vendored" "$HERE/sedna-$VERSION-py3-none-any.whl" >/dev/null
if diff -r -x RECORD "$SRC" "$WORK/vendored/sedna-$VERSION"; then
    echo "sedna-$VERSION-py3-none-any.whl matches sedna-$BASE + $PATCHES"
else
    echo "sedna-$VERSION-py3-none-any.whl does not match its patches" >&2
    exit 1
fi
