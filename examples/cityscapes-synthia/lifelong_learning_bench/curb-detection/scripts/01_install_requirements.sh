#!/usr/bin/env sh
set -eu

if [ "$#" -ne 1 ]; then
    echo "usage: $0 REQUIREMENTS_FILE" >&2
    exit 2
fi

ROOT=$(cd "$(dirname "$0")/../../../../.." && pwd)
SEDNA_VERSION=0.6.0.3

"${PYTHON:-python}" -m pip install -r "$1"

# Lifelong learning needs the patched Sedna wheel (see
# resources/third_party/README.md). It replaces 0.6.0.1 in this environment
# only; other examples keep installing 0.6.0.1.
"${PYTHON:-python}" -m pip install "$ROOT/resources/third_party/sedna-$SEDNA_VERSION-py3-none-any.whl"
"${PYTHON:-python}" -c "
import importlib.metadata as metadata
import sys
version = metadata.version('sedna')
if version != '$SEDNA_VERSION':
    sys.exit('installed sedna is ' + version + ', expected $SEDNA_VERSION')
print('sedna ' + version + ' installed')
"
