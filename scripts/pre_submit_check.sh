#!/bin/bash
# Local validation script for contributors to run before submitting PR

set -e

echo "🔍 Running pre-submission validation..."

# 1. Check Python version
echo "✓ Checking Python version..."
PY_VERSION=$(python3 --version 2>&1)
echo "Found: $PY_VERSION"
# Match 3.7-3.9 OR 3.10-3.99
echo "$PY_VERSION" | grep -E "3\.([7-9]|[1-9][0-9])" || {
    echo "❌ Python 3.7+ required"
    exit 1
}

# 2. Install dependencies
echo "✓ Installing dependencies..."
pip install -q -r requirements.txt
pip install -q -e .  # Install ianvs in editable mode so imports work

# 3. Run linting (if pylint is installed)
if command -v pylint &> /dev/null; then
    echo "✓ Running linting (Focused on your changes)..."
    # Lint core and ONLY the lifelong learning example
    TARGET_EXAMPLE="examples/robot/lifelong_learning_bench/semantic-segmentation"
    pylint core/ "$TARGET_EXAMPLE" || echo "⚠️  Linting warnings found (non-blocking)"
else
    echo "⚠️  Pylint not installed, skipping linting"
fi

# 4. Run quick smoke test
echo "✓ Running smoke test..."
python3 -c "import core; print('Ianvs core imported successfully')" || {
    echo "❌ Failed to import core module"
    exit 1
}
ianvs --help > /dev/null && echo "✓ Ianvs CLI working" || echo "⚠️  Ianvs CLI not found/working"


echo "✅ Pre-submission checks passed!"
