#!/bin/bash
# Local validation script for contributors to run before submitting PR

set -e

echo "🔍 Running pre-submission validation..."

# 1. Check Python version
echo "✓ Checking Python version..."
python3 --version | grep -E "3\.[7-9]|3\.10" || {
    echo "❌ Python 3.7+ required"
    exit 1
}

# 2. Install dependencies
echo "✓ Installing dependencies..."
pip install -q -r requirements.txt

# 3. Run linting (if pylint is installed)
if command -v pylint &> /dev/null; then
    echo "✓ Running linting..."
    pylint core/ examples/ || echo "⚠️  Linting warnings found (non-blocking)"
else
    echo "⚠️  Pylint not installed, skipping linting"
fi

# 4. Run quick smoke test
echo "✓ Running smoke test..."
python3 -c "import ianvs; print('Ianvs imported successfully')"

echo "✅ Pre-submission checks passed!"
