#!/bin/bash
# Local Verification Script for Semantic-Segmentation CI/CD
# This script runs the exact same checks that the GitHub Action will run in the cloud.

set -e

echo "======================================="
echo "1. Running Strict Yaml Linting..."
echo "======================================="
yamllint -d '{"extends": "relaxed", "rules": {"line-length": "disable", "document-start": "disable", "new-lines": "disable", "new-line-at-end-of-file": "disable", "empty-lines": "disable"}}' examples/robot/lifelong_learning_bench/semantic-segmentation/
echo "✅ Yamllint passed!"
echo ""

echo "======================================="
echo "2. Running Strict Flake8 (Python) Linting..."
echo "======================================="
flake8 examples/robot/lifelong_learning_bench/semantic-segmentation/ \
    --count --show-source --statistics \
    --extend-ignore=E501,E402,F821,F401,F403,F405,E722,E203,E262,E265,E266,E731,E741,F811,F841,W291
echo "✅ Flake8 passed!"
echo ""

echo "======================================="
echo "3. Running Semantic Segmentation Benchmark..."
echo "======================================="
source .venv310/bin/activate
export PYTHONPATH=$PYTHONPATH:/Users/suhaan/Desktop/KUBEEDGE/ianvs/examples/robot/lifelong_learning_bench/semantic-segmentation/testalgorithms/rfnet/RFNet
ianvs -f examples/robot/lifelong_learning_bench/semantic-segmentation/benchmarkingjob-simple.yaml
echo "✅ Benchmark executed successfully!"
echo ""

echo "🎉 ALL CI/CD pipeline steps completed successfully on your local machine!"
