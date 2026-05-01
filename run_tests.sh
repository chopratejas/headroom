#!/bin/bash
# Test runner script for parity audit

cd "C:/git/headroom/.worktrees/rust-rewrite" || exit 1

echo "===== ENVIRONMENT CHECK ====="
echo "Current Branch:"
git branch --show-current
echo ""
echo "Cargo version:"
cargo --version
echo ""
echo "Python version:"
python --version
echo ""
echo "Python executable:"
which python
echo ""

echo "===== PYTHON VENV SETUP ====="
if [ ! -f ".venv-parity/bin/activate" ]; then
    echo "Creating virtual environment..."
    python -m venv .venv-parity
fi
source .venv-parity/bin/activate || source .venv-parity/Scripts/activate
echo "Virtual environment activated"
which python
echo ""

echo "===== INSTALLING DEPENDENCIES ====="
echo "Installing headroom with dev extras..."
uv pip install -e ".[dev]" 2>&1 | head -50
echo ""
echo "Verifying headroom._core import..."
python -c "import headroom._core; print('OK: headroom._core imported successfully')" 2>&1
echo ""

echo "===== RUST TEST SUITE: WORKSPACE BROAD ====="
echo "Running: cargo test --workspace"
cargo test --workspace 2>&1
rust_ws_exit=$?
echo "Exit code: $rust_ws_exit"
echo ""

if [ $rust_ws_exit -ne 0 ]; then
    echo "RUST WORKSPACE FAILED - Attempting targeted tests"
    echo ""
    
    echo "===== RUST TEST SUITE: headroom-proxy tests ====="
    echo "Running: cargo test -p headroom-proxy --tests"
    cargo test -p headroom-proxy --tests 2>&1
    proxy_exit=$?
    echo "Exit code: $proxy_exit"
    echo ""
    
    echo "===== RUST TEST SUITE: headroom-runtime ====="
    echo "Running: cargo test -p headroom-runtime"
    cargo test -p headroom-runtime 2>&1
    runtime_exit=$?
    echo "Exit code: $runtime_exit"
    echo ""
    
    echo "===== RUST TEST SUITE: headroom-core ====="
    echo "Running: cargo test -p headroom-core --tests"
    cargo test -p headroom-core --tests 2>&1
    core_exit=$?
    echo "Exit code: $core_exit"
    echo ""
    
    echo "===== RUST TEST SUITE: headroom-parity ====="
    echo "Running: cargo test -p headroom-parity"
    cargo test -p headroom-parity 2>&1
    parity_exit=$?
    echo "Exit code: $parity_exit"
    echo ""
else
    echo "RUST WORKSPACE PASSED"
    echo ""
fi

echo "===== PYTHON TEST SUITE ====="
echo "Running: pytest -v --tb=short tests scripts/tests"
pytest -v --tb=short tests scripts/tests 2>&1
pytest_exit=$?
echo "Exit code: $pytest_exit"
echo ""

echo "===== CHECKING INSTALLED EXTRAS ====="
python -m pip show headroom-ai | grep -i "requires\|extras"
echo ""

echo "===== SUMMARY ====="
echo "Rust workspace exit code: $rust_ws_exit"
echo "pytest exit code: $pytest_exit"
