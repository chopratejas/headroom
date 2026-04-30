@echo off
REM Test runner batch script for parity audit
setlocal enabledelayedexpansion

cd /d "C:\git\headroom\.worktrees\rust-rewrite" || exit /b 1

echo ===== ENVIRONMENT CHECK =====
echo Current Branch:
git branch --show-current
echo.
echo Cargo version:
cargo --version
echo.
echo Python version:
python --version
echo.

echo ===== PYTHON VENV SETUP =====
if not exist ".venv-parity\Scripts\activate.bat" (
    echo Creating virtual environment...
    python -m venv .venv-parity
)
call .venv-parity\Scripts\activate.bat
echo Virtual environment activated
echo.

echo ===== INSTALLING DEPENDENCIES =====
echo Installing headroom with dev extras...
uv pip install -e ".[dev]" 2>&1 | head -50
echo.
echo Verifying headroom._core import...
python -c "import headroom._core; print('OK: headroom._core imported successfully')" 2>&1
echo.

echo ===== RUST TEST SUITE: WORKSPACE BROAD =====
cargo test --workspace 2>&1
set rust_ws_exit=!errorlevel!
echo Rust workspace test exit code: !rust_ws_exit!
echo.

if !rust_ws_exit! neq 0 (
    echo RUST WORKSPACE FAILED - attempting targeted tests
    echo.
    
    echo ===== RUST TEST SUITE: headroom-proxy tests =====
    cargo test -p headroom-proxy --tests 2>&1
    set proxy_exit=!errorlevel!
    echo headroom-proxy exit code: !proxy_exit!
    echo.
    
    echo ===== RUST TEST SUITE: headroom-runtime =====
    cargo test -p headroom-runtime 2>&1
    set runtime_exit=!errorlevel!
    echo headroom-runtime exit code: !runtime_exit!
    echo.
    
    echo ===== RUST TEST SUITE: headroom-core =====
    cargo test -p headroom-core --tests 2>&1
    set core_exit=!errorlevel!
    echo headroom-core exit code: !core_exit!
    echo.
) else (
    echo RUST WORKSPACE PASSED
)
echo.

echo ===== PYTHON TEST SUITE =====
echo pytest -v --tb=short tests scripts/tests
pytest -v --tb=short tests scripts/tests 2>&1
set pytest_exit=!errorlevel!
echo pytest exit code: !pytest_exit!
echo.

echo ===== SUMMARY =====
echo Rust workspace exit code: !rust_ws_exit!
echo pytest exit code: !pytest_exit!

endlocal
