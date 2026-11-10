"""Fallback runner for environments without pytest: calls every test_* function.

Usage (from the repository root):  python tests/run_without_pytest.py
"""
import importlib
import inspect
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

failed = passed = 0
for fname in sorted(os.listdir(HERE)):
    if not (fname.startswith("test_") and fname.endswith(".py")):
        continue
    mod = importlib.import_module(fname[:-3])
    for name, fn in inspect.getmembers(mod, inspect.isfunction):
        if not name.startswith("test_"):
            continue
        try:
            fn()
            passed += 1
            print(f"PASS {fname}::{name}")
        except Exception:
            failed += 1
            print(f"FAIL {fname}::{name}")
            traceback.print_exc()
print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
