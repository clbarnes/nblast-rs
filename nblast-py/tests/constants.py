from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent
PY_PROJECT_DIR = TEST_DIR.parent
WORKSPACE_DIR = PY_PROJECT_DIR.parent
DATA_DIR = WORKSPACE_DIR / "data"
