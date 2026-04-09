import os
import sys
from pathlib import Path
import pytest

# 1. SETUP PROJECT PATHS
# This ensures that when you run pytest from anywhere, 
# 'config' and 'RAG' are found.
HERE = Path(__file__).parent.resolve()          # tests/CASE_RAG/
REPO_ROOT = HERE.parent.parent.resolve()        # D:\FUCK!!\Grad\Code

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# 2. DEFINE THE DATA PATHS
# This matches where your real .txt files are stored
DATA_DIR = r"D:\FUCK!!\Grad\Backup\1st Iteration\1\Docs"

@pytest.fixture(scope="session", autouse=True)
def setup_environment():
    """
    Ensures the environment is ready before any tests run.
    Checks if the config folder exists and is treated as a package.
    """
    config_path = REPO_ROOT / "config"
    init_file = config_path / "__init__.py"
    
    # Critical Fix: Ensure __init__.py exists so 'config' is a package
    if config_path.exists() and not init_file.exists():
        init_file.touch()
        print(f"\n[BOOTSTRAP] Created missing {init_file}")

# 3. GLOBAL TEST CONSTANTS
@pytest.fixture
def case_id():
    return "2847_2024_civil_south_cairo"

@pytest.fixture
def doc_map():
    """Returns the mapping of real file paths to their legal titles."""
    return {
        str(DATA_DIR / "صحيفة_دعوى.txt"):               "صحيفة دعوى",
        str(DATA_DIR / "محضر_جلسة_25_03_2024.txt"):      "محضر جلسة",
        str(DATA_DIR / "تقرير_الخبير.txt"):              "تقرير خبير",
        str(DATA_DIR / "تقرير_الطب_الشرعي.txt"):        "تقرير الطب الشرعي",
        str(DATA_DIR / "حكم_المحكمة.txt"):               "حكم",
        str(DATA_DIR / "مذكرة_بدفاع_المدعى_عليه_الأول.txt"):    "مذكرة بدفاع",
        str(DATA_DIR / "مذكرة_بدفاع_المدعى_عليها_الثانية.txt"): "مذكرة بدفاع",
    }