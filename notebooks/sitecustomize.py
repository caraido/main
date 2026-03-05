import sys
from pathlib import Path

# Ensure notebook sessions can import sibling packages in ../models and ../utils.
project_root = Path(__file__).resolve().parent.parent
project_root_str = str(project_root)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)
