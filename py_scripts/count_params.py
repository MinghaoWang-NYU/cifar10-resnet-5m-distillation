"""Print the trainable-parameter count of one or more model configs.

    python count_params.py configs/main.yaml configs/teacher_wrn28_10.yaml
"""
import sys

from src.model import build_model, count_trainable_params
from src.utils import load_config

LIMIT = 5_000_000

for path in sys.argv[1:]:
    mcfg = load_config(path)["model"]
    n = count_trainable_params(build_model(mcfg))
    flag = "OK" if n <= LIMIT else "over 5M (teacher only)" if mcfg.get("param_limit") is None else "OVER LIMIT"
    print(f"{path:35s} widths={mcfg['widths']} blocks={mcfg['blocks']}  {n:>12,}  {flag}")
