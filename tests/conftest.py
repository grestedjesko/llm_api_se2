from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

_running_integration_tests = False
if len(sys.argv) > 1:
    args_str = " ".join(sys.argv)
    if ("-m" in sys.argv and "integration" in args_str) or "test_integration.py" in args_str:
        _running_integration_tests = True

if not _running_integration_tests:
    mock_torch = MagicMock()
    mock_torch.backends.mps.is_available.return_value = False
    mock_torch.cuda.is_available.return_value = False

    sys.modules.setdefault("torch", mock_torch)
    sys.modules.setdefault("transformers", MagicMock())
    sys.modules.setdefault("transformers.AutoModelForCausalLM", MagicMock())
    sys.modules.setdefault("transformers.AutoTokenizer", MagicMock())

