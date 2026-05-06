"""Dual-GPU launcher for cv_sgd_2marg_cbam_wo_conn_res.py.

This keeps the original training script unchanged and applies the temporary
multi-GPU behavior only at runtime by transforming the source in memory.

Usage:
    python cv_sgd_2marg_cbam_wo_conn_res_dual_gpu.py

Optional:
    set USE_DUAL_GPU=0 to fall back to single-GPU / CPU behavior.
"""

from __future__ import annotations

import os
import re
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
TARGET_SCRIPT = SCRIPT_DIR / "cv_sgd_2marg_cbam_wo_conn_res.py"


def transform_source(source: str) -> str:
    """Apply a minimal set of runtime-only patches for DataParallel safety."""
    source = source.replace(
        "\n\ndef generate_fundus_mask_batch",
        "\n\ndef unwrap_model(model):\n    \"\"\"Return the underlying module for DataParallel-wrapped models.\"\"\"\n    return model.module if isinstance(model, nn.DataParallel) else model\n\n\ndef generate_fundus_mask_batch",
        1,
    )

    source = source.replace(
        "    ).to(device)\n    \n    # Loss criterion",
        "    ).to(device)\n\n    use_dual_gpu = os.environ.get('USE_DUAL_GPU', '1') == '1'\n    if use_dual_gpu and torch.cuda.is_available() and torch.cuda.device_count() >= 2:\n        model = nn.DataParallel(model)\n        print(f\"  ✅ Dual-GPU mode enabled on {torch.cuda.device_count()} GPUs\")\n    elif use_dual_gpu:\n        print(f\"  ⚠️ Dual-GPU mode requested, but only {torch.cuda.device_count()} CUDA device(s) available\")\n\n    # Loss criterion",
        1,
    )

    source = source.replace(
        "embeddings = model.get_embedding(inputs)",
        "embeddings = unwrap_model(model).get_embedding(inputs)",
    )
    source = source.replace(
        "student_features_dict = model.forward_features(inputs, return_dict=True)",
        "student_features_dict = unwrap_model(model).forward_features(inputs, return_dict=True)",
    )
    source = source.replace(
        "_, teacher_features = teacher_model.extract_features(inputs)",
        "_, teacher_features = unwrap_model(teacher_model).extract_features(inputs)",
    )
    source = source.replace(
        "'model_state_dict': model.state_dict(),",
        "'model_state_dict': unwrap_model(model).state_dict(),",
    )

    return source


def main() -> None:
    os.environ.setdefault("USE_DUAL_GPU", "1")
    os.chdir(SCRIPT_DIR)

    source = TARGET_SCRIPT.read_text(encoding="utf-8")
    patched_source = transform_source(source)

    globals_dict = {
        "__name__": "__main__",
        "__file__": str(TARGET_SCRIPT),
        "__package__": None,
        "__cached__": None,
    }
    code = compile(patched_source, str(TARGET_SCRIPT), "exec")
    exec(code, globals_dict)


if __name__ == "__main__":
    main()
