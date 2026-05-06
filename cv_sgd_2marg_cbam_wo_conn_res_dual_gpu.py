"""Dual-GPU launcher for cv_sgd_2marg_cbam_wo_conn_res.py.

Áp dụng DataParallel tại runtime bằng cách patch source in-memory.
File gốc KHÔNG bị thay đổi.

Fix OOM so với phiên bản trước:
  - autocast dùng 'cuda' literal + dtype=float16 → propagate đúng sang cả 2 replicas
  - distillation_loss dùng _infer_device() thay vì biến global `device` (cuda:0)
  - output_device=1 → gather về GPU 1, giảm tải GPU 0 (nơi teacher chạy)
  - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True → giảm OOM do fragmentation
  - Giữ nguyên: unwrap_model, state_dict patches từ phiên bản trước

Cách dùng:
    python cv_sgd_2marg_cbam_wo_conn_res_dual_gpu.py

Tắt dual GPU:
    USE_DUAL_GPU=0 python cv_sgd_2marg_cbam_wo_conn_res_dual_gpu.py
"""

from __future__ import annotations

import os
import sys
import torch
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
TARGET_SCRIPT = SCRIPT_DIR / "cv_sgd_2marg_cbam_wo_conn_res.py"


def _safe_replace(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count == 0:
        raise RuntimeError(
            f"\n[Patch THẤT BẠI] '{label}'\n"
            f"  Không tìm thấy target string. File gốc có thể đã thay đổi.\n"
            f"  Target (80 ký tự đầu): {old[:80]!r}"
        )
    if count > 1:
        raise RuntimeError(
            f"\n[Patch THẤT BẠI] '{label}'\n"
            f"  Tìm thấy {count} lần (cần đúng 1).\n"
            f"  Target (80 ký tự đầu): {old[:80]!r}"
        )
    return source.replace(old, new, 1)


def transform_source(source: str) -> str:

    # PATCH 1: Inject unwrap_model() và _infer_device()
    source = _safe_replace(
        source,
        "\n\ndef generate_fundus_mask_batch",
        "\n\n"
        "def unwrap_model(m):\n"
        "    \"\"\"Trả về module gốc, bỏ qua DataParallel wrapper nếu có.\"\"\"\n"
        "    return m.module if isinstance(m, torch.nn.DataParallel) else m\n"
        "\n\n"
        "def _infer_device(*tensors):\n"
        "    \"\"\"Suy device từ tensor đầu tiên — an toàn với DataParallel replicas.\"\"\"\n"
        "    for t in tensors:\n"
        "        if isinstance(t, torch.Tensor):\n"
        "            return t.device\n"
        "        if isinstance(t, dict):\n"
        "            for v in t.values():\n"
        "                if isinstance(v, torch.Tensor):\n"
        "                    return v.device\n"
        "    return torch.device('cuda')\n"
        "\n\n"
        "def generate_fundus_mask_batch",
        label="inject unwrap_model() and _infer_device()",
    )

    # PATCH 2: DataParallel wrap sau .to(device), output_device=1 để balance tải
    source = _safe_replace(
        source,
        "    ).to(device)\n    \n    # Loss criterion",
        "    ).to(device)\n"
        "\n"
        "    # --- Dual-GPU: DataParallel wrapper ---\n"
        "    _use_dual_gpu = os.environ.get('USE_DUAL_GPU', '1') == '1'\n"
        "    if _use_dual_gpu and torch.cuda.is_available() and torch.cuda.device_count() >= 2:\n"
        "        torch.cuda.empty_cache()\n"
        "        model = torch.nn.DataParallel(model, device_ids=[0, 1], output_device=1)\n"
        "        print(f'  \u2705 Dual-GPU enabled: {torch.cuda.device_count()} GPUs | gather \u2192 GPU 1')\n"
        "    elif _use_dual_gpu:\n"
        "        print(f'  \u26a0\ufe0f  Dual-GPU y\u00eau c\u1ea7u nh\u01b0ng ch\u1ec9 c\u00f3 {torch.cuda.device_count()} GPU(s), b\u1ecf qua.')\n"
        "    # --- end Dual-GPU ---\n"
        "\n"
        "    # Loss criterion",
        label="DataParallel wrap after .to(device)",
    )

    # PATCH 3a: Fix autocast trong training loop
    source = _safe_replace(
        source,
        "            with autocast(device_type=device.type, enabled=config['training']['use_amp']):\n"
        "                outputs = model(inputs)\n"
        "                ce_loss = criterion(outputs, labels)",
        "            with autocast(device_type='cuda', dtype=torch.float16, enabled=config['training']['use_amp']):\n"
        "                outputs = model(inputs)\n"
        "                ce_loss = criterion(outputs, labels)",
        label="fix autocast training loop",
    )

    # PATCH 3b: Fix autocast trong validation loop
    source = _safe_replace(
        source,
        "                with autocast(device_type=device.type, enabled=config['training']['use_amp']):\n",
        "                with autocast(device_type='cuda', dtype=torch.float16, enabled=config['training']['use_amp']):\n",
        label="fix autocast validation loop",
    )

    # PATCH 4: distillation_loss dùng _infer_device thay vì biến global device
    source = _safe_replace(
        source,
        "                    distill_loss = distillation_loss_encoder_decoder_fusion(\n"
        "                        student_features_dict, teacher_features,\n"
        "                        student_connectors, teacher_connectors,\n"
        "                        teacher_cbam_modules, student_cbam_modules,\n"
        "                        layer_mapping, device, fundus_masks,",
        "                    distill_loss = distillation_loss_encoder_decoder_fusion(\n"
        "                        student_features_dict, teacher_features,\n"
        "                        student_connectors, teacher_connectors,\n"
        "                        teacher_cbam_modules, student_cbam_modules,\n"
        "                        layer_mapping, _infer_device(student_features_dict), fundus_masks,",
        label="distillation_loss: _infer_device instead of global device",
    )

    # PATCH 5: unwrap get_embedding
    source = _safe_replace(
        source,
        "                    embeddings = model.get_embedding(inputs)",
        "                    embeddings = unwrap_model(model).get_embedding(inputs)",
        label="unwrap model.get_embedding()",
    )

    # PATCH 6: unwrap forward_features
    source = _safe_replace(
        source,
        "                    student_features_dict = model.forward_features(inputs, return_dict=True)",
        "                    student_features_dict = unwrap_model(model).forward_features(inputs, return_dict=True)",
        label="unwrap model.forward_features()",
    )

    # PATCH 7: unwrap extract_features
    source = _safe_replace(
        source,
        "                            _, teacher_features = teacher_model.extract_features(inputs)",
        "                            _, teacher_features = unwrap_model(teacher_model).extract_features(inputs)",
        label="unwrap teacher_model.extract_features()",
    )

    # PATCH 8: state_dict best.pt
    source = _safe_replace(
        source,
        "            torch.save({\n"
        "                'epoch': epoch + 1,\n"
        "                'model_state_dict': model.state_dict(),\n"
        "                'optimizer_state_dict': optimizer.state_dict(),\n"
        "                'val_f1': val_f1,\n"
        "                'val_auc': val_auc\n"
        "            }, ckpt_dir / 'best.pt')",
        "            torch.save({\n"
        "                'epoch': epoch + 1,\n"
        "                'model_state_dict': unwrap_model(model).state_dict(),\n"
        "                'optimizer_state_dict': optimizer.state_dict(),\n"
        "                'val_f1': val_f1,\n"
        "                'val_auc': val_auc\n"
        "            }, ckpt_dir / 'best.pt')",
        label="unwrap state_dict → best.pt",
    )

    # PATCH 9: state_dict last.pt
    source = _safe_replace(
        source,
        "    torch.save({\n"
        "        'epoch': config['training']['epochs'],\n"
        "        'model_state_dict': model.state_dict(),\n"
        "        'optimizer_state_dict': optimizer.state_dict(),\n"
        "        'val_f1': val_f1,\n"
        "        'val_auc': val_auc\n"
        "    }, ckpt_dir / 'last.pt')",
        "    torch.save({\n"
        "        'epoch': config['training']['epochs'],\n"
        "        'model_state_dict': unwrap_model(model).state_dict(),\n"
        "        'optimizer_state_dict': optimizer.state_dict(),\n"
        "        'val_f1': val_f1,\n"
        "        'val_auc': val_auc\n"
        "    }, ckpt_dir / 'last.pt')",
        label="unwrap state_dict → last.pt",
    )

    return source


def _validate_patched_source(source: str) -> None:
    checks = {
        "unwrap_model defined":              "def unwrap_model(m):" in source,
        "_infer_device defined":             "def _infer_device(" in source,
        "DataParallel with output_device=1": "output_device=1" in source,
        "autocast train loop fixed":         "autocast(device_type='cuda', dtype=torch.float16" in source,
        "get_embedding unwrapped":           "unwrap_model(model).get_embedding" in source,
        "forward_features unwrapped":        "unwrap_model(model).forward_features" in source,
        "extract_features unwrapped":        "unwrap_model(teacher_model).extract_features" in source,
        "distill uses _infer_device":        "_infer_device(student_features_dict)" in source,
        "best.pt unwrapped":                 source.count("unwrap_model(model).state_dict()") >= 1,
        "last.pt unwrapped":                 source.count("unwrap_model(model).state_dict()") >= 2,
        "no bare model.state_dict()":        "model.state_dict()" not in source,
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise RuntimeError(
            "\n[Validation THẤT BẠI] Patches chưa được áp dụng:\n"
            + "\n".join(f"  ✗ {n}" for n in failed)
        )
    print("  ✅ Tất cả patches xác nhận thành công:")
    for name in checks:
        print(f"     ✓ {name}")


def main() -> None:
    if not TARGET_SCRIPT.exists():
        sys.exit(
            f"[Lỗi] Không tìm thấy:\n  {TARGET_SCRIPT}\n"
            f"Đảm bảo file gốc nằm cùng thư mục."
        )

    # Fix OOM fragmentation — gợi ý từ PyTorch OOM error message
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("USE_DUAL_GPU", "1")
    os.chdir(SCRIPT_DIR)

    use_dual_gpu = os.environ.get("USE_DUAL_GPU", "1") == "1"

    print("=" * 60)
    print("Dual-GPU Launcher")
    print("=" * 60)
    print(f"  File gốc   : {TARGET_SCRIPT.name}")
    print(f"  Dual GPU   : {'BẬT' if use_dual_gpu else 'TẮT'}")
    print(f"  ALLOC_CONF : {os.environ['PYTORCH_CUDA_ALLOC_CONF']}")

    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        for i in range(n_gpu):
            p = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}      : {p.name} ({p.total_memory/1024**3:.1f} GB)")
        if use_dual_gpu and n_gpu >= 2:
            print("  Phân bổ    : student → GPU 0+1 | teacher → GPU 0 | gather → GPU 1")
    print("=" * 60)

    print("\n[Bước 1] Áp dụng patches...")
    source = TARGET_SCRIPT.read_text(encoding="utf-8")
    patched = transform_source(source)

    print("\n[Bước 2] Xác nhận patches...")
    _validate_patched_source(patched)

    print("\n[Bước 3] Khởi động training...\n")
    print("=" * 60)

    code = compile(patched, str(TARGET_SCRIPT), "exec")
    exec(code, {  # noqa: S102
        "__name__": "__main__",
        "__file__": str(TARGET_SCRIPT),
        "__package__": None,
        "__cached__": None,
        "torch": torch,
    })


if __name__ == "__main__":
    main()