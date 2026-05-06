"""Dual-GPU launcher for cv_sgd_2marg_cbam_wo_conn_res.py.

Áp dụng DataParallel tại runtime bằng cách patch source in-memory.
File gốc KHÔNG bị thay đổi.

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


# ---------------------------------------------------------------------------
# Helper: safe replace — báo lỗi rõ ràng nếu patch không tìm thấy target
# ---------------------------------------------------------------------------

def _safe_replace(source: str, old: str, new: str, label: str) -> str:
    """Replace old→new trong source, raise lỗi nếu không tìm thấy đúng 1 lần."""
    count = source.count(old)
    if count == 0:
        raise RuntimeError(
            f"\n[Patch THẤT BẠI] '{label}'\n"
            f"  Không tìm thấy target string trong file gốc.\n"
            f"  File gốc có thể đã thay đổi. Cần cập nhật patch.\n"
            f"  Target (60 ký tự đầu): {old[:60]!r}"
        )
    if count > 1:
        raise RuntimeError(
            f"\n[Patch THẤT BẠI] '{label}'\n"
            f"  Tìm thấy {count} lần (cần đúng 1).\n"
            f"  Patch có thể gây ra thay đổi ngoài ý muốn.\n"
            f"  Target (60 ký tự đầu): {old[:60]!r}"
        )
    return source.replace(old, new, 1)


# ---------------------------------------------------------------------------
# Tất cả các patches
# ---------------------------------------------------------------------------

def transform_source(source: str) -> str:
    """
    Áp dụng tối thiểu các patch cần thiết để hỗ trợ nn.DataParallel.
    Thứ tự patch quan trọng: unwrap_model phải được inject trước khi dùng.
    """

    # ------------------------------------------------------------------
    # PATCH 1: Inject hàm unwrap_model() vào trước generate_fundus_mask_batch
    # Mục đích: dùng để gọi custom methods và lưu state_dict đúng
    # ------------------------------------------------------------------
    source = _safe_replace(
        source,
        "\n\ndef generate_fundus_mask_batch",
        "\n\n"
        "def unwrap_model(m):\n"
        "    \"\"\"Trả về module gốc, bỏ qua DataParallel wrapper nếu có.\"\"\"\n"
        "    return m.module if isinstance(m, torch.nn.DataParallel) else m\n"
        "\n\n"
        "def generate_fundus_mask_batch",
        label="inject unwrap_model()",
    )

    # ------------------------------------------------------------------
    # PATCH 2: Wrap model bằng DataParallel ngay sau .to(device)
    # Target chính xác: dòng 770-772 trong file gốc
    # ------------------------------------------------------------------
    source = _safe_replace(
        source,
        "    ).to(device)\n    \n    # Loss criterion",
        "    ).to(device)\n"
        "\n"
        "    # --- Dual-GPU: DataParallel wrapper ---\n"
        "    _use_dual_gpu = os.environ.get('USE_DUAL_GPU', '1') == '1'\n"
        "    if _use_dual_gpu and torch.cuda.is_available() and torch.cuda.device_count() >= 2:\n"
        "        model = torch.nn.DataParallel(model)\n"
        "        print(f\"  ✅ Dual-GPU enabled: {torch.cuda.device_count()} GPUs\")\n"
        "    elif _use_dual_gpu:\n"
        "        print(f\"  ⚠️  Dual-GPU yêu cầu nhưng chỉ có {torch.cuda.device_count()} GPU(s), bỏ qua.\")\n"
        "    # --- end Dual-GPU ---\n"
        "\n"
        "    # Loss criterion",
        label="DataParallel wrap after .to(device)",
    )

    # ------------------------------------------------------------------
    # PATCH 3: model.get_embedding() — cần unwrap để gọi custom method
    # Dòng 1021 trong file gốc
    # ------------------------------------------------------------------
    source = _safe_replace(
        source,
        "                    embeddings = model.get_embedding(inputs)",
        "                    embeddings = unwrap_model(model).get_embedding(inputs)",
        label="unwrap model.get_embedding()",
    )

    # ------------------------------------------------------------------
    # PATCH 4: model.forward_features() — cần unwrap để gọi custom method
    # Dòng 1029 trong file gốc
    # ------------------------------------------------------------------
    source = _safe_replace(
        source,
        "                    student_features_dict = model.forward_features(inputs, return_dict=True)",
        "                    student_features_dict = unwrap_model(model).forward_features(inputs, return_dict=True)",
        label="unwrap model.forward_features()",
    )

    # ------------------------------------------------------------------
    # PATCH 5: teacher_model.extract_features() — unwrap phòng trường hợp
    # teacher được wrap về sau; hiện tại teacher không dùng DataParallel
    # nhưng unwrap_model() an toàn khi không được wrap (trả về chính nó)
    # Dòng 1034 trong file gốc
    # ------------------------------------------------------------------
    source = _safe_replace(
        source,
        "                            _, teacher_features = teacher_model.extract_features(inputs)",
        "                            _, teacher_features = unwrap_model(teacher_model).extract_features(inputs)",
        label="unwrap teacher_model.extract_features()",
    )

    # ------------------------------------------------------------------
    # PATCH 6: Lưu state_dict đúng cho best.pt
    # Dòng 1192 trong file gốc — bên trong block `if val_f1 > best_val_f1`
    # ------------------------------------------------------------------
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
        label="unwrap model.state_dict() in best.pt",
    )

    # ------------------------------------------------------------------
    # PATCH 7: Lưu state_dict đúng cho last.pt
    # Dòng 1206 trong file gốc — sau vòng lặp epoch
    # ------------------------------------------------------------------
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
        label="unwrap model.state_dict() in last.pt",
    )

    return source


# ---------------------------------------------------------------------------
# Kiểm tra batch_size hợp lệ cho DataParallel
# ---------------------------------------------------------------------------

def _check_batch_size_for_dual_gpu(source: str) -> None:
    """
    Cảnh báo nếu batch_size trong config có thể không chia hết cho 2 GPU.
    Chỉ warn, không raise — vì batch_size đến từ YAML config runtime.
    """
    # Không thể đọc YAML ở đây vì chưa chạy script, nên chỉ nhắc nhở.
    print(
        "  ℹ️  Lưu ý: DataParallel yêu cầu batch_size chia hết cho số GPU (2).\n"
        "      Kiểm tra config 'training.batch_size' trước khi training."
    )


# ---------------------------------------------------------------------------
# Validate: đảm bảo tất cả patches đã được áp dụng thành công
# ---------------------------------------------------------------------------

def _validate_patched_source(source: str) -> None:
    """Kiểm tra các dấu hiệu cho thấy patch đã thành công."""
    checks = {
        "unwrap_model defined":          "def unwrap_model(m):" in source,
        "DataParallel wrap present":     "torch.nn.DataParallel(model)" in source,
        "get_embedding unwrapped":       "unwrap_model(model).get_embedding" in source,
        "forward_features unwrapped":    "unwrap_model(model).forward_features" in source,
        "extract_features unwrapped":    "unwrap_model(teacher_model).extract_features" in source,
        "best.pt state_dict unwrapped":  source.count("unwrap_model(model).state_dict()") >= 1,
        "last.pt state_dict unwrapped":  source.count("unwrap_model(model).state_dict()") >= 2,
        "no bare model.state_dict()":    "model.state_dict()" not in source,
    }

    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise RuntimeError(
            f"\n[Validation THẤT BẠI] Các patch sau không được áp dụng:\n"
            + "\n".join(f"  ✗ {name}" for name in failed)
        )

    print("  ✅ Tất cả patches đã được xác nhận thành công:")
    for name in checks:
        print(f"     ✓ {name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Kiểm tra file gốc tồn tại
    if not TARGET_SCRIPT.exists():
        sys.exit(
            f"[Lỗi] Không tìm thấy file gốc:\n  {TARGET_SCRIPT}\n"
            f"Đảm bảo file này nằm cùng thư mục với launcher."
        )

    os.environ.setdefault("USE_DUAL_GPU", "1")
    os.chdir(SCRIPT_DIR)

    use_dual_gpu = os.environ.get("USE_DUAL_GPU", "1") == "1"

    print("=" * 60)
    print("Dual-GPU Launcher")
    print("=" * 60)
    print(f"  File gốc  : {TARGET_SCRIPT}")
    print(f"  Dual GPU  : {'BẬT' if use_dual_gpu else 'TẮT (USE_DUAL_GPU=0)'}")
    if use_dual_gpu:
        n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
        print(f"  CUDA GPUs : {n_gpu}")
        if n_gpu >= 2:
            for i in range(n_gpu):
                props = torch.cuda.get_device_properties(i)
                vram = props.total_memory / 1024**3
                print(f"    GPU {i}: {props.name} ({vram:.1f} GB VRAM)")
    print("=" * 60)

    # Đọc và patch source
    source = TARGET_SCRIPT.read_text(encoding="utf-8")

    print("\n[Bước 1] Áp dụng patches...")
    patched_source = transform_source(source)

    print("\n[Bước 2] Xác nhận patches...")
    _validate_patched_source(patched_source)

    if use_dual_gpu:
        _check_batch_size_for_dual_gpu(patched_source)

    print("\n[Bước 3] Khởi động training...\n")
    print("=" * 60)

    # Compile và chạy patched source
    globals_dict = {
        "__name__": "__main__",
        "__file__": str(TARGET_SCRIPT),
        "__package__": None,
        "__cached__": None,
        # Inject torch vào namespace để unwrap_model() trong patched source
        # có thể dùng torch.nn.DataParallel ngay khi được định nghĩa
        "torch": torch,
    }

    code = compile(patched_source, str(TARGET_SCRIPT), "exec")
    exec(code, globals_dict)  # noqa: S102


if __name__ == "__main__":
    main()