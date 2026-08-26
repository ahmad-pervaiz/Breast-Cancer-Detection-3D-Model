"""
Optional ClearML experiment tracking - metrics + checkpoint artifacts.

Entirely gated behind cfg["clearml"]["enabled"] and wrapped defensively: if
clearml isn't installed, isn't configured (no ~/clearml.conf yet), or the
network/auth fails, training continues normally with a printed warning
instead of crashing. This keeps local/Kaggle runs working before you've
finished clearml-init.
"""
from pathlib import Path
from typing import Any, Dict, Optional


def init_clearml(cfg: dict):
    """Returns a clearml Task if enabled+available, else None."""
    clearml_cfg = cfg.get("clearml", {})
    if not clearml_cfg.get("enabled", False):
        return None

    try:
        from clearml import Task
    except ImportError:
        print("clearml.enabled=true in config but the `clearml` package isn't installed "
              "(pip install clearml) - continuing without experiment tracking.")
        return None

    try:
        task = Task.init(
            project_name=clearml_cfg.get("project_name", "Breast Tumor Segmentation"),
            task_name=clearml_cfg.get("task_name") or None,  # None -> ClearML auto-names it
            auto_connect_frameworks={"pytorch": False},  # we log checkpoints ourselves, explicitly
        )
        task.connect(cfg, name="config")
        print(f"ClearML tracking active: {task.get_output_log_web_page()}")
        return task
    except Exception as e:
        print(f"ClearML init failed ({e}) - continuing without experiment tracking. "
              f"Run `clearml-init` to configure credentials (see WALKTHROUGH.md).")
        return None


def log_epoch_metrics(task, epoch: int, train_metrics: Dict[str, Any],
                       val_metrics: Dict[str, Any], lr: float) -> None:
    if task is None:
        return
    try:
        logger = task.get_logger()
        logger.report_scalar("loss", "train", value=train_metrics["loss"], iteration=epoch)
        logger.report_scalar("loss", "valid", value=val_metrics["loss"], iteration=epoch)
        logger.report_scalar("dice", "train", value=train_metrics["dice"], iteration=epoch)
        logger.report_scalar("dice", "valid", value=val_metrics["dice"], iteration=epoch)
        logger.report_scalar("iou", "valid", value=val_metrics["iou"], iteration=epoch)
        if val_metrics.get("tumor_positive_dice") is not None:
            logger.report_scalar("dice_tumor_positive_only", "valid",
                                  value=val_metrics["tumor_positive_dice"], iteration=epoch)
        if train_metrics.get("tumor_positive_dice") is not None:
            logger.report_scalar("dice_tumor_positive_only", "train",
                                  value=train_metrics["tumor_positive_dice"], iteration=epoch)
        det = val_metrics.get("detection", {})
        for name in ["sensitivity_recall", "specificity", "accuracy", "f1"]:
            if det.get(name) is not None:
                logger.report_scalar(f"detection_{name}", "valid", value=det[name], iteration=epoch)
        logger.report_scalar("learning_rate", "lr", value=lr, iteration=epoch)
    except Exception as e:
        print(f"ClearML metric logging failed for epoch {epoch} ({e}) - continuing.")


def log_checkpoint_artifact(task, name: str, path: Path) -> None:
    if task is None:
        return
    try:
        task.upload_artifact(name=name, artifact_object=str(path))
    except Exception as e:
        print(f"ClearML artifact upload failed for {name} ({e}) - continuing.")


def close_clearml(task) -> None:
    if task is None:
        return
    try:
        task.close()
    except Exception:
        pass
