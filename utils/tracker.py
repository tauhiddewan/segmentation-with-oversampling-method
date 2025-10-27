# utils/result_tracker.py
import json
from pathlib import Path

class ResultTracker:
    """
    Writes results as:
    {
      "pre_oversample": {
        "num_training_images": int,
        "test_loss": float,
        "test_dice": float,
        "test_iou": float
      },
      "post_oversample": {
        "thresholds": {
          "0.50": {
            "iter1": { ...fields... },
            "iter2": { ... }
          },
          "0.60": { ... }
        }
      }
    }
    """
    def __init__(self, save_dir, model_name, model_config, variant):
        self.save_path = Path(save_dir) / f"{model_name}_{model_config}.{variant}.results.json"
        self.data = {
            "pre_oversample": {},
            "post_oversample": {"thresholds": {}}
        }
        self._save()

    def _save(self):
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.save_path, "w") as f:
            json.dump(self.data, f, indent=2)

    @staticmethod
    def _thr_key(thr: float) -> str:
        return f"{thr:.2f}"

    @staticmethod
    def _iter_key(i: int) -> str:
        return f"iter{i}"

    # ---------- PRE ----------
    def set_pre(self, num_training_images: int, test_loss: float, test_dice: float, test_iou: float):
        self.data["pre_oversample"] = {
            "num_training_images": int(num_training_images),
            "test_loss": float(test_loss),
            "test_dice": float(test_dice),
            "test_iou": float(test_iou),
        }
        self._save()

    # ---------- POST (per-iter, per-threshold) ----------
    def add_post_iter(self, thr: float, iter_idx: int, **fields):
        tkey = self._thr_key(thr)
        ikey = self._iter_key(iter_idx)
        self.data["post_oversample"]["thresholds"].setdefault(tkey, {})
        self.data["post_oversample"]["thresholds"][tkey][ikey] = {
            k: (float(v) if isinstance(v, (int, float)) else int(v) if isinstance(v, bool) else v)
            for k, v in fields.items()
        }
        self._save()

    # ---------- POST (test scores) ----------
    def set_post_threshold_test(self, thr: float, **fields):
        tkey = self._thr_key(thr)
        self.data["post_oversample"]["thresholds"].setdefault(tkey, {})
        # ensure plain types
        clean = {}
        for k, v in fields.items():
            if isinstance(v, bool):
                clean[k] = bool(v)
            elif isinstance(v, (int, float)):
                clean[k] = float(v) if isinstance(v, float) else int(v)
            else:
                clean[k] = v
        self.data["post_oversample"]["thresholds"][tkey]["test"] = clean
        self._save()
