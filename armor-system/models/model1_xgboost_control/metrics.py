"""Pure-NumPy classification metrics for the XGBoost control (no scikit-learn needed).

Conventions (fixed, used everywhere): labels are the Model 1 class ids 0..K-1; the confusion matrix has TRUE classes in
rows and PREDICTED classes in columns; precision/recall/F1 with an empty denominator are 0 (zero_division=0); macro-F1
averages over ALL K labels, including a class that has no support or no predictions.
"""
import numpy as np

CLASS_NAMES = ("Container Hardware", "Container Network", "Node CPU", "Node Disk", "Node Memory")


def family_of(name):
    """Fault family from the D1 category name: 'container' or 'node'."""
    if name.startswith("Container"):
        return "container"
    if name.startswith("Node"):
        return "node"
    raise ValueError(f"unknown fault category {name!r}")


def _check(y_true, y_pred, k):
    y_true, y_pred = np.asarray(y_true, np.int64), np.asarray(y_pred, np.int64)
    if y_true.shape != y_pred.shape or y_true.ndim != 1 or len(y_true) == 0:
        raise ValueError("y_true and y_pred must be non-empty 1-D arrays of equal length")
    if ((y_true < 0) | (y_true >= k) | (y_pred < 0) | (y_pred >= k)).any():
        raise ValueError(f"labels must lie in 0..{k - 1}")
    return y_true, y_pred


def confusion_matrix(y_true, y_pred, k=5):
    y_true, y_pred = _check(y_true, y_pred, k)
    cm = np.zeros((k, k), np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def _div(a, b):
    return np.divide(a, b, out=np.zeros(np.shape(a), np.float64), where=np.asarray(b) > 0)


def per_class(y_true, y_pred, k=5):
    """Dict of arrays: precision, recall, f1, support, predicted (all length k)."""
    cm = confusion_matrix(y_true, y_pred, k)
    tp = np.diag(cm).astype(np.float64)
    support, predicted = cm.sum(1), cm.sum(0)
    p, r = _div(tp, predicted), _div(tp, support)
    return {"precision": p, "recall": r, "f1": _div(2 * p * r, p + r), "support": support, "predicted": predicted}


def summary(y_true, y_pred, k=5):
    """Scalar metrics plus per-class tables and the confusion matrix."""
    pc = per_class(y_true, y_pred, k)
    n = pc["support"].sum()
    acc = float(np.trace(confusion_matrix(y_true, y_pred, k)) / n)
    macro = float(pc["f1"].mean())
    weighted = float((pc["f1"] * pc["support"]).sum() / n)
    return {"n_cases": int(n), "accuracy": acc, "macro_f1": macro, "weighted_f1": weighted,
            "per_class": {key: v.tolist() for key, v in pc.items()}, "confusion_matrix": confusion_matrix(y_true, y_pred, k).tolist()}


def macro_f1_excluding(y_true, y_pred, exclude, k=5):
    """Mean F1 over the other labels, computed from ALL cases (a prediction of the excluded class still counts as an error)."""
    pc = per_class(y_true, y_pred, k)
    keep = [i for i in range(k) if i != exclude]
    return float(pc["f1"][keep].mean())


def majority_class(y_fit, k=5):
    """Most frequent FITTING class; ties go to the lowest class id (deterministic)."""
    return int(np.argmax(np.bincount(np.asarray(y_fit, np.int64), minlength=k)))


def family_diagnostic(y_true, y_pred, names=CLASS_NAMES):
    """Family-level accounting. Predictions outside the true family are kept, never discarded.

    For each family: the confusion rows of its true classes against ALL predicted classes; counts of exactly-correct,
    wrong-inside-family and wrong-outside-family predictions; family-level accuracy (prediction landed in the true
    family; denominator = all cases of the family) and conditional within-family accuracy (exactly correct among cases
    whose prediction landed inside the true family; denominator stated). Also the 2x2 family confusion.
    """
    k = len(names)
    y_true, y_pred = _check(y_true, y_pred, k)
    fam = np.array([family_of(n) for n in names])
    cm = confusion_matrix(y_true, y_pred, k)
    out = {"families": {}}
    for f in ("container", "node"):
        cls = [i for i in range(k) if fam[i] == f]
        sel = np.isin(y_true, cls)
        n = int(sel.sum())
        yt, yp = y_true[sel], y_pred[sel]
        correct = int((yt == yp).sum())
        inside = int(np.isin(yp, cls).sum())
        within_wrong = inside - correct
        cross = n - inside
        out["families"][f] = {
            "classes": [names[i] for i in cls], "n_cases": n,
            "confusion_true_rows_vs_all_predicted_columns": cm[cls].tolist(), "row_labels": [names[i] for i in cls], "column_labels": list(names),
            "exactly_correct": correct, "wrong_inside_family": within_wrong, "wrong_outside_family": cross,
            "family_level_accuracy": {"value": (inside / n) if n else None, "numerator": inside, "denominator": n,
                                      "meaning": "share of this family's cases whose predicted class is in the same family (ignores which class inside)"},
            "conditional_within_family_accuracy": {"value": (correct / inside) if inside else None, "numerator": correct, "denominator": inside,
                                                   "meaning": "among this family's cases predicted inside the family, share with the exactly right class; cases predicted in the other family are not in the denominator, so this is not overall accuracy"},
        }
    fam_ids = np.array([0 if fam[i] == "container" else 1 for i in range(k)])
    fc = np.zeros((2, 2), np.int64)
    np.add.at(fc, (fam_ids[y_true], fam_ids[y_pred]), 1)
    out["family_confusion_true_rows_pred_cols"] = {"labels": ["container", "node"], "matrix": fc.tolist()}
    out["note"] = "Families follow the D1 category names. They do not prove a fault physically affects only pods or only VMs."
    return out
