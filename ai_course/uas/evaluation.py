from ultralytics import YOLO
import numpy as np
import os

def save_evaluation_report(metrics, class_names, save_path):
    p = np.array(metrics.box.p)
    r = np.array(metrics.box.r)
    f1 = np.array(metrics.box.f1)

    nc = metrics.box.nc  # number of real classes

    # Confusion matrix includes background -> remove it
    cm_full = metrics.confusion_matrix.matrix
    cm = cm_full[:nc, :nc]  # remove background row/col

    # Support = GT objects per class
    supports = cm.sum(axis=1).astype(int)
    total_support = supports.sum()

    # Accuracy
    accuracy = np.trace(cm) / total_support if total_support > 0 else 0.0

    # Macro averages
    macro_p = p.mean()
    macro_r = r.mean()
    macro_f1 = f1.mean()

    # Weighted averages
    weighted_p = np.sum(p * supports) / total_support
    weighted_r = np.sum(r * supports) / total_support
    weighted_f1 = np.sum(f1 * supports) / total_support

    lines = []
    lines.append("              precision    recall  f1-score   support\n\n")

    for i, name in enumerate(class_names):
        lines.append(
            f"{name:>14} "
            f"{p[i]:10.4f} "
            f"{r[i]:9.4f} "
            f"{f1[i]:9.4f} "
            f"{supports[i]:9d}\n"
        )

    lines.append("\n")
    lines.append(f"{'accuracy':>14}{accuracy:24.4f}{total_support:9d}\n")
    lines.append(
        f"{'macro avg':>14} "
        f"{macro_p:10.4f} "
        f"{macro_r:9.4f} "
        f"{macro_f1:9.4f} "
        f"{total_support:9d}\n"
    )
    lines.append(
        f"{'weighted avg':>14} "
        f"{weighted_p:10.4f} "
        f"{weighted_r:9.4f} "
        f"{weighted_f1:9.4f} "
        f"{total_support:9d}\n"
    )

    with open(save_path, "w") as f:
        f.writelines(lines)


if __name__ == "__main__":
    model = YOLO("models/Face_detect_v2.pt")

    metrics = model.val(
        data="config.yaml",
        split="val",
        device=0
    )

    class_names = list(metrics.names.values())
    save_path = os.path.join(metrics.save_dir, "evaluation_report.txt")

    save_evaluation_report(metrics, class_names, save_path)

    print(f"Evaluation report saved to:\n{save_path}")
