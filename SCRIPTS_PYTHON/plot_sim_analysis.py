import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# File paths
gt_path = "/home/nastia/datasets/rosbags/sim_longer/gt_tum.txt"
pred_path = "/home/nastia/datasets/rosbags/sim_longer/aligned_superpoint.txt"
rotated_path = "/home/nastia/datasets/rosbags/sim_longer/aligned_orb.txt"
path4_path = "/home/nastia/datasets/rosbags/sim_longer/aligned_finetuned.txt"

# Colors from plot_kitti.py
colors = {
    "groundtruth": "firebrick",
    "superpoint": "royalblue",
    "orb": "forestgreen",
    "finetuned": "mediumorchid"
}

cols = ["t", "x", "y", "z", "qx", "qy", "qz", "qw"]

# Read data
gt = pd.read_csv(gt_path, sep=r"\s+", header=None, usecols=range(8))
pr = pd.read_csv(pred_path, sep=r"\s+", header=None, usecols=range(8))
rt = pd.read_csv(rotated_path, sep=r"\s+", header=None, usecols=range(8))
p4 = pd.read_csv(path4_path, sep=r"\s+", header=None, usecols=range(8)) if path4_path else None

gt.columns = cols
pr.columns = cols
rt.columns = cols
if p4 is not None:
    p4.columns = cols

# Extract positions with exchanged XYZ (z, x, y)
gtx = gt["z"].to_numpy()
gty = gt["x"].to_numpy()
gtz = gt["y"].to_numpy() 

prx = pr["x"].to_numpy()
pry = pr["y"].to_numpy()
prz = pr["z"].to_numpy()

rtx = rt["x"].to_numpy()
rty = rt["y"].to_numpy()
rtz = rt["z"].to_numpy()

p4x = p4["x"].to_numpy() if p4 is not None else None
p4y = p4["y"].to_numpy() if p4 is not None else None
p4z = p4["z"].to_numpy() if p4 is not None else None


def plot_start(ax, xs, zs):
    """Mark the starting point as a red dot if data is available."""
    if xs is None or zs is None or len(xs) == 0 or len(zs) == 0:
        return
    ax.scatter(xs[0], zs[0], color="red", marker="o", s=60, zorder=6)

# Compute metrics function
def compute_metrics(gt, est, name):
    """Compute ATE (RMSE), MAE, and translational RPE on positions."""
    n = min(len(gt), len(est))
    if n < 2:
        print(f"[{name}] Not enough data to compute metrics (need >= 2 frames)")
        return None
    
    gt = gt[:n]
    est = est[:n]
    diff = est - gt

    # ATE: RMSE of position error
    ate_rmse = float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))

    # MAE: mean absolute error per axis + overall mean of axis-wise MAE
    mae_axis = np.mean(np.abs(diff), axis=0)
    mae_mean = float(np.mean(mae_axis))

    # RPE (translation): frame-to-frame delta error
    gt_delta = np.diff(gt, axis=0)
    est_delta = np.diff(est, axis=0)
    delta_err = est_delta - gt_delta
    rpe = np.linalg.norm(delta_err, axis=1)
    rpe_rmse = float(np.sqrt(np.mean(rpe ** 2)))
    rpe_mean = float(np.mean(rpe))

    return {
        "ate_rmse": ate_rmse,
        "mae_axis": mae_axis,
        "mae_mean": mae_mean,
        "rpe_rmse": rpe_rmse,
        "rpe_mean": rpe_mean,
        "used_frames": n,
    }

def report(label, metrics):
    """Print metrics report"""
    if metrics is None:
        print(f"[{label}] Not enough data to compute metrics (need >= 2 frames)")
        return
    print(f"[{label}] frames used: {metrics['used_frames']}")
    print(f"[{label}] ATE RMSE (m): {metrics['ate_rmse']:.4f}")
    print(f"[{label}] MAE mean (m): {metrics['mae_mean']:.4f} | per-axis [x y z] (m): {metrics['mae_axis'][0]:.4f} {metrics['mae_axis'][1]:.4f} {metrics['mae_axis'][2]:.4f}")
    print(f"[{label}] RPE transl. RMSE (m): {metrics['rpe_rmse']:.4f} | mean: {metrics['rpe_mean']:.4f}")

# Compute metrics for all trajectories
gt_pos = np.column_stack([gtx, gty, gtz])
pr_pos = np.column_stack([prx, pry, prz])
rt_pos = np.column_stack([rtx, rty, rtz])
p4_pos = np.column_stack([p4x, p4y, p4z]) if p4 is not None else None

print("\n=== METRICS COMPUTATION ===\n")
metrics_pr = compute_metrics(gt_pos, pr_pos, "SuperPoint")
metrics_rt = compute_metrics(gt_pos, rt_pos, "ORB")
metrics_p4 = compute_metrics(gt_pos, p4_pos, "Finetuned") if p4_pos is not None else None
metrics_p4_vs_pr = compute_metrics(pr_pos, p4_pos, "Finetuned_vs_SuperPoint") if p4_pos is not None else None

report("SuperPoint", metrics_pr)
print()
report("ORB", metrics_rt)
print()
report("Finetuned", metrics_p4)
print()
report("Finetuned vs SuperPoint", metrics_p4_vs_pr)
print()

# ============================================
# PLOT 1: SuperPoint and Finetuned
# ============================================
fig1, ax1 = plt.subplots(figsize=(10, 8))

ax1.plot(prx, prz, label='Pretrained', linewidth=2.5, color=colors["superpoint"])
if p4x is not None:
    ax1.plot(p4x, p4z, label='Finetuned', linewidth=2.5, color=colors["finetuned"])
plot_start(ax1, prx, prz)
if p4x is not None:
    plot_start(ax1, p4x, p4z)
ax1.set_xlabel('X', fontsize=12)
ax1.set_ylabel('Z', fontsize=12)
ax1.set_title('Comparison of pretrained SuperPoint model to finetuned model', fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.axis('equal')

# ============================================
# PLOT 2: SuperPoint, Finetuned, and Groundtruth
# ============================================
fig2, ax2 = plt.subplots(figsize=(10, 8))

ax2.plot(gtx, gtz, label='Groundtruth', linewidth=2.5, color=colors["groundtruth"])
ax2.plot(prx, prz, label='Pretrained', linewidth=2.5, color=colors["superpoint"])
if p4x is not None:
    ax2.plot(p4x, p4z, label='Finetuned', linewidth=2.5, color=colors["finetuned"])
plot_start(ax2, gtx, gtz)
plot_start(ax2, prx, prz)
if p4x is not None:
    plot_start(ax2, p4x, p4z)
ax2.set_xlabel('X', fontsize=12)
ax2.set_ylabel('Z', fontsize=12)
ax2.set_title('Comparison of pretrained SuperPoint model to finetuned model', fontsize=14)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.axis('equal')

# ============================================
# PLOT 3: Groundtruth, SuperPoint, and ORB
# ============================================
fig3, ax3 = plt.subplots(figsize=(10, 8))

ax3.plot(gtx, gtz, label='Groundtruth', linewidth=2.5, color=colors["groundtruth"])
ax3.plot(prx, prz, label='Pretrained SuperPoint', linewidth=2.5, color=colors["superpoint"])
ax3.plot(rtx, rtz, label='ORB', linewidth=2.5, color=colors["orb"])
plot_start(ax3, gtx, gtz)
plot_start(ax3, prx, prz)
plot_start(ax3, rtx, rtz)
ax3.set_xlabel('X', fontsize=12)
ax3.set_ylabel('Z', fontsize=12)
ax3.set_title('ORB vs Pretrained SuperPoint performance comparison', fontsize=14)
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3)
ax3.axis('equal')

# ============================================
# PLOT 4: Groundtruth, ORB, SuperPoint, Finetuned
# ============================================
fig4, ax4 = plt.subplots(figsize=(10, 8))

ax4.plot(gtx, gtz, label='Groundtruth', linewidth=2.5, color=colors["groundtruth"])
ax4.plot(rtx, rtz, label='ORB', linewidth=2.5, color=colors["orb"])
ax4.plot(prx, prz, label='SuperPoint', linewidth=2.5, color=colors["superpoint"])
if p4x is not None:
    ax4.plot(p4x, p4z, label='Finetuned', linewidth=2.5, color=colors["finetuned"])
plot_start(ax4, gtx, gtz)
plot_start(ax4, rtx, rtz)
plot_start(ax4, prx, prz)
if p4x is not None:
    plot_start(ax4, p4x, p4z)
ax4.set_xlabel('X', fontsize=12)
ax4.set_ylabel('Z', fontsize=12)
ax4.set_title('All trajectories comparison', fontsize=14)
ax4.legend(fontsize=11)
ax4.grid(True, alpha=0.3)
ax4.axis('equal')

plt.show()
