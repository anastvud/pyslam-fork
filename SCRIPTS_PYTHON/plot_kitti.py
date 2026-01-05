import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot

def read_kitti_groundtruth(filepath):
    """
    Read KITTI ground truth file in format: R11 R12 R13 T1 R21 R22 R23 T2 R31 R32 R33 T3
    Returns positions as Nx3 array
    """
    poses = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                values = [float(x) for x in line.split()]
                if len(values) != 12:
                    continue
                
                # Extract translation (last column of the 3x4 pose matrix)
                # Format: [R11 R12 R13 T1 R21 R22 R23 T2 R31 R32 R33 T3]
                # Position is at indices [3, 7, 11]
                x = values[3]
                y = values[7]
                z = values[11]
                
                poses.append([x, y, z])
    
    except FileNotFoundError:
        print(f"Error: Could not find '{filepath}'")
        return None
    
    return np.array(poses)

def read_tum_trajectory(filepath):
    """
    Read TUM format trajectory: timestamp x y z qx qy qz qw
    Returns positions as Nx3 array
    """
    poses = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if len(parts) < 8:
                    continue
                
                # Extract position (indices 1, 2, 3)
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])
                
                poses.append([x, y, z])
    
    except FileNotFoundError:
        print(f"Error: Could not find '{filepath}'")
        return None
    
    return np.array(poses)

def plot_trajectories(groundtruth_file, estimated_file1, estimated_file2):
    """
    Plot KITTI ground truth and two TUM format estimated trajectories
    """
    # Read trajectories
    print(f"Reading ground truth from {groundtruth_file}...")
    gt_poses = read_kitti_groundtruth(groundtruth_file)
    
    print(f"Reading first estimated trajectory from {estimated_file1}...")
    est_poses1 = read_tum_trajectory(estimated_file1)
    
    print(f"Reading second estimated trajectory from {estimated_file2}...")
    est_poses2 = read_tum_trajectory(estimated_file2)
    
    if gt_poses is None or est_poses1 is None or est_poses2 is None:
        print("Error reading trajectories")
        return
    
    print(f"Ground truth poses: {len(gt_poses)}")
    print(f"Estimated poses 1: {len(est_poses1)}")
    print(f"Estimated poses 2: {len(est_poses2)}")

    # --- Metrics ---
    def compute_metrics(gt, est):
        """Compute ATE (RMSE), MAE, and translational RPE on positions."""
        n = min(len(gt), len(est))
        if n < 2:
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
        if metrics is None:
            print(f"[{label}] Not enough data to compute metrics (need >= 2 frames)")
            return
        print(f"[{label}] frames used: {metrics['used_frames']}")
        print(f"[{label}] ATE RMSE (m): {metrics['ate_rmse']:.4f}")
        print(f"[{label}] MAE mean (m): {metrics['mae_mean']:.4f} | per-axis [x y z] (m): {metrics['mae_axis'][0]:.4f} {metrics['mae_axis'][1]:.4f} {metrics['mae_axis'][2]:.4f}")
        print(f"[{label}] RPE transl. RMSE (m): {metrics['rpe_rmse']:.4f} | mean: {metrics['rpe_mean']:.4f}")

    print("\nComputing metrics vs ground truth (truncated to common length)...")
    metrics_orb = compute_metrics(gt_poses, est_poses1)
    metrics_sp = compute_metrics(gt_poses, est_poses2)
    report("ORB-SLAM", metrics_orb)
    report("SuperPoint", metrics_sp)
    
    # Create figure for XZ plane view
    fig = plt.figure(figsize=(10, 8))
    
    # XZ plane view
    ax = fig.add_subplot(111)
    ax.plot(gt_poses[:, 0], gt_poses[:, 2], color='firebrick', label='Groundtruth', linewidth=2.5)
    ax.plot(est_poses1[:, 0], est_poses1[:, 2], color='forestgreen', label='ORB', linewidth=2.5)
    ax.plot(est_poses2[:, 0], est_poses2[:, 2], color='royalblue', label='SuperPoint', linewidth=2.5)
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Z (m)', fontsize=12)
    ax.set_title('KITTI Sequence 03', fontsize=14)
    # Add RPE summaries on-plot (mean values)
    # if metrics_orb is not None:
    #     ax.text(0.02, 0.96, f"ORB RPE mean: {metrics_orb['rpe_mean']:.3f} m", transform=ax.transAxes,
    #             fontsize=11, color='forestgreen', verticalalignment='top')
    # if metrics_sp is not None:
    #     ax.text(0.02, 0.90, f"SuperPoint RPE mean: {metrics_sp['rpe_mean']:.3f} m", transform=ax.transAxes,
    #             fontsize=11, color='royalblue', verticalalignment='top')

    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Define your file paths here
    groundtruth_file = "/home/nastia/datasets/kitti/poses/03.txt"
    estimated_file1 = "/home/nastia/datasets/kitti/sequences/03/orb_predicted_tum.txt"
    estimated_file2 = "/home/nastia/datasets/kitti/sequences/03/superpoint_predicted_tum.txt"
    
    plot_trajectories(groundtruth_file, estimated_file1, estimated_file2)
