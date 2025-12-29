import numpy as np
from scipy.spatial.transform import Rotation as Rot

def transform_tum_trajectory():
    # --- 1. Define the Alignment Parameters ---
    # Rotation Matrix (R_align)
    R_align = np.array([
        [ 0.14783384, -0.76998699, -0.62069734],
        [ 0.94948702, -0.06515419,  0.30696796],
        [-0.27680237, -0.63472432,  0.72146066]
    ])
    
    # Translation Vector (t_align)
    t_align = np.array([28.39438401, 3.96675942, 0.65894437])
    
    # Scale Factor (s_align)
    s_align = 2.183457177130708

    input_file = "/data/20251130_1/orb_predicted_tum.txt"
    output_file = "data/20251130_1/tum_aligned_as_in_evo.txt"

    print(f"Reading from {input_file}...")
    
    aligned_poses = [] # To store data for the second pass
    
    try:
        # --- PASS 1: Read and Apply Initial Alignment (S, R, t) ---
        with open(input_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if len(parts) < 8:
                    continue

                timestamp = parts[0]
                pos = np.array([float(x) for x in parts[1:4]])
                quat = np.array([float(x) for x in parts[4:8]]) # [qx, qy, qz, qw]

                # 1. Apply Alignment to Position: P_new = s * R * P_old + t
                pos_aligned = s_align * (R_align @ pos) + t_align

                # 2. Apply Alignment to Orientation: R_new = R_align * R_old
                r_old = Rot.from_quat(quat)
                r_aligned = Rot.from_matrix(R_align @ r_old.as_matrix())
                quat_aligned = r_aligned.as_quat()
                
                # Store for next step
                aligned_poses.append({
                    'timestamp': timestamp,
                    'pos': pos_aligned,
                    'quat': quat_aligned
                })

        if not aligned_poses:
            print("Error: No valid data found in input file.")
            return

        # --- PASS 2: Reset Trajectory to Start at Identity (T0_inverse * Ti) ---
        print("Normalizing trajectory to start at identity...")
        
        # 1. Construct the Transformation Matrix of the first frame (T0)
        p0 = aligned_poses[0]['pos']
        q0 = aligned_poses[0]['quat']
        
        R0 = Rot.from_quat(q0).as_matrix()
        T0 = np.eye(4)
        T0[:3, :3] = R0
        T0[:3, 3] = p0
        
        # 2. Compute Inverse of T0
        # T_inv = [ R^T  -R^T*t ]
        #         [  0      1   ]
        T0_inv = np.eye(4)
        T0_inv[:3, :3] = R0.T
        T0_inv[:3, 3] = -R0.T @ p0

        final_lines = []
        
        for pose in aligned_poses:
            # Construct Ti (Current Pose Matrix)
            Ri = Rot.from_quat(pose['quat']).as_matrix()
            ti = pose['pos']
            
            Ti = np.eye(4)
            Ti[:3, :3] = Ri
            Ti[:3, 3] = ti
            
            # Apply Relative Transform: T_final = T0_inv @ Ti
            T_final = T0_inv @ Ti
            
            # Extract final Position and Quaternion
            pos_final = T_final[:3, 3]
            quat_final = Rot.from_matrix(T_final[:3, :3]).as_quat() # Normalized automatically
            
            # Format output line
            new_line = (
                f"{pose['timestamp']} "
                f"{pos_final[0]:.6f} {pos_final[1]:.6f} {pos_final[2]:.6f} "
                f"{quat_final[0]:.6f} {quat_final[1]:.6f} {quat_final[2]:.6f} {quat_final[3]:.6f}"
            )
            final_lines.append(new_line)

        # --- Write Output ---
        with open(output_file, 'w') as f:
            f.write("\n".join(final_lines))
            
        print(f"Success! Transformed and normalized data saved to {output_file}")

    except FileNotFoundError:
        print(f"Error: Could not find '{input_file}' in the current directory.")

if __name__ == "__main__":
    transform_tum_trajectory()