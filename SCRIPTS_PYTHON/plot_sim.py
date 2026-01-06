import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

gt_path = "/home/nastia/datasets/rosbags/sim_longer/gt_tum.txt"
pred_path = "/home/nastia/datasets/rosbags/sim_longer/aligned_superpoint.txt"
rotated_path = "/home/nastia/datasets/rosbags/sim_longer/aligned_orb.txt"
path4_path = "/home/nastia/datasets/rosbags/sim_longer/aligned_finetuned.txt"

cols = ["t","x","y","z","qx","qy","qz","qw"]

gt = pd.read_csv(gt_path, sep=r"\s+", header=None, usecols=range(8))
pr = pd.read_csv(pred_path, sep=r"\s+", header=None, usecols=range(8))
rt = pd.read_csv(rotated_path, sep=r"\s+", header=None, usecols=range(8))
p4 = pd.read_csv(path4_path, sep=r"\s+", header=None, usecols=range(8)) if path4_path else None

gt.columns = cols
pr.columns = cols
rt.columns = cols
if p4 is not None:
    p4.columns = cols

gtx = gt["x"].to_numpy()
gty = gt["y"].to_numpy()
gtz = gt["z"].to_numpy()

prx = pr["x"].to_numpy()
pry = pr["y"].to_numpy()
prz = pr["z"].to_numpy()


rtx = rt["x"].to_numpy()
rty = rt["y"].to_numpy()
rtz = rt["z"].to_numpy()

p4x = p4["x"].to_numpy() if p4 is not None else None
p4y = p4["y"].to_numpy() if p4 is not None else None
p4z = p4["z"].to_numpy() if p4 is not None else None

fig, axs = plt.subplots(2, 2, figsize=(15, 5))

axs[0, 0].plot(gtx, gty)
axs[0, 0].plot(prx, pry)
axs[0, 0].plot(rtx, rty)
if p4x is not None:
    axs[0, 0].plot(p4x, p4y)
axs[0, 0].set_xlabel("x")
axs[0, 0].set_ylabel("y")
axs[0, 0].set_title("XY")

axs[1, 0].plot(gtx, gtz)
axs[1, 0].plot(prx, prz)
axs[1, 0].plot(rtx, rtz)
if p4x is not None:
    axs[1, 0].plot(p4x, p4z)
axs[1, 0].set_xlabel("x")
axs[1, 0].set_ylabel("z")
axs[1, 0].set_title("XZ")

axs[0, 1].plot(gty, gtz)
axs[0, 1].plot(pry, prz)
axs[0, 1].plot(rty, rtz)
if p4y is not None:
    axs[0, 1].plot(p4y, p4z)
axs[0, 1].set_xlabel("y")
axs[0, 1].set_ylabel("z")
axs[0, 1].set_title("YZ")

axs[1, 1].plot(-gty, -gtz)
axs[1, 1].plot(-prz, -prx)
axs[1, 1].plot(-rtz, -rtx)
if p4z is not None:
    axs[1, 1].plot(-p4z, -p4x)
axs[1, 1].set_xlabel("x")
axs[1, 1].set_ylabel("z")

labels = ["groundtruth", "estimated", "aligned"]
if p4x is not None:
    labels.append("path4")
axs[1, 1].legend(labels)

plt.tight_layout()
plt.show()

# 3D plot
fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(121, projection='3d')
ax1.plot(gtz, gtx, gty, label='groundtruth', linewidth=2)
ax1.plot(prx, pry, prz, label='estimated', linewidth=2)
ax1.plot(rtx, rty, rtz, label='aligned', linewidth=2)
if p4x is not None:
    ax1.plot(p4x, p4y, p4z, label='path4', linewidth=2)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
ax1.set_title('3D Trajectories')
ax1.legend()
ax1.grid(True)

# Another 3D view with different rotation
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot(gtz, gtx, gty, label='groundtruth', linewidth=2)
ax2.plot(prx, pry, prz, label='estimated', linewidth=2)
ax2.plot(rtx, rty, rtz, label='aligned', linewidth=2)
if p4x is not None:
    ax2.plot(p4x, p4y, p4z, label='path4', linewidth=2)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')
ax2.set_title('3D Trajectories (Different View)')
ax2.view_init(elev=0, azim=90)
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
