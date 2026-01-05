import pandas as pd
import matplotlib.pyplot as plt

gt_path = "/home/nastia/datasets/rosbags/sim_longer/gt_tum.txt"
pred_path = "/home/nastia/datasets/rosbags/sim_longer/aligned_superpoint.txt"
rotated_path = "/home/nastia/datasets/rosbags/sim_longer/aligned_finetuned.txt"

cols = ["t","x","y","z","qx","qy","qz","qw"]

gt = pd.read_csv(gt_path, sep=r"\s+", header=None, usecols=range(8))
pr = pd.read_csv(pred_path, sep=r"\s+", header=None, usecols=range(8))
rt = pd.read_csv(rotated_path, sep=r"\s+", header=None, usecols=range(8))

gt.columns = cols
pr.columns = cols
rt.columns = cols

gtx = gt["x"].to_numpy()
gty = gt["y"].to_numpy()
gtz = gt["z"].to_numpy()

prx = pr["x"].to_numpy()
pry = pr["y"].to_numpy()
prz = pr["z"].to_numpy()


rtx = rt["x"].to_numpy()
rty = rt["y"].to_numpy()
rtz = rt["z"].to_numpy()

fig, ax = plt.subplots(figsize=(15, 5))

ax.plot(gty, gtz, label="groundtruth")
ax.plot(prz, rtx, label="aligned_superpoint")
ax.plot(rtz, rtx, label="aligned_finetuned")
ax.set_xlabel("x")
ax.set_ylabel("z")
ax.legend()

plt.tight_layout()
plt.show()
