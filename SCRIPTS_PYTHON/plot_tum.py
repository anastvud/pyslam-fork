import pandas as pd
import matplotlib.pyplot as plt

gt_path = "/home/nastia/datasets/rosbags/barka/20251130_1/gt_tum.txt"
pred_path = "/home/nastia/datasets/rosbags/barka/20251130_1/orb_predicted_tum.txt"
rotated_path = "/home/nastia/datasets/rosbags/barka/20251130_1/tum_aligned_as_in_evo.txt"

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

fig, axs = plt.subplots(2, 2, figsize=(15, 5))

axs[0, 0].plot(gtx, gty)
axs[0, 0].plot(prx, pry)
axs[0, 0].plot(rtx, rty)
axs[0, 0].set_xlabel("x")
axs[0, 0].set_ylabel("y")
axs[0, 0].set_title("XY")

axs[1, 0].plot(gtx, gtz)
axs[1, 0].plot(-prz, -prx)
axs[1, 0].plot(rtx, rtz)
axs[1, 0].set_xlabel("x")
axs[1, 0].set_ylabel("z")
axs[1, 0].set_title("XZ")

axs[0, 1].plot(gty, gtz)
axs[0, 1].plot(pry, prz)
axs[0, 1].plot(rty, rtz)
axs[0, 1].set_xlabel("y")
axs[0, 1].set_ylabel("z")
axs[0, 1].set_title("YZ")

axs[1, 1].plot(gty, gtz)
axs[1, 1].plot(-prz, -prx)
axs[1, 1].plot(-rtz, -rtx)
axs[1, 1].set_xlabel("x")
axs[1, 1].set_ylabel("z")


axs[1, 1].legend(["groundtruth", "estimated", "aligned"])

plt.tight_layout()
plt.show()
