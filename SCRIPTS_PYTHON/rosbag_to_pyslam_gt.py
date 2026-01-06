#!/usr/bin/env python3
import os
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from cv_bridge import CvBridge
from geographiclib.geodesic import Geodesic
import rosbag2_py
from rclpy.serialization import deserialize_message
import re
import pandas as pd



class RosbagReader:
    def __init__(self, bag_path):
        self.bag_path = bag_path

    def reader(self):
        storage = rosbag2_py.StorageOptions(uri=self.bag_path, storage_id='sqlite3')
        conv = rosbag2_py.ConverterOptions('', '')
        r = rosbag2_py.SequentialReader()
        r.open(storage, conv)
        return r

    def msg_class(self, t):
        pkg, msg = t.split('/msg/')
        mod = __import__(f"{pkg}.msg", fromlist=[msg])
        return getattr(mod, msg)


def extract_gps(bag, topic):
    r = bag.reader()
    tmap = {t.name: t.type for t in r.get_all_topics_and_types()}
    mt = bag.msg_class(tmap[topic])

    first = None
    out = []

    while r.has_next():
        tpc, data, _ = r.read_next()
        if tpc != topic:
            continue
        msg = deserialize_message(data, mt)

        ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        lat, lon, alt = msg.latitude, msg.longitude, msg.altitude

        if first is None:
            first = (lat, lon, alt, ts)
            out.append([0, 0, 0, 0])
            continue

        g = Geodesic.WGS84.Inverse(first[0], first[1], lat, lon)
        dist = g["s12"]
        azi = np.deg2rad(g["azi1"])
        x = dist * np.cos(azi)
        y = dist * np.sin(azi)
        z = alt - first[2]
        out.append([ts - first[3], x, y, z])

    return np.array(out)


def extract_imu(bag, topic):
    r = bag.reader()
    tmap = {t.name: t.type for t in r.get_all_topics_and_types()}
    mt = bag.msg_class(tmap[topic])

    first = None
    out = []

    while r.has_next():
        tpc, data, _ = r.read_next()
        if tpc != topic:
            continue
        msg = deserialize_message(data, mt)

        ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if first is None:
            first = ts

        q = msg.orientation
        out.append([ts - first, q.x, q.y, q.z, q.w])

    return np.array(out)


def extract_image_timestamps(bag, topic):
    r = bag.reader()
    tmap = {t.name: t.type for t in r.get_all_topics_and_types()}
    mt = bag.msg_class(tmap[topic])

    first = None
    out = []

    while r.has_next():
        tpc, data, _ = r.read_next()
        if tpc != topic:
            continue
        msg = deserialize_message(data, mt)

        ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if first is None:
            first = ts
        out.append(ts - first)

    return np.array(out)


def merge(gps, imu):
    ts = imu[:, 0]

    gx = np.interp(ts, gps[:, 0], gps[:, 1])
    gy = np.interp(ts, gps[:, 0], gps[:, 2])
    gz = np.interp(ts, gps[:, 0], gps[:, 3])

    # axis rotation: x, z, y
    rx = gy
    ry = gz
    rz = gx

    out = np.column_stack([ts, gx, gy, gz, imu[:, 1:]])
    return out


def match_to_images(gt, image_ts):
    out = []
    for t in image_ts:
        i = np.argmin(np.abs(gt[:, 0] - t))
        out.append(gt[i])
    return np.array(out)

def extract_images_and_timestamps(bag, topic, outdir):
    os.makedirs(outdir, exist_ok=True)

    r = bag.reader()
    tmap = {t.name: t.type for t in r.get_all_topics_and_types()}
    mt = bag.msg_class(tmap[topic])
    bridge = CvBridge()

    first_ts = None
    timestamps = []
    msg_idx = 0

    while r.has_next():
        tpc, data, _ = r.read_next()
        if tpc != topic:
            continue

        msg_idx += 1
        msg = deserialize_message(data, mt)

        ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if first_ts is None:
            first_ts = ts
        timestamps.append(ts - first_ts)

        img_data = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        filename = os.path.join(outdir, f"image_{msg_idx:06d}.png")
        cv2.imwrite(filename, img_data)

    return timestamps


def drop_every_second_images_and_records(images_dir, timestamps_path, groundtruth_path, drop_even=True):
    """
    Drop every second image/record. If drop_even is True (default) the function
    removes images with even numeric ids (2,4,6,...) and keeps odd (1,3,5,...).
    If drop_even is False, it removes odd ids and keeps even ones.
    """

    # collect and sort image files by numeric index
    imgs = [f for f in os.listdir(images_dir) if re.match(r'image_\d+\.png$', f)]
    def idx_of(name):
        return int(re.search(r'(\d+)', name).group(1))
    imgs.sort(key=idx_of)

    # decide which parity to keep
    if drop_even:
        keep_pred = lambda i: (i % 2) == 1  # keep odd ids
    else:
        keep_pred = lambda i: (i % 2) == 0  # keep even ids

    # split kept and removed
    kept = []
    for name in imgs:
        i = idx_of(name)
        if keep_pred(i):
            kept.append(name)
        else:
            try:
                os.remove(os.path.join(images_dir, name))
            except OSError:
                pass

    # safe two-step rename kept images to sequential numbering starting at 1
    # first to temporary names to avoid collisions
    for new_i, old_name in enumerate(kept, start=1):
        src = os.path.join(images_dir, old_name)
        tmp = os.path.join(images_dir, f".tmp_image_{new_i:06d}.png")
        if os.path.exists(src):
            os.rename(src, tmp)

    # then finalize names
    for new_i in range(1, len(kept) + 1):
        tmp = os.path.join(images_dir, f".tmp_image_{new_i:06d}.png")
        dst = os.path.join(images_dir, f"image_{new_i:06d}.png")
        if os.path.exists(tmp):
            os.rename(tmp, dst)

    # update timestamps file
    if timestamps_path and os.path.exists(timestamps_path):
        ts = np.loadtxt(timestamps_path)
        ts = np.atleast_1d(ts).reshape(-1)  # ensure 1D
        ts_new = ts[::2] if drop_even else ts[1::2]
        np.savetxt(timestamps_path, ts_new.reshape(-1, 1), fmt="%.6f")

    # update groundtruth file
    if groundtruth_path and os.path.exists(groundtruth_path):
        gt = np.loadtxt(groundtruth_path)
        if gt.ndim == 1:
            gt = gt.reshape(1, -1)
        gt_new = gt[::2] if drop_even else gt[1::2]
        np.savetxt(groundtruth_path, gt_new, fmt="%.6f")




def gt_tum(timestamps_file, groundtruth_file, out_file, ts_col=0):

    def read_table(path):
        return pd.read_csv(
            path,
            header=None,
            comment='#',
            sep=r"\s+",
            engine='python',
            skip_blank_lines=True,
            dtype=str
        )

    ts_df = read_table(timestamps_file)
    data_df = read_table(groundtruth_file)

    ts = ts_df.iloc[:, ts_col].reset_index(drop=True)

    if data_df.shape[1] > 1:
        data_no_last = data_df.iloc[:, :-1].reset_index(drop=True)
    else:
        data_no_last = pd.DataFrame(index=range(data_df.shape[0]))

    n = min(len(ts), len(data_no_last))

    ts_out = ts.iloc[:n].reset_index(drop=True)
    data_out = data_no_last.iloc[:n].reset_index(drop=True)

    out_df = pd.concat([ts_out, data_out], axis=1)
    out_df.to_csv(out_file, sep=' ', header=False, index=False, encoding='utf-8')



def drop_n_records(gt_path, timestamps_path, n):
    gt = np.loadtxt(gt_path)
    ts = np.loadtxt(timestamps_path)

    gt_new = gt[n:]
    ts_new = ts[n:]

    np.savetxt(gt_path, gt_new, fmt="%.6f")
    np.savetxt(timestamps_path, ts_new.reshape(-1, 1), fmt="%.6f")



SIM_TOPICS = {
    "gps": "/wamv/sensors/gps/gps/fix",
    "imu": "/wamv/sensors/imu/imu/data_rotated",
    "images": "/wamv/sensors/cameras/front_right_camera_sensor/image_raw"
}

BARKA_TOPICS = {
    "gps": "/gnss",
    "imu": "/imu/data",
    "images": "/zed/zed_node/rgb/color/rect/image"
}


def main():

    # BARKA
    dir = "/home/nastia/datasets/rosbags/barka/20251130_3/"

    bag_path = dir + "20251130_3.db3"
    bag = RosbagReader(bag_path)

    gps = extract_gps(bag, BARKA_TOPICS["gps"])
    imu = extract_imu(bag, BARKA_TOPICS["imu"])
    img_ts = extract_image_timestamps(bag, BARKA_TOPICS["images"])
    gt = merge(gps, imu)
    aligned = match_to_images(gt, img_ts)

    timestamps = extract_images_and_timestamps(bag, BARKA_TOPICS["images"], dir + "images")
    np.savetxt(dir + "groundtruth_pyslam.txt", aligned, fmt="%.6f")
    np.savetxt(dir + "images/groundtruth_pyslam.txt", aligned, fmt="%.6f")
    np.savetxt(dir + "timestamps.txt", np.array(timestamps).reshape(-1, 1), fmt="%.6f")
    np.savetxt(dir + "images/timestamps.txt", np.array(timestamps).reshape(-1, 1), fmt="%.6f")
    gt_tum(
        dir + "timestamps.txt",
        dir + "groundtruth_pyslam.txt",
        dir + "gt_tum.txt"
    )

    # SIM
    # dir = "/home/nastia/datasets/rosbags/sim_longer/"

    # bag_path = dir + "sim_longer_0.db3"
    # bag = RosbagReader(bag_path)

    # gps = extract_gps(bag, SIM_TOPICS["gps"])
    # imu = extract_imu(bag, SIM_TOPICS["imu"])
    # img_ts = extract_image_timestamps(bag, SIM_TOPICS["images"])
    # gt = merge(gps, imu)
    # aligned = match_to_images(gt, img_ts)

    # timestamps = extract_images_and_timestamps(bag, SIM_TOPICS["images"], dir + "images")

    # np.savetxt(dir + "images/groundtruth_pyslam.txt", aligned, fmt="%.6f")
    # np.savetxt(dir + "images/timestamps.txt", np.array(timestamps).reshape(-1, 1), fmt="%.6f")


    # drop_every_second_images_and_records(
    #     images_dir=dir + "images",
    #     timestamps_path=dir + "images/timestamps.txt",
    #     groundtruth_path=dir + "images/groundtruth_pyslam.txt",
    #     drop_even=True)
    


    # gt_tum(
    #     dir + "images/timestamps.txt",
    #     dir + "images/groundtruth_pyslam.txt",
    #     dir + "gt_tum.txt"
    # )

    

if __name__ == "__main__":
    main()
