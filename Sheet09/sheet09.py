import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from prettytable import PrettyTable

from tqdm import tqdm


def load_FLO_file(filename):
    assert os.path.isfile(filename), 'file does not exist: ' + filename   
    flo_file = open(filename, 'rb')
    magic = np.fromfile(flo_file, np.float32, count=1)
    assert magic == 202021.25,  'Magic number incorrect. .flo file is invalid'
    w = np.fromfile(flo_file, np.int32, count=1)
    h = np.fromfile(flo_file, np.int32, count=1)
    data = np.fromfile(flo_file, np.float32, count=2*w[0]*h[0])
    flow = np.resize(data, (int(h[0]), int(w[0]), 2))
    flo_file.close()
    return flow

class OpticalFlow:
    def __init__(self):
        self.EIGEN_THRESHOLD = 0.01
        self.WINDOW_SIZE = (25, 25)
        self.EPSILON = 1e-3
        self.MAX_ITERS = 1000
        self.ALPHA = 1.0
        self.prev = None
        self.next = None

    def next_frame(self, img):
        self.prev = self.next
        self.next = img

        if self.prev is None:
            return False

        frames = np.float32([self.prev, self.next]) / 255.0
        self.Ix = cv.Sobel(frames[0], cv.CV_32F, 1, 0, ksize=3)
        self.Iy = cv.Sobel(frames[0], cv.CV_32F, 0, 1, ksize=3)
        self.It = frames[1] - frames[0]
        return True

    def Lucas_Kanade_flow(self):
        flow = np.zeros((*self.next.shape, 2), dtype=np.float32)
        half_w = self.WINDOW_SIZE[0] // 2
        
        for y in range(half_w, self.next.shape[0] - half_w):
            for x in range(half_w, self.next.shape[1] - half_w):
                Ix_window = self.Ix[y-half_w:y+half_w+1, x-half_w:x+half_w+1].flatten()
                Iy_window = self.Iy[y-half_w:y+half_w+1, x-half_w:x+half_w+1].flatten()
                It_window = self.It[y-half_w:y+half_w+1, x-half_w:x+half_w+1].flatten()

                A = np.vstack((Ix_window, Iy_window)).T
                b = -It_window

                if np.linalg.cond(A.T @ A) < 1 / self.EIGEN_THRESHOLD:
                    nu = np.linalg.pinv(A.T @ A) @ (A.T @ b)
                    flow[y, x] = nu
        return flow, self.flow_map_to_bgr(flow)

    def Horn_Schunck_flow(self):
        u = np.zeros_like(self.next, dtype=np.float32)
        v = np.zeros_like(self.next, dtype=np.float32)

        kernel = np.array([[0, 1/4, 0],
                           [1/4, 0, 1/4],
                           [0, 1/4, 0]])

        for _ in range(self.MAX_ITERS):
            u_avg = cv.filter2D(u, -1, kernel)
            v_avg = cv.filter2D(v, -1, kernel)

            P = self.Ix * u_avg + self.Iy * v_avg + self.It
            D = self.ALPHA**2 + self.Ix**2 + self.Iy**2

            u_new = u_avg - (self.Ix * P) / D
            v_new = v_avg - (self.Iy * P) / D

            if np.linalg.norm(u_new - u) < self.EPSILON and np.linalg.norm(v_new - v) < self.EPSILON:
                break

            u, v = u_new, v_new

        flow = np.stack((u, v), axis=-1)
        return flow, self.flow_map_to_bgr(flow)

    def calculate_angular_error(self, estimated_flow, groundtruth_flow):
        dot_product = np.sum(estimated_flow * groundtruth_flow, axis=-1)
        mag_est = np.linalg.norm(estimated_flow, axis=-1)
        mag_gt = np.linalg.norm(groundtruth_flow, axis=-1)
        cos_theta = dot_product / (mag_est * mag_gt + 1e-6)
        aae_per_point = np.arccos(np.clip(cos_theta, -1, 1))
        return np.mean(aae_per_point), aae_per_point

    def calculate_endpoint_error(self, estimated_flow, groundtruth_flow):
        diff = estimated_flow - groundtruth_flow
        epe_per_point = np.linalg.norm(diff, axis=-1)
        return np.mean(epe_per_point), epe_per_point

    def flow_map_to_bgr(self, flow):
        h, w = flow.shape[:2]
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = 255
        hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
        bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        return bgr



if __name__ == "__main__":

    data_list = [
        'data/frame_0001.png',
        'data/frame_0002.png',
        'data/frame_0003.png',
    ]

    gt_list = [
        'data/frame_0001.flo',
        'data/frame_0002.flo',
        'data/frame_0003.flo',
    ]

    Op = OpticalFlow()
    tab = PrettyTable()
    tab.field_names = ["Run", "AAE_lucas_kanade", "AEE_lucas_kanade", "AAE_horn_schunk", "AEE_horn_schunk"]


    results = []
    count = 0
    for (i, (frame_filename, gt_filename)) in tqdm(enumerate(zip(data_list, gt_list)), total=len(data_list)):
        groundtruth_flow = load_FLO_file(gt_filename)
        img = cv.cvtColor(cv.imread(frame_filename), cv.COLOR_BGR2GRAY)
        if not Op.next_frame(img):
            continue

        flow_lucas_kanade, flow_lucas_kanade_bgr = Op.Lucas_Kanade_flow()
        aae_lucas_kanade, aae_lucas_kanade_per_point = Op.calculate_angular_error(flow_lucas_kanade, groundtruth_flow)
        aee_lucas_kanade, aee_lucas_kanade_per_point = Op.calculate_endpoint_error(flow_lucas_kanade, groundtruth_flow)

        flow_horn_schunck, flow_horn_schunck_bgr = Op.Horn_Schunck_flow()
        aae_horn_schunk, aae_horn_schunk_per_point = Op.calculate_angular_error(flow_horn_schunck, groundtruth_flow) 
        aee_horn_schunk, aee_horn_schunk_per_point = Op.calculate_endpoint_error(flow_horn_schunck, groundtruth_flow)
        count += 1

        tab.add_row([count,
                     np.round(aae_lucas_kanade, decimals=2),
                     np.round(aee_lucas_kanade, decimals=2),
                     np.round(aae_horn_schunk, decimals=2),
                     np.round(aee_horn_schunk, decimals=2)])

        flow_bgr_gt = Op.flow_map_to_bgr(groundtruth_flow)

        results.append({
            "Flow GT": flow_bgr_gt,
            "Lucas Kanade": flow_lucas_kanade_bgr,
            "AAE Lucas Kanade": aae_lucas_kanade_per_point,
            "AEE Luca Kanade": aee_lucas_kanade_per_point,
            "Horn-Schunk": flow_horn_schunck_bgr,
            "AAE Horn-Schunk": aae_horn_schunk_per_point,
            "AEE Horn-Schunk": aee_horn_schunk_per_point,
        })

    fig, axes = plt.subplots(nrows=len(results),
                             ncols=7,
                             figsize=(30, 5))
    for r, res in enumerate(results):
        for c, (name, value) in enumerate(res.items()):
            ax = axes[r][c]
            ax.imshow(value)
            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_title(name)
    fig.subplots_adjust(wspace=0.00, hspace=0.0, left=0, right=1.0, top=1.0, bottom=0)
    plt.savefig("results.png")
    # plt.show()
    print(tab)
