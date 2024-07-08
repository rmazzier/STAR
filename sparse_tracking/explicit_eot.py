import numpy as np
import scipy as sp
from .utils import isNaN


class KalmanTracker:
    def __init__(self, id_, birth_idx, x0=None, X0=None, body_parts_md=False):

        self.F = np.eye(4)
        self.H = np.zeros((2, 4))
        self.H[:2, :2] = np.eye(2)

        self.R = np.diag([0.08**2, 0.08**2])
        self.P = np.eye(4) * 0.1
        self.sigma_a = 8
        self.Q = np.eye(4)
        self.S = 0.1 * np.eye(2)
        self.n = self.F.shape[1]
        self.m = self.H.shape[1]

        self.s = np.zeros((self.n, 1)) if x0 is None else x0.reshape(-1, 1)

        #########################################################
        # Tracker-related parameters
        self.detection_history = []
        self.misses_number = 0
        self.hits = 0
        self.id = id_
        self.birth_step = birth_idx
        self.state_memory = []
        self.alive_steps = []
        self.body_md = body_parts_md
        self.micro_doppler = [] if not self.body_md else [[] for _ in range(11)]
        self.micro_range = []
        self.best_AP = []
        self.current_dbp_index = None
        self.assoc_meas = []
        self.last_h = None
        self.spec_window_times = []
        self.h_signal = []

    def compute_FQ(self, dt):
        self.F = np.kron(np.array([[1, dt], [0, 1]]), np.eye(2))
        G = np.array([0.5 * (dt**2), dt]).reshape(2, 1)
        self.Q = np.kron((G @ G.T) * self.sigma_a**2, np.eye(2))

    def predict(self, dt):
        self.compute_FQ(dt)
        assert self.s.shape[0] == self.n
        self.s = self.F @ self.s
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z, dt):
        self.compute_FQ(dt)
        err = z - self.H @ self.s
        self.S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(self.S)
        self.s += K @ err
        self.P -= K @ self.H @ self.P
        self.state_memory.append(self.s)
        return self.s

    def get_S(self):
        S = self.H @ self.P @ self.H.T + self.R
        if isNaN(S[0, 0]):
            S = 1e3 * np.eye(2)
        return S

    def get_ni(self, z):
        return z - self.H @ self.s

    def get_mahalanobis(self, z):
        return (
            self.get_ni(z).T @ np.linalg.inv(self.get_S()) @ self.get_ni(z)
        ).squeeze()

    def get_distance(self, z):
        ni = self.get_ni(z)
        return np.sqrt((ni[0] ** 2 + ni[1] ** 2)).squeeze()

    @staticmethod
    def hungarian_assignment(score_matrix):
        # print(score_matrix.astype(np.float32))
        det_idx, tr_idx = sp.optimize.linear_sum_assignment(score_matrix)
        unmatched, undetected = [], []
        for t in range(score_matrix.shape[1]):
            if t not in tr_idx:
                undetected.append(t)
        for d in range(score_matrix.shape[0]):
            if d not in det_idx:
                unmatched.append(d)
        matches = []
        for d, t in zip(det_idx, tr_idx):
            if (1 - score_matrix[d, t]) > 0.3:
                matches.append(np.array([d, t]).reshape(1, 2))
            else:
                undetected.append(t)
        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)
        return matches, np.array(undetected), np.array(unmatched)


if __name__ == "__main__":
    import pickle
    import matplotlib
    from matplotlib import pyplot as plt
    from matplotlib import rc
    from matplotlib.patches import Ellipse
    from sklearn.cluster import DBSCAN

    rc("text", usetex=True)
    matplotlib.rcParams["font.family"] = "serif"

    with open("results/example_track.obj", "rb") as f:
        obs = pickle.load(f)

    obs = obs[:-30]
    tracker = ETKalmanTracker(
        id_=0,
        x0=np.array(
            [
                obs[0]["center"][0],
                obs[0]["center"][1],
                0,
                0,
                obs[0]["ellipse"][0],
                obs[0]["ellipse"][1],
                obs[0]["ellipse"][2],
            ]
        ),
    )

    Rseq = np.zeros((len(obs), 2, 2))
    for i, o in enumerate(obs):
        tracker.predict()
        tracker.update(get_obs(o))
        Rseq[i] = tracker.set_R(get_obs(o))

    plt.plot(Rseq[:, 0, 0], label="R00")
    plt.plot(Rseq[:, 1, 1], label="R11")
    plt.plot(Rseq[:, 1, 0], label="R10")
    plt.xlabel("time-steps")
    plt.grid()
    plt.legend()
    plt.show()
