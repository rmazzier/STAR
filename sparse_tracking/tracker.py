import time

import matplotlib
import matplotlib.patches as patches
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib.patches import Ellipse

from .cjpdaf import NN_CJPDAF
from .cs_stft import *
from .explicit_eot import KalmanTracker
from .md_extractor import *
from .utils import *

rc("text", usetex=True)
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.size"] = 14


class Tracker:
    def __init__(self, params, ap, sparse_md):
        self.sel_list = []
        self.tlist = []
        self.hist = []
        self.idlist = list(range(1000))
        self.timestep = 0
        self.ap = ap
        self.params = params
        self.current_positions = []
        self.sparse_md = sparse_md

    def tracking_step(self, meas, dt):
        slist, trlist = self._tstep(meas, dt)
        self.sel_list = slist
        self.tlist = trlist
        self.timestep += 1
        self.update_distbp()

    def _tstep(self, meas, dt):
        meas = np.nan_to_num(meas)
        # print(len(meas))

        if len(self.tlist) == 0:
            for z in meas:
                self.tlist.append(
                    KalmanTracker(
                        id_=self.idlist.pop(0),
                        birth_idx=self.timestep,
                        x0=np.array([z[0, 0], z[1, 0], 0, 0]),
                    )
                )
                self.tlist[-1].detection_history.append(1)
            self.sel_list = np.copy(self.tlist)

        else:
            # absolute threshold on the number of consecutive misses
            self.tlist = [
                t for t in self.tlist if t.misses_number < self.params["TRCK_DEL_THR"]
            ]
            # compute association scores with nn-cjpda
            nn_cjpdaf = NN_CJPDAF(trackers=self.tlist, b=self.params["b"])
            prob_matrix = nn_cjpdaf.compute_assoc(meas=meas)
            # set to zero very small values
            prob_matrix[prob_matrix < 1e-3] = 0
            # Hungarian alg. for lowest cost association
            matches, undet, unmatch = KalmanTracker.hungarian_assignment(
                1 - prob_matrix
            )
            # print(prob_matrix)

            # handle matched pairs (det, traj)
            if len(matches) > 0:
                for detec_idx, track_idx in matches:
                    obs = meas[detec_idx]
                    current_tracker = self.tlist[track_idx]
                    current_tracker.predict(dt)
                    current_tracker.update(obs, dt)
                    current_tracker.alive_steps.append(self.timestep)
                    current_tracker.detection_history.append(1)
                    # reset no. of misses
                    current_tracker.misses_number = 0
                    current_tracker.assoc_meas.append(obs)

            # deal with unmatched detections
            if len(unmatch) > 0:
                for idx in unmatch:
                    # extract kinematc and ext. shape from detection
                    obs = meas[idx]
                    # init new tracker
                    new_tracker = KalmanTracker(
                        id_=self.idlist.pop(0),
                        birth_idx=self.timestep,
                        x0=np.array([obs[0, 0], obs[1, 0], 0, 0]),
                    )
                    new_tracker.predict(dt)
                    self.tlist.append(new_tracker)

            # deal with undetected tracks
            if len(undet) > 0:
                for track_idx in undet:
                    old_tracker = self.tlist[track_idx]
                    old_tracker.detection_history.append(0)
                    old_tracker.misses_number += 1
                    old_tracker.predict(dt)
                    old_tracker.state_memory.append(old_tracker.s)
                    old_tracker.alive_steps.append(self.timestep)

            # avoid merging tracks killing the least consolidated one
            self.kill_merging_tracks()
            # self.tlist = [t for t in self.tlist if ((t.misses_number <= MAX_AGE) and (t.hits >= MIN_DET_NUMBER))]
            self.sel_list = [
                t
                for t in self.tlist
                if np.sum(t.detection_history[-self.params["N_THR"] :])
                >= self.params["M_THR"]
            ]

            for sl in self.sel_list:
                state = sl.s[:2].squeeze()
                ap_coords = (
                    self.params["AP_POS"]
                    if self.ap == 0
                    else self.params["AP_POS"] + np.array([[1.8, 0.0]])
                )
                dists = np.linalg.norm(state - ap_coords, axis=1)
                sl.best_AP.append(dists.argmin())

        return self.sel_list, self.tlist

    def update_distbp(self):
        for t in self.sel_list:
            t.current_dbp_index = self.ra2distbp_index(t)

    def ra2distbp_index(self, t):
        state = t.s[:2]
        ra = cart2polar(state[0, 0], state[1, 0])

        distbp = np.zeros_like(ra)
        rng, ang = ra
        range_diff = np.abs(self.params["DIST_VEC"] - rng)
        distbp[0] = np.argmin(range_diff)
        restr_bp_ang = np.copy(self.params["BP_ANGLES"])
        restr_bp_ang[:, :60] = np.inf
        restr_bp_ang[:, 180:] = np.inf
        ang_diff = np.abs(self.params["BP_ANGLES"] - ang)
        ang_idx = np.argmin(ang_diff)
        distbp[1] = np.argmax(self.params["BP_NORM"][:, ang_idx])
        return distbp.astype(int)

    def update_micro_doppler(self, tstep, chunk):
        for t in self.tlist:
            if t.current_dbp_index is not None:
                if self.sparse_md:
                    compute_mdspec_sparse(tstep, t, chunk, self.params)
                else:
                    compute_mdspec_us(tstep, t, chunk, self.params)

    def update_micro_range(self, chunk):
        for t in self.tlist:
            if t.current_dbp_index is not None:
                compute_mrspec(t, chunk, self.params)

    def update_hist(self):
        for sel in self.sel_list:
            if len(self.hist) == 0:
                self.hist.append(sel)
            else:
                ids_hist = [x.id for x in self.hist]
                if sel.id in ids_hist:
                    rem = [x for x in self.hist if x.id == sel.id][0]
                    self.hist.append(sel)
                    self.hist.remove(rem)
                else:
                    self.hist.append(sel)

    def kill_merging_tracks(self, n=5):
        for t1 in self.tlist:
            for t2 in self.tlist:
                if t1.id != t2.id:  # skip self-checks
                    d = np.linalg.norm(t1.s[:2] - t2.s[:2])
                    if d <= self.params["MERGE_DIST"]:
                        if len(t1.state_memory) >= n and len(t2.state_memory) >= n:
                            var1 = np.mean(
                                np.std(np.asarray(t1.state_memory[-n:]), axis=1)
                            )
                            var2 = np.mean(
                                np.std(np.asarray(t2.state_memory[-n:]), axis=1)
                            )
                            # var1 = np.linalg.det(t1.P)
                            # var2 = np.linalg.det(t2.P)
                            if var1 > var2:
                                self.tlist.remove(t1)
                            else:
                                self.tlist.remove(t2)


def plot(path, data_points, t_list, index, params):
    fig, ax = plt.subplots()
    plt.scatter(
        data_points[:, 0],
        data_points[:, 1],
        marker=".",
        alpha=0.5,
        label="Detected points",
    )
    plt.xlabel(r"$x$ [m]")
    plt.ylabel(r"$y$ [m]")
    plt.grid()
    plt.xlim([-4, 4])
    plt.ylim([0, 6])
    clist = ["r", "b", "g", "c", "m", "y", "k"] * 10
    prop_cycle = plt.rcself.Params["axes.prop_cycle"]
    clist = prop_cycle.by_key()["color"]
    for i in range(len(t_list)):
        cx = t_list[i].s[0]
        cy = t_list[i].s[1]

        plt.scatter(cx, cy, marker="o", c=clist[i])
        plt.text(cx, cy + 0.5, f" track {t_list[i].id}", fontdict={"color": clist[i]})

    plt.title(f"$k = {index}$")
    plt.legend(loc="upper right")
    # plt.savefig(path / f'pdf/figure_{index}.pdf')
    plt.savefig(path / f"figure_{index}.png")
    plt.close()
