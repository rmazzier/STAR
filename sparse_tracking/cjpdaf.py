import numpy as np
from scipy.stats import norm, multivariate_normal, chi2
from .utils import isNaN


class NN_CJPDAF():

    def __init__(self, trackers, b):

        self.trackers = trackers
        self.betas = None
        self.b = b

    def compute_assoc(self, meas):
        m_k = len(meas)
        T_k = len(self.trackers)

        G = np.zeros((m_k, T_k))
        for i in range(m_k):
            z = meas[i]
            for t in range(T_k):
                track = self.trackers[t]
                
                if len(z.shape) == 2:
                    ni = track.get_ni(z)
                else:
                    ni = track.get_ni(z[..., np.newaxis])
                G[i, t] = self.N(ni, np.zeros_like(ni), track.get_S())
        S_t = np.sum(G, axis=0)
        S_i = np.sum(G, axis=1)

        self.betas = np.zeros((m_k, T_k))
        for i in range(m_k):
            for t in range(T_k):
                # print(self.trackers[t].get_mahalanobis(meas[i]))
                # if self.trackers[t].get_mahalanobis(meas[i]) < 4.605:   # gating confidence 95%
                if self.trackers[t].get_distance(meas[i]) < 1.5:
                    val = G[i, t]/(S_t[t] + S_i[i] - G[i, t] + self.b)
                else:
                    val = 0
                self.betas[i, t] = 0.0 if isNaN(val) else val
                
        return self.betas

    def N(self, x, mu, Sigma):
        p = multivariate_normal(mean=mu.squeeze(), cov=Sigma.squeeze())
        prob = p.pdf(x.squeeze())
        return prob if prob >= 1e-8 else 1e-8





