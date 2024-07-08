import numpy as np
from sklearn.linear_model import OrthogonalMatchingPursuit
from scipy.linalg import dft
from matplotlib import pyplot as plt
import cvxpy as cvx

class cs_stft():

    def __init__(self, s_win, mask, last_h, sparsity=3, solver='convex'):
        # for j, el in enumerate(s_win):
        #     if np.abs(el) < 1e-8:
        #         if mask[j]:
        #             print(mask[j], el)
        #             print('error')
        self.s_win = s_win
        # print(s_win[mask])
        self.N = len(self.s_win)
        # self.keep_idx = np.random.choice(np.arange(self.N), 
        #                                  size=no_meas, 
        #                                  replace=False)
        self.keep_idx = np.argwhere(mask)
        # self.keep_idx.sort()
        self.meas = self.s_win[self.keep_idx].squeeze()
        self.psi = self.partial_fourier(self.N) 
        self.H = None
        self.solver = solver
        self.last_h = last_h
        self.sparsity = sparsity

    def solve(self):
        if self.solver == 'convex':
            self.H = self.convex_solver(self.psi, self.meas)
        elif self.solver == 'iht':
            self.H = self.iht(psi=self.psi, y=self.meas, s=self.sparsity)
        elif self.solver == 'cosamp':
            self.H = self.cosamp(self.psi, self.meas)
        elif self.solver == 'omp':
            self.H = self.omp(self.psi, self.meas)
        elif self.solver == 'ls_cs':
            self.H = self.ls_cs(self.psi, self.meas)
        elif self.solver == 'mcs':
            self.H = self.mcs(self.psi, self.meas)
        return self.H

    def partial_fourier(self, N):
        F = np.conj(dft(N, scale='sqrtn'))
        F_part = F[self.keep_idx] #* np.sqrt(N / len(self.keep_idxs))
        return F_part.squeeze()

    def convex_solver(self, psi, meas):
        lmbda = 3
        H = cvx.Variable(self.N, complex=True)
        objective = cvx.Minimize(lmbda*cvx.norm(H, 1) + 0.5*cvx.norm(psi @ H - meas, 2)**2)
        prob = cvx.Problem(objective)
        result = prob.solve(verbose=False)
        return H.value

    # def convex_solver(self, psi, meas):
    #     epsilon = 3e-1
    #     norm_epsilon = epsilon * np.linalg.norm(meas)
    #     H = cvx.Variable(self.N, complex=True)
    #     objective = cvx.Minimize(cvx.norm(H, 1))
    #     constraints = [cvx.norm(psi @ H - meas, 2) <= norm_epsilon]   # residual error
    #     prob = cvx.Problem(objective, constraints)
    #     result = prob.solve(verbose=False)
    #     return H.value

    def mcs(self, psi, meas):
        alpha = 0.01
        alpha_rej = alpha
        gamma = 0.5

        if self.last_h is None:
            out = self.convex_solver(psi, meas)
            return out
        last_support = self.est_support(self.last_h, alpha).squeeze()
        if len(last_support) == 0:
            out = self.convex_solver(psi, meas)
            return out
        else:
            comp_last_support = list(set(np.arange(self.N)) - set(last_support))
            epsilon = 3e-1
            norm_epsilon = epsilon * np.linalg.norm(meas)
            H = cvx.Variable(self.N, complex=True)
            objective = cvx.Minimize(cvx.norm(H[comp_last_support], 1) 
                                        + gamma * cvx.norm(H[last_support] - self.last_h[last_support], 2)**2)
            constraints = [cvx.norm(psi @ H - meas, 2) <= norm_epsilon]   # residual error
            prob = cvx.Problem(objective, constraints)
            result = prob.solve(verbose=False)
            return H.value

    def ls_cs(self, psi, meas):
        alpha = 0.05
        alpha_rej = alpha
        if self.last_h is None:
            out = self.convex_solver(psi, meas)
            return out
        elif (self.last_h == 0).all():
            out = self.convex_solver(psi, meas)
            return out
        else:
            last_support = self.est_support(self.last_h, alpha).squeeze()
            x_init = np.zeros(self.N, dtype=np.complex128)
            x_init[last_support], _, _, _ = np.linalg.lstsq(psi[:, last_support], meas)
            yt_res = meas - psi @ x_init
            beta_t = self.convex_solver(psi, yt_res)
            xt_csres = beta_t + x_init
            adds = self.est_support(xt_csres, alpha)
            T_tilde = np.array(list(set(last_support).union(set(adds))), dtype=int)
            x_det = np.zeros(self.N, dtype=np.complex128)
            x_det[T_tilde], _, _, _ = np.linalg.lstsq(psi[:, T_tilde], yt_res)
            Nt = np.array(list(set(T_tilde) - set(self.est_support(x_det, alpha_rej, mode='lower'))), dtype=int)
            x_hat = np.zeros(self.N, dtype=np.complex128)
            x_hat[Nt], _, _, _ = np.linalg.lstsq(psi[:, Nt], meas)
            return x_hat

    def est_support(self, h, alpha, mode='greater'):
        if mode == 'greater':
            idx = np.argwhere(np.abs(h) > alpha)
        elif mode == 'lower':
            idx = np.argwhere(np.abs(h) <= alpha)
        return idx.squeeze()

    # def hard_thresholding(self, vector, sparsity_level):
    #     tozero = np.argpartition(np.abs(vector), -sparsity_level)[:self.N - sparsity_level]
    #     vector[tozero] = complex(0, 0)
    #     return vector

    def hard_thresholding(self, vector, sparsity_level):
        max_idx = np.argmax(np.abs(vector))
        # not_tozero = np.array([max_idx - 1, max_idx, max_idx + 1])
        not_tozero = np.arange(start=max_idx - int(sparsity_level//2), stop=max_idx + int(sparsity_level//2))
        not_tozero = not_tozero[np.logical_and(not_tozero >= 0, not_tozero < self.N)]
        new_vector = np.zeros_like(vector, dtype=complex)
        new_vector[not_tozero] = vector[not_tozero]
        return new_vector

    def iht(self, psi, y, s=1, mu=1, maxit=300, eps=0.1, 
            change_conv=1e-4, tau=0.99, momentum=False):
        norm_eps = eps * np.linalg.norm(y)
        it = 0
        end = False
        z_old = np.zeros(self.N, dtype=np.complex128)
        x_old = np.zeros(self.N, dtype=np.complex128)
        z = np.zeros(self.N, dtype=np.complex128)
        residuals = []
        while not end:
            it += 1
            if momentum:
                x_new = z + mu * np.conj(psi).T @ (y - psi @ z)
                x_new = self.hard_thresholding(x_new, sparsity_level=s)
                temp = psi @ x_new - psi @ x_old
                tau = np.abs((y - psi @ x_new).dot(temp)) / np.linalg.norm(temp)**2
                z = x_new + tau * (x_new - x_old)
                x_old = np.copy(x_new)
                z_old = np.copy(z)
            else:
                z += mu * np.conj(psi).T @ (y - psi @ z)
                z = self.hard_thresholding(z, sparsity_level=s)
                change = np.linalg.norm(z - z_old)
                z_old = np.copy(z)
            end_maxit = it >= maxit 
            residuals.append(np.linalg.norm(psi @ z - y))
            end_converged_error = np.linalg.norm(psi @ z - y) < norm_eps
            end_converged_change = change < change_conv * np.linalg.norm(z_old)
            end = end_maxit or end_converged_change

        # print(f'End condition {"maxit" if end_maxit else "converged"}')
        # print(it, residuals)
        # plt.plot(np.array(residuals)/np.linalg.norm(y))
        # plt.ylim([0, 1])
        # plt.show()
        return z


    # def norm_iht(self, psi, y, s=5, mu=1, maxit=300, eps=0.3, block_sparse=False):
    #     norm_eps = eps * np.linalg.norm(y)
    #     Ti = np.argsort(np.abs(np.conj(self.psi.T) @ self.meas))[:self.N - s]
    #     it = 0
    #     end = False
    #     x = np.zeros(self.N, dtype=np.complex128)
    #     while not end:
    #         it += 1
    #         gi = np.conj(self.psi.T) @ (self.meas - self.psi @ x)
    #         mui = np.linalg.norm(gi[Ti])**2 / np.linalg.norm(self.psi[:, Ti] @ gi[Ti])**2
    #         x += mui * gi
    #         tozero = np.argsort(np.abs(x))[:self.N - s]
    #         x[tozero] = complex(0, 0)
    #         Ti = np.argwhere(np.abs(x) != complex(0, 0))




    #         x += mu * np.conj(psi).T @ (y - psi @ x)
    #         tozero = np.argsort(np.abs(x))[:self.N - s]
    #         x[tozero] = complex(0, 0)

    #         end_maxit = it >= maxit 
    #         end_converged = np.linalg.norm(psi @ x - y) < norm_eps
    #         end = end_maxit or end_converged
    #     return x

    def block_sparse_pruning(self, x, K=4, J=8):
        dim = len(x) // J
        X = np.reshape(x, (dim, J))
        norms = np.linalg.norm(X, axis=1)
        idx_tz = np.argsort(norms)[:len(norms) - K]
        X[idx_tz, :] = complex(0, 0)
        out = np.reshape(X, X.shape[0] * X.shape[1])
        return out

    def cosamp(self, phi, u, s=20, epsilon=0.05, max_iter=100):
    
        a = np.zeros(phi.shape[1], dtype=np.complex128)
        v = u
        it = 0 # count
        halt = False
        while not halt:
            it += 1            
            y = np.dot(np.transpose(phi), v)
            omega = np.argsort(np.abs(y))[-(2*s):] # large components
            omega = np.union1d(omega, a.nonzero()[0]) # use set instead?
            phiT = phi[:, omega]
            b = np.zeros(phi.shape[1], dtype=np.complex128)
            # Solve Least Square
            b[omega], _, _, _ = np.linalg.lstsq(phiT, u)
            
            # Get new estimate
            b[np.argsort(b)[:-s]] = complex(0, 0)
            a = b
            
            # Halt criterion
            v_old = v
            v = u - np.dot(phi, a)

            halt = (np.linalg.norm(v - v_old) < epsilon*np.linalg.norm(u)) or \
                np.linalg.norm(v) < epsilon*np.linalg.norm(u) or \
                it > max_iter
        return a

    def omp(self, psi, y, ncoef=5, maxit=30, tol=0.05, ztol=0.005):
        active = []
        coef = np.zeros(psi.shape[1], dtype=np.complex128) 
        residual = y
        ynorm = np.linalg.norm(y) #/ np.sqrt(len(y))
        err = np.zeros(maxit, dtype=float) 

        tol = tol * ynorm     
        ztol = ztol * ynorm  

        for it in range(maxit):
            rcov = np.abs(psi.T @ residual)
            i = np.argmax(rcov)
            # rc = np.abs(rcov[i])
            # if rc < ztol:
            #     print(4)
            #     break

            if i not in active:
                active.append(i)
                
            coefi, _, _, _ = np.linalg.lstsq(psi[:, active], y)
            coef[active] = coefi 
            # print(len(active))

            residual = y - np.dot(psi[:,active], coefi)
            err[it] = np.linalg.norm(residual) / (np.sqrt(len(residual)) * ynorm)

            if err[it] < tol: 
                # print(1)
                break
            if len(active) >= ncoef: 
                # print(2)
                break
            if it == maxit-1: 
                # print(3)
                break
        # plt.plot(err)
        # plt.show()
        return coef

    def bomp(self, psi, y, nblocks=16, block_dim=4, tol=0.001, ztol=1e-4):
        active = []
        x = np.zeros(psi.shape[1], dtype=np.complex128) 
        x_block = np.stack(np.split(x, nblocks), axis=0)[..., np.newaxis]
        psi_block = np.stack(np.split(psi, nblocks, axis=1), axis=0)
        residual = y[..., np.newaxis]
        ynorm = np.linalg.norm(y) / np.sqrt(len(y))
        err = np.zeros(nblocks, dtype=float) 

        tol = tol * ynorm     
        ztol = ztol * ynorm  

        for it in range(nblocks):
            # rcov = np.dot(np.conj(psi.T), residual)
            rcov = np.transpose(np.conj(psi_block), axes=(0, 2, 1)) @ residual
            i = np.argmax(np.linalg.norm(rcov, axis=1))
            # rc = np.abs(rcov[i])
            # if rc < ztol:
            #     break
            
            flat_idxs = np.arange(i * block_dim, (i + 1) * block_dim)
            for k in flat_idxs:
                if k not in active:
                    active.append(k)
            
            xi, _, _, _ = np.linalg.lstsq(psi[:, active], y) 
            x[active] = xi 

            residual = y - np.dot(psi[:,active], xi)
            err[it] = np.linalg.norm(residual) / (np.sqrt(len(residual)) * ynorm)

            if err[it] < tol: 
                break
            if len(active) >= nblocks * block_dim: 
                break
        return x

if __name__ == "__main__":
    
    mask = np.ones(64)
    test = cs_stft(np.random.normal(size=64), mask, np.zeros(64))

    F = test.psi
    print(F.shape)
    inner = []
    for i in range(64):
        for j in range(64):
            # if i != j:
            v = F[:, i]
            u = F[:, j]
            inner.append(v.dot(u))

    plt.plot(np.abs(inner))
    plt.show()
