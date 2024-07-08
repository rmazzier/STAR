import torch
import os

# from torch.nn import MultiheadAttention
import numpy as np
import hyperparams as hpm

from md_extraction.utils_jp import (
    complex_to_real_matrix,
)

from md_extraction.sparsity_based import (
    partial_fourier,
)

"""
Summary of theory and notation from Gregor, Lecun:

Iteration step is:

Z[k+1] = h_theta * (matmul(W_e, X) + matmul(S, Z(k))), Z[0] = 0

X is the partial signal, aka the input of the network.
In the paper, matmul(W_e, X) = B.

Learnable parameters are
1) W_e = 1/L * W_d (Filter Matrix)
2) S = 1 - 1/L W_d.T @ W_d (Mutual inhibition Matrix)
3) theta (threshold of the soft thresholding operation)

Where:
    - W_d: is the (m x n) dictionary matrix
    - L controls the learning rate, and in theory, in classic ISTA, 
        it is required that L > max(eigenvalues(W_d.T @ W_d))

NB: In standard ISTA thresholds are set to theta = alpha/L, 
where alpha is the parameter controlling the sparsity regularization component weight.
"""


class SoftThr_Layer(torch.nn.Module):
    """Soft thresholding operation. Threshold is set as a learnable parameter."""

    def __init__(self, soft_approx=False):
        super(SoftThr_Layer, self).__init__()
        self.soft_approx = soft_approx

        thresholds = torch.clamp(
            torch.normal(
                0.0,
                1.0,
                # (hpm.NRANGE, 1, 1),
                (
                    # hpm.NRANGE,
                    1,
                    hpm.W * 2,
                    1,
                ),
            ),
            min=0,
        )
        self.thresholds = torch.nn.Parameter(thresholds)
        self.thresholds.requires_grad = True

        # To enforce threshold positivity
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        if self.soft_approx:
            return (
                torch.tanh(x + self.activation(self.thresholds))
                + torch.tanh(x - self.activation(self.thresholds))
            ) / 2
        else:
            return torch.sign(x) * torch.maximum(
                torch.abs(x) -
                self.activation(self.thresholds), torch.zeros_like(x)
            )


class HardThr_Layer(torch.nn.Module):
    """
    Hard thresholding operation.
    Sets all but the \Omega largest values of the tensor to 0.
    \Omega is set as a learnable parameter. (?)"""

    def __init__(self, omega=10):
        super(HardThr_Layer, self).__init__()
        self.Omega = omega

    def forward(self, x):
        w = x.size()[-2]
        inv_omega = w - self.Omega
        idxs = torch.topk(-torch.abs(x), inv_omega, dim=-2)
        x2 = x.clone()
        x2.scatter_(-2, idxs.indices, 0)
        # x2 = x2 / torch.norm(x2, dim=-2, keepdim=True)

        return x2


class STAR(torch.nn.Module):
    def __init__(
        self,
        n_iters,
        omega=10,
        learn_L=False,
        learn_S=False,
        learn_W=True,
        only_add=False,
        only_mult=False,
        init_W_d_as_fourier=False,
        use_attention=True,
        learn_attention=True,
        learn_W_transposed=False,
    ):

        super(STAR, self).__init__()
        self.only_add = only_add
        self.only_mult = only_mult
        self.n_iters = n_iters
        self.use_attention = use_attention
        self.learn_attention = learn_attention
        self.learn_W_transposed = learn_W_transposed
        if init_W_d_as_fourier:
            W_d_cpx = partial_fourier(hpm.W, np.arange(hpm.W))
            W_d = complex_to_real_matrix(W_d_cpx)
            W_d = torch.tensor(W_d).float().unsqueeze(0)
        else:
            W_d = torch.normal(
                0.0,
                0.5,
                (
                    1,
                    hpm.W * 2,
                    hpm.W * 2,
                ),
            )

        if learn_W:
            self.W_d = torch.nn.Parameter(W_d)
            self.W_d.requires_grad = True
        else:
            # self.W_d = W_d
            self.register_buffer("W_d", W_d)

        I = torch.unsqueeze(torch.eye(hpm.W * 2), 0)
        self.register_buffer("I", I)

        self.hard_threshold_layer = HardThr_Layer(omega=omega)

        if self.learn_attention:
            self.attn_layer = torch.nn.MultiheadAttention(hpm.W, 1)

        self.post_attn_layer_mult = torch.nn.Linear(hpm.W, hpm.W, bias=False)
        self.post_attn_layer_add = torch.nn.Linear(hpm.W, hpm.W, bias=True)

        if learn_L:
            self.L = torch.nn.Parameter(torch.tensor(hpm.L).float())
            self.L.requires_grad = True
        else:
            self.L = hpm.L

        # Initialize S if we want to learn it
        # otherwise compute it inside the forward pass
        if learn_S:
            S = self.I - 1 / self.L * torch.matmul(
                torch.transpose(self.W_d, 1, 2), self.W_d
            )
            self.S = torch.nn.Parameter(S)
            self.S.requires_grad = True

    def forward(self, x, prev_windows):
        # make x a column vector
        x = torch.unsqueeze(x, -1)

        if self.learn_W_transposed:
            B = torch.matmul(1 / self.L * torch.transpose(self.W_d, 1, 2), x)
        else:
            B = torch.matmul(1 / self.L * self.W_d, x)

        # S is equal to self.S if we want to learn it otherwise compute it
        # based on W_d
        if hasattr(self, "S"):
            S = self.S
        else:
            S = self.I - 1 / self.L * torch.matmul(
                torch.transpose(self.W_d, 1, 2), self.W_d
            )

        z = self.hard_threshold_layer(B)

        zs = []

        for _ in range(self.n_iters):
            c = B + torch.matmul(S, z)
            z = self.hard_threshold_layer(c)
            zs.append(z.squeeze())

        # convert to tensor
        zs = torch.stack(zs)

        # Compute the mD column from IHT output
        cpx_crop = zs[-1].reshape(zs.shape[1], 2, zs.shape[2] // 2).clone()
        cpx_crop[:, 1, :] *= -1

        p = torch.norm(cpx_crop, dim=1) ** 2
        # 3.2) Sum along range axis
        mD = p.sum(0)

        # min max normalize
        mD_normalized = (mD - mD.min()) / (mD.max() - mD.min() + 1e-8)

        # apply attention
        if self.use_attention:
            if len(prev_windows) > 0:

                # convert IHT outputs to mD columns
                cpx_prev_wins = prev_windows.reshape(
                    prev_windows.shape[0], prev_windows.shape[1], 2, zs.shape[2] // 2
                ).clone()
                cpx_prev_wins[:, :, 1, :] *= -1

                p = torch.norm(cpx_prev_wins, dim=2) ** 2
                # 3.2) Sum along range axis
                mD: torch.Tensor = p.sum(1)

                # min max normalize
                prev_windows_mD = (mD - mD.min(dim=1, keepdim=True)[0]) / (
                    mD.max(dim=1, keepdim=True)[0]
                    - mD.min(dim=1, keepdim=True)[0]
                    + 1e-8
                )

                if self.learn_attention:
                    # compute keys, values and queries
                    query = mD_normalized.unsqueeze(0).unsqueeze(1)
                    keys = prev_windows_mD.unsqueeze(1)
                    values = prev_windows_mD.unsqueeze(1)
                    attn_output, attn_output_weights = self.attn_layer(
                        query, keys, values, need_weights=True
                    )
                else:
                    att_weights = torch.matmul(
                        prev_windows_mD, mD_normalized.unsqueeze(1)
                    )
                    att_weights = torch.nn.Softmax(dim=0)(
                        att_weights / np.sqrt(mD_normalized.shape[0])
                    )
                    attn_output = torch.matmul(
                        torch.transpose(prev_windows_mD, 0, 1), att_weights
                    )
                    attn_output_weights = att_weights
                attn_output_weights = attn_output_weights.squeeze()

                # # TODO: THIS IS A TEMPORARY LOG FOR SHOWING SPARSITY OF ATTENTION OUTPUT
                # out_attn_path = "./results/attn_output_test_logs"
                # os.makedirs(out_attn_path, exist_ok=True)
                # if len(os.listdir(out_attn_path)) == 0:
                #     i = 0
                # else:
                #     i = len(os.listdir(out_attn_path))
                # torch.save(
                #     attn_output, os.path.join(out_attn_path, f"test_attn_output_{i}.pt")
                # )

                # Filtering part of the model
                if self.only_add:
                    attn_output_add = torch.nn.ReLU()(
                        self.post_attn_layer_add(attn_output.squeeze())
                    )

                    mD_out = mD_normalized + attn_output_add.squeeze()

                elif self.only_mult:
                    attn_output_mult = torch.nn.Sigmoid()(
                        self.post_attn_layer_mult(attn_output.squeeze())
                    )

                    mD_out = mD_normalized * attn_output_mult.squeeze()

                else:
                    attn_output_mult = torch.nn.Sigmoid()(
                        self.post_attn_layer_mult(attn_output.squeeze())
                    )
                    attn_output_add = torch.nn.ReLU()(
                        self.post_attn_layer_add(attn_output.squeeze())
                    )

                    mD_out = (
                        mD_normalized + attn_output_add.squeeze()
                    ) * attn_output_mult.squeeze()

                    mD_out = (mD_out - mD_out.min()) / (
                        mD_out.max() - mD_out.min() + 1e-8
                    )

            else:
                mD_out = mD_normalized
                attn_output_weights = None
        else:
            mD_out = mD_normalized
            attn_output_weights = None

        # mD_out = self.hard_threshold_layer(mD_out.unsqueeze(0).unsqueeze(-1)).squeeze()

        return mD_out, zs[-1]


class DUST(torch.nn.Module):
    """Transformer model from  :
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10094712

    How it works:
    0) First, one single LIHT iteration to obtain sequence of zs (our mD columns);
    1) Self attention module where there is no extra learnable matrix involved (i.e. just dot product
    between all the different zs (they do it with (W_d * z) but our setup is slightly different))
    (also, they don't seem to use scaled dot product attention)
    2) Use output of (1) as initialization for simple LIHT layer with W_d and S learned,
    applied individually to each z
    """

    def __init__(
        self,
        n_iters,
        omega=10,
    ):

        super(DUST, self).__init__()
        self.n_iters = n_iters
        W_d_cpx = partial_fourier(hpm.W, np.arange(hpm.W))
        W_d = complex_to_real_matrix(W_d_cpx)
        W_d = torch.tensor(W_d).float().unsqueeze(0)

        self.W_d = torch.nn.Parameter(W_d)
        self.W_d.requires_grad = True

        # self.V = torch.nn.Parameter(1 / self.L * torch.transpose(self.W_d, 1, 2))
        # self.V.requires_grad = True

        self.lambda2 = torch.nn.Parameter(torch.tensor(1.0))
        self.lambda2.requires_grad = True

        I = torch.unsqueeze(torch.eye(hpm.W * 2), 0)
        self.register_buffer("I", I)

        self.hard_threshold_layer = HardThr_Layer(omega=omega)

        self.L = torch.nn.Parameter(torch.tensor(hpm.L).float())
        self.L.requires_grad = False

        # Initialize S
        S = self.I - 1 / self.L * torch.matmul(
            torch.transpose(self.W_d, 1, 2), self.W_d
        )
        self.S = torch.nn.Parameter(S)
        self.S.requires_grad = True

    def forward(self, x, prev_windows):
        # make x a column vector
        x = torch.unsqueeze(x, -1)

        B = torch.matmul(1 / self.L * self.W_d, x)
        S = self.S
        z = self.hard_threshold_layer(B)

        # 1 iteration of LIHT to get the current z
        c = B + torch.matmul(S, z)
        z = self.hard_threshold_layer(c)

        # apply attention
        if len(prev_windows) > 0:

            # get reconstructed signal from current z (out shape = (10, 128))
            s_star_t = torch.matmul(
                torch.transpose(self.W_d, 1, 2), z).squeeze()

            s_star_t_prev = torch.matmul(
                self.W_d.unsqueeze(0), prev_windows.unsqueeze(-1)
            ).squeeze(-1)

            # Att weights (shape = (n_wins, 10))
            att_weights = torch.matmul(
                s_star_t_prev, torch.transpose(s_star_t, 0, 1)
            ).squeeze(-1)
            att_weights = torch.diagonal(att_weights, dim1=1, dim2=2)
            att_weights = torch.nn.Softmax(dim=1)(att_weights)
            # attn_output = s_star_t_prev * att_weights.unsqueeze(-1)

            # clamp prev_windows between -15 and 15 to avoid nan stuff
            prev_windows = torch.clamp(prev_windows, -150, 150)

            attn_output = prev_windows * att_weights.unsqueeze(-1)
            attn_output = attn_output.sum(dim=0).unsqueeze(-1) * self.lambda2

            attn_output_weights = att_weights
            # plt.show()
        else:
            attn_output_weights = None
            attn_output = z

        # Actual IHT iteration with initialization from attention
        zs = []
        z = attn_output
        for _ in range(self.n_iters):
            c = B + torch.matmul(S, z)
            z = self.hard_threshold_layer(c)
            zs.append(z.squeeze())
        zs = torch.stack(zs)

        # Compute the mD column from IHT output
        cpx_crop = zs[-1].reshape(zs.shape[1], 2, zs.shape[2] // 2).clone()
        cpx_crop[:, 1, :] *= -1

        p = torch.norm(cpx_crop, dim=1) ** 2
        # 3.2) Sum along range axis
        mD = p.sum(0)

        # min max normalize
        mD_normalized = (mD - mD.min()) / (mD.max() - mD.min() + 1e-8)

        return mD_normalized, zs[-1]


class DUST_V2(torch.nn.Module):
    """Same of dust, but attention is applied in the frequency domain like in LIHT"""

    def __init__(
        self,
        n_iters,
        omega=10,
    ):

        super(DUST_V2, self).__init__()
        self.n_iters = n_iters
        W_d_cpx = partial_fourier(hpm.W, np.arange(hpm.W))
        W_d = complex_to_real_matrix(W_d_cpx)
        W_d = torch.tensor(W_d).float().unsqueeze(0)

        self.W_d = torch.nn.Parameter(W_d)
        self.W_d.requires_grad = True

        # self.V = torch.nn.Parameter(1 / self.L * torch.transpose(self.W_d, 1, 2))
        # self.V.requires_grad = True

        self.lambda2 = torch.nn.Parameter(torch.tensor(1.0))
        self.lambda2.requires_grad = True

        I = torch.unsqueeze(torch.eye(hpm.W * 2), 0)
        self.register_buffer("I", I)

        self.hard_threshold_layer = HardThr_Layer(omega=omega)

        self.L = torch.nn.Parameter(torch.tensor(hpm.L).float())
        self.L.requires_grad = False

        # Initialize S
        S = self.I - 1 / self.L * torch.matmul(
            torch.transpose(self.W_d, 1, 2), self.W_d
        )
        self.S = torch.nn.Parameter(S)
        self.S.requires_grad = True

    def forward(self, x, prev_windows):
        # make x a column vector
        x = torch.unsqueeze(x, -1)

        B = torch.matmul(1 / self.L * self.W_d, x)
        S = self.S
        z = self.hard_threshold_layer(B)

        # 1 iteration of LIHT to get the current z
        c = B + torch.matmul(S, z)
        z = self.hard_threshold_layer(c)

        # Compute the mD column from IHT output
        cpx_crop = z.reshape(z.shape[0], 2, z.shape[1] // 2).clone()
        cpx_crop[:, 1, :] *= -1

        p = torch.norm(cpx_crop, dim=1) ** 2
        # 3.2) Sum along range axis
        mD = p.sum(0)

        # min max normalize
        mD_normalized = (mD - mD.min()) / (mD.max() - mD.min() + 1e-8)

        # apply attention
        if len(prev_windows) > 0:

            # convert IHT outputs to mD columns
            cpx_prev_wins = prev_windows.reshape(
                prev_windows.shape[0], prev_windows.shape[1], 2, z.shape[1] // 2
            ).clone()
            cpx_prev_wins[:, :, 1, :] *= -1

            p = torch.norm(cpx_prev_wins, dim=2) ** 2
            # 3.2) Sum along range axis
            mD: torch.Tensor = p.sum(1)

            # min max normalize
            prev_windows_mD = (mD - mD.min(dim=1, keepdim=True)[0]) / (
                mD.max(dim=1, keepdim=True)[0] -
                mD.min(dim=1, keepdim=True)[0] + 1e-8
            )

            att_weights = torch.matmul(
                prev_windows_mD, mD_normalized.unsqueeze(1))
            att_weights = torch.nn.Softmax(dim=0)(
                att_weights / np.sqrt(mD_normalized.shape[0])
            )

            # clamp prev_windows between -15 and 15 to avoid nan stuff
            prev_windows = torch.clamp(prev_windows, -150, 150)

            attn_output = prev_windows * att_weights.unsqueeze(-1)
            attn_output = attn_output.sum(dim=0).unsqueeze(-1) * self.lambda2

            # attn_output = torch.matmul(
            #     torch.transpose(prev_windows_mD, 0, 1), att_weights
            # )
            attn_output_weights = att_weights.squeeze()

        else:
            attn_output = z
            attn_output_weights = None

        # Actual IHT iteration with initialization from attention
        zs = []
        z = attn_output
        for _ in range(self.n_iters):
            c = B + torch.matmul(S, z)
            z = self.hard_threshold_layer(c)
            zs.append(z.squeeze())
        zs = torch.stack(zs)

        # Compute the mD column from IHT output
        cpx_crop = zs[-1].reshape(zs.shape[1], 2, zs.shape[2] // 2).clone()
        cpx_crop[:, 1, :] *= -1

        p = torch.norm(cpx_crop, dim=1) ** 2
        # 3.2) Sum along range axis
        mD = p.sum(0)

        # min max normalize
        mD_normalized = (mD - mD.min()) / (mD.max() - mD.min() + 1e-8)

        return mD_normalized, zs[-1]


def init_model(cfg):
    if cfg["MODEL_TYPE"] == "LIHT":
        model = STAR(
            n_iters=cfg["N_LIHT_ITERS"],
            omega=cfg["LIHT_OMEGA"],
            learn_L=False,
            learn_S=cfg["LEARN_LIHT_S"],
            learn_W=cfg["LEARN_W"],
            only_add=cfg["ONLY_ADD"],
            only_mult=cfg["ONLY_MULT"],
            init_W_d_as_fourier=cfg["INIT_W_D_AS_FOURIER"],
            use_attention=cfg["USE_ATTENTION"],
            learn_attention=cfg["LEARN_ATTENTION"],
            learn_W_transposed=cfg["LEARN_W_TRANSPOSED"],
        ).to(hpm.DEVICE)
    elif cfg["MODEL_TYPE"] == "DUST":
        model = DUST(
            n_iters=cfg["N_LIHT_ITERS"],
            omega=cfg["LIHT_OMEGA"],
        ).to(hpm.DEVICE)
    elif cfg["MODEL_TYPE"] == "DUST_V2":
        model = DUST_V2(
            n_iters=cfg["N_LIHT_ITERS"],
            omega=cfg["LIHT_OMEGA"],
        ).to(hpm.DEVICE)

    return model
