import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function


def ada_hermite(m, n, params, dtype=torch.float, device=None):
    """ada is a user-supplied function which computes the values and the derivatives of
    the function system matrix 'Phi'.
    ada_hermite computes the values and the derivatives of the classical Hermite functions
    parametrized by dilation and translation.

    Input
    ----------
    m: int
        Number of samples, i.e., row dimension of 'Phi'.
    n: int
        Number of basis functions, i.e., column dimension of 'Phi'.
    device: torch.device, optional
        the desired device of returned tensor. Default: None
    params: torch Tensor of floats
        nonlinear parameters of the basic functions, e.g., params = torch.tensor([dilation, translation])

    Output
    -------
    Phi: 2D torch Tensor of floats, whose [i,j] entry is equal to the jth basic function evaluated
        at the ith time instance t[i], e.g., each column of the matrix 'Phi' contains a sampling of the
        parametrized Hermite functions for a given 'params'.

    dPhi: 2D torch Tensor of floats, whose kth column contains the partial derivative of the jth basic function
        with respect to the ith nonlinear parameter, where j = ind[0,k] and i = ind[1,k],
        e.g., each column of the matrix 'dPhi' contains a sampling of the partial derivatives of the
         Hermite functions with respect to the dilation or to the translation parameter.

    ind: 2D torch Tensor of floats, auxiliary matrix for dPhi, i.e., column dPhi[:,k] contains
        the partial derivative of Phi[:,j]
        with respect to params[i], where j=ind[0,k] and i=ind[1,k],
        e.g., for the first three parametrized Hermite functions:
        ind = torch.tensor([[0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1]])
    """

    dilation, translation = params[:2]
    t = torch.arange(-(m // 2), m // 2 + 1, dtype=dtype) if m % 2 else torch.arange(-(m / 2), m / 2, dtype=dtype,
                                                                                    device=device)
    x = dilation * (t - translation * m / 2)
    w = torch.exp(-0.5 * x ** 2)
    dw = -x * w
    pi_sqrt = torch.sqrt(torch.sqrt(torch.tensor(math.pi, device=device)))

    # Phi, dPhi
    Phi = torch.zeros((m, n), dtype=dtype, device=device)
    Phi[:, 0] = 1
    Phi[:, 1] = 2 * x
    for j in range(1, n - 1):
        Phi[:, j + 1] = 2 * (x * Phi[:, j] - j * Phi[:, j - 1])

    Phi[:, 0] = w * Phi[:, 0] / pi_sqrt
    dPhi = torch.zeros(m, 2 * n, dtype=dtype, device=device)
    dPhi[:, 0] = dw / pi_sqrt
    dPhi[:, 1] = dPhi[:, 0]

    f = 1
    for j in range(1, n):
        f *= j
        Phi[:, j] = w * Phi[:, j] / \
            torch.sqrt(torch.tensor(2 ** j * f, dtype=dtype, device=device)) / pi_sqrt
        dPhi[:, 2 * j] = torch.sqrt(torch.tensor(2 * j, dtype=dtype, device=device)) * Phi[:, j - 1] - x * Phi[:, j]
        dPhi[:, 2 * j + 1] = dPhi[:, 2 * j]

    t = t[:, None]
    dPhi[:, 0::2] = dPhi[:, 0::2] * (t - translation * m / 2)
    dPhi[:, 1::2] = -dPhi[:, 1::2] * dilation * m / 2

    # ind
    ind = torch.zeros((2, 2 * n), dtype=torch.int64, device=device)
    ind[0, 0::2] = torch.arange(n, dtype=torch.int64, device=device)
    ind[0, 1::2] = torch.arange(n, dtype=torch.int64, device=device)
    ind[1, 0::2] = torch.zeros((1, n), dtype=torch.int64, device=device)
    ind[1, 1::2] = torch.ones((1, n), dtype=torch.int64, device=device)

    return Phi, dPhi, ind


class vpfun(Function):
    """Performs orthogonal projection, i.e. projects the input 'x' to the
    space spanned by the columns of 'Phi', where the matrix 'Phi' is provided by the 'ada' function.
    Then performs the inverse projection thus filtering the input 'x' through the spanned space.

    Input
    ----------
    x: torch Tensor of size (N,C,L) where
        N is the batch_size,
        C is the number of channels, and
        L is the number of signal samples
    params: torch Tensor of floats
          Contains the nonlinear parameters of the function system stored in Phi.
          For instance, if Phi(params) is provided by 'ada_hermite',
          then 'params' is a tensor of size (2,) that contains the dilation and the translation
          parameters of the Hermite functions.
    ada: callable
        Builder for the function system. For a given set of parameters 'params',
        it computes the matrix Phi(params) and its derivatives dPhi(params).
        For instance, in this package 'ada = ada_hermite' could be used.
    device: torch.device
             The desired device of the returned tensor(s).
    penalty: L2 regularization penalty that is added to the training loss.
              For instance, in the case of classification, the training loss is calculated as

                  loss = cross-entropy loss + penalty * ||x - projected_input||_2 / ||x||_2,

              where the projected_input is equal to the orthogonal projection of
              the 'x' to the columnspace of 'Phi(params)',
              i.e., projected_input =  Phi.mm( torch.linalg.pinv(Phi(params).mm(x) )

    Output
    -------
    projected_input: torch Tensor
             The filtered input signal:

                 projected_input =  Phi.mm( torch.linalg.pinv(Phi(params).mm(x) ),

             where coeffs = torch.linalg.pinv(Phi(params).mm(x)
    """

    @staticmethod
    def forward(ctx, x, params, ada, device, penalty):
        ctx.device = device
        ctx.penalty = penalty
        phi, dphi, ind = ada(params)
        phip = torch.linalg.pinv(phi)
        coeffs = phip @ torch.transpose(x, 1, 2)
        print(coeffs.shape)
        y_est = torch.transpose(phi @ coeffs, 1, 2)
        nparams = torch.tensor(max(params.shape))
        ctx.save_for_backward(x, phi, phip, dphi, ind, coeffs, y_est, nparams)

        return y_est

    @staticmethod
    def backward(ctx, dy):
        x, phi, phip, dphi, ind, coeffs, y_est, nparams = ctx.saved_tensors
        dx = dy @ (phi @ phip)

        batch = x.shape[0]
        channels = x.shape[1]
        jac1 = torch.zeros(
            batch, channels, phi.shape[0], nparams, dtype=x.dtype, device=ctx.device)

        jac = torch.zeros(
            batch, channels, phi.shape[0], nparams, dtype=x.dtype, device=ctx.device)

        for j in range(nparams):
            rng = ind[1, :] == j
            indrows = ind[0, rng]
            dphi_phip = dphi[:, rng] @ phip
            prod = dphi_phip - phi @ (phip @ dphi_phip)
            # jacobian (N, C, L, P)
            jac[:, :, :, j] = torch.transpose(
                (prod + torch.transpose(prod, -1, -2)) @ torch.transpose(x, 1, 2), 1, 2)
            # for l2 penalty
            jac1[:, :, :, j] = torch.transpose(dphi[:, rng] @ coeffs[:, indrows, :], 1, 2)  # (N,C,L)

        dy = dy.unsqueeze(-1)
        res = (x - y_est) / (x ** 2).sum(dim=2, keepdim=True)
        res = res.unsqueeze(-1)
        dp = (jac * dy).mean(dim=0).sum(dim=1) - 2 * \
            ctx.penalty * (jac1 * res).mean(dim=0).sum(dim=1)

        return dx, dp, None, None, None


class vp_layer(nn.Module):
    """Basic Variable Projection (VP) layer class.
    The output of a single VP operator is forwarded to the subsequent layers.

        Input
        ----------
        ada: callable
            Builder for the function system and its derivatives (see e.g., 'ada_hermite').
        n_in: int
            Input dimension of the VP layer.
        n_out: int
            Output dimension of the VP layer.
        nparams: int
            Number of trainable weights,
            e.g., nparams=2 in the case of 'ada_hermite' function.
        penalty: L2 regularization penalty that is added directily to the training loss (see e.g., 'vpfun').
            This can be intepreted as a skip connection from this layer to the cost function. Default: 0.0.
        device: torch.device. Default: None.
            The desired device of the returned tensor(s).
        init: a list of values to initialize the VP layer.
            Default for Hermite functions: init=[0.1, 0.0].
        """

    def __init__(self, ada, n_in, n_bases, nparams, penalty=0.0, dtype=torch.float32, device=None, init=None):
        if init is None:
            init = [0.1, 0.0]
        super().__init__()
        self.device = device
        self.n_in = n_in
        self.n_out = n_bases
        self.nparams = nparams
        self.penalty = penalty
        self.ada = lambda params: ada(n_in, n_bases, params, dtype=dtype, device=self.device)
        self.weight = nn.Parameter(torch.tensor(init))

    def forward(self, x):
        return vpfun.apply(x, self.weight, self.ada, self.device, self.penalty)
