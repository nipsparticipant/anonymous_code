from utils.utility import *

class CovFunction(nn.Module):
    def __init__(self, n_dim, c=None, ls=None, sig_opt=False, scale_opt=True):
        super(CovFunction, self).__init__()
        self.n_dim = n_dim
        self.device = get_cuda_device()
        if ls is None:  # all parameters are assumed to be stored in log form
            self.weights = nn.Parameter(ts(np.random.randn(self.n_dim, 1)).to(self.device),
                                        requires_grad=scale_opt)
        else:
            self.weights = nn.Parameter(ls, requires_grad = scale_opt)
        if c is None:
            self.sn = nn.Parameter(ts(0.0).to(self.device), requires_grad = sig_opt)
        else:
            self.sn = nn.Parameter(c, requires_grad=sig_opt)

    def forward(self, U, V = None):  # U is of size _ by n_dim, V is of size _ by n_dim
        if V is None:
            V = U
        scales = torch.exp(-1.0 * self.weights).float().view(1, -1)
        a = torch.sum((U * scales) ** 2, 1).reshape(-1, 1)
        b = torch.sum((V * scales) ** 2, 1) - 2 * torch.mm((U * scales), (V * scales).t())
        res = torch.exp(2.0 * self.sn) * torch.exp(-0.5 * (a.float() + b.float()))
        return res

class SpectralCov(CovFunction):
    def __init__(self, n_dim, eps=None, n_eps=100, c=None, ls=None):
        super(SpectralCov, self).__init__(n_dim, c, ls)
        if eps is None:
            if n_eps == 0:
                n_eps = 100
            self.n_eps = n_eps
            self.eps = torch.rand((n_dim, int(n_eps))).to(self.device)
        else:
            self.n_eps = eps.shape[1]  # each column of eps is a eps sample of size n_dim by 1
            self.eps = eps

    def phi(self, X):
        diag = torch.exp(-0.5 * self.weights.view(1, -1))
        phi = X * diag.float()
        phi = torch.mm(phi.float(), self.eps)
        res = torch.cat([torch.cos(phi), torch.sin(phi)], dim=1)
        return res * torch.exp(self.sn) / np.sqrt(self.n_eps)  # n_data by 2 * n_eps

    def forward(self, U, V = None):
        pu = self.phi(U)
        if V is None:
            return torch.mm(pu, pu.t())
        else:
            pv = self.phi(V)
            return torch.mm(pu, pv.t())


class MeanFunction(nn.Module):  # simply a constant mean function
    def __init__(self, c=None, mean_opt=True):
        super(MeanFunction, self).__init__()
        self.device = get_cuda_device()
        if c is None:
            self.mean = nn.Parameter(ts(0.0).to(self.device), requires_grad=mean_opt)
        else:
            self.mean = nn.Parameter(c, requires_grad = mean_opt)

    def forward(self, U):  # simply the constant mean function; input form: (_, x_dim)
        return torch.ones(U.size(0), 1).to(self.device) * self.mean


class LikFunction(nn.Module):  # return log likelihood of a Gaussian N(z | x, noise^2 * I)
    def __init__(self, c = None, noise_opt=True):
        super(LikFunction, self).__init__()
        self.device = get_cuda_device()
        if c is None:
            self.noise = nn.Parameter(ts(0.0).to(self.device), requires_grad=noise_opt)
        else:
            self.noise = nn.Parameter(c, requires_grad=noise_opt)

    def forward(self, o, x):
        diff = o - x
        n, d = o.size(0), o.size(1)
        res = -0.5 * n * d * torch.log(torch.tensor(2 * np.pi).to(self.device)) - 0.5 * n * self.noise \
                 -0.5 * torch.exp(-2.0 * self.noise) * torch.trace(torch.mm(diff.t(), diff))
        return res

