from covs.kernel import *


class Sampler(nn.Module):
    def __init__(self, noise, n_dim):
        super(Sampler, self).__init__()
        self.device = get_cuda_device()
        self.noise = ts(noise).to(self.device)
        self.n_dim = n_dim
        self.cov = CovFunction(n_dim).to(self.device)

    def forward(self, X):
        pass

    def recursive_sampling(self, X, tolerance):
        m = X.shape[0]
        if m <= 192 * np.log(1.0 / tolerance):
            return torch.eye(m)

        selected = sample_rows(0.5 * np.ones(m), m)
        Xs = X[selected, :].view(-1, self.n_dim)
        Ss = torch.eye(m)[:, selected].view(m, -1).to(self.device)
        St = self.recursive_sampling(Xs, tolerance / 3.0)
        Sh = torch.mm(Ss, St)
        selected = torch.argmax(Sh, dim=1)
        Xh = X[selected, :].view(-1, self.n_dim)
        Khh = self.cov(Xh)
        phi = torch.inverse(Khh + torch.exp(2.0 * self.noise) * torch.eye(selected.shape[0]))
        l = np.zeros(m)
        for i in range(m):
            Kii = self.cov(X[i, :].view(1, -1))[0, 0]
            Kih = self.cov(X[i, :].view(1, -1), Xh)
            l[i] = Kii - torch.mm(Kih, torch.mm(phi, Kih.t()))[0, 0]
        scale = 16.0 * np.log(np.sum(l) / tolerance)
        p = np.minimum(np.ones(m), l * scale)
        selected = sample_rows(p, m)
        S = (1.0 / torch.sqrt(torch.tensor(p[selected]).view(1, -1))).float() * torch.eye(m)[:, selected].view(m, -1).float()
        return S.float()


class SGP(nn.Module):
    def __init__(self, n_dim):
        super(SGP, self).__init__()
        self.device = get_cuda_device()
        self.n_dim = n_dim
        self.cov = CovFunction(self.n_dim).to(self.device)
        self.mean = MeanFunction().to(self.device)
        self.lik = LikFunction().to(self.device)
        self.sampler = Sampler(self.lik.noise, self.n_dim).to(self.device)

    def compute_log_det_inv(self, X):
        chol_X = torch.cholesky(X, upper=False)
        chol_X_inv = torch.inverse(chol_X)
        X_inv = torch.mm(chol_X_inv.t(), chol_X_inv)
        logdet_X = 2.0 * torch.sum(torch.log(chol_X.diag()))
        return X_inv, logdet_X

    def NLL(self, X, Y, Xs=None):
        Kss, Kxs = self.nystrom(X, Xs=Xs)
        Yc = Y - self.mean.mean.float()
        I = torch.eye(X.shape[0]).to(self.device)
        Is = torch.eye(Kss.shape[0]).to(self.device)
        Kss = Kss + 0.01 * Is
        T = Kss + torch.exp(-2.0 * self.lik.noise) * torch.mm(Kxs.t(), Kxs)
        T_inv, logdet_T = self.compute_log_det_inv(T)
        Kss_inv, logdet_Kss = self.compute_log_det_inv(Kss)
        Q_inv = torch.exp(-2.0 * self.lik.noise) * I - torch.exp(-4.0 * self.lik.noise) * torch.mm(Kxs, torch.mm(T_inv, Kxs.t()))
        logdet_Q_inv = - logdet_T + logdet_Kss - 2.0 * X.shape[0] * self.lik.noise
        res = -0.5 * logdet_Q_inv + 0.5 * torch.mm(Yc.t(), torch.mm(Q_inv, Yc))
        return res

    def nystrom(self, X, Xs=None, Xt = None):
        if Xs is None:
            S = self.sampler.recursive_sampling(X, tolerance=0.1)
            selected = torch.argmax(S, dim=0)
            Xs = X[selected, :]
        Kss = self.cov(Xs)
        Kxs = self.cov(X, Xs)
        if Xt is None:
            return Kss, Kxs
        Kts = self.cov(Xt, Xs)
        return Kts, Kss, Kxs

    def predict(self, Xt, X, Y, Xs=None, var=False):
        with torch.no_grad():
            Yc = Y - self.mean.mean.float()
            Kts, Kss, Kxs = self.nystrom(X, Xs=Xs, Xt=Xt)
            Ksx = Kxs.t()
            T = torch.inverse(Kss + torch.exp(-2.0 * self.lik.noise) * torch.mm(Ksx, Kxs))
            Yt = self.mean.mean.float() + torch.exp(-2.0 * self.lik.noise) * torch.mm(Kts, torch.mm(T, torch.mm(Ksx, Yc)))
            if var is False:
                return Yt
            return Yt, None  # ignore predictive variance for now

    def forward(self, Xt, X, Y, Xs=None, grad=False, var=False):
        if grad is False:
            return self.predict(Xt, X, Y, var = var)
        Yc = Y - self.mean.mean.float()
        Kts, Kss, Kxs = self.nystrom(X, Xs=Xs, Xt=Xt)
        Ksx = Kxs.t()
        T = torch.inverse(Kss + torch.exp(-2.0 * self.lik.noise) * torch.mm(Ksx, Kxs))
        Yt = self.mean.mean.float() + torch.exp(-2.0 * self.lik.noise) * torch.mm(Kts, torch.mm(T, torch.mm(Ksx, Yc)))
        if var is False:
            return Yt
        return Yt, None  # ignore predictive variance for now
