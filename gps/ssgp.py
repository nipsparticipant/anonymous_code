from covs.kernel import *


class SSGP(nn.Module):
    def __init__(self, n_dim, n_eps, ls=None):
        super(SSGP, self).__init__()
        self.device = get_cuda_device()
        self.n_dim = n_dim
        self.n_eps = n_eps
        self.cov = SpectralCov(self.n_dim, n_eps=self.n_eps, ls=ls).to(self.device)
        self.mean = MeanFunction().to(self.device)
        self.lik = LikFunction().to(self.device)

    def NLL(self, X, Y):  # efficient computation of NLL -- this only costs nm^2 where m = 2 * n_eps
        Yc = Y.float() - self.mean.mean.float()
        Phi = self.cov.phi(X)
        w = torch.inverse(torch.mm(Phi.t(), Phi) + torch.exp(2.0 * self.lik.noise) *
                      torch.eye(2 * self.n_eps).to(self.device))
        w = torch.mm(w, torch.mm(Phi.t(), Yc))
        res = 0.5 * X.shape[0] * np.log(2.0 * np.pi) + X.shape[0] * self.lik.noise + \
              0.5 * torch.exp(-2.0 * self.lik.noise) * (torch.norm(Yc - torch.mm(Phi, w)) ** 2)
        return res

    def predict(self, Xt, X, Y, var = False):
        with torch.no_grad():
            Yc = Y.float() - self.mean.mean.float()
            Phi = self.cov.phi(X)
            w = torch.inverse(torch.mm(Phi.t(), Phi) + torch.exp(2.0 * self.lik.noise) *
                          torch.eye(2 * self.n_eps).to(self.device))
            w = torch.mm(w, torch.mm(Phi.t(), Yc))
            Yt = torch.mm(self.cov.phi(Xt), w) + self.mean.mean.float()
            if var is False:
                return Yt
            return Yt, None  # ignore predictive variance

    def forward(self, Xt, X, Y, grad=False, var=False):
        if grad is False:
            return self.predict(Xt, X, Y, var=var)
        Yc = Y.float() - self.mean.mean.float()
        Phi = self.cov.phi(X)
        w = torch.inverse(torch.mm(Phi.t(), Phi) + torch.exp(2.0 * self.lik.noise) *
                      torch.eye(2 * self.n_eps).to(self.device))
        w = torch.mm(w, torch.mm(Phi.t(), Yc))
        Yt = torch.mm(self.cov.phi(Xt), w) + self.mean.mean.float()
        if var is False:
            return Yt
        return Yt, None  # ignore predictive variance
