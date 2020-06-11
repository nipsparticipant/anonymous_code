from experiments import *
from torch.nn.utils.clip_grad import clip_grad_value_


class VAEGP(nn.Module):
    def __init__(self, train, vae_cluster=8, embed_dim=4, gp_method='vaegp_32', batch_size=200):
        super(VAEGP, self).__init__()
        self.device = get_cuda_device()
        self.dataset = TensorDataset(train['X'], train['Y'])
        self.data = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        self.vae = MixtureVAE(self.data, train['X'].shape[1], embed_dim, vae_cluster, n_sample=50)
        self.original = train
        self.train = {'X': self.vae(train['X'], grad=False), 'Y': train['Y']}
        self.gp = Experiment.create_gp_object(self.original, gp_method)
        self.model = nn.ModuleList([self.vae.qz_x,
                                    self.vae.px_z,
                                    self.vae.qz_x.weights,
                                    self.gp.cov,
                                    self.gp.mean] + self.vae.qz_x.components)
        self.gp_model = nn.ModuleList([self.gp.cov, self.gp.mean])
        self.history = []

    def train_gp(self, seed=0, n_iter=100, n_epoch=50, lmbda=1.0, pred_interval=5, test=None, verbose=True):
        set_seed(seed)
        print('SEED=', seed)
        optimizer = opt.Adam(self.model.parameters())
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        pred_start = torch.cuda.Event(enable_timing=True)
        pred_end = torch.cuda.Event(enable_timing=True)
        pred_time = 0.0
        start.record()
        for i in range(n_iter):
            batch_nll = 0.0
            batch_elbo = 0.0
            batch_loss = 0.0
            epoch = 0
            for (X, Y) in self.data:
                if epoch > n_epoch:
                    break
                epoch += 1
                X.to(self.device)
                Y.to(self.device)
                Xr = self.vae(self.vae(X), encode=False)
                self.model.train()
                optimizer.zero_grad()
                delbo = self.vae.dsELBO(X, alpha=1.0, beta=1.0, gamma=1.0, verbose=False)
                nll = self.gp.NLL(Xr, Y)
                loss = - 0.01 * delbo + lmbda * nll
                batch_nll += nll * X.shape[0]
                batch_elbo += delbo * X.shape[0]
                batch_loss += loss * X.shape[0]
                loss.backward()
                clip_grad_value_(self.model.parameters(), 10)
                optimizer.step()
                torch.cuda.empty_cache()
            if i % pred_interval == 0:
                torch.cuda.synchronize()
                record = {'iter': i,
                          'nll': batch_nll.item() / self.train['X'].shape[0],
                          'elbo': batch_elbo.item() / self.train['X'].shape[0],
                          'loss': batch_loss.item() / self.train['X'].shape[0],
                          }
                if test is not None:
                    pred_start.record()
                    Xr = self.vae(self.vae(self.original['X'], grad=False), encode=False, grad=False)
                    Xtr = self.vae(self.vae(test['X'], grad=False), encode=False, grad=False)
                    Ypred= self.gp(Xtr, Xr, self.original['Y'])
                    record['rmse'] = rmse(Ypred, test['Y']).item()
                    pred_end.record()
                    torch.cuda.synchronize()
                    pred_time += pred_start.elapsed_time(pred_end)
                end.record()
                torch.cuda.synchronize()
                record['time'] = start.elapsed_time(end) - pred_time
                if verbose:
                    print(record)
                self.history.append(record)
        return self.history

    def forward(self, X):
        return self.gp(X, self.original['X'], self.original['Y'])

class GP_wrapper(nn.Module):
    def __init__(self, train, gp_method='ssgp_32', batch_size=200):
        super(GP_wrapper, self).__init__()
        self.device = get_cuda_device()
        self.train = train
        self.gp_method = gp_method
        self.gp = Experiment.create_gp_object(self.train, self.gp_method)
        self.dataset = TensorDataset(self.train['X'], self.train['Y'])
        self.data = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        self.model = nn.ModuleList([self.gp.cov, self.gp.mean])
        self.history = []

    def train_gp(self, seed=0, n_iter=300, pred_interval=5, test=None, verbose=True, n_epoch=20):
        set_seed(seed)
        print('SEED=', seed)
        optimizer = opt.Adam(self.model.parameters())
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        pred_start = torch.cuda.Event(enable_timing=True)
        pred_end = torch.cuda.Event(enable_timing=True)
        pred_time = 0.0
        start.record()
        for i in range(n_iter):
            batch_nll = 0.0
            epoch = 0
            for (X, Y) in self.data:
                if epoch > n_epoch:
                    break
                epoch += 1
                self.model.train()
                optimizer.zero_grad()
                nll = self.gp.NLL(X, Y)
                batch_nll += nll * X.shape[0]
                nll.backward()
                clip_grad_value_(self.model.parameters(), 10)
                optimizer.step()
                torch.cuda.empty_cache()
            if i % pred_interval == 0:
                record = {'nll': batch_nll.item() / self.train['X'].shape[0],
                          'iter': i}
                if test is not None:
                    pred_start.record()
                    Ypred = self.gp(test['X'], self.train['X'], self.train['Y'])
                    record['rmse'] = rmse(Ypred, test['Y']).item()
                    pred_end.record()
                    torch.cuda.synchronize()
                    pred_time += pred_start.elapsed_time(pred_end)
                end.record()
                torch.cuda.synchronize()
                record['time'] = start.elapsed_time(end) - pred_time
                if verbose:
                    print(record)
                self.history.append(record)
        return self.history


def deploy(snum, prefix, method, dataset, batch_size):
    if not os.path.isdir(prefix):
        os.mkdir(prefix)

    train, test = Experiment.load_data(dataset)
    seed = [int(snum)]
    res = dict()
    for s in seed:
        if 'vaegp' in method:
            vaegp = VAEGP(train, gp_method=method, batch_size=batch_size)
            try:
                res[s] = vaegp.train_gp(seed=s, n_iter=300, lmbda=1.0, pred_interval=10, test=test, verbose=True)
                torch.save(vaegp, prefix + str(s) + '_' + method + '.pth')
            except Exception as e:
                print(e)
                torch.save(vaegp, prefix + str(s) + '_' + method + '.pth')
        else:
            gp = GP_wrapper(train, gp_method=method, batch_size=batch_size)
            try:
                res[s] = gp.train_gp(seed=s, n_iter=300, pred_interval=10, test=test, verbose=True)
                torch.save(gp, prefix + str(s) + '_' + method + '.pth')
            except Exception as e:
                print(e)
                torch.save(gp, prefix + str(s) + '_' + method + '.pth')


if __name__ == '__main__':
    if len(sys.argv) < 3:
        #torch.cuda.set_device(1)
        deploy(1001, prefix='./results/gas', method='vaegp_32', dataset='gas10', batch_size=1000)
    else:
        torch.cuda.set_device(int(sys.argv[4]))
        deploy(sys.argv[5], sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[6]))

