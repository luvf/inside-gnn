
class GraphLIME:
    """ GraphLIME explainer - code taken from original repository
    Explains only node features
    """

    def __init__(self, data, model, gpu=False, hop=2, rho=0.1, cached=True):
        self.data = data
        self.model = model
        self.hop = hop
        self.rho = rho
        self.cached = cached
        self.cached_result = None
        self.M = self.data.num_features
        self.F = self.data.num_features
        self.gpu = gpu

        self.model.eval()

    def __flow__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                return module.flow
        return 'source_to_target'

    def __subgraph__(self, node_idx, x, y, edge_index, **kwargs):
        num_nodes, num_edges = x.size(0), edge_index.size(1)

        subset, edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx, self.hop, edge_index, relabel_nodes=True,
            num_nodes=num_nodes, flow=self.__flow__())

        x = x[subset]
        y = y[subset]

        for key, item in kwargs:
            if torch.is_tensor(item) and item.size(0) == num_nodes:
                item = item[subset]
            elif torch.is_tensor(item) and item.size(0) == num_edges:
                item = item[edge_mask]
            kwargs[key] = item

        return x, y, edge_index, mapping, edge_mask, kwargs

    def __init_predict__(self, x, edge_index, **kwargs):
        if self.cached and self.cached_result is not None:
            if x.size(0) != self.cached_result.size(0):
                raise RuntimeError(
                    'Cached {} number of nodes, but found {}.'.format(
                        x.size(0), self.cached_result.size(0)))

        if not self.cached or self.cached_result is None:
            # Get the initial prediction.
            with torch.no_grad():
                if self.gpu:
                    device = torch.device(
                        'cuda' if torch.cuda.is_available() else 'cpu')
                    self.model = self.model.to(device)
                    log_logits = self.model(
                        x=x.cuda(), edge_index=edge_index.cuda(), **kwargs)
                else:
                    log_logits = self.model(
                        x=x, edge_index=edge_index, **kwargs)
                probas = log_logits.exp()

            self.cached_result = probas

        return self.cached_result

    def __compute_kernel__(self, x, reduce):
        assert x.ndim == 2, x.shape

        n, d = x.shape

        dist = x.reshape(1, n, d) - x.reshape(n, 1, d)  # (n, n, d)
        dist = dist ** 2

        if reduce:
            dist = np.sum(dist, axis=-1, keepdims=True)  # (n, n, 1)

        std = np.sqrt(d)

        # (n, n, 1) or (n, n, d)
        K = np.exp(-dist / (2 * std ** 2 * 0.1 + 1e-10))

        return K

    def __compute_gram_matrix__(self, x):
        # unstable implementation due to matrix product (HxH)
        # n = x.shape[0]
        # H = np.eye(n, dtype=np.float) - 1.0 / n * np.ones(n, dtype=np.float)
        # G = np.dot(np.dot(H, x), H)

        # more stable and accurate implementation
        G = x - np.mean(x, axis=0, keepdims=True)
        G = G - np.mean(G, axis=1, keepdims=True)

        G = G / (np.linalg.norm(G, ord='fro', axis=(0, 1), keepdims=True) + 1e-10)

        return G

    def explain(self, node_index, hops, num_samples, info=False, multiclass=False, *unused, **kwargs):
        # hops, num_samples, info are useless: just to copy graphshap pipeline
        x = self.data.x
        edge_index = self.data.edge_index

        probas = self.__init_predict__(x, edge_index, **kwargs)

        x, probas, _, _, _, _ = self.__subgraph__(
            node_index, x, probas, edge_index, **kwargs)

        x = x.cpu().detach().numpy()  # (n, d)
        y = probas.cpu().detach().numpy()  # (n, classes)

        n, d = x.shape

        if multiclass:
            K = self.__compute_kernel__(x, reduce=False)  # (n, n, d)
            L = self.__compute_kernel__(y, reduce=False)  # (n, n, 1)

            K_bar = self.__compute_gram_matrix__(K)  # (n, n, d)
            L_bar = self.__compute_gram_matrix__(L)  # (n, n, 1)

            K_bar = K_bar.reshape(n ** 2, d)  # (n ** 2, d)
            L_bar = L_bar.reshape(n ** 2, self.data.num_classes)  # (n ** 2,)

            solver = LassoLars(self.rho, fit_intercept=False,
                               normalize=False, positive=True)
            solver.fit(K_bar * n, L_bar * n)

            return solver.coef_.T

        else:
            K = self.__compute_kernel__(x, reduce=False)  # (n, n, d)
            L = self.__compute_kernel__(y, reduce=True)  # (n, n, 1)

            K_bar = self.__compute_gram_matrix__(K)  # (n, n, d)
            L_bar = self.__compute_gram_matrix__(L)  # (n, n, 1)

            K_bar = K_bar.reshape(n ** 2, d)  # (n ** 2, d)
            L_bar = L_bar.reshape(n ** 2,)  # (n ** 2,)

            solver = LassoLars(self.rho, fit_intercept=False,
                               normalize=False, positive=True)
            solver.fit(K_bar * n, L_bar * n)

            return solver.coef_


class LIME:
    """ LIME explainer - adapted to GNNs
    Explains only node features
    """

    def __init__(self, data, model, gpu=False, cached=True):
        self.data = data
        self.model = model
        self.gpu = gpu
        self.M = self.data.num_features
        self.F = self.data.num_features

        self.model.eval()

    def __init_predict__(self, x, edge_index, *unused, **kwargs):

        # Get the initial prediction.
        with torch.no_grad():
            if self.gpu:
                device = torch.device(
                    'cuda' if torch.cuda.is_available() else 'cpu')
                self.model = self.model.to(device)
                log_logits = self.model(
                    x=x.cuda(), edge_index=edge_index.cuda(), **kwargs)
            else:
                log_logits = self.model(x=x, edge_index=edge_index, **kwargs)
            probas = log_logits.exp()

        return probas

    def explain(self, node_index, hops, num_samples, info=False, multiclass=False, **kwargs):
        num_samples = num_samples//2
        x = self.data.x
        edge_index = self.data.edge_index

        probas = self.__init_predict__(x, edge_index, **kwargs)
        proba, label = probas[node_index, :].max(dim=0)

        x_ = deepcopy(x)
        original_feats = x[node_index, :]

        if multiclass:
            sample_x = [original_feats.detach().numpy()]
            #sample_y = [proba.item()]
            sample_y = [probas[node_index, :].detach().numpy()]

            for _ in range(num_samples):
                x_[node_index, :] = original_feats + \
                    torch.randn_like(original_feats)

                with torch.no_grad():
                    if self.gpu:
                        log_logits = self.model(
                            x=x_.cuda(), edge_index=edge_index.cuda(), **kwargs)
                    else:
                        log_logits = self.model(
                            x=x_, edge_index=edge_index, **kwargs)
                    probas_ = log_logits.exp()

                #proba_ = probas_[node_index, label]
                proba_ = probas_[node_index]

                sample_x.append(x_[node_index, :].detach().numpy())
                # sample_y.append(proba_.item())
                sample_y.append(proba_.detach().numpy())

        else:
            sample_x = [original_feats.detach().numpy()]
            sample_y = [proba.item()]
            # sample_y = [probas[node_index, :].detach().numpy()]

            for _ in range(num_samples):
                x_[node_index, :] = original_feats + \
                    torch.randn_like(original_feats)

                with torch.no_grad():
                    if self.gpu:
                        log_logits = self.model(
                            x=x_.cuda(), edge_index=edge_index.cuda(), **kwargs)
                    else:
                        log_logits = self.model(
                            x=x_, edge_index=edge_index, **kwargs)
                    probas_ = log_logits.exp()

                proba_ = probas_[node_index, label]
                # proba_ = probas_[node_index]

                sample_x.append(x_[node_index, :].detach().numpy())
                sample_y.append(proba_.item())
                # sample_y.append(proba_.detach().numpy())

        sample_x = np.array(sample_x)
        sample_y = np.array(sample_y)

        solver = Ridge(alpha=0.1)
        solver.fit(sample_x, sample_y)

        return solver.coef_.T
