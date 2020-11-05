import numpy as np
from scipy.special import softmax
from scipy.stats import zscore, multivariate_normal as mv_norm


class DPMix:
    def __init__(self, data, truncation, obs_param, allo_param, allo_hyper, obs_prior):
        # x
        self.data = zscore(data)
        # K
        self.truncation = truncation
        # {tau hat sub k, nu hat sub k} for k=1 to K
        self.obs_count, self.obs_shape = obs_param
        # {eta hat sub k} for k=1 to K
        self.allo_param = allo_param
        # gamma
        self.allo_hyper = allo_hyper
        # {tau bar, nu bar}
        self.prior_count, self.prior_shape = obs_prior
        # N
        self.N = self.data.shape[0]
        # D
        self.D = self.data.shape[1]

    def _update(self):
        # local step
        # C
        # approximation of prior mean from Eq. 2.51
        log_cluster_weight = mv_norm.logpdf(self.data, self.obs_count / self.obs_shape)
        # from Eq. 3.17-18
        log_u_hat = np.log(self.allo_param[:, 1] / np.sum(self.allo_param, axis=1))
        # W
        log_post_weight = log_cluster_weight + \
            np.asarray([log_u_hat[i] + np.sum(1 - log_u_hat[:i]) for i in range(self.truncation)])
        # r hat
        resp = softmax(log_post_weight, axis=1)

        # summary step
        # S
        weighted_stat = self.data.T @ resp
        # N
        count = np.sum(resp, axis=0)
        # N greater than subscript
        count_gt = np.asarray([np.sum(count[i+1:]) for i in range(self.truncation)])

        # global step for observation params
        # tau hat
        self.obs_count = weighted_stat + self.prior_count
        # nu hat
        self.obs_shape = count + self.prior_shape
        # global step for allocation params
        # eta hat
        self.allo_param[:, 0] = count_gt + self.allo_hyper
        self.allo_param[:, 1] = count + 1
