import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from scipy.special import softmax
from scipy.stats import multivariate_normal as mv_norm


class DPMix:
    def __init__(self, data, truncation, obs_param, allo_param, allo_hyper, obs_prior):
        # x
        self.data = data
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
        # TODO use actual distribution instead of approximation
        cluster_means = self.obs_mean()
        log_cluster_weight = np.asarray([mv_norm.logpdf(self.data, cluster_means[i])
                                         for i in range(self.truncation)]).T
        # from Eq. 3.17-18
        log_u_hat = np.log(self.allo_param[:, 1] / np.sum(self.allo_param, axis=1))
        # W
        log_post_weight = log_cluster_weight + \
            np.asarray([[log_u_hat[i] + np.sum(1 - log_u_hat[:i])
                        for i in range(self.truncation)]])
        # r hat
        resp = softmax(log_post_weight, axis=0)

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

    def plot_data(self):
        # scatter plot of data
        plt.scatter(self.data[:, 0], self.data[:, 1])
        # scatter plot of cluster means
        cluster_means = self.obs_mean()
        plt.scatter(cluster_means[:, 0], cluster_means[:, 1], c="red")
        # ellipses for 1 STD around cluster
        ax = plt.gca()
        for i in range(self.truncation):
            std_ellipse = Ellipse((cluster_means[i, 0], cluster_means[i, 1]), 1, 1,
                                  color="red", alpha=.25)
            ax.add_patch(std_ellipse)
        plt.show()

    def obs_mean(self):
        return self.obs_count / self.obs_shape[:, np.newaxis]
