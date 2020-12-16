import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from scipy.special import softmax, digamma


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
        resp = self.responsibility()

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

    def responsibility(self):
        # C
        # np.sum(obs_shape) should become np.trace if full covariance matrix is used
        cluster_means = self.obs_mean()
        log_cluster_weight = np.log(1/(2 * np.math.pi)) \
            - np.sum(self.data * self.data, axis=1, keepdims=True)/2 \
            + self.data @ cluster_means.T \
            - (np.sum(cluster_means * cluster_means, axis=1) + np.sum(self.obs_shape))/2
        # expectations over Dirichlet
        log_u = digamma(self.allo_param[:, 1]) - digamma(np.sum(self.allo_param, axis=1))
        log_1_minus_u = digamma(self.allo_param[:, 0]) - digamma(np.sum(self.allo_param, axis=1))
        # W
        log_post_weight = log_cluster_weight + \
            np.asarray([[log_u[i] + np.sum(log_1_minus_u[:i])
                        for i in range(self.truncation)]])
        # r hat
        return softmax(log_post_weight, axis=0)

    # TODO implement objective function
    def objective(self):
        resp = self.responsibility()
        # data loss
        data_loss = 0
        # entropy loss
        entropy_loss = -1 * np.sum(resp * np.log(resp + 1e-10))
        # DP allocation loss
        dp_alloc_loss = 0
        return data_loss + entropy_loss + dp_alloc_loss
