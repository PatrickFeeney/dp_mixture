import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from scipy.special import softmax, digamma, loggamma


class DPMix:
    def __init__(self, data, truncation, obs_param, allo_param, allo_hyper, obs_prior):
        # x
        self.data = data
        # K
        self.truncation = truncation
        # {tau hat sub k, nu hat sub k} for k=1 to K
        self.obs_shape, self.obs_count = obs_param
        # {eta hat sub k} for k=1 to K
        self.allo_param = allo_param
        # gamma
        self.allo_hyper = allo_hyper
        # {tau bar, nu bar}
        self.prior_shape, self.prior_count = obs_prior
        # N
        self.N = self.data.shape[0]
        # D
        self.D = self.data.shape[1]
        # fixed variance for Gaussian prior
        self.prior_fixed_var = 1
        # fixed variance for Gaussian likelihood
        self.like_fixed_var = 1

    def _update(self):
        # summary step, from results of local step
        # S, N, N greater than subscript
        weighted_stat, count, count_gt = self.resp_summary()

        # global step for observation params
        # tau hat
        self.obs_shape = weighted_stat + self.prior_shape
        # nu hat
        self.obs_count = count + self.prior_count
        # global step for allocation params
        # eta hat
        self.allo_param[:, 0] = count_gt + self.allo_hyper
        self.allo_param[:, 1] = count + 1
        # print objective, should outperform N([0, 0], I2) which has logpdf -771.9025620633421
        print("Objective: " + str(self.objective()))

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
        return self.obs_shape / self.obs_count[:, np.newaxis]

    def responsibility(self):
        # C
        # np.sum(obs_count) should become np.trace if full covariance matrix is used
        cluster_means = self.obs_mean()
        log_cluster_weight = np.log(1/(2 * np.math.pi)) \
            - np.sum(self.data * self.data, axis=1, keepdims=True)/2 \
            + self.data @ cluster_means.T \
            - (np.sum(cluster_means * cluster_means, axis=1) + np.sum(self.obs_count))/2
        # expectations over Dirichlet
        log_u, log_1_minus_u = self.dirichlet_expectation()
        # W
        log_post_weight = log_cluster_weight + \
            np.asarray([[log_u[i] + np.sum(log_1_minus_u[:i])
                        for i in range(self.truncation)]])
        # r hat from Eq 3.49
        return softmax(log_post_weight, axis=1)

    def objective(self):
        # r hat
        resp = self.responsibility()
        # S, N, N greater than subscript
        weighted_stat, count, count_gt = self.resp_summary()
        # simplified data loss from Eq 3.42
        ref_sum = np.sum(self.ref_measure())
        cum_dif = np.sum(self.prior_cumulant(self.obs_shape, self.obs_count)
                         - self.prior_cumulant(self.prior_shape, self.prior_count))
        data_loss = ref_sum + cum_dif

        # entropy loss from Eq 3.29
        entropy_loss = -1 * np.sum(resp * np.log(resp + 1e-10))

        # simplified DP allocation loss from Eq 3.44
        dp_alloc_loss = np.sum(
            self.beta_cumulant(1, self.allo_hyper)
            - self.beta_cumulant(self.allo_param[:, 1], self.allo_param[:, 0])
        )
        print()
        print()
        print("Data Loss: " + str(data_loss))
        print("\tRef Measure: " + str(ref_sum))
        print("\tPrior Cumulant: " + str(cum_dif))
        print("Entropy Loss: " + str(entropy_loss))
        print("DP Alloc Loss: " + str(dp_alloc_loss))
        print()
        return data_loss + entropy_loss + dp_alloc_loss

    def dirichlet_expectation(self):
        # expectations from Eq 3.32-33
        log_u = digamma(self.allo_param[:, 1]) - digamma(np.sum(self.allo_param, axis=1))
        log_1_minus_u = digamma(self.allo_param[:, 0]) - digamma(np.sum(self.allo_param, axis=1))
        return log_u, log_1_minus_u

    def beta_cumulant(self, alpha, beta):
        # cumulant function for beta distribution from Eq 3.34
        return loggamma(beta + alpha) - loggamma(beta) - loggamma(alpha)

    def resp_summary(self):
        # local step
        resp = self.responsibility()
        # summary step
        # S
        weighted_stat = resp.T @ self.data
        # N
        count = np.sum(resp, axis=0)
        # N greater than subscript
        count_gt = np.asarray([np.sum(count[i+1:]) for i in range(self.truncation)])
        return weighted_stat, count, count_gt

    def prior_cumulant(self, shape, count):
        # cumulant function for univariate Gaussian prior for fixed-variance Gaussian likelihood
        # from Eq 2.49
        # modified for multivariate case
        # change count from (K) vector to (K, 1) matrix
        count = count[:, np.newaxis]
        return (
            self.D * np.log(2 * np.math.pi)
            - self.D * np.log(count)
            + self.D * np.log(self.prior_fixed_var)
            + np.sum(shape ** 2, axis=1, keepdims=True) * self.prior_fixed_var / count
        )/2

    def ref_measure(self):
        # reference measure for univariate Gaussian with known variance from Eq 2.24
        # modified for multivariate case
        return (
            np.sum(self.data ** 2, axis=1, keepdims=True)/self.like_fixed_var
            + self.D * np.log(self.like_fixed_var)
            + self.D * np.log(2 * np.math.pi)
        )/-2
