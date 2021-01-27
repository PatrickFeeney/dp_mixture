import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.vq import kmeans
from scipy.stats import zscore

import dp_mix.data.load as load
import dp_mix.models as models

K = 2
data = zscore(load.load_faithful())

num_models = 10
num_updates = 10
# hold objective evaluation for model after each update
objs = np.zeros((num_models, num_updates))

for i in range(num_models):
    # initialize new model
    init_means = kmeans(data, K)[0]
    dp_mix = models.DPMix(data, K, (init_means, np.asarray([1] * K)),
                          np.asarray([[1, .5]] * K), .5, (np.asarray([[0, 0]]), np.asarray([1])))
    # visualize model before and after updates
    dp_mix.plot_data()
    for j in range(num_updates):
        obj = dp_mix._update()
        objs[i, j] = obj
    dp_mix.plot_data()

# plot objective from each run
for i in range(num_models):
    plt.plot(range(10), objs[i])
plt.show()
