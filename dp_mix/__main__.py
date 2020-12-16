import numpy as np
from scipy.cluster.vq import kmeans
from scipy.stats import zscore

import dp_mix.data.load as load
import dp_mix.models as models

K = 2
data = zscore(load.load_faithful())
init_means = kmeans(data, K)[0]
dp_mix = models.DPMix(data, K, (init_means, np.asarray([1] * K)),
                      np.asarray([[1, .5]] * K), .5, (np.asarray([0, 0]), np.asarray([1])))
dp_mix.plot_data()
for i in range(10):
    dp_mix._update()
    dp_mix.plot_data()
