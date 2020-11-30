import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
mu, sigma = 0, .1
s_a = np.random.normal(loc=mu, scale=sigma, size=10000)
s = np.abs(s_a)

count, bins, _ = plt.hist(s, 30, density='False')
        # normed是进行拟合的关键
        # count统计某一bin出现的次数，在Normed为True时，可能其值会略有不同
# plt.plot(bins, 1/(np.sqrt(2*np.pi)*sigma)*np.exp(-(bins-mu)**2/(2*sigma**2)), lw=2, c='r')
# plt.show()

# fig_2 = plt.figure()
mu_2, sigma_2 = 0, .3
s_a_2 = np.random.normal(loc=mu_2, scale=sigma_2, size=10000)
s_2 = np.abs(s_a_2)
count_2, bins_2, _2 = plt.hist(s_2, 30, density='False')
        # normed是进行拟合的关键
        # count统计某一bin出现的次数，在Normed为True时，可能其值会略有不同

# plt.plot(bins_2, 1/(np.sqrt(2*np.pi)*sigma_2)*np.exp(-(bins_2-mu_2)**2/(2*sigma_2**2)), lw=2, c='r')
fig_2 = plt.figure()
s_a_3 = s_a - s_a_2
s_a_3 = np.abs(s_a_3)
count_3, bins_3, _3 = plt.hist(s_a_3, 30, density='False')

plt.show()



