"""
Use mocked scatter plot to illustrate `lgbm_pu` algorithm for PU learning

Algorithm
1. Train g(x) to classify positive versus unlabeled data
2. Calculate c = mean(g(x)), x is positive samples
3. f(x) = g(x) / c, where f(x) is the classifier for positive versus negative data
"""

import random
import matplotlib.pyplot as plt


if __name__ == '__main__':
    n_tn = 40           # number of true negative samples (under `f_threshold`)
    n_tp = 10           # number of true positive samples (over `g_threshold`)
    n_fn = 5            # number of false negative samples (between `f_threshold` and `g_threshold`)
    x_max = 5.0
    y_max = 5.0
    g_threshold = 3.0   # threshold of g(x)
    f_threshold = 2.5   # threshold of f(x)
    tn_x = [random.random() for _ in range(n_tn)]                                                   # true negative x
    tn_y = [random.random() * (f_threshold - .2) for _ in range(n_tn)]                              # true negative y
    tp_x = [random.random() for _ in range(n_tp)]                                                   # true positive x
    tp_y = [random.random() * (y_max - g_threshold) + g_threshold for _ in range(n_tp)]             # true positive y
    fn_x = [random.random() for _ in range(n_fn)]                                                   # false negative x
    fn_y = [random.random() * (g_threshold - f_threshold) + f_threshold for _ in range(n_fn)]       # false negative y
    plt.figure(0)
    plt.plot([.1 * i for i in range(10)], [f_threshold for _ in range(10)], color='red')
    plt.plot([.1 * i for i in range(10)], [g_threshold for _ in range(10)], linestyle='--', color='red')
    plt.scatter(tn_x, tn_y)
    plt.scatter(tp_x, tp_y)
    plt.scatter(fn_x, fn_y)
    plt.legend(['f(x)',  'g(x)', 'tn', 'tp', 'fn'])
    plt.show()