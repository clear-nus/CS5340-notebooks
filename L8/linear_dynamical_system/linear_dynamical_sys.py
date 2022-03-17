import random

import numpy as np
import matplotlib.pyplot as plt


def generate_linear_data(C, A, Gamma, Sigma, mu, V, horizon=25):
    z_g = np.random.normal(loc=mu, scale=V)[:, np.newaxis]  # mu[:, np.newaxis]
    z = np.random.normal(loc=mu, scale=V)[:, np.newaxis]
    x = C @ z + np.random.normal(loc=0, scale=Sigma)[:, np.newaxis]
    x_g = C @ z

    zs = [z.copy()]
    xs = [x.copy()]
    z_gs = [z_g.copy()]
    x_gs = [x_g.copy()]

    for i in range(horizon):
        z_g = A @ z_g
        x_g = C @ z
        z = A @ z + np.random.normal(loc=0, scale=Gamma)[:, np.newaxis]
        x = C @ z + np.random.normal(loc=0, scale=Sigma)[:, np.newaxis]

        zs += [z.copy()]
        xs += [x.copy()]
        z_gs += [z_g.copy()]
        x_gs += [x_g.copy()]

    zs = np.stack(zs, axis=0)
    xs = np.stack(xs, axis=0)
    z_gs = np.stack(z_gs, axis=0)
    x_gs = np.stack(x_gs, axis=0)

    return zs, xs, z_gs, x_gs


def generate_nonlinear_data(C, A, Gamma, Sigma, mu, V, horizon=25):
    z_g = np.random.normal(loc=mu, scale=V)[:, np.newaxis]
    z = np.random.normal(loc=mu, scale=V)[:, np.newaxis]
    x = C @ np.exp(z) + np.random.normal(loc=0, scale=Sigma)[:, np.newaxis]
    x_g = C @ np.exp(z)

    zs = [z.copy()]
    xs = [x.copy()]
    z_gs = [z_g.copy()]
    x_gs = [x_g.copy()]

    for i in range(horizon):
        z_g = A @ z_g
        x_g = C @ np.exp(z)
        z = A @ z + np.random.normal(loc=0, scale=Gamma)[:, np.newaxis]
        x = C @ np.exp(z) + np.random.normal(loc=0, scale=Sigma)[:, np.newaxis]

        zs += [z.copy()]
        xs += [x.copy()]
        z_gs += [z_g.copy()]
        x_gs += [x_g.copy()]

    zs = np.stack(zs, axis=0)
    xs = np.stack(xs, axis=0)
    z_gs = np.stack(z_gs, axis=0)
    x_gs = np.stack(x_gs, axis=0)

    return zs, xs, z_gs, x_gs


def inference(mu_l, V_l, A_l, C_l, Gamma_l, Sigma_l, x_noise):
    # compute forward messages
    def forward_kalman_gain_matrix(_P, C, Sigma):
        return _P @ C.transpose() @ np.linalg.inv(C @ _P @ C.transpose() + Sigma)

    def forward_update_P(A, V, Gamma):
        return A @ V @ A.transpose() + Gamma

    def forward_update_V(K, C, _P):
        return (np.eye(C.shape[-1]) - K @ C) @ _P

    def forward_update_mu(A, K, x, C, _mu):
        return A @ _mu + K @ (x - C @ A @ _mu)

    def forward_update_V_prior(K, C, _V):
        return (np.eye(C.shape[-1]) - K @ C) @ _V

    def forward_update_mu_prior(K, x, C, _mu):
        return _mu + K @ (x - C @ _mu)

    def forward_kalman_gain_matrix_prior(_V, C, Sigma):
        return _V @ C.transpose() @ np.linalg.inv(C @ _V @ C.transpose() + Sigma)

    # t = 0
    forward_K = forward_kalman_gain_matrix_prior(_V=A_l @ V_l @ A_l.transpose() + Gamma_l, C=C_l, Sigma=Sigma_l)
    forward_mu = forward_update_mu_prior(K=forward_K, x=x_noise[0], C=C_l, _mu=mu_l)
    forward_V = forward_update_V_prior(K=forward_K, C=C_l, _V=V_l)
    forward_P = forward_update_P(A=A_l, Gamma=Gamma_l, V=forward_V)

    forward_mus = [forward_mu.copy()]
    forward_Vs = [forward_V.copy()]
    forward_Ps = [forward_P.copy()]

    # t > 0
    for t in range(1, x_noise.shape[0]):
        forward_K = forward_kalman_gain_matrix(_P=forward_P, C=C_l, Sigma=Sigma_l)
        forward_mu = forward_update_mu(A=A_l, K=forward_K, x=x_noise[t], C=C_l, _mu=forward_mu)
        forward_V = forward_update_V(K=forward_K, C=C_l, _P=forward_P)
        forward_P = forward_update_P(A=A_l, V=forward_V, Gamma=Gamma_l)

        forward_mus += [forward_mu.copy()]
        forward_Vs += [forward_V.copy()]
        forward_Ps += [forward_P.copy()]

    forward_mus = np.stack(forward_mus, axis=0)
    forward_Vs = np.stack(forward_Vs, axis=0)
    forward_Ps = np.stack(forward_Ps, axis=0)

    # compute smooth messages
    def smooth_update_mu(forward_mu, J, smooth_mu_, A):
        return forward_mu + J @ (smooth_mu_ - A @ forward_mu)

    def smooth_update_V(forward_V, J, smooth_V_, forward_P):
        return forward_V + J @ (smooth_V_ - forward_P) @ J.transpose()

    def smooth_update_J(forward_V, A, forward_P):
        return forward_V @ A.transpose() @ np.linalg.inv(forward_P)

    # t = T
    smooth_J = smooth_update_J(forward_V=forward_Vs[-1], A=A_l, forward_P=forward_Ps[-1])
    smooth_mu = forward_mus[-1]
    smooth_V = forward_Vs[-1]

    smooth_Js = [smooth_J.copy()]
    smooth_mus = [smooth_mu.copy()]
    smooth_Vs = [smooth_V.copy()]

    for t in reversed(range(1, x_noise.shape[0])):
        smooth_J = smooth_update_J(forward_V=forward_Vs[t], A=A_l, forward_P=forward_Ps[t])
        smooth_mu = smooth_update_mu(forward_mu=forward_mus[t], J=smooth_J, smooth_mu_=smooth_mu, A=A_l)
        smooth_V = smooth_update_V(forward_V=forward_Vs[t], J=smooth_J, smooth_V_=smooth_V, forward_P=forward_Ps[t])

        smooth_Js += [smooth_J.copy()]
        smooth_mus += [smooth_mu.copy()]
        smooth_Vs += [smooth_V.copy()]

    smooth_mus = np.stack(smooth_mus[::-1], axis=0)
    smooth_Vs = np.stack(smooth_Vs[::-1], axis=0)
    smooth_Js = np.stack(smooth_Js[::-1], axis=0)

    return forward_mus, forward_Vs[:, np.newaxis], forward_Ps[:, np.newaxis], smooth_mus, smooth_Vs[:,
                                                                                          np.newaxis], smooth_Js[:,
                                                                                                       np.newaxis]


def plot_traj(x_noise, x_truth, smooth_xs, predict_xs, predict_smooth_xs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3.2))
    # smoothing results
    ax1.set(title="Smoothing")
    ax1.plot(x_noise[:, 0, 0], x_noise[:, 1, 0],
             'o-', color='lightgrey', label='noisy observation')
    ax1.plot(x_truth[:, 0, 0], x_truth[:, 1, 0],
             'o-', color='green', label='ground truth')
    ax1.plot(smooth_xs[:, 0, 0], smooth_xs[:, 1, 0],
             'o-', color='red', label='smooth observation')
    # ax1.legend()

    # prediction results
    ax2.set(title="Prediction")
    ax2.plot(x_noise[:, 0, 0], x_noise[:, 1, 0],
             'o-', color='lightgrey', label='noisy observation')
    ax2.plot(x_truth[:, 0, 0], x_truth[:, 1, 0],
             'o-', color='green', label='ground truth')
    ax2.plot(predict_xs[:, 0, 0], predict_xs[:, 1, 0],
             'o-', color='blue', label='predict observation')
    ax2.plot(predict_smooth_xs[:, 0, 0], predict_smooth_xs[:, 1, 0],
             'o-', color='red', label='smooth observation')
    ax2.legend()

    plt.tight_layout()

    plt.pause(1)
    plt.close()


def E_step(smooth_mus, smooth_Js, smooth_Vs):
    # from t=1 to t=T
    post_mean = smooth_mus
    # from t=2 to t=T
    post_dot_trans = smooth_Js[:-1] @ smooth_Vs[1:] + smooth_mus[1:] @ smooth_mus[:-1].transpose((0, 1, -1, -2))
    # from t=1 to t=T
    post_dot = smooth_Vs + smooth_mus @ smooth_mus.transpose((0, 1, -1, -2))
    return post_mean, post_dot_trans, post_dot


def M_step(x_noise, post_mean, post_dot, post_dot_trans):
    mu_l = post_mean[0]

    V_l = post_dot[0] - post_mean[0] @ post_mean[0].transpose((0, -1, -2))

    T = x_noise.shape[0]

    A_l = post_dot_trans.sum(axis=0) @ np.linalg.inv(
        post_dot[:-1].sum(axis=0))

    Gamma_l = (post_dot[1:].sum(axis=0) - A_l @ post_dot_trans.transpose((0, 1, -1, -2)).sum(
        axis=0) - post_dot_trans.sum(
        axis=0) @ A_l.transpose((0, -1, -2)) + A_l @ post_dot[:-1].sum(axis=0) @ A_l.transpose((0, -1, -2))) / (T - 1)

    C_l = (x_noise @ post_mean.transpose((0, 1, -1, -2))).sum(axis=0) @ np.linalg.inv(post_dot.sum(axis=0))

    Sigma_l = ((x_noise @ x_noise.transpose((0, 1, -1, -2))).sum(axis=0)
               - C_l @ (post_mean @ x_noise.transpose((0, 1, -1, -2))).sum(axis=0)
               - (x_noise @ post_mean.transpose((0, 1, -1, -2))).sum(axis=0) @ C_l.transpose((0, -1, -2))
               + C_l @ post_dot.sum(axis=0) @ C_l.transpose((0, -1, -2))) / T
    return mu_l.mean(axis=0), V_l.mean(axis=0), A_l.mean(axis=0), Gamma_l.mean(axis=0), C_l.mean(axis=0), Sigma_l.mean(
        axis=0)


def EM(x_noise, x_truth, x_noise_test, x_truth_test, mu_l, V_l, A_l, Gamma_l, C_l, Sigma_l):
    itr_count = 0
    max_count = 50
    epsilon = 0.005
    param_diff = 1.0
    plot_freq = 1

    A_l_ = A_l.copy()
    C_l_ = C_l.copy()
    mu_l_ = mu_l.copy()
    V_l_ = V_l.copy()
    Sigma_l_ = Sigma_l.copy()
    Gamma_l_ = Gamma_l.copy()
    while itr_count < max_count and param_diff > epsilon:
        forward_mus, forward_Vs, forward_Ps, smooth_mus, smooth_Vs, smooth_Js = inference(mu_l, V_l, A_l, C_l, Gamma_l,
                                                                                          Sigma_l,
                                                                                          x_noise)
        # E-step
        post_mean, post_dot_trans, post_dot = E_step(smooth_mus, smooth_Js, smooth_Vs)

        # M-step
        mu_l, V_l, A_l, Gamma_l, C_l, Sigma_l = M_step(x_noise, post_mean, post_dot, post_dot_trans)

        itr_count += 1
        param_diff = np.abs(
            (A_l - A_l_).sum() + (C_l - C_l_).sum() + (mu_l - mu_l_).sum() + (V_l - V_l_).sum() + (
                    Sigma_l - Sigma_l_).sum() + (
                    Gamma_l - Gamma_l_).sum())

        A_l_ = A_l.copy()
        C_l_ = C_l.copy()
        mu_l_ = mu_l.copy()
        V_l_ = V_l.copy()
        Sigma_l_ = Sigma_l.copy()
        Gamma_l_ = Gamma_l.copy()
        print(f'iteration:{itr_count}, param_difference:{param_diff}')
        if itr_count % plot_freq == 0:
            forward_mus_test, forward_Vs_test, forward_Ps_test, smooth_mus_test, smooth_Vs_test, smooth_Js_test = inference(
                mu_l, V_l, A_l, C_l,
                Gamma_l,
                Sigma_l,
                x_noise_test)

            # smoothing results
            smooth_xs = C_l @ smooth_mus_test

            # prediction results
            start_t = 25
            predict_mu_test = smooth_mus_test[start_t]
            predict_mus_test = [predict_mu_test]
            for t in range(smooth_mus_test.shape[0] - start_t - 1):
                predict_mu_test = A_l @ predict_mu_test
                predict_mus_test += [predict_mu_test]
            predict_mus_test = np.stack(predict_mus_test, axis=0)

            predict_smooth_xs = C_l @ smooth_mus_test[:start_t+1]
            predict_xs = C_l @ predict_mus_test

            batch_id = 0
            plot_traj(x_noise_test[:, batch_id], x_truth_test[:, batch_id], smooth_xs[:, batch_id],
                      predict_xs[:, batch_id], predict_smooth_xs[:, batch_id])
    return A_l, C_l, mu_l, V_l, Sigma_l, Gamma_l


batch_size = 50
# linear case
theta = np.pi / 36
C_g = np.array([[1, 0], [0, 1]])
A_g = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(np.sin(theta))]])
Gamma_g = np.ones(2) * 0.005
Sigma_g = np.ones(2) * 0.25
V_g = np.ones(2) * 1.0
mu_g = np.array([0, 1])
x_noises = []
x_truths = []
for i in range(batch_size):
    _, x_noise, _, x_truth = generate_linear_data(C_g, A_g, Gamma_g, Sigma_g, mu_g, V_g, horizon=50)
    x_noises += [x_noise]
    x_truths += [x_truth]

x_noises = np.stack(x_noises, axis=1)
x_truths = np.stack(x_truths, axis=1)

x_noises_test = []
x_truths_test = []
for i in range(batch_size):
    _, x_noise, _, x_truth = generate_linear_data(C_g, A_g, Gamma_g, Sigma_g, mu_g, V_g, horizon=50)
    x_noises_test += [x_noise]
    x_truths_test += [x_truth]

x_noises_test = np.stack(x_noises_test, axis=1)
x_truths_test = np.stack(x_truths_test, axis=1)

mu_l = np.random.rand(A_g.shape[1])[:, np.newaxis]
V_l = np.diag(V_g)
A_l = np.random.rand(A_g.shape[0], A_g.shape[1])
C_l = np.random.rand(C_g.shape[0], C_g.shape[1])
Gamma_l = np.exp2(np.diag(np.random.rand(Gamma_g.shape[0])))
Sigma_l = np.exp2(np.diag(np.random.rand(Sigma_g.shape[0])))

# EM(x_noises, x_truths, x_noises_test, x_truths_test, mu_l, V_l, A_l, Gamma_l, C_l, Sigma_l)

# non-linear case
x_noises = []
x_truths = []
mu_g = np.array([0, 0])
V_g = np.ones(2) * 1.0
for i in range(batch_size):
    _, x_noise, _, x_truth = generate_nonlinear_data(C_g, A_g, Gamma_g, Sigma_g, mu_g, V_g, horizon=50)
    x_noises += [x_noise]
    x_truths += [x_truth]
x_noises = np.stack(x_noises, axis=1)
x_truths = np.stack(x_truths, axis=1)

x_noises_test = []
x_truths_test = []
for i in range(batch_size):
    _, x_noise, _, x_truth = generate_nonlinear_data(C_g, A_g, Gamma_g, Sigma_g, mu_g, V_g, horizon=50)
    x_noises_test += [x_noise]
    x_truths_test += [x_truth]

x_noises_test = np.stack(x_noises_test, axis=1)
x_truths_test = np.stack(x_truths_test, axis=1)

mu_l = np.random.rand(A_g.shape[1])[:, np.newaxis]
V_l = np.diag(V_g)
A_l = np.random.rand(A_g.shape[0], A_g.shape[1])
C_l = np.random.rand(C_g.shape[0], C_g.shape[1])
Gamma_l = np.exp2(np.diag(np.random.rand(Gamma_g.shape[0])))
Sigma_l = np.exp2(np.diag(np.random.rand(Sigma_g.shape[0])))

plt.plot(x_noises_test[:, 0, 0, 0], x_noises_test[:, 0, 1, 0],
         'o-', color='lightgrey', label='noisy observation')
plt.plot(x_truths_test[:, 0, 0, 0], x_truths_test[:, 0, 1, 0],
         'o-', color='green', label='ground truth')
plt.legend()

plt.tight_layout()

plt.pause(-1)
plt.close()

EM(x_noises, x_truths, x_noises_test, x_truths_test, mu_l, V_l, A_l, Gamma_l, C_l, Sigma_l)
