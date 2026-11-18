"""Question 2: direction of motion on a 40-sensor ring road."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

N = 40
FIG_DIR = os.path.join("outputs", "figures")


def true_step(x):
    """Backward motion: x_i(t+1) = x_{i+1}(t)."""
    return np.roll(x, -1)


def forward_step(x):
    """Forward motion: x_i(t+1) = x_{i-1}(t)."""
    return np.roll(x, 1)


def symmetric_layer(x, a, b):
    return a * x + b * (np.roll(x, 1) + np.roll(x, -1))


def q2a_fit():
    """Least squares fit of (a, b) on the single-impulse snapshot."""
    x = np.zeros(N)
    x[20] = 1.0
    y = true_step(x)
    # Design matrix: column 0 = x_i, column 1 = x_{i-1} + x_{i+1}
    X = np.stack([x, np.roll(x, 1) + np.roll(x, -1)], axis=1)
    (a, b), *_ = np.linalg.lstsq(X, y, rcond=None)
    sse = ((symmetric_layer(x, a, b) - y) ** 2).sum()
    print(f"Q2(a)(i) a={a:.4f}, b={b:.4f}, SSE={sse:.4f}")
    return a, b


def q2a_rollout(a, b, steps=20):
    i = np.arange(N)
    x0 = np.exp(-(i - 20) ** 2 / 8)
    pred, true = [x0], [x0]
    for _ in range(steps):
        pred.append(symmetric_layer(pred[-1], a, b))
        true.append(true_step(true[-1]))
    pred, true = np.array(pred), np.array(true)

    vmin, vmax = min(pred.min(), true.min()), max(pred.max(), true.max())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, data, title in [(axes[0], pred, f"Symmetric layer (a={a:.2f}, b={b:.2f})"),
                            (axes[1], true, "True motion")]:
        im = ax.imshow(data, aspect="auto", origin="lower", cmap="viridis",
                       vmin=vmin, vmax=vmax, extent=[-0.5, N - 0.5, -0.5, steps + 0.5])
        ax.set_title(title)
        ax.set_xlabel("sensor index $i$")
    axes[0].set_ylabel("time step $t$")
    fig.colorbar(im, ax=axes, label="slowdown")
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, "q2a_heatmaps.pdf"), bbox_inches="tight")
    plt.close(fig)

    com = lambda x: (x * i).sum() / x.sum()
    print(f"Q2(a)(ii) t=20: predicted peak={pred[-1].max():.3f} at {pred[-1].argmax()}, "
          f"centre={com(pred[-1]):.2f}, mass={pred[-1].sum():.3f}")
    print(f"          t=20: true peak={true[-1].max():.3f} at {true[-1].argmax()}, "
          f"mass={true[-1].sum():.3f}")


def q2b(rng):
    a, w_minus, w_plus = 0.0, 0.0, 1.0
    for _ in range(100):
        x = rng.normal(size=N)
        pred = a * x + w_minus * np.roll(x, 1) + w_plus * np.roll(x, -1)
        assert np.allclose(pred, true_step(x))

    # Same thing as a circular 1D convolution (cross-correlation, as in PyTorch's Conv1d).
    kernel = np.array([w_minus, a, w_plus])  # multiplies [x_{i-1}, x_i, x_{i+1}]
    x = rng.normal(size=N)
    padded = np.concatenate([x[-1:], x, x[:1]])  # circular padding of 1 on each side
    conv = np.array([kernel @ padded[i:i + 3] for i in range(N)])
    assert np.allclose(conv, true_step(x))
    try:
        import torch
        layer = torch.nn.Conv1d(1, 1, 3, padding=1, padding_mode="circular", bias=False)
        with torch.no_grad():
            layer.weight[:] = torch.tensor(kernel, dtype=torch.float32).view(1, 1, 3)
            out = layer(torch.tensor(x, dtype=torch.float32).view(1, 1, N)).view(-1).numpy()
        assert np.allclose(out, true_step(x), atol=1e-6)
        print("Q2(b) torch Conv1d(circular) check passed")
    except ImportError:
        pass
    print(f"Q2(b) a={a}, w-={w_minus}, w+={w_plus}, kernel={kernel.tolist()} exact for 100 random inputs")


def history_layer(x_t, x_prev, a, b, c):
    return a * x_t + b * (np.roll(x_t, 1) + np.roll(x_t, -1)) + c * x_prev


def q2c(rng):
    a, b, c = 0.0, 1.0, -1.0
    for _ in range(100):
        # Backward motion pair
        x_prev = rng.normal(size=N)
        x_t = true_step(x_prev)
        assert np.allclose(history_layer(x_t, x_prev, a, b, c), true_step(x_t))
        # Forward motion pair, same coefficients
        x_t = forward_step(x_prev)
        assert np.allclose(history_layer(x_t, x_prev, a, b, c), forward_step(x_t))
    print(f"Q2(c) a={a}, b={b}, c={c} exact for backward and forward motion (100 random pairs each)")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    a, b = q2a_fit()
    q2a_rollout(a, b)
    q2b(rng)
    q2c(rng)
