"""
Contains plots that show different sampled images from Flow-Models, 
latent space exchanges and reconstruction for autoencoders
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def plot_data_sampling(
    model,
    n_rows=8,
):
    """
    Sample from model and show generated images
    """
    model.eval()

    device = model.device

    d = torch.randint(0, model.hparams.n_envs, (n_rows * n_rows,)).to(device)
    y = torch.randint(0, model.hparams.n_classes, (n_rows * n_rows,)).to(device)

    d = F.one_hot(d, num_classes=model.hparams.n_envs).float()
    y = F.one_hot(y, num_classes=model.hparams.n_classes).float()

    n_samples = n_rows * n_rows
    #  Generate Samples
    with torch.no_grad():
        dec = model.generate_samples(
            y=y,
            d=d,
            n_samples=n_samples,
        )

    z_rec = torch.clip(dec.detach(), 0, 1).cpu().numpy()

    z_rec = np.swapaxes(z_rec, 1, 2)
    z_rec = np.swapaxes(z_rec, 2, 3)

    fig, axs = plt.subplots(nrows=n_rows, ncols=n_rows, figsize=(15, 15))

    ij = 0
    for i in range(n_rows):
        for j in range(n_rows):
            # if ij < num_classes:
            axs[i, j].imshow((z_rec[ij]))
            axs[i, j].set_title(
                f"Y: {torch.argmax(y[ij]).item()}; E: {torch.argmax(d[ij]).item()}"
            )
            ij += 1
            axs[i, j].axis("off")
    fig.tight_layout()

    return fig


def plot_reconstruction(model, dataset, n_rows=6):
    """
    Reconstruction plots for Autoencoder
    """
    device = model.device

    x, y, d = dataset[: n_rows * n_rows]
    x, y, d = x.to(model.device), y.to(model.device), d.to(model.device)

    with torch.no_grad():
        x_rec = model.forward_autoencoder(x)

    x_rec = x_rec.cpu().numpy()
    x = x.cpu().numpy()

    fig, ax = plt.subplots(nrows=n_rows, ncols=n_rows, figsize=(15, 15))

    x_rec = np.swapaxes(x_rec, 1, 2)
    x_rec = np.swapaxes(x_rec, 2, 3)
    x_rec = np.clip(x_rec, 0, 1)

    x = np.swapaxes(x, 1, 2)
    x = np.swapaxes(x, 2, 3)
    x = np.clip(x, 0, 1)

    e = 0
    for j in range(n_rows):
        for i in range(n_rows):
            if i % 2 == 1:
                ax[j, i].imshow((x_rec[e - 1]))
                ax[j, i].set_title("Reconstruction")
                ax[j, i].axis("off")
            else:
                ax[j, i].set_title("Original")
                ax[j, i].imshow(x[e])
                ax[j, i].axis("off")
            e += 1
    fig.tight_layout()
    return fig


def plot_latent_space_exchange(
    model,
    dataloader,
    n_rows=12,
    generic=False,
):
    """
    latent space exchange
    """

    model.eval()

    x, y, e = dataloader[: 8 * n_rows]
    x, y, e = x.to(model.device), y.to(model.device), e.to(model.device)

    with torch.no_grad():
        enc = model.forward_encoder(x)
        z_rec = model.forward_decoder(enc).detach().cpu().numpy()

        if not generic:
            x_env, x_class, _ = model.latent_exchange(x[::2], x[1::2])
        else:
            x_env, x_class = model.latent_exchange_generic(x[::2], y[1::2], e[1::2])

    x = np.clip(x.cpu().numpy(), 0, 1)
    x = np.swapaxes(x, 1, 2)
    x = np.swapaxes(x, 2, 3)

    z_rec = np.clip(z_rec, 0, 1)
    z_rec = np.swapaxes(z_rec, 1, 2)
    z_rec = np.swapaxes(z_rec, 2, 3)

    x_class = np.clip(x_class.cpu().numpy(), 0, 1)
    x_class = np.swapaxes(x_class, 1, 2)
    x_class = np.swapaxes(x_class, 2, 3)

    x_env = np.clip(x_env.cpu().numpy(), 0, 1)
    x_env = np.swapaxes(x_env, 1, 2)
    x_env = np.swapaxes(x_env, 2, 3)

    fig, axs = plt.subplots(nrows=n_rows, ncols=5, figsize=(10, 25))

    for j in range(n_rows):
        axs[j, 0].imshow(x[2 * j])
        axs[j, 0].axis("off")

        axs[j, 1].imshow(x[2 * j + 1])
        axs[j, 1].axis("off")

        axs[j, 2].imshow(z_rec[2 * j])
        axs[j, 2].axis("off")

        axs[j, 3].imshow(x_env[j])
        axs[j, 3].axis("off")

        axs[j, 4].imshow(x_class[j])
        axs[j, 4].axis("off")

        if j == 0:
            axs[j, 0].set_title(r"$X_1$")
            axs[j, 1].set_title(r"$X_2$")
            axs[j, 2].set_title(r"$X_1^{rec}$")
            axs[j, 3].set_title(r"$X_2 \to X_1$ (Env)")
            axs[j, 4].set_title(r"$X_2 \to X_1$ (Class)")
    plt.tight_layout()
    return fig


def plot_latent_space_exchange_generic(model, dataloader, n_rows=8):
    """
    Latent Space Exchange for generic environments
    """
    model.eval()
    device = model.device

    x, y, d = dataloader[:]
    x, y, d = (
        x.to(model.device)[:n_rows],
        y.to(model.device)[:n_rows],
        d.to(model.device)[:n_rows],
    )

    d_orig = d.argmax(1)
    y_orig = y.argmax(1)

    d = torch.arange(0, model.hparams.n_envs).view(-1)
    d = F.one_hot(d, num_classes=model.hparams.n_envs).float().to(device)

    with torch.no_grad():
        enc = model.forward_encoder(x)
        z_rec = model.forward_decoder(enc).detach().cpu().numpy()

        xs_exchange = []
        for i in range(model.hparams.n_envs):
            x_exchange = model.latent_exchange_generic(
                x, d=d[i : i + 1].repeat(n_rows, 1), location="env"
            )
            x_exchange = np.clip(x_exchange.cpu().numpy(), 0, 1)
            x_exchange = np.swapaxes(x_exchange, 1, 2)
            x_exchange = np.swapaxes(x_exchange, 2, 3)
            xs_exchange.append(x_exchange)

    x = np.clip(x.cpu().numpy(), 0, 1)
    x = np.swapaxes(x, 1, 2)
    x = np.swapaxes(x, 2, 3)

    z_rec = np.clip(z_rec, 0, 1)
    z_rec = np.swapaxes(z_rec, 1, 2)
    z_rec = np.swapaxes(z_rec, 2, 3)

    fig, axs = plt.subplots(
        nrows=n_rows, ncols=2 + model.hparams.n_envs, figsize=(10, 25)
    )

    for j in range(n_rows):
        axs[j, 0].imshow(x[j])
        axs[j, 0].axis("off")

        axs[j, 1].imshow(z_rec[j])
        axs[j, 1].axis("off")

        for i in range(model.hparams.n_envs):
            axs[j, 2 + i].imshow(xs_exchange[i][j])
            axs[j, 2 + i].axis("off")

        if j == 0:
            axs[j, 1].set_title(r"$X_1^{rec}$")
            for i in range(model.hparams.n_envs):
                axs[j, 2 + i].set_title(f"Env. {i}")
        axs[j, 0].set_title(rf"$X_1$l; E: {d_orig[j].item()} Y:{y_orig[j].item()}")
    plt.tight_layout()
    return fig
