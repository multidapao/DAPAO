#!/usr/bin/env python3

def attack():
    import csv
    import os
    from pathlib import Path
    import time

    import kornia
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    import torchvision
    from tqdm import tqdm

    from color import DenseLoss, cluster_channels, project
    from dp_trans import AttackDataset
    import pytorch_ssim

    device = torch.device("cuda:0")
    image_size = int(os.environ["IMAGE_SIZE"])
    eps = float(os.environ["EPS"])
    epochs = int(os.environ["EPOCH"])
    clusters = int(os.environ["K_CLUSTER"])
    selected_clusters = int(os.environ["N_CLUSTER"])
    output_dir = Path(os.environ["OUT_PATH"])
    experiment_name = os.environ["EXPERIMENT_NAME"]
    experiment_dir = Path("exp") / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    dataset = AttackDataset(os.environ["TRAIN"], image_size)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    dense_loss = DenseLoss().to(device)

    start = time.time()
    for image, name in tqdm(loader):
        delta = torch.nn.Parameter(0.01 * torch.randn((1, 3, image_size, image_size)).to(device))
        optimizer = torch.optim.AdamW([delta], lr=0.15, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs // 10, eta_min=0.049,
        )
        ssim_loss = pytorch_ssim.SSIM(window_size=256)
        image = image.to(device)
        image_feature = dense_loss(image)
        labels = cluster_channels(image_feature[0], k=clusters)
        unique_labels, counts = np.unique(np.array(labels), return_counts=True)
        count_by_label = dict(zip(unique_labels, counts))
        selected_labels = sorted(count_by_label, key=count_by_label.get)[:selected_clusters]
        indices_to_keep = np.where(np.isin(labels, selected_labels))[0]
        weights = torch.zeros(image_feature.shape[1]).to(device)
        weights[indices_to_keep] = 1.0
        weights = weights.view(1, -1, 1, 1)
        print(f"k = {clusters}, n = {selected_clusters}")

        for step in range(epochs):
            image = image.to(device)
            optimizer.zero_grad()
            image_feature = dense_loss(image)
            attacked_feature = dense_loss(image + delta)
            loss = ssim_loss(attacked_feature * weights, image_feature * weights)
            print("loss:", loss.item())
            loss.backward()
            optimizer.step()
            scheduler.step()

            perturbation = delta.clone().detach()
            norm = torch.norm(perturbation, float("inf")).cpu()
            if norm >= eps:
                projected = project(image + perturbation, image, eps / norm)
                delta.data = projected - image

            if step % 5 == 0:
                with torch.no_grad(), (experiment_dir / f"{experiment_name}-record.csv").open("a", newline="") as record:
                    saved_delta = delta.mean(dim=0, keepdim=True).cpu()
                    image = image.cpu()
                    attacked = image + saved_delta
                    perturbation_norm = torch.norm(saved_delta, float("inf")).numpy()
                    psnr = -kornia.losses.psnr_loss(attacked, image, 2).item()
                    ssim = 1 - 2 * kornia.losses.ssim_loss(attacked, image, window_size=11, reduction="mean").item()
                    stem = Path(name[0]).name.split(".")[0]
                    csv.writer(record).writerow([f"{stem}-{step}", perturbation_norm, psnr, ssim])
                    torchvision.utils.save_image(torch.cat([image, saved_delta * 10, attacked]),
                                                 str(experiment_dir / f"{stem}-{step}.png"), nrow=2)
                    torchvision.utils.save_image(attacked, str(output_dir / f"{stem}-8.png"))
    print("time cost", time.time() - start, "s")


if __name__ == "__main__":
    from launcher import main
    main("remove")
