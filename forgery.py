#!/usr/bin/env python3

def attack_dual():
    import csv
    import os
    from pathlib import Path
    import time

    import kornia
    import torch
    from torch.utils.data import DataLoader
    import torchvision
    from tqdm import tqdm

    from color import DenseLoss, project, select_channel_weights
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
    dataset = AttackDataset(os.environ["TRAIN"], image_size, os.environ["CARRIER_LIST"])
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    dense_loss = DenseLoss().to(device)
    start = time.time()

    for image, name, carrier in tqdm(loader):
        delta1 = torch.nn.Parameter(0.05 * torch.randn((1, 3, image_size, image_size)).to(device))
        delta2 = torch.nn.Parameter(0.05 * torch.randn((1, 3, image_size, image_size)).to(device))
        optimizer1 = torch.optim.AdamW([delta1], lr=0.15, weight_decay=0.0001)
        optimizer2 = torch.optim.AdamW([delta2], lr=0.15, weight_decay=0.0001)
        scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer1, T_max=epochs // 10, eta_min=0.049,
        )
        scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer2, T_max=epochs // 10, eta_min=0.049,
        )
        ssim_loss = pytorch_ssim.SSIM(window_size=256)
        image = image.to(device)
        with torch.no_grad():
            weights, indices = select_channel_weights(
                dense_loss(image)[0], clusters, selected_clusters,
            )
        print(f"selected channels for {Path(name[0]).name}: {indices.tolist()} ({len(indices)}/{weights.shape[1]})")

        for _ in range(epochs):
            image = image.to(device)
            optimizer1.zero_grad()
            image_feature = dense_loss(image)
            attacked_feature = dense_loss(image + delta1)
            loss = -ssim_loss(attacked_feature * weights, image_feature * weights)
            print("donor loss:", loss.item())
            loss.backward()
            optimizer1.step()
            scheduler1.step()
            perturbation = delta1.clone().detach()
            norm = torch.norm(perturbation, float("inf")).cpu()
            if norm >= eps:
                projected = project(image + perturbation, image, eps / norm)
                delta1.data = projected - image

        for step in range(epochs):
            image = image.to(device)
            carrier = carrier.to(device)
            optimizer2.zero_grad()
            donor_feature = dense_loss(image + delta1)
            carrier_feature = dense_loss(carrier + delta2)
            loss = -ssim_loss(carrier_feature * (1 - weights), donor_feature * (1 - weights))
            print("carrier loss:", loss.item())
            loss.backward()
            optimizer2.step()
            scheduler2.step()
            donor_delta = delta1.clone().detach()
            carrier_delta = delta2.clone().detach()
            norm = torch.norm(carrier_delta + donor_delta, float("inf")).cpu()
            if norm >= eps:
                norm = torch.norm(donor_delta + carrier_delta, float("inf")).cpu()
                projected = project(carrier + donor_delta + carrier_delta, carrier, eps / norm)
                delta2.data = projected - carrier - donor_delta

            if step % 5 == 0:
                with torch.no_grad(), (experiment_dir / f"{experiment_name}-record.csv").open("a", newline="") as record:
                    saved_delta1 = delta1.mean(dim=0, keepdim=True).cpu()
                    saved_delta2 = delta2.mean(dim=0, keepdim=True).cpu()
                    image = image.cpu()
                    carrier = carrier.cpu()
                    perturbation = saved_delta2 + saved_delta1
                    attacked = carrier + perturbation
                    perturbation_norm = torch.norm(perturbation, float("inf")).numpy()
                    psnr = -kornia.losses.psnr_loss(attacked, carrier, 2).item()
                    ssim = 1 - 2 * kornia.losses.ssim_loss(attacked, carrier, window_size=11, reduction="mean").item()
                    stem = Path(name[0]).name.split(".")[0]
                    csv.writer(record).writerow([f"{stem}-{step}", perturbation_norm, psnr, ssim])
                    torchvision.utils.save_image(torch.cat([carrier, saved_delta1 * 10, saved_delta2 * 10, attacked]),
                                                 str(experiment_dir / f"{stem}-{step}.png"), nrow=2)
                    torchvision.utils.save_image(attacked, str(output_dir / f"{stem}-8.png"))
    print("time cost", time.time() - start, "s")


def attack_transfer():
    import csv
    import os
    from pathlib import Path
    import time

    import kornia
    import torch
    from torch.utils.data import DataLoader
    import torchvision
    from tqdm import tqdm

    from color import DenseLoss, project, select_channel_weights
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
    dataset = AttackDataset(os.environ["TRAIN"], image_size, os.environ["CARRIER_LIST"])
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    dense_loss = DenseLoss().to(device)
    start = time.time()

    for image, name, carrier in tqdm(loader):
        delta = torch.nn.Parameter(0.05 * torch.randn((1, 3, image_size, image_size)).to(device))
        optimizer = torch.optim.AdamW([delta], lr=0.15, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs // 10, eta_min=0.049,
        )
        ssim_loss = pytorch_ssim.SSIM(window_size=256)
        image = image.to(device)
        with torch.no_grad():
            weights, indices = select_channel_weights(
                dense_loss(image)[0], clusters, selected_clusters,
            )
        print(f"selected channels for {Path(name[0]).name}: {indices.tolist()} ({len(indices)}/{weights.shape[1]})")

        for step in range(epochs):
            image = image.to(device)
            carrier = carrier.to(device)
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
                    carrier = carrier.cpu()
                    perturbation = -saved_delta
                    attacked = carrier + perturbation
                    perturbation_norm = torch.norm(perturbation, float("inf")).numpy()
                    psnr = -kornia.losses.psnr_loss(attacked, carrier, 2).item()
                    ssim = 1 - 2 * kornia.losses.ssim_loss(attacked, carrier, window_size=11, reduction="mean").item()
                    stem = Path(name[0]).name.split(".")[0]
                    csv.writer(record).writerow([f"{stem}-{step}", perturbation_norm, psnr, ssim])
                    torchvision.utils.save_image(torch.cat([carrier, saved_delta * 10, attacked]),
                                                 str(experiment_dir / f"{stem}-{step}.png"), nrow=2)
                    torchvision.utils.save_image(attacked, str(output_dir / f"{stem}-8.png"))
    print("time cost", time.time() - start, "s")


if __name__ == "__main__":
    from launcher import main
    main("forgery")
