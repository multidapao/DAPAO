from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class AttackDataset(Dataset):
    def __init__(self, directory, image_size=256, carrier_list=None):
        self.images = sorted(Path(directory).iterdir())
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)), transforms.ToTensor(),
        ])
        self.carriers = None
        if carrier_list is not None:
            self.carriers = Path(carrier_list).read_text().splitlines()[:100]
            if len(self.carriers) < len(self.images):
                raise ValueError("Not enough clean carriers for the donor images")

    def load_image(self, path):
        with Image.open(str(path)) as image:
            # Retain the original PIL resize before the torchvision transform.
            image = image.convert("RGB").resize((256, 256))
            return self.transform(image)

    def __getitem__(self, index):
        path = self.images[index]
        image = self.load_image(path)
        if self.carriers is None:
            return image, str(path)
        return image, str(path), self.load_image(self.carriers[index])

    def __len__(self):
        return len(self.images)
