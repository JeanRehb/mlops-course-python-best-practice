import os
from PIL import Image
import torch
from torchvision import transforms, models
from torchvision.models.resnet import ResNet18_Weights
from torch.utils.tensorboard import SummaryWriter


class ImageLoader:
    def __init__(self, image_dir):
        self.D = image_dir

    def load_images(self):
        imgs = []
        for F in os.listdir(self.D):
            if F.endswith(".jpg") or F.endswith(".png") or F.endswith(".JPG"):
                imgs.append(Image.open(os.path.join(self.D, F)))
        return imgs


class ImageProcessor:
    def __init__(self, size):
        self.s = size

    def apply_preprocessing(self, img_list):
        """

        Args:
            img_list:

        Returns:

        """
        p_images = []
        for img in img_list:
            t = transforms.Compose(
                [
                    transforms.Resize((self.s, self.s)),
                    transforms.Grayscale(num_output_channels=3),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
            p_images.append(t(img))
        return p_images


class Resnet18Predictor:
    def __init__(self):
        self.mdl = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.mdl.eval()

    def predict(self, tensor_list):
        results = []
        for img_tensor in tensor_list:
            pred = self.mdl(img_tensor.unsqueeze(0))
            results.append(torch.argmax(pred, dim=1).item())
        return results


if __name__ == "__main__":
    writer = SummaryWriter("tensorboard/runs/image_classification")

    loader = ImageLoader(image_dir="data/")
    images = loader.load_images()

    processor = ImageProcessor(256)
    preprocessed_tensor = processor.apply_preprocessing(images)

    predictor = Resnet18Predictor()
    results = predictor.predict(tensor_list=preprocessed_tensor)

    for i, tensor in enumerate(preprocessed_tensor):
        writer.add_image(f"{results[i]}", tensor, 0)

    writer.close()

    print(f"Predicted {len(results)} images: {results}")
