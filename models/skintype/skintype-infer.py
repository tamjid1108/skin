import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Load the model
model = torch.load('saved_models/skintype/best_20.pt')
model.eval()  # Set the model to evaluation mode


def predict_skin_type(x):
    preprocess = transforms.Compose([transforms.ToPILImage(),
                                     transforms.ToTensor(),
                                     transforms.Resize((224, 224)),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])])

    img = Image.open(x).convert("RGB")
    img = preprocess(np.array(img))
    img = img.view(1, 3, 224, 224)

    if torch.cuda.is_available():
        img = img.cuda()

    with torch.no_grad():
        out = model(img)

        return out.argmax(1).item()


image_path = "datasets/Oily-Dry-Skin-Types/test/normal/normal_7d00a675b9fabb9780d8_jpg.rf.3ba7f8e5db8941144c0690dc2ce4b1e5.jpg"

print(predict_skin_type(image_path))
