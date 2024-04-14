import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from acne_model import MyNet

# Load the model
model = torch.load('saved_models/acne-severity/best_6.pt')
model.eval()  # Set the model to evaluation mode


def predict_acne_level(image_path):
    # Define image preprocessing
    preprocess = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    class_name = ['Minimal', 'Moderate', 'Severe', 'Extreme']
    image = Image.open(image_path)
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # Add batch dimension

    # If GPU is available, move the input to GPU
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    # Perform inference
    with torch.no_grad():
        output = model(input_batch)

    # Post-process the output
    # applying Softmax to results
    prob = nn.Softmax(dim=1)
    probs = prob(output)
    # print(probs)
    predicted_class = torch.argmax(probs, axis=1).tolist()[0]
    return class_name[predicted_class]


image_path = 'acne-severe.jpg'

while True:
    input_image = input('Image path: ')
    print(predict_acne_level(input_image))
