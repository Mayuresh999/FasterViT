import torch
from torchvision import transforms
from PIL import Image
from fastervit import create_model

num_classes = 3
class_names = ['paper', 'rock', 'scissors']

model = create_model("faster_vit_0_224", pretrained=True)
model.head = torch.nn.Linear(model.head.in_features, num_classes)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

model.load_state_dict(torch.load("faster_vit_custon_model.pth"))
model.eval()

preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def load_image(image_path):
    image = Image.open(image_path)
    image = preprocess(image).unsqueeze(0)
    return image.to(device)

def predict(image_path, model, class_names):
    image = load_image(image_path)
    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, dim=1)
        predicted_class = class_names[preds.item()]
    return predicted_class


image_path = r"Rock Paper Scissors.v1-raw-300x300.folder\test\paper\paper1_png.rf.6771d0803db4a4cb6e5e96339f785b9d.jpg"
predicted_class = predict(image_path, model, class_names)
print(predicted_class)