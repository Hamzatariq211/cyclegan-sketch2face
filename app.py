from flask import Flask, request, send_file
from torchvision import transforms
from PIL import Image
from io import BytesIO
import torch
from generator import Generator

app = Flask(__name__, static_folder='.', static_url_path='')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load both models
G_AB = Generator().to(device)  # Face → Sketch
G_AB.load_state_dict(torch.load("G_AB_trained.pth", map_location=device))
G_AB.eval()

G_BA = Generator().to(device)  # Sketch → Face
G_BA.load_state_dict(torch.load("G_BA_trained.pth", map_location=device))
G_BA.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def is_sketch(tensor):
    """
    Stronger sketch detection:
    - Sketches have near-equal R, G, B values (low color variance)
    - Sketches often have lots of white (background) and thin black lines
    """
    r, g, b = tensor[0], tensor[1], tensor[2]
    grayscale_variance = torch.mean(torch.abs(r - g) + torch.abs(r - b) + torch.abs(g - b))
    
    # High ratio of bright pixels (i.e., mostly white background)
    white_ratio = (tensor > 0.9).float().mean()

    # Low color variance AND high whiteness = likely sketch
    print(f"Grayscale Var: {grayscale_variance:.4f}, White Ratio: {white_ratio:.4f}")
    return grayscale_variance < 0.05 and white_ratio > 0.7



@app.route("/")
def index():
    return app.send_static_file("index.html")

@app.route("/convert", methods=["POST"])
def convert_image():
    file = request.files["file"]
    img = Image.open(file).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        if is_sketch(tensor[0]):
            print("Detected Sketch → Converting to Face")
            output = G_BA(tensor)
        else:
            print("Detected Face → Converting to Sketch")
            output = G_AB(tensor)

    output = output.squeeze().cpu() * 0.5 + 0.5
    buffer = BytesIO()
    transforms.ToPILImage()(output).save(buffer, format="PNG")
    buffer.seek(0)
    return send_file(buffer, mimetype="image/png")

if __name__ == "__main__":
    app.run(debug=True)
