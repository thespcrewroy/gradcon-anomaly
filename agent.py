import torch
from crewai import Agent, Task, Crew
from PIL import Image
import torchvision.transforms as transforms
import os

from models import GradConCAE  # Correctly import your model

# ----------- Load model correctly ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set parameters manually here since args is undefined
ckpt_path = 'C:\\Users\\hetpa\\DLFinal\\gradcon-anomaly\\save\\maldeb\\GradConCAE_test_benign\\model_best.pth.tar'
in_channel = 1  # 'maldeb' is grayscale
input_size = (252, 252)  # Match training input

# Load model
model = GradConCAE(in_channel=in_channel)
model = torch.nn.DataParallel(model).to(device)

# Load checkpoint
checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['state_dict'])
ref_grad = checkpoint.get('ref_grad', None)  # Only needed if you want GradCon scoring
model.eval()

# ----------- Image transform ----------
transform = transforms.Compose([
    transforms.Resize(input_size),
    transforms.ToTensor(),
])

# ----------- Image prediction function ----------
def predict_image(image_path):
    image = Image.open(image_path).convert("L")  # "L" for grayscale
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        recon_error = ((output - image) ** 2).view(output.size(0), -1).mean(dim=1).item()

    # Using a placeholder threshold; ideally you'd determine this via validation
    threshold = 0.25 
    return "Benign" if recon_error < threshold else "Malicious"

# ----------- Define a CrewAI Task ----------
task = Task(
    description="Classify a given image as malicious or benign using a trained deep learning model.",
    expected_output="A classification result: either 'Malicious' or 'Benign'.",
    tools=[],
    async_execution=False,
    agent=None
)

# ----------- Define the Agent ----------
agent = Agent(
    role="Image Security Analyst",
    goal="Detect malicious content in images.",
    backstory="This agent uses a trained GradConCAE model to detect anomalies in images to prevent cybersecurity threats.",
    tasks=[task]
)

task.agent = agent

# ----------- Create a Crew ----------
crew = Crew(
    agents=[agent],
    tasks=[task],
    verbose=True
)

# ----------- Run the Prediction ----------
image_path = 'C:\\Users\\hetpa\\DLFinal\\gradcon-anomaly\\datasets\\Maldeb\\val\Malicious\\0b27fbcb91a355c14ee1367af92b962f1901e4d48a08e719f6e4dada1d599130_L.png'
result = predict_image(image_path)

# Inject the prediction result into the task description
task.description += f"\n\nThe image reconstruction error indicates this image is: **{result}**."

# Kick off CrewAI to generate a response with that context
crew.kickoff()

