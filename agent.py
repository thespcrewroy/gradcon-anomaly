'''Import necessary libraries and modules for the CrewAI agent.'''
import torch  # Import PyTorch for deep learning
from crewai import Agent, Task, Crew # Import CrewAI classes for agent and task management
from PIL import Image # Import PIL for image processing
import torchvision.transforms as transforms # Import torchvision for image transformations
import os # Import os for file path operations

from models import GradConCAE  # Correctly import your model

# ----------- Load model correctly ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set parameters manually here since args is undefined
ckpt_path = './save/maldeb/GradConCAE/model_best.pth.tar'
in_channel = 1  # 'maldeb' is grayscale
input_size = (252, 252)  # Match training input

# Load model
model = GradConCAE(in_channel=in_channel) # Initialize the model
model = torch.nn.DataParallel(model).to(device) # Use DataParallel for multi-GPU support if available

# Load checkpoint
checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False) # Load the checkpoint
model.load_state_dict(checkpoint['state_dict']) # Load the model state dict
ref_grad = checkpoint.get('ref_grad', None)  # Only needed if you want GradCon scoring
model.eval() # Set model to evaluation mode

# ----------- Image transform ----------
transform = transforms.Compose([
    transforms.Resize(input_size),
    transforms.ToTensor(),
]) # Define the image transformation pipeline

# ----------- Image prediction function ----------
def predict_image(image_path): # Function to predict if an image is malicious or benign
    image = Image.open(image_path).convert("L")  # "L" for grayscale
    image = transform(image).unsqueeze(0).to(device) # Apply transformations and add batch dimension

    with torch.no_grad(): # Disable gradient calculation for inference
        output = model(image) # Forward pass through the model
        recon_error = ((output - image) ** 2).view(output.size(0), -1).mean(dim=1).item() # Calculate reconstruction error

    # Using a placeholder threshold; ideally you'd determine this via validation
    threshold = 0.25 # Set a threshold for classification
    return "Benign" if recon_error < threshold else "Malicious" # Classify based on reconstruction error

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
image_path = './datasets/Maldeb/val/Malicious/0b27fbcb91a355c14ee1367af92b962f1901e4d48a08e719f6e4dada1d599130_L.png'
result = predict_image(image_path)

# Inject the prediction result into the task description
task.description += f"\n\nThe image reconstruction error indicates this image is: **{result}**."

# Kick off CrewAI to generate a response with that context
crew.kickoff()

