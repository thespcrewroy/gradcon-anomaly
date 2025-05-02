import torch
from crewai import Agent, Task, Crew
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from models import GradConCAE
from ae_grad_reg import gradcon_score

# ----------- Setup ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define relative paths
root_dir = Path(__file__).parent.resolve()
ckpt_path = root_dir / 'save' / 'maldeb' / 'GradConCAE_test' / 'model_best.pth.tar'
image_path = root_dir / 'datasets' / 'Maldeb' / 'val' / 'Malicious' / '0b27fbcb91a355c14ee1367af92b962f1901e4d48a08e719f6e4dada1d599130_L.png'

# Load model
model = GradConCAE(in_channel=1)  # grayscale
model = torch.nn.DataParallel(model).to(device)
checkpoint = torch.load(ckpt_path, map_location=device)
model.load_state_dict(checkpoint['state_dict'])
ref_grad = checkpoint.get('ref_grad', None)
model.eval()

# ----------- Image Transform ----------
transform = transforms.Compose([
    transforms.Resize((252, 252)),
    transforms.ToTensor(),
])

# ----------- Single-image Dataset ----------
class SingleImageDataset(Dataset):
    def __init__(self, image_tensor, label=1):
        self.image = image_tensor
        self.label = label

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.image, self.image, torch.tensor(self.label)

# ----------- GradCon Prediction ----------
def predict_image(image_path):
    image = Image.open(image_path).convert("L")
    image_tensor = transform(image).unsqueeze(0).to(device)
    loader = DataLoader(SingleImageDataset(image_tensor), batch_size=1)

    result = gradcon_score(
        model,
        in_cls=1,
        grad_loss_weight=0.05,
        ref_grad=ref_grad,
        nlayer=4,
        device=device,
        test_loader=loader
    )

    score = result[0, 1]
    threshold = 0.15  # empirical threshold
    return "Benign" if score < threshold else "Malicious"

# ----------- CrewAI Setup ----------
task = Task(
    description="Classify a given image as malicious or benign using a trained deep learning model.",
    expected_output="A classification result: either 'Malicious' or 'Benign'.",
    tools=[],
    async_execution=False,
    agent=None
)

agent = Agent(
    role="Image Security Analyst",
    goal="Detect malicious content in images.",
    backstory="This agent uses a trained GradConCAE model to detect anomalies in images to prevent cybersecurity threats.",
    tasks=[task]
)

task.agent = agent
crew = Crew(agents=[agent], tasks=[task], verbose=True)

# ----------- Run ----------
result = predict_image(image_path)
task.description += f"\n\nThe image reconstruction error indicates this image is: **{result}**."
crew.kickoff()
