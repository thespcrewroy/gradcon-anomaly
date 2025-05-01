'''Import dataset for anomaly detection'''
import os # Used to get the path of the dataset

from PIL import Image # Used to open the image
import pickle # Used to load the dataset
import torch.utils.data as data # Used to create a dataset
from torch.utils.data import Dataset # Used to create a dataset

'''
This function is used to load the dataset for anomaly detection.
It takes the following arguments:
 - root: The root directory of the dataset
 - split: The type of split to load (e.g. 'train', 'val')
 - in_channel: The number of input channels (e.g. 1 for grayscale data, 3 for RGB data)
 - transform: The transform to apply to the input
 - target_transform: The transform to apply to the target image of reconstruction
 - inlier_class: The inlier class
It returns a dataset object that can be used to load the data.
'''
class AnomalyDataset(data.Dataset):
    def __init__(self, root, split='train', in_channel=3, transform=None, target_transform=None, inlier_class=None):
        self.root = os.path.expanduser(root) # Expand the user directory
        self.transform = transform # Transform to apply to the input
        self.target_transform = target_transform # Transform to apply to the target image of reconstruction
        self.split = split  # training set or test set
        self.in_channel = in_channel # Number of input channels
        self.inlier_class = inlier_class # Inlier class
        self.label_img_data = [] # List to store the image data

        with open(os.path.join(self.root, 'data_split_%s.pkl' % split), 'rb') as pkl: # Load the dataset
            split_data = pickle.load(pkl) # Load the split data

        if split.split('_')[0] == 'test': # If the split is test
            self.label_img_data = split_data # Load all the data
        else: # If the split is train or val
            # Load inlier class samples
            for x in split_data: # For each sample in the split data
                if x[0] == self.inlier_class: # If the sample is inlier class
                    self.label_img_data.append(x) # Append the sample to the list

    """
    Args:
        index (int): Index
    Returns:
        img (tensor): Input image after transform
        target (tensor): Target image after transform for the reconstruction
        label (tensor): Class label for input image
    """
    def __getitem__(self, index):
        label, img = self.label_img_data[index][0], self.label_img_data[index][1] # Get the label and image data

        if self.in_channel == 1: # If the input channel is 1 (grayscale)
            img = Image.fromarray(img, mode='L') # Convert the image to grayscale
        elif self.in_channel == 3: # If the input channel is 3 (RGB)
            img = Image.fromarray(img, mode='RGB') # Convert the image to RGB
        target = img # Target image is the same as input image

        if self.transform is not None: # If transform is not None
            img = self.transform(img) # Apply the transform to the input image

        if self.target_transform is not None: # If target transform is not None
            target = self.target_transform(target) # Apply the target transform to the target image

        return img, target, label # Return the input image, target image, and label

    def __len__(self): # Get the length of the dataset
        return len(self.label_img_data) # Return the length of the label image data
    
'''
This function is used to load the dataset for anomaly detection.
It takes the following arguments:
 - root_dir: The root directory of the dataset
 - split: The type of split to load (e.g. 'train', 'val')
 - transform: The transform to apply to the input
It returns a dataset object that can be used to load the data.
'''    
class MaldebDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.transform = transform # Transform to apply to the input
        self.samples = [] # List to store the samples

        benign_dir = os.path.join(root_dir, split, 'Benign') # Path to benign images
        malicious_dir = os.path.join(root_dir, split, 'Malicious') # Path to malicious images

        if os.path.exists(benign_dir): # Check if benign directory exists
            for fname in os.listdir(benign_dir): # For each file in the benign directory
                if fname.lower().endswith(('.png', '.jpg')): # Check if the file is an image
                    self.samples.append((os.path.join(benign_dir, fname), 1))  # label 1 = benign

        if os.path.exists(malicious_dir): # Check if malicious directory exists
            for fname in os.listdir(malicious_dir): # For each file in the malicious directory
                if fname.lower().endswith(('.png', '.jpg')): # Check if the file is an image
                    self.samples.append((os.path.join(malicious_dir, fname), 0))  # label 0 = malicious

    def __getitem__(self, index): # Get the item at the specified index
        path, label = self.samples[index] # Get the path and label of the sample
        image = Image.open(path).convert('L')  # grayscale
        if self.transform: # If transform is not None
            image = self.transform(image) # Apply the transform to the image
        return image, image, label  # input, target, label

    def __len__(self): # Get the length of the dataset
        return len(self.samples) # Return the length of the samples
