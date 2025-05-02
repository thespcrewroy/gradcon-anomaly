'''Import Libraries'''
import os # Importing os for file path operations

import shutil # Importing shutil for file operations
import torch # Importing PyTorch for deep learning


class AverageMeter(object):
    """
    Computes and stores the average and current value
    """
    def __init__(self): # Constructor to initialize the AverageMeter object
        self.reset() # Resetting the values to default
        self.val = 0 # Current value
        self.avg = 0 # Average value
        self.sum = 0 # Sum of values
        self.prev = 0 # Previous value
        self.count = 0 # Count of values

    def reset(self): # Resetting the values to default
        self.val = 0 # Current value
        self.avg = 0 # Average value
        self.sum = 0 # Sum of values
        self.count = 0 # Count of values

    def update(self, val, n=1): # Updating the values
        self.val = val # Current value
        self.sum += val * n # Sum of values
        self.count += n # Count of values
        self.avg = self.sum / self.count # Average value


def save_checkpoint(state, is_best, checkpointdir): # Function to save the model checkpoint
    fullpath = os.path.join(checkpointdir, 'checkpoint.pth.tar') # Full path for the checkpoint
    fullpath_best = os.path.join(checkpointdir, 'model_best.pth.tar') # Full path for the best model
    torch.save(state, fullpath) # Saving the state

    if is_best: # If the model is the best, save it as the best model
        shutil.copyfile(fullpath, fullpath_best) # Copying the file to the best model path
