'Importing the required libraries'
import torch # Importing the PyTorch library
import torch.nn.functional as func # Importing the functional module from PyTorch
import numpy as np # Importing the NumPy library

import utils # Importing the utils module

'''
This function is used to train the model.
It takes as input ars
* The model - this is the model to be trained (VAE)
* Device - this is the device on which the model is trained (CPU or GPU)
* Train_loader - this is the data loader for training data
* Optimizer - this is the optimizer
* Epoch - this is the number of epochs (int)
* Print_freq - this is how often to print the loss (int)
* Grad_loss_weight - this is the weight for the gradient loss (float)
* Ref_grad - this is a list of average gradients generated while training (tensor)
* nlayer - this is the number of decoder layers in the model (int)

The function trains the model for one epoch and prints the loss.
'''
def train(model, device, train_loader, optimizer, epoch, print_freq, grad_loss_weight, ref_grad, nlayer):
    model.train() # Set the model to training mode
    losses = utils.AverageMeter() # Create an AverageMeter object to track the loss
    recon_losses = utils.AverageMeter() # Create an AverageMeter object to track the reconstruction loss
    grad_losses = utils.AverageMeter() # Create an AverageMeter object to track the gradient loss

    for batch_idx, (data, target_data, label) in enumerate(train_loader): # Loop through the training data
        data = data.to(device) # Move the data to the device (CPU or GPU)
        target_data = target_data.to(device) # Move the target data to the device (CPU or GPU)
        optimizer.zero_grad() # Zero the gradients of the optimizer
        model.zero_grad() # Zero the gradients of the model

        recon_batch = model(data) # Forward pass through the model
        recon_loss = func.mse_loss(recon_batch, target_data) # Calculate the reconstruction loss

        # Calculate the gradient loss for each layer
        grad_loss = 0 # Initialize the gradient loss
        for i in range(nlayer): # Loop through each layer
            wrt = model.module.up[int(2*i)].weight # Get the weight of the layer
            target_grad = torch.autograd.grad(recon_loss, wrt, create_graph=True, retain_graph=True)[0] # Calculate the gradient of the reconstruction loss with respect to the weight
            grad_loss += -1 * func.cosine_similarity(target_grad.view(-1, 1),
                                                     ref_grad[i].avg.view(-1, 1), dim=0) # Calculate the cosine similarity between the target gradient and the reference gradient

        # In the first iteration, since there is no history of training gradients, gradient loss is not utilized
        if ref_grad[0].count == 0: # If this is the first iteration
            grad_loss = torch.FloatTensor([0.0]).to(device) # Set the gradient loss to 0
        else: # If this is not the first iteration
            grad_loss = grad_loss / nlayer # Normalize the gradient loss
        loss = recon_loss + grad_loss_weight * grad_loss # Calculate the total loss

        losses.update(loss.item(), data.size(0))  # data.size(0): Batch size
        recon_losses.update(recon_loss.item(), data.size(0)) # Update the reconstruction loss
        grad_losses.update(grad_loss.item(), data.size(0)) # Update the gradient loss
        loss.backward() # Backward propagation to calculate the gradients

        # Update the reference gradient
        for i in range(nlayer): # Loop through each layer
            ref_grad[i].update(model.module.up[2*i].weight.grad, 1) # Update the reference gradient with the current gradient

        optimizer.step() # Step the optimizer

        if batch_idx % print_freq == 0: # If the batch index is a multiple of the print frequency
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f}) Recon Loss {recon_loss.val:.4f} ({recon_loss.avg:.4f}) '
                  'Grad Loss {grad_loss.val:.4f} ({grad_loss.avg:.4f})'
                  .format(epoch, batch_idx, len(train_loader), loss=losses, recon_loss=recon_losses,
                          grad_loss=grad_losses)) # Print the loss

'''
This function is used to test the model.
It takes as input args
* The model - this is the model to be tested (VAE)
* Device - this is the device on which the model is tested (CPU or GPU)
* Test_loader - this is the data loader for test data
* Epoch - this is the number of epochs (int)
* Print_freq - this is how often to print the loss (int)
* Grad_loss_weight - this is the weight for the gradient loss (float)
* Ref_grad - this is a list of average gradients generated while training (tensor)
* nlayer - this is the number of decoder layers in the model (int)
The function tests the model and prints the loss.
'''
def test(model, device, test_loader, epoch, print_freq, grad_loss_weight, ref_grad, nlayer):
    model.eval() # Set the model to evaluation mode
    losses = utils.AverageMeter() # Create an AverageMeter object to track the loss
    recon_losses = utils.AverageMeter() # Create an AverageMeter object to track the reconstruction loss
    grad_losses = utils.AverageMeter() # Create an AverageMeter object to track the gradient loss

    for batch_idx, (data, target_data, label) in enumerate(test_loader): # Loop through the test data
        data = data.to(device) # Move the data to the device (CPU or GPU)
        target_data = target_data.to(device) # Move the target data to the device (CPU or GPU)
        model.zero_grad() # Zero the gradients of the model

        recon_batch = model(data) # Forward pass through the model
        recon_loss = func.mse_loss(recon_batch, target_data) # Calculate the reconstruction loss

        # Calculate the gradient loss for each layer
        grad_loss = 0 # Initialize the gradient loss
        for i in range(nlayer): # Loop through each layer
            wrt = model.module.up[int(2*i)].weight # Get the weight of the layer
            target_grad = torch.autograd.grad(recon_loss, wrt, create_graph=True, retain_graph=True)[0] # Calculate the gradient of the reconstruction loss with respect to the weight
            grad_loss += -1 * func.cosine_similarity(target_grad.view(-1, 1),
                                                     ref_grad[i].avg.view(-1, 1), dim=0) # Calculate the cosine similarity between the target gradient and the reference gradient

        grad_loss = grad_loss / nlayer # Normalize the gradient loss
        loss = recon_loss + grad_loss_weight * grad_loss # Calculate the total loss

        losses.update(loss.item(), data.size(0)) # data.size(0): Batch size
        recon_losses.update(recon_loss.item(), data.size(0)) # Update the reconstruction loss
        grad_losses.update(grad_loss.item(), data.size(0)) # Update the gradient loss

        # Update the reference gradient
        if batch_idx == 0: # Only visualize the first batch
            nimg = 3  # Visualize three sample images on Tensorboard
            input_img = data[:nimg] # Input images
            recon_img = recon_batch[:nimg, :].view(nimg, data.shape[1], data.shape[2], data.shape[3]) # Reconstructed images
            target_img = target_data[:nimg] # Target images

        if batch_idx % print_freq == 0: # If the batch index is a multiple of the print frequency
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f}) Recon Loss {recon_loss.val:.4f} ({recon_loss.avg:.4f}) '
                  'Grad Loss {grad_loss.val:.4f} ({grad_loss.avg:.4f})'
                  .format(epoch, batch_idx, len(test_loader), loss=losses, recon_loss=recon_losses,
                          grad_loss=grad_losses)) # Print the loss

    print(' * Loss {loss.avg:.3f}'.format(loss=losses)) # Print the average loss
    return losses.avg, recon_losses.avg, grad_losses.avg, input_img, recon_img, target_img # Return the average loss, reconstruction loss, gradient loss, input images, reconstructed images, and target images

'''
This function is used to calculate the gradient loss.
It takes as input args
* The model - this is the model to be tested (AE)
* In_cls - this is the inlier class (int)
* Grad_loss_weight - this is the weight for the gradient loss (float)
* Ref_grad - this is a list of average gradients generated while training (tensor)
* Nlayer - this is the number of decoder layers in the model (int)
* Device - this is the device on which the model is tested (CPU or GPU)
* Test_loader - this is the data loader for test data (DataLoader)
The function calculates the gradient loss and returns the results.

Returns:
* The results of the gradient loss calculation (ndarray) - (number of samples) x 2 (label, estimated score)
'''
def gradcon_score(model, in_cls, grad_loss_weight, ref_grad, nlayer, device, test_loader):
    model.eval() # Set the model to evaluation mode

    results = np.zeros([len(test_loader.dataset), 2]) # Create an array to store the results
    for batch_idx, (data, target_data, class_label) in enumerate(test_loader): # Loop through the test data
        if batch_idx % 10 == 0: # Print the progress every 10 batches
            print('Evaluation inlier {0}: [{1} / {2}]...'.format(in_cls, batch_idx, len(test_loader))) # Print the progress

        data = data.to(device) # Move the data to the device (CPU or GPU)
        target_data = target_data.to(device) # Move the target data to the device (CPU or GPU)

        model.zero_grad() # Zero the gradients of the model

        recon_batch = model(data) # Forward pass through the model
        recon_loss = func.mse_loss(recon_batch, target_data) # Calculate the reconstruction loss

        recon_loss.backward() # Backward pass to calculate the gradients

        grad_loss = 0 # Initialize the gradient loss
        for i in range(nlayer): # Loop through each layer
            target_grad = model.module.up[int(2*i)].weight.grad # Get the gradient of the layer
            grad_loss += 1 * func.cosine_similarity(target_grad.view(-1, 1), ref_grad[i].avg.view(-1, 1), dim=0) # Calculate the cosine similarity between the target gradient and the reference gradient

        grad_loss = grad_loss / nlayer # Normalize the gradient loss

        score = -1 * recon_loss + grad_loss_weight * grad_loss # Calculate the custom anomaly score
        inout_label = 1 if class_label == in_cls else 0 # Set the in/outlier label of the sample

        results[batch_idx, 0] = inout_label # Store the in/outlier label
        results[batch_idx, 1] = score.cpu().detach().numpy() # Store the score

    return results # Return the results
