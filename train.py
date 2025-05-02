'''Import necessary libraries'''
import argparse # for command line argument parsing
import os # for file and directory operations
import time # for time operations
import errno # for error handling

import torch # for tensor operations and GPU support
import torchvision.utils as vutils # for visualization of images
import torch.utils.data as data # for data loading and processing
import torchvision.transforms as transforms # for data transformations
from torch import optim # for optimization algorithms
from tensorboardX import SummaryWriter
import numpy as np # for numerical operations
from sklearn.metrics import precision_recall_curve # for precision-recall curve calculation
from sklearn.metrics import precision_recall_curve  # for precision-recall curve calculation
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay  # for confusion matrix calculation
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt  # for plotting

import models # for defining the model architecture
import utils # for utility functions
import datasets # for loading datasets
from datasets import MaldebDataset # for loading the Maldeb dataset
import ae_grad_reg # for training and testing the model


parser = argparse.ArgumentParser(description='Training GradCon') # create argument parser
parser.add_argument('-e', '--epochs', default=1000, type=int, metavar='N',
                    help='number of total epochs to run') # total number of epochs
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)') # starting epoch
parser.add_argument('--print-freq', '-pf', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)') # frequency of printing training status
parser.add_argument('--write-freq', '-wf', default=5, type=int,
                    metavar='N', help='write frequency (default: 5)') # frequency of writing logs
parser.add_argument('-r', '--resume', default='', type=str, help='Resume training from a checkpoint') # path to resume training
parser.add_argument('--dataset', default='', type=str, help='Dataset to be used for training (e.g. cifar-10, mnist)') # dataset name
parser.add_argument('--dataset_dir', default='./datasets', type=str, help='Path for the dataset') # path to dataset
parser.add_argument('--save_dir', default='./save', type=str, help='Path to save the data') # path to save the model and logs
parser.add_argument('--save_name', default='GradConCAE', type=str, help='Save name') # name for saving the model
parser.add_argument('--grad-loss-weight', '-gw', default=3e-2, type=float, help='gradient loss weight') # weight for gradient loss


def main():
    args = parser.parse_args() # parse command line arguments

    if args.dataset not in ['cifar-10', 'mnist', 'fmnist', 'maldeb']:
        raise ValueError('Dataset should be one of the followings: cifar-10, mnist, fmnist, maldeb')

    dataset = args.dataset # dataset name
    grad_loss_weight = args.grad_loss_weight # weight for gradient loss
    batch_size = 64 # batch size for training
    num_decoder_layers = 4 # number of decoder layers in the model
    in_channel = 3 if dataset == 'cifar-10' else 1  # cifar-10: RGB, mnist, fminst: Graysacle
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # set device to GPU if available, else CPU

    if dataset == 'maldeb': # if dataset is Maldeb
        print("Training with Maldeb dataset (Benign only)") # print message
        dataset_dir = os.path.join(args.dataset_dir, 'maldeb') # path to dataset directory
        save_dir = os.path.join(args.save_dir, dataset, args.save_name) # path to save directory
        log_dir = os.path.join(save_dir, 'logs') # path to logs directory
        os.makedirs(save_dir, exist_ok=True) # create save directory if it does not exist
        writer = SummaryWriter(log_dir=log_dir) # create a SummaryWriter for logging

        ae = models.GradConCAE(in_channel=in_channel) # create an instance of the model
        ae = torch.nn.DataParallel(ae).to(device) # parallelize the model across multiple GPUs if available
        best_loss = 1e20 # initialize best loss to a large value
        optimizer = optim.Adam(ae.parameters(), lr=1e-3) # create an optimizer for the model parameters

        ref_grad = [] # list to store reference gradients
        for i in range(num_decoder_layers): # loop through the decoder layers
            layer_grad = utils.AverageMeter() # create an AverageMeter to keep track of gradients
            layer_grad.avg = torch.zeros(ae.module.up[2 * i].weight.shape).to(device) # initialize average gradient to zero
            ref_grad.append(layer_grad) # append to the list of reference gradients

        if args.resume: # if resume flag is set
            ae_resume_ckpt = os.path.join(args.resume, 'model_best.pth.tar') # path to the checkpoint
            if os.path.isfile(ae_resume_ckpt): # check if the checkpoint file exists
                print("=> loading checkpoint '{}'".format(ae_resume_ckpt)) # print checkpoint path
                checkpoint = torch.load(ae_resume_ckpt) # load the checkpoint
                ae.load_state_dict(checkpoint['state_dict']) # load the model state
                ref_grad = checkpoint['ref_grad'] # load the reference gradients
                optimizer.load_state_dict(checkpoint['optimizer']) # load the optimizer state
                print("=> loaded checkpoint '{}' (epoch {}, best_loss {})".format(
                    ae_resume_ckpt, checkpoint['epoch'], checkpoint['best_loss'])) # print checkpoint details
            else: # if checkpoint file does not exist
                print("=> no checkpoint found at '{}'".format(ae_resume_ckpt)) # print error message
                return # exit the program

        # Define transform to resize and convert to tensor
        transform = transforms.Compose([
            transforms.Resize((252, 252)),  # Match the model's output size
            transforms.ToTensor() # Convert to tensor
        ]) # create a transform to resize and convert to tensor

        # Training and validation loaders for Maldeb
        in_train_loader = torch.utils.data.DataLoader(
            MaldebDataset(dataset_dir, split='train', transform=transform),
            batch_size=batch_size, shuffle=True) # create dataloader for training data

        in_val_loader = torch.utils.data.DataLoader(
            MaldebDataset(dataset_dir, split='val', transform=transform),
            batch_size=batch_size, shuffle=False) # create dataloader for validation data

        classes = ['Malicious', 'Benign'] # list of classes

        timestart = time.time() # start time for training
        for epoch in range(args.start_epoch, args.epochs): # loop through epochs
            print('\n*** Start Training *** Epoch: [%d/%d]\n' % (epoch + 1, args.epochs)) # print epoch details
            train_loss = ae_grad_reg.train(ae, device, in_train_loader, optimizer, epoch + 1, args.print_freq,
                              grad_loss_weight, ref_grad, num_decoder_layers) # train the model
            writer.add_scalar('train_loss', train_loss, epoch + 1) # log training loss

            print('\n*** Start Testing *** Epoch: [%d/%d]\n' % (epoch + 1, args.epochs)) # print epoch details
            loss, recon_loss, grad_loss, input_img, recon_img, target_img = ae_grad_reg.test(
                ae, device, in_val_loader, epoch + 1, args.print_freq, grad_loss_weight,
                ref_grad, num_decoder_layers) # test the model
            
            # Step 1: Pre-pass to compute recon_errors and threshold
            recon_errors = []
            ae.eval()
            with torch.no_grad():
                for images, targets, labels in in_val_loader:
                    images, targets, labels = images.to(device), targets.to(device), labels.to(device)
                    
                    outputs = ae(images)
                    errors = ((outputs - targets) ** 2).view(outputs.size(0), -1).mean(dim=1)

                    # --- after you compute `errors` ---
                    errors = errors.view(-1)          # 1-D  (N,)
                    labels_flat = labels.view(-1)     # 1-D  (N,)

                    # Make sure they really match; crash here if something is wrong
                    assert errors.numel() == labels_flat.numel(), \
                        f"shape mismatch: errors={errors.shape}, labels={labels_flat.shape}"

                    benign_mask = (labels_flat == 1)          # boolean mask, same length
                    recon_errors.extend(errors[benign_mask].cpu().numpy())

            threshold = np.percentile(recon_errors, 95)  # Compute 95th percentile threshold

            # Step 2: Evaluation and metric collection
            class_correct = [0, 0]
            class_total = [0, 0]
            all_labels = []
            all_scores = []
            all_predictions = []

            with torch.no_grad():
                for images, targets, labels in in_val_loader:
                    images, targets, labels = images.to(device), targets.to(device), labels.to(device)
                    outputs = ae(images)
                    recon_error = ((outputs - targets) ** 2).view(outputs.size(0), -1).mean(dim=1)

                    all_labels.extend(labels.cpu().numpy())
                    all_scores.extend(recon_error.cpu().numpy())

                    predicted = (recon_error < threshold).long()  # 1 = benign, 0 = malicious
                    all_predictions.extend(predicted.cpu().numpy())

                    for i in range(len(labels)):
                        label = labels[i].item()
                        class_total[label] += 1
                        class_correct[label] += (predicted[i] == label).item()
                        
            # Log PR Curve
            precision, recall, _ = precision_recall_curve(all_labels, all_scores, pos_label=1)

            writer.add_pr_curve('Validation_PR_Curve',
                                torch.tensor(all_labels),
                                torch.tensor(all_scores),
                                global_step=epoch + 1)

            # Log Confusion Matrix
            cm = confusion_matrix(all_labels, all_predictions)
            fig, ax = plt.subplots() # create a figure for confusion matrix
            ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes).plot(ax=ax)
            writer.add_figure('Confusion_Matrix', fig, global_step=epoch + 1)

            print("\nPer-class Accuracy:") # print per-class accuracy
            for i in range(2): # loop through classes
                if class_total[i] > 0: # if there are samples for the class
                    acc = 100 * class_correct[i] / class_total[i] # calculate accuracy
                    print(f"  {classes[i]}: {acc:.2f}% ({class_correct[i]}/{class_total[i]})") # print accuracy
                    writer.add_scalar(f'val_acc/{classes[i]}', acc, epoch + 1) # log accuracy
                else: # if there are no samples for the class
                    print(f"  {classes[i]}: No samples") # print message

            # Log AUROC Curve
            fpr, tpr, _ = roc_curve(all_labels, all_scores, pos_label=1)
            roc_auc = auc(fpr, tpr)
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('Receiver Operating Characteristic')
            ax.legend(loc="lower right")
            writer.add_figure('AUROC_Curve', fig, global_step=epoch + 1)

            is_best = loss < best_loss # check if current loss is the best
            best_loss = min(loss, best_loss) # update best loss
            if is_best: # if current loss is the best
                best_epoch = epoch + 1 # update best epoch

            if (epoch % args.write_freq == 0) or (epoch == args.epochs - 1): # if epoch is a multiple of write frequency or last epoch
                writer.add_scalar('loss', loss, epoch + 1) # log loss
                writer.add_scalar('recon_loss', recon_loss, epoch + 1) # log reconstruction loss
                writer.add_scalar('grad_loss', grad_loss, epoch + 1) # log gradient loss

                writer.add_image('input_img', vutils.make_grid(input_img, nrow=3), epoch + 1) # log input images
                writer.add_image('recon_img', vutils.make_grid(recon_img, nrow=3), epoch + 1) # log reconstructed images
                writer.add_image('target_img', vutils.make_grid(target_img, nrow=3), epoch + 1) # log target images

                utils.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': ae.state_dict(),
                    'best_loss': best_loss,
                    'last_loss': loss,
                    'optimizer': optimizer.state_dict(),
                    'ref_grad': ref_grad,
                }, is_best, save_dir) # save the model checkpoint

        writer.close() # close the SummaryWriter
        print('Best test loss: %.3f at epoch %d' % (best_loss, best_epoch)) # print best test loss and epoch
        print('Best epoch: ', best_epoch) # print best epoch
        print('Total processing time: %.4f' % (time.time() - timestart)) # print total processing time
        return # exit the program
    
    
    # Default path for other datasets
    dataset_dir = os.path.join(args.dataset_dir, dataset, 'splits') # path to dataset directory

    for in_cls in [0]: # inlier class for training
        print('Training with inlier class: %d' % in_cls) # print inlier class
        save_dir = os.path.join(args.save_dir, dataset, args.save_name + '_inlier-%d' % in_cls) # path to save directory
        log_dir = os.path.join(save_dir, 'logs') # path to logs directory
        try: # create save directory
            os.makedirs(save_dir) # create save directory
        except OSError as exception: # handle error if directory already exists
            if exception.errno != errno.EEXIST: # if error is not "directory already exists"
                raise # re-raise the exception
        writer = SummaryWriter(log_dir=log_dir) # create a SummaryWriter for logging
        print('Save directory: %s' % save_dir) # print save directory

        # Define an autoencoder model
        ae = models.GradConCAE(in_channel=in_channel) # create an instance of the model
        ae = torch.nn.DataParallel(ae).to(device) # parallelize the model across multiple GPUs if available
        best_loss = 1e20 # initialize best loss to a large value
        optimizer = optim.Adam(ae.parameters(), lr=1e-3) # create an optimizer for the model parameters

        # Keep track of traning gradients to calculate the gradient loss
        ref_grad = [] # list to store reference gradients
        for i in range(num_decoder_layers): # loop through the decoder layers
            layer_grad = utils.AverageMeter() # create an AverageMeter to keep track of gradients
            layer_grad.avg = torch.zeros(ae.module.up[2 * i].weight.shape).to(device) # initialize average gradient to zero
            ref_grad.append(layer_grad) # append to the list of reference gradients

        # Resume training from a checkpoint
        if args.resume: # if resume flag is set
            ae_resume_ckpt = os.path.join(args.resume, 'model_best.pth.tar') # path to the checkpoint
            if os.path.isfile(ae_resume_ckpt): # check if the checkpoint file exists
                print("=> loading checkpoint '{}'".format(ae_resume_ckpt)) # print checkpoint path
                checkpoint = torch.load(ae_resume_ckpt) # load the checkpoint
                ae.load_state_dict(checkpoint['state_dict']) # load the model state
                ref_grad = checkpoint['ref_grad'] # load the reference gradients
                optimizer.load_state_dict(checkpoint['optimizer']) # load the optimizer state
                print("=> loaded checkpoint '{}' (epoch {}, best_loss {})"
                      .format(ae_resume_ckpt, checkpoint['epoch'], checkpoint['best_loss'])) # print checkpoint details
            else: # if checkpoint file does not exist
                print("=> no checkpoint found at '{}'".format(ae_resume_ckpt)) # print error message
                return # exit the program

        # Dataloader for training and validation
        in_train_loader = torch.utils.data.DataLoader(
            datasets.AnomalyDataset(dataset_dir, split='train', in_channel=in_channel,
                                 transform=transforms.ToTensor(),
                                 target_transform=transforms.ToTensor(),
                                 inlier_class=in_cls),
            batch_size=batch_size, shuffle=True) # create dataloader for training data

        in_val_loader = torch.utils.data.DataLoader(
            datasets.AnomalyDataset(dataset_dir, split='val', in_channel=in_channel,
                                 transform=transforms.ToTensor(),
                                 target_transform=transforms.ToTensor(),
                                 inlier_class=in_cls),
            batch_size=batch_size, shuffle=True) # create dataloader for validation data

        # Start training
        timestart = time.time() # start time for training
        for epoch in range(args.start_epoch, args.epochs): # loop through epochs

            print('\n*** Start Training *** Epoch: [%d/%d]\n' % (epoch + 1, args.epochs)) # print epoch details
            ae_grad_reg.train(ae, device, in_train_loader, optimizer, epoch + 1, args.print_freq,
                              grad_loss_weight, ref_grad, num_decoder_layers) # train the model

            print('\n*** Start Testing *** Epoch: [%d/%d]\n' % (epoch + 1, args.epochs)) # print epoch details
            loss, recon_loss, grad_loss, input_img, recon_img, target_img = ae_grad_reg.test(
                ae, device, in_val_loader, epoch + 1, args.print_freq, grad_loss_weight,
                ref_grad, num_decoder_layers) # test the model

            is_best = loss < best_loss # check if current loss is the best
            best_loss = min(loss, best_loss) # update best loss

            if is_best: # if current loss is the best
                best_epoch = epoch + 1 # update best epoch

            if (epoch % args.write_freq == 0) or (epoch == args.epochs - 1): # if epoch is a multiple of write frequency or last epoch
                writer.add_scalar('loss', loss, epoch + 1) # log loss
                writer.add_scalar('recon_loss', recon_loss, epoch + 1) # log reconstruction loss
                writer.add_scalar('grad_loss', grad_loss, epoch + 1) # log gradient loss

                writer.add_image('input_img', vutils.make_grid(input_img, nrow=3), epoch + 1) # log input images
                writer.add_image('recon_img', vutils.make_grid(recon_img, nrow=3), epoch + 1) # log reconstructed images
                writer.add_image('target_img', vutils.make_grid(target_img, nrow=3), epoch + 1) # log target images

                utils.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': ae.state_dict(),
                    'best_loss': best_loss,
                    'last_loss': loss,
                    'optimizer': optimizer.state_dict(),
                    'ref_grad': ref_grad,
                }, is_best, save_dir) # save the model checkpoint

        writer.close() # close the SummaryWriter

        print('Best test loss: %.3f at epoch %d' % (best_loss, best_epoch)) # print best test loss and epoch
        print('Best epoch: ', best_epoch) # print best epoch
        print('Total processing time: %.4f' % (time.time() - timestart)) # print total processing time

if __name__ == '__main__':
    main()