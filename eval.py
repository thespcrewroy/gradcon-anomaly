'''Import Necessary Libraries'''
import argparse # for command line argument parsing
import os # for file and directory operations
import errno # for error handling
import time # for time operations

import numpy as np # for numerical operations
import torch # for tensor operations
import torch.utils.data # for data loading and processing
import torchvision.transforms as transforms # for data transformations
import matplotlib.pyplot as plt # for plotting
from sklearn.metrics import roc_curve, auc # for ROC curve and AUC calculation
from thop import profile # for model complexity calculation

import models # for model definitions
import ae_grad_reg # for gradient regularization
import datasets # for dataset definitions


'''
Argument Parser - for command line arguments
'''
parser = argparse.ArgumentParser(description='Evaluation of GradCon (Fixed Inlier Class)')
parser.add_argument('--print-freq', '-pf', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--dataset', default='', type=str, help='Dataset to be used for training '
                                                            '(e.g. cifar-10, mnist, fmnist)')
parser.add_argument('--dataset_dir', default='./datasets', type=str, help='Path for the dataset')
parser.add_argument('--ckpt_dir', default='./save', type=str, help='Path to the folder that contains saved models')
parser.add_argument('--ckpt_name', default='GradConCAE', type=str, help='Checkpoint name')
parser.add_argument('--output_dir', default='./results', type=str, help='Path to save the result file')
parser.add_argument('--grad-loss-weight', '-gw', default=0.12, type=float,
                    metavar='N', help='gradient loss weight for the anomaly score')


def main():
    args = parser.parse_args() # parse command line arguments

    if args.dataset not in ['cifar-10', 'mnist', 'fmnist', 'maldeb']:
        raise ValueError('Dataset should be one of the followings: cifar-10, mnist, fmnist, maldeb')

    dataset = args.dataset # dataset name
    grad_loss_weight = args.grad_loss_weight # gradient loss weight
    trained_inlier_class = 0 # inlier class for training (0 for MNIST, 1 for Fashion-MNIST, 2 for CIFAR-10)
    in_channel = 3 if dataset == 'cifar-10' else 1 # number of input channels (3 for CIFAR-10, 1 for MNIST and Fashion-MNIST)
    num_decoder_layers = 4 # number of decoder layers in the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # use GPU if available

    if dataset == 'maldeb': # special case for Maldeb dataset
        dataset_dir = os.path.join(args.dataset_dir, 'maldeb') # path to the Maldeb dataset
        ae_ckpt = os.path.join(args.ckpt_dir, dataset, f'{args.ckpt_name}/model_best.pth.tar') # path to the saved model checkpoint
        ae = models.GradConCAE(in_channel=in_channel) # initialize the model
        ae = torch.nn.DataParallel(ae).to(device) # wrap the model in DataParallel for multi-GPU training
        ae.eval() # set the model to evaluation mode

        if os.path.isfile(ae_ckpt): # check if the model checkpoint exists
            print(f"=> loading checkpoint '{ae_ckpt}'") # load the model checkpoint
            checkpoint_ae = torch.load(ae_ckpt) # load the checkpoint
            ae.load_state_dict(checkpoint_ae['state_dict']) # load the model state dict
            ref_grad = checkpoint_ae['ref_grad'] # load the reference gradient
            print(f"=> loaded checkpoint '{ae_ckpt}' (epoch {checkpoint_ae['epoch']}, best_loss {checkpoint_ae['best_loss']})") # print the loaded checkpoint info
        else:
            print(f"=> no checkpoint found at '{ae_ckpt}'") # print error message if checkpoint not found
            return # exit the program

        test_loader = torch.utils.data.DataLoader(
            datasets.MaldebDataset(dataset_dir, split='test',
                                   transform=transforms.Compose([
                                       transforms.Resize((252, 252)),
                                       transforms.ToTensor()])),
            batch_size=1, shuffle=False) # load the test dataset

        # --- Measure MACs and Parameters ---
        dummy_input = torch.randn(1, in_channel, 252, 252).to(device) # create a dummy input tensor
        macs, params = profile(ae.module, inputs=(dummy_input,)) # calculate the model complexity
        print(f"\nModel Complexity:") # print model complexity info
        print(f"Params: {params:,} ({params/1e6:.2f}M)") # print number of parameters
        print(f"MACs: {macs:,} ({macs/1e6:.2f}M)") # print number of MACs

        # --- Inference and scoring ---
        total_start = time.time() # start timer for total inference time
        preprocess_time = 0 # initialize preprocessing time
        inference_time = 0 # initialize inference time
        postprocess_time = 0 # initialize postprocessing time

        all_labels = [] # initialize list to store labels
        all_scores = [] # initialize list to store scores

        with torch.no_grad(): # disable gradient calculation
            for images, targets, labels in test_loader: # iterate over the test dataset
                t0 = time.time() # start timer for preprocessing
                images, targets, labels = images.to(device), targets.to(device), labels.to(device) # move data to GPU
                t1 = time.time() # end timer for preprocessing
                outputs = ae(images) # forward pass through the model
                t2 = time.time() # end timer for inference
                recon_error = ((outputs - targets) ** 2).view(outputs.size(0), -1).mean(dim=1) # calculate reconstruction error
                t3 = time.time() # end timer for postprocessing

                all_labels.extend(labels.cpu().numpy()) # store labels
                all_scores.extend(recon_error.cpu().numpy()) # store scores

                preprocess_time += (t1 - t0) # update preprocessing time
                inference_time += (t2 - t1) # update inference time
                postprocess_time += (t3 - t2) # update postprocessing time

        total_time = time.time() - total_start # end timer for total inference time

        total_start = time.time() # start timer for total inference time
        all_labels = [] # initialize lists to store labels and scores
        all_scores = [] # initialize lists to store labels and scores

        result = ae_grad_reg.gradcon_score(
            ae, in_cls=1, grad_loss_weight=grad_loss_weight,
            ref_grad=ref_grad, nlayer=num_decoder_layers,
            device=device, test_loader=test_loader
        ) # calculate the anomaly score using gradient regularization

        total_time = time.time() - total_start # end timer for total inference time
        labels = result[:, 0] # extract labels from the result
        scores = result[:, 1] # extract scores from the result
        fpr, tpr, _ = roc_curve(labels, scores, pos_label=1) # calculate the ROC curve
        score_auc = auc(fpr, tpr) # calculate the AUC score

        os.makedirs(args.output_dir, exist_ok=True) # create output directory if it doesn't exist
        metrics_output_path = os.path.join(args.output_dir, f"{args.dataset}_{args.ckpt_name}_metrics.txt") # path to save the metrics

        '''
        Save metrics to a file
        '''
        with open(metrics_output_path, "w") as f:
            f.write(f"AUROC: {score_auc:.4f}\n")
            f.write(f"MACs: {macs/1e6:.2f}M\n")
            f.write(f"Parameters: {params/1e6:.2f}M\n")
            f.write(f"Preprocessing time: {preprocess_time:.6f}s\n")
            f.write(f"Inference time: {inference_time:.6f}s\n")
            f.write(f"Postprocessing time: {postprocess_time:.6f}s\n")
            f.write(f"Total inference time: {total_time:.6f}s\n")

        print(f"\nMaldeb AUROC: {score_auc:.4f}") # print AUC score
        print(f"Total Inference Time: {total_time:.4f} sec") # print total inference time
        return # exit the program

    dataset_dir = os.path.join(args.dataset_dir, dataset, 'splits') # path to the dataset splits
    in_channel = 3 if dataset == 'cifar-10' else 1 # number of input channels
    batch_size = 1 # batch size for data loading
    num_decoder_layers = 4 # number of decoder layers in the model

    auroc_results = np.zeros([1, 11])  # 10 classes + avg
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # use GPU if available

    # Load the trained model (only once)
    ae_ckpt = os.path.join(args.ckpt_dir, dataset, f'{args.ckpt_name}_inlier-{trained_inlier_class}/model_best.pth.tar') # path to the saved model checkpoint

    ae = models.GradConCAE(in_channel=in_channel) # initialize the model
    ae = torch.nn.DataParallel(ae).to(device) # wrap the model in DataParallel for multi-GPU training
    ae.eval() # set the model to evaluation mode

    if os.path.isfile(ae_ckpt): # check if the model checkpoint exists
        print(f"=> loading checkpoint '{ae_ckpt}'") # load the model checkpoint
        checkpoint_ae = torch.load(ae_ckpt) # load the checkpoint
        best_loss = checkpoint_ae['best_loss'] # load the best loss
        ae.load_state_dict(checkpoint_ae['state_dict']) # load the model state dict
        ref_grad = checkpoint_ae['ref_grad'] # load the reference gradient
        print(f"=> loaded checkpoint '{ae_ckpt}' (epoch {checkpoint_ae['epoch']}, best_loss {best_loss})")
    else: # print error message if checkpoint not found
        print(f"=> no checkpoint found at '{ae_ckpt}'") # exit the program
        return # exit the program

    # Evaluate on all test classes (0 to 9)
    for test_cls in range(10): # iterate over all test classes
        print(f"Evaluating test set for class {test_cls}...") # print current test class

        test_loader = torch.utils.data.DataLoader(
            datasets.AnomalyDataset(
                root=dataset_dir,
                split=f'test_{test_cls}',
                in_channel=in_channel,
                transform=transforms.ToTensor(),
                target_transform=transforms.ToTensor(),
                inlier_class=trained_inlier_class  # still treated as class 0 is "normal"
            ),
            batch_size=batch_size,
            shuffle=False
        ) # load the test dataset

        result = ae_grad_reg.gradcon_score(
            ae, trained_inlier_class, grad_loss_weight, ref_grad,
            num_decoder_layers, device, test_loader
        ) # calculate the anomaly score using gradient regularization

        in_pred = result[np.where(result[:, 0] == 1)] # extract inlier predictions
        out_pred = result[np.where(result[:, 0] == 0)] # extract outlier predictions

        label = np.concatenate((np.ones([in_pred.shape[0]]), np.zeros([out_pred.shape[0]])), axis=0) # create labels for inliers and outliers
        score = np.concatenate((in_pred[:, 1], out_pred[:, 1]), axis=0) # concatenate scores for inliers and outliers

        fpr_auc, tpr_auc, _ = roc_curve(label, score, pos_label=1) # calculate the ROC curve
        auroc_results[0, test_cls] = auc(fpr_auc, tpr_auc) # calculate the AUC score

    auroc_results[:, -1] = np.mean(auroc_results[:, :-1], axis=1) # calculate the average AUC score

    try: # create output directory if it doesn't exist
        os.makedirs(args.output_dir) # create output directory
    except OSError as exception: # check if the error is due to directory already existing
        if exception.errno != errno.EEXIST: # check if the error is not due to directory already existing
            raise # raise error if it's not due to directory already existing

    save_path = os.path.join(args.output_dir, f"{dataset}_{args.ckpt_name}_result.txt") # path to save the results
    np.savetxt(save_path, auroc_results, fmt='%.4f') # save the results to a text file
    print(f"Saved AUROC results to {save_path}") # print message indicating results saved
    print("AUROC per class:", auroc_results[0, :-1]) # print AUC scores for each class
    print("Average AUROC:", auroc_results[0, -1]) # print average AUC score


if __name__ == '__main__':
    main()