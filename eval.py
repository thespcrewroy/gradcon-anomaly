'''Import Necessary Libraries'''
import argparse
import os
import errno
import time

import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from thop import profile

import models
import ae_grad_reg
import datasets


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
    args = parser.parse_args()

    if args.dataset not in ['cifar-10', 'mnist', 'fmnist', 'maldeb']:
        raise ValueError('Dataset should be one of the followings: cifar-10, mnist, fmnist, maldeb')

    dataset = args.dataset
    grad_loss_weight = args.grad_loss_weight
    trained_inlier_class = 0
    in_channel = 3 if dataset == 'cifar-10' else 1
    num_decoder_layers = 4
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if dataset == 'maldeb':
        dataset_dir = os.path.join(args.dataset_dir, 'maldeb')
        ae_ckpt = os.path.join(args.ckpt_dir, dataset, f'{args.ckpt_name}/model_best.pth.tar')
        ae = models.GradConCAE(in_channel=in_channel)
        ae = torch.nn.DataParallel(ae).to(device)
        ae.eval()

        if os.path.isfile(ae_ckpt):
            print(f"=> loading checkpoint '{ae_ckpt}'")
            checkpoint_ae = torch.load(ae_ckpt)
            ae.load_state_dict(checkpoint_ae['state_dict'])
            ref_grad = checkpoint_ae['ref_grad']
            print(f"=> loaded checkpoint '{ae_ckpt}' (epoch {checkpoint_ae['epoch']}, best_loss {checkpoint_ae['best_loss']})")
        else:
            print(f"=> no checkpoint found at '{ae_ckpt}'")
            return

        test_loader = torch.utils.data.DataLoader(
            datasets.MaldebDataset(dataset_dir, split='test',
                                   transform=transforms.Compose([
                                       transforms.Resize((252, 252)),
                                       transforms.ToTensor()])),
            batch_size=1, shuffle=False)

        # --- Measure MACs and Parameters ---
        dummy_input = torch.randn(1, in_channel, 252, 252).to(device)
        macs, params = profile(ae.module, inputs=(dummy_input,))
        print(f"\nModel Complexity:")
        print(f"Params: {params:,} ({params/1e6:.2f}M)")
        print(f"MACs: {macs:,} ({macs/1e6:.2f}M)")

        # --- Inference and scoring ---
        total_start = time.time()
        preprocess_time = 0
        inference_time = 0
        postprocess_time = 0

        all_labels = []
        all_scores = []

        with torch.no_grad():
            for images, targets, labels in test_loader:
                t0 = time.time()
                images, targets, labels = images.to(device), targets.to(device), labels.to(device)
                t1 = time.time()
                outputs = ae(images)
                t2 = time.time()
                recon_error = ((outputs - targets) ** 2).view(outputs.size(0), -1).mean(dim=1)
                t3 = time.time()

                all_labels.extend(labels.cpu().numpy())
                all_scores.extend(recon_error.cpu().numpy())

                preprocess_time += (t1 - t0)
                inference_time += (t2 - t1)
                postprocess_time += (t3 - t2)

        total_time = time.time() - total_start

        total_start = time.time()
        all_labels = []
        all_scores = []

        result = ae_grad_reg.gradcon_score(
            ae, in_cls=1, grad_loss_weight=grad_loss_weight,
            ref_grad=ref_grad, nlayer=num_decoder_layers,
            device=device, test_loader=test_loader
        )

        total_time = time.time() - total_start
        labels = result[:, 0]
        scores = result[:, 1]
        fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
        score_auc = auc(fpr, tpr)

        os.makedirs(args.output_dir, exist_ok=True)
        metrics_output_path = os.path.join(args.output_dir, f"{args.dataset}_{args.ckpt_name}_metrics.txt")
        with open(metrics_output_path, "w") as f:
            f.write(f"AUROC: {score_auc:.4f}\n")
            f.write(f"MACs: {macs/1e6:.2f}M\n")
            f.write(f"Parameters: {params/1e6:.2f}M\n")
            f.write(f"Preprocessing time: {preprocess_time:.6f}s\n")
            f.write(f"Inference time: {inference_time:.6f}s\n")
            f.write(f"Postprocessing time: {postprocess_time:.6f}s\n")
            f.write(f"Total inference time: {total_time:.6f}s\n")

        print(f"\nMaldeb AUROC: {score_auc:.4f}")
        print(f"Total Inference Time: {total_time:.4f} sec")
        return

    dataset_dir = os.path.join(args.dataset_dir, dataset, 'splits')
    in_channel = 3 if dataset == 'cifar-10' else 1
    batch_size = 1
    num_decoder_layers = 4

    auroc_results = np.zeros([1, 11])  # 10 classes + avg
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the trained model (only once)
    ae_ckpt = os.path.join(args.ckpt_dir, dataset, f'{args.ckpt_name}_inlier-{trained_inlier_class}/model_best.pth.tar')

    ae = models.GradConCAE(in_channel=in_channel)
    ae = torch.nn.DataParallel(ae).to(device)
    ae.eval()

    if os.path.isfile(ae_ckpt):
        print(f"=> loading checkpoint '{ae_ckpt}'")
        checkpoint_ae = torch.load(ae_ckpt)
        best_loss = checkpoint_ae['best_loss']
        ae.load_state_dict(checkpoint_ae['state_dict'])
        ref_grad = checkpoint_ae['ref_grad']
        print(f"=> loaded checkpoint '{ae_ckpt}' (epoch {checkpoint_ae['epoch']}, best_loss {best_loss})")
    else:
        print(f"=> no checkpoint found at '{ae_ckpt}'")
        return

    # Evaluate on all test classes (0 to 9)
    for test_cls in range(10):
        print(f"Evaluating test set for class {test_cls}...")

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
        )

        result = ae_grad_reg.gradcon_score(
            ae, trained_inlier_class, grad_loss_weight, ref_grad,
            num_decoder_layers, device, test_loader
        )

        in_pred = result[np.where(result[:, 0] == 1)]
        out_pred = result[np.where(result[:, 0] == 0)]

        label = np.concatenate((np.ones([in_pred.shape[0]]), np.zeros([out_pred.shape[0]])), axis=0)
        score = np.concatenate((in_pred[:, 1], out_pred[:, 1]), axis=0)

        fpr_auc, tpr_auc, _ = roc_curve(label, score, pos_label=1)
        auroc_results[0, test_cls] = auc(fpr_auc, tpr_auc)

    auroc_results[:, -1] = np.mean(auroc_results[:, :-1], axis=1)

    try:
        os.makedirs(args.output_dir)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

    save_path = os.path.join(args.output_dir, f"{dataset}_{args.ckpt_name}_result.txt")
    np.savetxt(save_path, auroc_results, fmt='%.4f')
    print(f"Saved AUROC results to {save_path}")
    print("AUROC per class:", auroc_results[0, :-1])
    print("Average AUROC:", auroc_results[0, -1])


if __name__ == '__main__':
    main()