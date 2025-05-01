import argparse
import os
import errno

import numpy as np
from sklearn.metrics import roc_curve, auc
import torch
import torch.utils.data
import torchvision.transforms as transforms

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

    if args.dataset not in ['cifar-10', 'mnist', 'fmnist']:
        raise ValueError('Dataset should be one of the followings: cifar-10, mnist, fmnist')

    dataset = args.dataset
    grad_loss_weight = args.grad_loss_weight
    trained_inlier_class = 0  # ✅ You only trained on this class

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
