'Importing libraries'
import argparse # For argument parsing
import os # For file and directory operations
import random # For random sampling
import errno # For error handling

from torchvision import datasets # For loading datasets
import numpy as np  # For numerical operations
import pickle # For saving and loading data
from PIL import Image # For image processing


parser = argparse.ArgumentParser(description='Download datasets and create splits') # Create an argument parser
parser.add_argument('--dataset', default='', type=str, help='Dataset to be downloaded or preprocessed (e.g. cifar-10, mnist, fmnist, maldeb)') # Dataset name
parser.add_argument('--save_dir', default='./datasets', type=str, help='Path to save the data') # Directory to save the dataset
parser.add_argument('--outlier_ratio', default=50, type=int, help='Outlier ratio in the test set') # Ratio of outliers in the test set


def main():
    args = parser.parse_args() # Parse the command line arguments

    if args.dataset not in ['cifar-10', 'mnist', 'fmnist', 'maldeb']: # Check if the dataset is valid
        raise ValueError('Dataset should be one of the followings: cifar-10, mnist, fmnist, madleb')

    if args.dataset != 'maldeb': # If the dataset is not 'maldeb'
        dataset_dir = os.path.join(args.save_dir, args.dataset) # Directory to save the dataset
        split_dir = os.path.join(dataset_dir, 'splits')  # Folder where train, val, test splits are saved
        try: # Create the directory for splits
            os.makedirs(split_dir) # Create the directory if it doesn't exist
        except OSError as exception: # Handle the error if the directory already exists
            if exception.errno != errno.EEXIST: # If the error is not "directory already exists"
                raise # Raise the error

    if args.dataset == 'cifar-10': # If the dataset is 'cifar-10'
        trainset = datasets.CIFAR10(dataset_dir, download=True, train=True) # Load the CIFAR-10 training set
        testset = datasets.CIFAR10(dataset_dir, download=True, train=False) # Load the CIFAR-10 test set

    elif args.dataset == 'mnist': # If the dataset is 'mnist'
        trainset = datasets.MNIST(dataset_dir, download=True, train=True) # Load the MNIST training set
        testset = datasets.MNIST(dataset_dir, download=True, train=False) # Load the MNIST test set

    elif args.dataset == 'fmnist': # If the dataset is 'fmnist'
        trainset = datasets.FashionMNIST(dataset_dir, download=True, train=True) # Load the Fashion-MNIST training set
        testset = datasets.FashionMNIST(dataset_dir, download=True, train=False) # Load the Fashion-MNIST test set

    elif args.dataset == 'maldeb': # If the dataset is 'maldeb'
        raw_root = os.path.join(args.save_dir, 'Maldeb')  # where the original Benign/Malicious folders are
        out_root = os.path.join(args.save_dir, 'maldeb')  # destination split folder

        train_ratio = 0.7 # Ratio of training data
        val_ratio = 0.15 # Ratio of validation data
        test_ratio = 0.15 # Ratio of test data

        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(out_root, split, 'Benign'), exist_ok=True)
            os.makedirs(os.path.join(out_root, split, 'Malicious'), exist_ok=True)

        for cls in ['Benign', 'Malicious']:
            full_cls_dir = os.path.join(raw_root, cls)
            files = sorted(os.listdir(full_cls_dir))
            random.shuffle(files)
            n = len(files)

            train_files = files[:int(n * train_ratio)]
            val_files = files[int(n * train_ratio):int(n * (train_ratio + val_ratio))]
            test_files = files[int(n * (train_ratio + val_ratio)):]

            for f in train_files:
                shutil.copy(os.path.join(full_cls_dir, f), os.path.join(out_root, 'train', cls, f))
            for f in val_files:
                shutil.copy(os.path.join(full_cls_dir, f), os.path.join(out_root, 'val', cls, f))
            for f in test_files:
                shutil.copy(os.path.join(full_cls_dir, f), os.path.join(out_root, 'test', cls, f))

        print("Maldeb dataset has been split into train/val/test under:", out_root)
        return
    np_train = []
    np_test = []
    for img, label in trainset:
        if args.dataset == 'cifar-10':
            img = img.resize((28, 28), Image.ANTIALIAS)
        np_img = np.asarray(img, dtype='uint8')
        np_train.append((label, np_img))  # A tuple (label, img(ndarray)) is appended

    for img, label in testset:
        if args.dataset == 'cifar-10':
            img = img.resize((28, 28), Image.ANTIALIAS)
        np_img = np.asarray(img, dtype='uint8')
        np_test.append((label, np_img))

    # Protocols for splitting cifar-10, mnist and that for fmnist are different.
    if args.dataset in ['cifar-10', 'mnist']:
        # Categorize samples based on their classes
        class_bins_train = {}
        random.shuffle(np_train)
        for x in np_train:
            if x[0] not in class_bins_train:
                class_bins_train[x[0]] = []
            class_bins_train[x[0]].append(x)

        train_split = []
        val_split = []
        val_ratio = 0.1

        for _class, data in class_bins_train.items():
            count = len(data)
            print("(Train set) Class %d has %d samples" % (_class, count))

            # Create a validation set by taking a small portion of data from the training set
            count_per_class = int(count * val_ratio)
            val_split += data[:count_per_class]
            train_split += data[count_per_class::]

        # Create a test split
        class_bins_test = {}
        for x in np_test:
            if x[0] not in class_bins_test:
                class_bins_test[x[0]] = []
            class_bins_test[x[0]].append(x)

        test_split = []
        for _class, data in class_bins_test.items():
            count = len(data)
            print("(Test set) Class %d has %d samples" % (_class, count))
            test_split += data

    elif args.dataset == 'fmnist':
        num_folds = 5
        class_bins = {}
        np_all = np_train + np_test

        random.shuffle(np_all)
        for x in np_all:
            if x[0] not in class_bins:
                class_bins[x[0]] = []
            class_bins[x[0]].append(x)

        # Create 5 different folds (3 folds (60% of data) will be used for training and remaining folds are
        # for validation and test)
        data_folds = [[] for _ in range(num_folds)]
        for _class, data in class_bins.items():
            count = len(data)
            print("Class %d has %d samples" % (_class, count))
            count_per_fold = count // num_folds

            for i in range(num_folds):
                data_folds[i] += data[i * count_per_fold: (i + 1) * count_per_fold]

        train_split = data_folds[0] + data_folds[1] + data_folds[2]
        val_split = data_folds[3]
        test_split = data_folds[4]

    # Save train and validation splits
    output_train = open(os.path.join(split_dir, 'data_split_train.pkl'), 'wb')
    pickle.dump(train_split, output_train)
    output_train.close()

    output_val = open(os.path.join(split_dir, 'data_split_val.pkl'), 'wb')
    pickle.dump(val_split, output_val)
    output_val.close()

    # Create test splits for each inlier class
    for cls in range(10):
        print('Creating a test split with inlier class %d' % cls)
        cls_balanced_data = []
        cls_cnt = [0] * 10
        random.shuffle(test_split)

        # First add all the inlier samples
        for x in test_split:
            if x[0] == cls:
                cls_balanced_data.append(x)
                cls_cnt[cls] += 1

        # Add the same number of outlier samples by sampling from all other classes
        num_inlier = len(cls_balanced_data)
        num_outlier = int(num_inlier * args.outlier_ratio // (100 - args.outlier_ratio))
        outlier_cnt = 0
        for x in test_split:
            if x[0] != cls and outlier_cnt < num_outlier:
                cls_balanced_data.append(x)
                outlier_cnt += 1
                cls_cnt[x[0]] += 1

        print('Number of samples for each class:  ', cls_cnt)

        # Save the test split for each inlier class
        output_test = open(os.path.join(split_dir, 'data_split_test_%d.pkl' % cls), 'wb')
        pickle.dump(cls_balanced_data, output_test)
        output_test.close()

    print('\nTrain, validation, and test splits have been successfully created.')


if __name__ == '__main__':
    main()
