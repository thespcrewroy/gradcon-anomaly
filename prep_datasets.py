'Importing libraries'
import argparse # For argument parsing
import os # For file and directory operations
import random # For random sampling
import shutil # For file operations
import errno # For error handling

from torchvision import datasets # For loading datasets
import numpy as np  # For numerical operations
import pickle # For saving and loading data
from PIL import Image # For image processing


parser = argparse.ArgumentParser(description='Download datasets and create splits')
parser.add_argument('--dataset', default='', type=str, help='Dataset to be downloaded or preprocessed (e.g. cifar-10, mnist, fmnist, maldeb)')
parser.add_argument('--save_dir', default='./datasets', type=str, help='Path to save the data')
parser.add_argument('--outlier_ratio', default=50, type=int, help='Outlier ratio in the test set')
parser.add_argument('--benign_cap', default=None, type=int, help='Maximum number of benign images to use (for maldeb only)')
parser.add_argument('--malicious_cap', default=None, type=int, help='Maximum number of malicious images to use (for maldeb only)')



def main():
    args = parser.parse_args() # Parse the command line arguments

    if args.dataset not in ['cifar-10', 'mnist', 'fmnist', 'maldeb']: # Check if the dataset is valid
        raise ValueError('Dataset should be one of the followings: cifar-10, mnist, fmnist, maldeb')

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
        raw_root = os.path.join(args.save_dir, 'Maldeb') # Directory where the raw Maldeb dataset is stored
        out_root = os.path.join(args.save_dir, 'maldeb') # Directory to save the processed Maldeb dataset

        train_ratio = 0.7 # Ratio of training data
        val_ratio = 0.15 # Ratio of validation data
        test_ratio = 0.15 # Ratio of test data

        for split in ['train', 'val', 'test']: # For each split (train, val, test)
            os.makedirs(os.path.join(out_root, split, 'Benign'), exist_ok=True) # Create directories for benign data
            if split != 'train': # If the split is not 'train'
                os.makedirs(os.path.join(out_root, split, 'Malicious'), exist_ok=True) # Create directories for malicious data

        for cls in ['Benign', 'Malicious']: # For each class (Benign, Malicious)
            full_cls_dir = os.path.join(raw_root, cls) # Directory where the raw class data is stored
            files = sorted([f for f in os.listdir(full_cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]) # List of files in the class directory
            random.shuffle(files) # Shuffle the files
            n = len(files) # Number of files in the class directory

            cap = args.benign_cap if cls == 'Benign' and args.benign_cap else args.malicious_cap if cls == 'Malicious' and args.malicious_cap else n
            files = files[:min(cap, n)] # Limit the number of files to the specified cap

            train_files = files[:int(len(files) * train_ratio)] # Training files
            val_files = files[int(len(files) * train_ratio):int(len(files) * (train_ratio + val_ratio))] # Validation files
            test_files = files[int(len(files) * (train_ratio + val_ratio)) :] # Test files

            if cls == 'Benign': # If the class is 'Benign'
                for f in train_files: # For each training file
                    shutil.copy(os.path.join(full_cls_dir, f), os.path.join(out_root, 'train', cls, f)) # Copy the file to the training directory

            for f in val_files: # For each validation file
                shutil.copy(os.path.join(full_cls_dir, f), os.path.join(out_root, 'val', cls, f)) # Copy the file to the validation directory
            for f in test_files: # For each test file
                shutil.copy(os.path.join(full_cls_dir, f), os.path.join(out_root, 'test', cls, f)) # Copy the file to the test directory

        print("Maldeb dataset has been split into train/val/test under:", out_root) # Print success message
        return # End of the function
    
    np_train = [] # List to store training data
    np_test = [] # List to store test data
    for img, label in trainset: # For each image and label in the training set
        if args.dataset == 'cifar-10': # If the dataset is 'cifar-10'
            img = img.resize((28, 28), Image.ANTIALIAS) # Resize the image to 28x28
        np_img = np.asarray(img, dtype='uint8') # Convert the image to a numpy array
        np_train.append((label, np_img))  # A tuple (label, img(ndarray)) is appended

    for img, label in testset: # For each image and label in the test set
        if args.dataset == 'cifar-10': # If the dataset is 'cifar-10'
            img = img.resize((28, 28), Image.ANTIALIAS) # Resize the image to 28x28
        np_img = np.asarray(img, dtype='uint8') # Convert the image to a numpy array
        np_test.append((label, np_img)) # A tuple (label, img(ndarray)) is appended

    # Protocols for splitting cifar-10, mnist and that for fmnist are different.
    if args.dataset in ['cifar-10', 'mnist']: # If the dataset is 'cifar-10' or 'mnist'
        # Categorize samples based on their classes
        class_bins_train = {} # Dictionary to store training data for each class
        random.shuffle(np_train) # Shuffle the training data
        for x in np_train: # For each sample in the training data
            if x[0] not in class_bins_train: # If the class is not in the dictionary
                class_bins_train[x[0]] = [] # Create a new list for the class
            class_bins_train[x[0]].append(x) # Append the sample to the class list

        train_split = [] # List to store training split
        val_split = [] # List to store validation split
        val_ratio = 0.1 # Ratio of validation data

        for _class, data in class_bins_train.items(): # For each class in the training data
            count = len(data) # Number of samples in the class
            print("(Train set) Class %d has %d samples" % (_class, count)) # Print the number of samples

            # Create a validation set by taking a small portion of data from the training set
            count_per_class = int(count * val_ratio) # Number of samples for validation
            val_split += data[:count_per_class] # Append the validation samples
            train_split += data[count_per_class::] # Append the remaining samples to the training split

        # Create a test split
        class_bins_test = {} # Dictionary to store test data for each class
        for x in np_test: # For each sample in the test data
            if x[0] not in class_bins_test: # If the class is not in the dictionary
                class_bins_test[x[0]] = [] # Create a new list for the class
            class_bins_test[x[0]].append(x) # Append the sample to the class list

        test_split = [] # List to store test split
        for _class, data in class_bins_test.items(): # For each class in the test data
            count = len(data) # Number of samples in the class
            print("(Test set) Class %d has %d samples" % (_class, count)) # Print the number of samples
            test_split += data # Append the samples to the test split

    elif args.dataset == 'fmnist': # If the dataset is 'fmnist'
        num_folds = 5 # Number of folds for cross-validation
        class_bins = {} # Dictionary to store data for each class
        np_all = np_train + np_test # Combine training and test data

        random.shuffle(np_all) # Shuffle the combined data
        for x in np_all:    # For each sample in the combined data
            if x[0] not in class_bins: # If the class is not in the dictionary
                class_bins[x[0]] = [] # Create a new list for the class
            class_bins[x[0]].append(x) # Append the sample to the class list

        # Create 5 different folds (3 folds (60% of data) will be used for training and remaining folds are
        # for validation and test)
        data_folds = [[] for _ in range(num_folds)] # List to store data for each fold
        for _class, data in class_bins.items(): # For each class in the combined data
            count = len(data) # Number of samples in the class
            print("Class %d has %d samples" % (_class, count)) # Print the number of samples
            count_per_fold = count // num_folds # Number of samples per fold

            for i in range(num_folds): # For each fold
                data_folds[i] += data[i * count_per_fold: (i + 1) * count_per_fold] # Append the samples to the fold

        train_split = data_folds[0] + data_folds[1] + data_folds[2] # Training data (3 folds)
        val_split = data_folds[3] # Validation data (1 fold)
        test_split = data_folds[4] # Test data (1 fold)

    # Save train and validation splits
    output_train = open(os.path.join(split_dir, 'data_split_train.pkl'), 'wb') # Open the output file for training split
    pickle.dump(train_split, output_train) # Save the training split
    output_train.close() # Close the output file

    output_val = open(os.path.join(split_dir, 'data_split_val.pkl'), 'wb') # Open the output file for validation split
    pickle.dump(val_split, output_val) # Save the validation split
    output_val.close() # Close the output file

    # Create test splits for each inlier class
    for cls in range(10): # For each class (0 to 9)
        print('Creating a test split with inlier class %d' % cls) # Print the class
        cls_balanced_data = [] # List to store balanced test data
        cls_cnt = [0] * 10 # List to store the count of samples for each class
        random.shuffle(test_split) # Shuffle the test data

        # First add all the inlier samples
        for x in test_split: # For each sample in the test data
            if x[0] == cls: # If the sample is an inlier
                cls_balanced_data.append(x) # Append the sample to the balanced data
                cls_cnt[cls] += 1 # Increment the count for the inlier class

        # Add the same number of outlier samples by sampling from all other classes
        num_inlier = len(cls_balanced_data) # Number of inlier samples
        num_outlier = int(num_inlier * args.outlier_ratio // (100 - args.outlier_ratio)) # Number of outlier samples
        outlier_cnt = 0 # Count of outlier samples
        for x in test_split: # For each sample in the test data
            if x[0] != cls and outlier_cnt < num_outlier: # If the sample is an outlier and the count is less than the required number
                cls_balanced_data.append(x) # Append the sample to the balanced data
                outlier_cnt += 1 # Increment the count of outlier samples
                cls_cnt[x[0]] += 1 # Increment the count for the outlier class

        print('Number of samples for each class:  ', cls_cnt) # Print the count of samples for each class

        # Save the test split for each inlier class
        output_test = open(os.path.join(split_dir, 'data_split_test_%d.pkl' % cls), 'wb') # Open the output file for test split
        pickle.dump(cls_balanced_data, output_test) # Save the test split
        output_test.close() # Close the output file

    print('\nTrain, validation, and test splits have been successfully created.') # Print success message


if __name__ == '__main__':
    main()
