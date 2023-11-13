# Author: Curtis Helsel
# 11/10/2021
# Implementation of k-nearest neighbor algorithm using
# the 8x8 MNIST dataset from sklearn. 
#
# Run python nearest_neighbor.py -h to see all the options.

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
import argparse

# Parses the arguments supplied by the commandline. The default values are
# for the nearest neighbor algorithm with 500 test images. 
def argument_parse():

    parser = argparse.ArgumentParser('Nearest Neighbor Classification')

    parser.add_argument('--k_neighbors','-k', 
            type=int, default=1, 
            help='Select number of neighbors.')

    parser.add_argument('--test_size','-t', 
            type=int, default=500, 
            help='Select number of test images.')

    parser.add_argument('--iterations','-i', 
            type=int, default=1, 
            help='Select number of iterations for averaging accuracy.')
    

    args, unparsed = parser.parse_known_args()

    return args

# Gets the mnist 8x8 dataset from sklearn datasets and returns the
# data and labels.
def get_data():

    digits = load_digits()

    return digits.data, digits.target

# Splits the data based on the test size supplied and returns
# train data and labels and test data and labels.
def split_data(data, label, test_size):

    return train_test_split(data, label, stratify=label, test_size=test_size)

# Computes the L2-norm distance between two images
# sqrt(sum(test image - train image)^2) for all corresponding pixels.
def l2_norm_distance(test_image, train_image):

    return np.sqrt(sum((test_image - train_image) ** 2))

# Finds the nearest neighbor for each test image and returns the prediction
# based on the kth nearest neighbor.
def nearest_neighbor(train_data, test_data, train_labels, test_labels, k):
    
    predictions = []
    
    # For each test image, compare the pixel distance of every training
    # image and find the closest k neighbors
    for test_image, test_label in zip(test_data, test_labels):

        neighbors = []

        for train_image, train_label in zip(train_data, train_labels):

            distance = l2_norm_distance(test_image, train_image)
            neighbors.append((distance, train_label))

        # Sort by the distance and get labels of closest neighbor
        neighbors.sort(key=lambda x:x[0])
        neighbor_labels = [label for (distance, label)in neighbors]

        # Find neighbor with the highest count closest to the test image
        prediction = np.bincount(neighbor_labels[:k]).argmax()
        predictions.append(prediction)
    
    return predictions

# Gets the prediction accuracy by measuring correct predictions over
# total predictions.
def get_prediction_accuracy(predictions, test_labels):
    
    correct_predictions = sum(predictions == test_labels)
    total_predictions = len(test_labels)
    
    return correct_predictions / total_predictions

# Prints statistics of current iteration to terminal
def display_iteration_statistics(iteration, accuracy):

    print('Iteration {} Complete'.format(iteration + 1))
    print('Iteration Accuracy {:0.2f}\n'.format(accuracy * 100))

# Prints statistics about accuracy with settings 
# provided from command line.
def display_statistics(accuracy, k, test_size, iterations):

    print('Size of Test Image Set: {}'.format(test_size))
    print('Number of Iterations: {}'.format(iterations))
    print('Number of Nearest Neighbors: {}'.format(k))
    print('Accuracy: {:0.2f}%'.format(accuracy * 100))

if __name__ == '__main__':

    args = argument_parse()
    data, label = get_data()

    average_accuracy = 0
    
    # Loop for specified iteration amount and calculate the overall
    # accuracy for a k value 
    for iteration in range(args.iterations):
        train_data, test_data, train_labels, test_labels = split_data(data, 
                label, args.test_size)

        predictions = nearest_neighbor(train_data, test_data, 
                train_labels, test_labels, args.k_neighbors)

        accuracy = get_prediction_accuracy(predictions, test_labels)

        average_accuracy += accuracy

        # If user is doing k-nearest neighbor instead of nearest neighbor,
        # display the iteration statistics
        if args.iterations != 1:
            display_iteration_statistics(iteration, accuracy)

    overall_accuracy = average_accuracy / args.iterations


    display_statistics(overall_accuracy, args.k_neighbors, 
            args.test_size, args.iterations)


    
