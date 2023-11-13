# Nearest Neighbor Classification

## Overview

This project implements the k-nearest neighbor algorithm using the 8x8 MNIST dataset from scikit-learn (sklearn). The script allows you to customize various parameters such as the number of neighbors, the size of the test image set, and the number of iterations for averaging accuracy. Additionally, the project includes a detailed breakdown of the implementation, highlighting key functions and strategies employed.

## Project Details

### Nearest Neighbor Classification

This project implements a nearest neighbor classifier and a k-nearest neighbor classifier using pixel features on the 8x8 MNIST dataset. The classifiers are tested for classification accuracy, with specific evaluation for k values of 1, 3, 5, and 7.

#### Functions and Organization

For this project, the implementation is organized into multiple functions, each serving a specific utility in the overall algorithm. The process involves gathering and splitting the data into training and test sets, with the test set consisting of 500 images. Stratified sampling is employed to ensure a balanced distribution of classes in the test set.

The nearest neighbor and k-nearest neighbor algorithms share the same implementation, with the k value being adjustable through the command line argument `--k_neighbors` or `-k`. The algorithms compute pixel-wise distances between test and training images, determining predictions based on the class label of the k-nearest neighbors with the highest count.

To enhance accuracy assessment, the program supports multiple iterations, averaging the accuracy over these iterations. This approach provides a representative evaluation of the 8x8 MNIST dataset in comparison to the k-nearest neighbor algorithm.

### Results and Observations

After testing the program with different k-values and conducting multiple iterations, the following accuracy outputs were obtained:

- **k = 1:**
  - Size of Test Image Set: 500
  - Number of Iterations: 100
  - Accuracy: 98.72%

- **k = 3:**
  - Size of Test Image Set: 500
  - Number of Iterations: 100
  - Accuracy: 98.68%

- **k = 5:**
  - Size of Test Image Set: 500
  - Number of Iterations: 100
  - Accuracy: 98.54%

- **k = 7:**
  - Size of Test Image Set: 500
  - Number of Iterations: 100
  - Accuracy: 98.29%

The algorithm demonstrates high accuracy in predicting labels based solely on pixel representation. However, it's noted that the current implementation is relatively slow, especially as image resolution increases. Interestingly, as the number of neighbors increases, the accuracy tends to decrease, suggesting potential efficiency gains by favoring nearest neighbor over k-nearest neighbor when examining unknown images.

## Usage

Run the script using the following command:

```bash
python nearest_neighbor.py [-h] [--k_neighbors K_NEIGHBORS] [--test_size TEST_SIZE] [--iterations ITERATIONS]
```

- `--k_neighbors` or `-k`: Select the number of neighbors (default is 1).
- `--test_size` or `-t`: Select the number of test images (default is 500).
- `--iterations` or `-i`: Select the number of iterations for averaging accuracy (default is 1).

Example:

```bash
python nearest_neighbor.py --k_neighbors 3 --test_size 300 --iterations 5
```

## Dependencies

Make sure you have the required dependencies installed:

```bash
pip install numpy scikit-learn
```

## Implementation Details

- **Data Loading**: Utilizes the 8x8 MNIST dataset from scikit-learn.
- **Nearest Neighbor Algorithm**: Computes the L2-norm distance between test and train images to find the k-nearest neighbors.
- **Accuracy Calculation**: Measures the accuracy of predictions based on correct predictions over total predictions.
- **Iteration Averaging**: Optionally performs multiple iterations and averages the accuracy over these iterations.
- **Statistics Display**: Displays statistics such as test set size, number of iterations, number of neighbors, and accuracy.

## Example

```bash
python nearest_neighbor.py --k_neighbors 3 --test_size 300 --iterations 5
```

This command will run the k-nearest neighbor algorithm with 3 neighbors on a test set of 300 images, repeating the process 5 times and displaying iteration-wise accuracy. Finally, it will show the overall accuracy and the chosen parameter settings.

Feel free to explore and modify the script based on your needs!
