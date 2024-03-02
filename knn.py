import numpy as np
from sklearn.model_selection import train_test_split


def load_data(file_path, haberman: bool):
    """
     Load data from a file.

     Parameters:
     - file_path (str): The path to the file containing the data.
     - haberman (bool): If True, the data is in the Haberman's Survival format, else circle_separator.

     Returns:
     - coordinates (numpy.ndarray): The coordinates data extracted from the file.
     - labels (numpy.ndarray): The labels data extracted from the file.
     """
    if haberman:
        data = np.loadtxt(file_path, delimiter=',')
        coordinates, labels = data[:, :3], data[:, 3]

        return coordinates, labels

    else:
        data = np.loadtxt(file_path)
        coordinates, labels = data[:, :2], data[:, 2]
        return coordinates, labels


def manhattan_distance(p1, p2):
    """
    When p = 1
     Calculate the Manhattan distance between two points.
     Parameters:
     - p1 (numpy.ndarray): Coordinates of the first point.
     - p2 (numpy.ndarray): Coordinates of the second point.
     Returns:
     - dist (float): The Manhattan distance between the two points.
       """
    return np.sum(np.abs(p1 - p2))


def euclidean_distance(p1, p2):
    """
    When p = 2.
    Calculate the Euclidean distance between two points.

    Parameters:
    - p1 (numpy.ndarray): Coordinates of the first point.
    - p2 (numpy.ndarray): Coordinates of the second point.

    Returns:
    - dist (float): The Euclidean distance between the two points.
    """

    return np.sqrt(np.sum((p1 - p2) ** 2))


def infinity_distance(p1, p2):
    """
    When p = infinity.
       Calculate the Infinity distance between two points.

       Parameters:
       - p1 (numpy.ndarray): Coordinates of the first point.
       - p2 (numpy.ndarray): Coordinates of the second point.

       Returns:
       - dist (float): The Infinity distance between the two points.
       """
    return np.max(np.abs(p1 - p2))


def get_distance(pair):
    """
      Extract the distance from a tuple.

      Parameters:
      - pair (tuple): A tuple containing the distance and label.

      Returns:
      - distance (float): The distance value.
      """
    return pair[0]


def knn(k, p, train_X, train_y, test_X):
    """
       k-Nearest Neighbors algorithm implementation.

       Parameters:
       - k (int): The number of nearest neighbors to consider.
       - p (int): The value of p for the distance.
       - train_X (numpy.ndarray): Training data points.
       - train_y (numpy.ndarray): Labels corresponding to training data points.
       - test_X (numpy.ndarray): Test data points.

       Returns:
       - predicted_label: The predicted label for the test data point.
       """
    distances = []
    for i in range(len(train_X)):
        if p == 1:
            dist = manhattan_distance(test_X, train_X[i])
        elif p == 2:
            dist = euclidean_distance(test_X, train_X[i])
        else:
            dist = infinity_distance(test_X, train_X[i])
        distances.append((dist, train_y[i]))
    # Sort the list of distances based on the distance
    distances.sort(key=get_distance)
    neighbors = distances[:k]

    # Count the occurrences of each label among the k nearest neighbors
    label_counts = {}
    for neighbor in neighbors:
        label = neighbor[1]
        label_counts[label] = label_counts.get(label, 0) + 1
    # Find the label with the highest count
    max_count = 0
    predicted_label = None
    for label, count in label_counts.items():
        if count > max_count:
            max_count = count
            predicted_label = label
    return predicted_label


def evaluate_knn(k, p, train_X, train_y, test_X, test_y):
    """
        Evaluate the k-Nearest Neighbors algorithm.

        Parameters:
        - k (int): The number of nearest neighbors to consider.
        - p (int): The value of p for the distance.
        - train_X (numpy.ndarray): Training data points.
        - train_y (numpy.ndarray): Labels corresponding to training data points.
        - test_X (numpy.ndarray): Test data points.
        - test_y (numpy.ndarray): True labels for test data points.

        Returns:
        - error (float): The classification error rate.
        """
    error = 0
    for i in range(len(test_X)):

        prediction = knn(k, p, train_X, train_y, test_X[i])
        if prediction != test_y[i]:
            error += 1

    return error / len(test_y)


def run_task(X, y, data_set):
    """
      Run k-Nearest Neighbors algorithm on a dataset.

      Parameters:
      - X (numpy.ndarray): Input features.
      - y (numpy.ndarray): Labels.
      - data_set (str): Name of the dataset.
      """
    print(f"\n-----run knn on {data_set} data set--------------- ")

    runs = 100
    neighbors = [1, 3, 5, 7, 9]

    best_true_error = np.inf
    best_p = None
    best_k = None

    best_empirical_error = np.inf
    best_p_of_empirical_error = None
    best_k_of_empirical_error = None

    for p in [1, 2, np.inf]:
        for k in neighbors:
            avg_empirical_errors = 0
            avg_true_errors = 0
            for run in range(runs):
                train_X, test_X, train_Y, test_Y = train_test_split(X, y, test_size=0.5)
                avg_empirical_errors += evaluate_knn(k, p, train_X, train_Y, train_X, train_Y)
                avg_true_errors += evaluate_knn(k, p, train_X, train_Y, test_X, test_Y)

            # Set the average value
            avg_empirical_errors /= runs
            avg_true_errors /= runs
            difference = avg_empirical_errors - avg_true_errors

            print(
                f"p: {p}, k: {k}, Average Empirical Errors: {np.round(avg_empirical_errors, 4)}, Average True Error:"
                f" {np.round(avg_true_errors, 4)},"
                f" Difference between them: {np.round(difference, 4)} ")

            # Update best parameters if current true error is the lowest
            if avg_true_errors < best_true_error:
                best_true_error = avg_true_errors
                best_p = p
                best_k = k

            if avg_empirical_errors < best_empirical_error:
                best_empirical_error = avg_empirical_errors
                best_p_of_empirical_error = p
                best_k_of_empirical_error = k

    print(
        f"\nBest parameters of true_errors with p: {best_p}, k: {best_k}, True Error: {np.round(best_true_error, 4)}")
    print(
        f"Best parameters of empirical errors with p: {best_p_of_empirical_error}, k: {best_k_of_empirical_error},"
        f" Empirical Errors: {np.round(best_empirical_error, 4)}")


if __name__ == '__main__':
    X, y = load_data('dataset/haberman.data', True)
    run_task(X, y, 'haberman')

    X, y = load_data('dataset/circle_separator.txt', False)
    run_task(X, y, 'circle_separator')

"""
-----run knn on haberman data set--------------- 
p: 1, k: 1, Average Empirical Errors: 0.0095, Average True Error: 0.3431, Difference between them: -0.3337 
p: 1, k: 3, Average Empirical Errors: 0.1691, Average True Error: 0.2945, Difference between them: -0.1254 
p: 1, k: 5, Average Empirical Errors: 0.2022, Average True Error: 0.2734, Difference between them: -0.0712 
p: 1, k: 7, Average Empirical Errors: 0.218, Average True Error: 0.2603, Difference between them: -0.0424 
p: 1, k: 9, Average Empirical Errors: 0.2244, Average True Error: 0.2608, Difference between them: -0.0363 
p: 2, k: 1, Average Empirical Errors: 0.0103, Average True Error: 0.3225, Difference between them: -0.3122 
p: 2, k: 3, Average Empirical Errors: 0.1631, Average True Error: 0.2956, Difference between them: -0.1325 
p: 2, k: 5, Average Empirical Errors: 0.2044, Average True Error: 0.2752, Difference between them: -0.0708 
p: 2, k: 7, Average Empirical Errors: 0.2144, Average True Error: 0.2652, Difference between them: -0.0507 
p: 2, k: 9, Average Empirical Errors: 0.2203, Average True Error: 0.2586, Difference between them: -0.0382 
p: inf, k: 1, Average Empirical Errors: 0.0078, Average True Error: 0.3351, Difference between them: -0.3273 
p: inf, k: 3, Average Empirical Errors: 0.1642, Average True Error: 0.2888, Difference between them: -0.1245 
p: inf, k: 5, Average Empirical Errors: 0.1984, Average True Error: 0.2752, Difference between them: -0.0767 
p: inf, k: 7, Average Empirical Errors: 0.2126, Average True Error: 0.2589, Difference between them: -0.0463 
p: inf, k: 9, Average Empirical Errors: 0.2228, Average True Error: 0.2581, Difference between them: -0.0353 

Best parameters of true_errors with p: inf, k: 9, True Error: 0.2581
Best parameters of empirical errors with p: inf, k: 1, Empirical Errors: 0.0078

-----run knn on circle_separator data set--------------- 
p: 1, k: 1, Average Empirical Errors: 0.0, Average True Error: 0.0667, Difference between them: -0.0667 
p: 1, k: 3, Average Empirical Errors: 0.0316, Average True Error: 0.0944, Difference between them: -0.0628 
p: 1, k: 5, Average Empirical Errors: 0.0505, Average True Error: 0.1109, Difference between them: -0.0604 
p: 1, k: 7, Average Empirical Errors: 0.0723, Average True Error: 0.1307, Difference between them: -0.0584 
p: 1, k: 9, Average Empirical Errors: 0.0817, Average True Error: 0.1372, Difference between them: -0.0555 
p: 2, k: 1, Average Empirical Errors: 0.0, Average True Error: 0.0652, Difference between them: -0.0652 
p: 2, k: 3, Average Empirical Errors: 0.032, Average True Error: 0.0859, Difference between them: -0.0539 
p: 2, k: 5, Average Empirical Errors: 0.0507, Average True Error: 0.0957, Difference between them: -0.0451 
p: 2, k: 7, Average Empirical Errors: 0.0663, Average True Error: 0.1189, Difference between them: -0.0527 
p: 2, k: 9, Average Empirical Errors: 0.0853, Average True Error: 0.1435, Difference between them: -0.0581 
p: inf, k: 1, Average Empirical Errors: 0.0, Average True Error: 0.0743, Difference between them: -0.0743 
p: inf, k: 3, Average Empirical Errors: 0.0313, Average True Error: 0.0776, Difference between them: -0.0463 
p: inf, k: 5, Average Empirical Errors: 0.0504, Average True Error: 0.1021, Difference between them: -0.0517 
p: inf, k: 7, Average Empirical Errors: 0.0673, Average True Error: 0.1359, Difference between them: -0.0685 
p: inf, k: 9, Average Empirical Errors: 0.0928, Average True Error: 0.1471, Difference between them: -0.0543 

Best parameters of true_errors with p: 2, k: 1, True Error: 0.0652
Best parameters of empirical errors with p: 1, k: 1, Empirical Errors: 0.0

Process finished with exit code 0





"""
