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

    print(f"\nBest Parameters with p: {best_p}, k: {best_k}, True Error: {np.round(best_true_error, 4)}\n")


if __name__ == '__main__':
    X, y = load_data('dataset/haberman.data', True)
    run_task(X, y, 'haberman')

    X, y = load_data('dataset/circle_separator.txt', False)
    run_task(X, y, 'circle_separator')



"""
-----run knn on haberman data set--------------- 
p: 1, k: 1, Average Empirical Errors: 0.0092, Average True Error: 0.3389, Difference between them: -0.3297 
p: 1, k: 3, Average Empirical Errors: 0.168, Average True Error: 0.2971, Difference between them: -0.129 
p: 1, k: 5, Average Empirical Errors: 0.2029, Average True Error: 0.2767, Difference between them: -0.0737 
p: 1, k: 7, Average Empirical Errors: 0.2162, Average True Error: 0.2673, Difference between them: -0.051 
p: 1, k: 9, Average Empirical Errors: 0.2208, Average True Error: 0.2605, Difference between them: -0.0397 
p: 2, k: 1, Average Empirical Errors: 0.0094, Average True Error: 0.3243, Difference between them: -0.3149 
p: 2, k: 3, Average Empirical Errors: 0.1639, Average True Error: 0.3018, Difference between them: -0.1378 
p: 2, k: 5, Average Empirical Errors: 0.2052, Average True Error: 0.2782, Difference between them: -0.073 
p: 2, k: 7, Average Empirical Errors: 0.2184, Average True Error: 0.2625, Difference between them: -0.0442 
p: 2, k: 9, Average Empirical Errors: 0.2194, Average True Error: 0.2628, Difference between them: -0.0434 
p: inf, k: 1, Average Empirical Errors: 0.0093, Average True Error: 0.3385, Difference between them: -0.3292 
p: inf, k: 3, Average Empirical Errors: 0.164, Average True Error: 0.2939, Difference between them: -0.1299 
p: inf, k: 5, Average Empirical Errors: 0.2014, Average True Error: 0.2692, Difference between them: -0.0678 
p: inf, k: 7, Average Empirical Errors: 0.2075, Average True Error: 0.2654, Difference between them: -0.0578 
p: inf, k: 9, Average Empirical Errors: 0.2209, Average True Error: 0.2551, Difference between them: -0.0342 

Best Parameters with p: inf, k: 9, True Error: 0.2551


-----run knn on circle_separator data set--------------- 
p: 1, k: 1, Average Empirical Errors: 0.0, Average True Error: 0.0707, Difference between them: -0.0707 
p: 1, k: 3, Average Empirical Errors: 0.0315, Average True Error: 0.0935, Difference between them: -0.062 
p: 1, k: 5, Average Empirical Errors: 0.0591, Average True Error: 0.1093, Difference between them: -0.0503 
p: 1, k: 7, Average Empirical Errors: 0.0695, Average True Error: 0.1283, Difference between them: -0.0588 
p: 1, k: 9, Average Empirical Errors: 0.0773, Average True Error: 0.1423, Difference between them: -0.0649 
p: 2, k: 1, Average Empirical Errors: 0.0, Average True Error: 0.0681, Difference between them: -0.0681 
p: 2, k: 3, Average Empirical Errors: 0.0317, Average True Error: 0.0827, Difference between them: -0.0509 
p: 2, k: 5, Average Empirical Errors: 0.0492, Average True Error: 0.1009, Difference between them: -0.0517 
p: 2, k: 7, Average Empirical Errors: 0.0676, Average True Error: 0.1196, Difference between them: -0.052 
p: 2, k: 9, Average Empirical Errors: 0.0904, Average True Error: 0.1507, Difference between them: -0.0603 
p: inf, k: 1, Average Empirical Errors: 0.0, Average True Error: 0.0771, Difference between them: -0.0771 
p: inf, k: 3, Average Empirical Errors: 0.0339, Average True Error: 0.0865, Difference between them: -0.0527 
p: inf, k: 5, Average Empirical Errors: 0.0508, Average True Error: 0.0981, Difference between them: -0.0473 
p: inf, k: 7, Average Empirical Errors: 0.0757, Average True Error: 0.1305, Difference between them: -0.0548 
p: inf, k: 9, Average Empirical Errors: 0.0951, Average True Error: 0.1571, Difference between them: -0.062 

Best Parameters with p: 2, k: 1, True Error: 0.0681


Process finished with exit code 0


"""