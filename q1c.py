"""
@author Antoun Obied

This program implements softmax regression to train a model, or perform classification on a set of features.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle
from Optimal_Parameters import Optimal_Parameters
from math import floor


def generate_data(filename):
    """
    Reads in a .dat file, and adds features to one array, and adds the classes (last index of row) to another array
    :param filename: Name of .dat file
    :return: The arrays of features and class values
    """
    filedata = np.genfromtxt(filename, dtype=None, delimiter=",")

    features = []
    class_list = []

    # For each row, add the last index to the class list, and all other entries to the feature list
    for i in filedata:
        sample = list(i)
        sample.pop(-1)
        features.append(sample)
        class_list.append(float(i[-1]))

    # Convert the lists to numpy arrays for easier manipulation
    features = np.array(features)
    class_list = np.array(class_list)

    return features, class_list


def get_num_classes(class_list):
    """
    Calculates the number of different classes in a data set
    :param class_list: List of all class values
    :return: Number of different classes in the data set
    """
    unique_classes = []

    for i in class_list:
        if i not in unique_classes:
            unique_classes.append(i)

    return len(unique_classes)


def logit_score_matrix(features, weights, bias):
    """
    Calculates the logits, or net inputs, from the features, weights, and bias
    :param features: Array of features read from file
    :param weights: Weight matrix
    :param bias: Bias vector
    :return: Logits matrix
    """
    return features.dot(weights) + bias


def softmax(X):
    """
    Calculates matrix output using the softmax equation
    :param X: Matrix of logits
    :return: Output matrix after being passed through softmax function
    """
    return np.exp(X) / np.sum(np.exp(X), axis=1, keepdims=True)


def one_hot_encoding(class_list, num_classes):
    """
    Return the one-hot encoding of a class list, where each row is all zeros except for the index of the class, which
    is set to 1
    :param class_list: List of class values
    :param num_classes: Number of different classes in class list
    :return: One hot encoded matrix
    """

    # Returns true for the class index, false otherwise
    booleans = (np.arange(num_classes) == class_list[:, None])

    # Converts all false entries to 0, and all true entries to 1
    encoded = booleans.astype(float)
    return encoded


def sigmoid(X):
    return 1 / (1 + np.exp(-X))


def sigmoid_derivative(X):
    return sigmoid(X) * (1 - sigmoid(X))


def cost_function(results, class_list):
    """
    Calculates the cost given the current results and weights
    :param results: Probability matrix from softmax function
    :param weights: Current weights
    :return: Cost
    """

    # From cost function defined in assignment. 0.01 is lambda, the regularization parameter
    num_samples = len(class_list)
    smax = softmax(results)
    cost = np.sum(-np.log(smax[range(num_samples), class_list])) / num_samples

    return cost


def cost_derivative_estimate(results, epsilon, class_list):
    """
    Estimates the current derivative of the cost using secant approximation
    :param results: Probability matrix from softmax function
    :param weights: Current weights
    :param epsilon: Small value used to change the cost slightly to calculate gradient
    :return:
    """
    return (cost_function(results, class_list) - cost_function(results,class_list)) / (2 * epsilon)


def gradient_descent(features, one_hot_encoded, weights_input, bias_input, weights_hidden, bias_hidden,
                     learning_rate, max_iterations):
    """
    Gradient descent algorithm to find the optimal weight and bias values
    :param features: Features used for training
    :param one_hot_encoded: One hot encoded matrix from class list
    :param weights_input: Current weights
    :param bias_input: Current bias
    :param learning_rate: Step size with each iteration
    :param max_iterations: The maximum number of iterations before the loop breaks
    :return: Optimal weights, optimal bias, cost history, cost history estimated using secant approximation
    """
    # List of all calculated costs
    cost_history = []

    # List of all estimated costs using secant approximation
    cost_estimate_history = []

    # Weight that is estimated using secant approximation
    input_weight_estimate = weights_input
    input_bias_estimate = bias_input

    hidden_weight_estimate = weights_hidden
    hidden_bias_estimate = bias_hidden

    class_list = one_hot_encoded.argmax(axis=1)

    for i in range(max_iterations):
        # Forward Propagation

        # Calculate the logits, and from that the probability matrix
        input_results = sigmoid(logit_score_matrix(features, weights_input, bias_input))

        hidden_results = softmax(logit_score_matrix(input_results, weights_hidden, bias_hidden))

        # Back Propagation

        # Calculate the partial cost derivative with respect to weight, and with respect to bias
        hidden_weight_gradient = input_results.T @ (hidden_results - one_hot_encoded)
        hidden_bias_gradient = np.sum(hidden_results - one_hot_encoded)

        input_weight_gradient = features.T @ \
                                (sigmoid_derivative(logit_score_matrix(features, weights_input, bias_input)) *
                                 ((hidden_results - one_hot_encoded) @ weights_hidden.T))

        input_bias_gradient = np.sum(((hidden_results - one_hot_encoded) @ weights_hidden.T) * sigmoid_derivative(
            logit_score_matrix(features, weights_input, bias_input)))

        # Modify the current weight and bias values
        weights_input -= learning_rate * input_weight_gradient
        bias_input -= learning_rate * input_bias_gradient

        weights_hidden -= learning_rate * hidden_weight_gradient
        bias_hidden -= learning_rate * hidden_bias_gradient

        # Estimate the gradient using secant approximation
        gradient_estimate = cost_derivative_estimate(hidden_results, 0.001, class_list)

        # Calculate the weight from the estimated gradient
        input_weight_estimate -= learning_rate * gradient_estimate
        input_bias_estimate -= learning_rate * gradient_estimate

        hidden_weight_estimate -= learning_rate * gradient_estimate
        hidden_bias_estimate -= learning_rate * gradient_estimate

        # Calculate the cost using the modified weight, and the estimated weight using secant approximation, and append
        # them to separate lists
        cost_history.append(cost_function(hidden_results, class_list))
        cost_estimate_history.append(cost_function(hidden_results, class_list))

    return weights_input, bias_input, weights_hidden, bias_hidden, cost_history, \
           cost_estimate_history, input_weight_estimate, input_bias_estimate, hidden_weight_estimate, hidden_bias_estimate


def get_predictions(softmax_matrix):
    predictions = np.argmax(softmax_matrix, axis=1)

    return predictions


def calculate_accuracy(predictions, class_list):
    correct = 0
    for i in range(len(predictions)):
        if predictions[i] == class_list[i]:
            correct += 1

    return correct / predictions.shape[0]


def create_mini_batch(features, class_list):
    """
    Split the data set to a mini batch for training, and a validation batch for testing accuracy
    :param features: Full array of features
    :param class_list: Full array of class values
    :return: Mini batches of features, class list, as well as validation batches of features and class list
    """

    # About 70% of the full training set is used for training. The remaining is used for validation
    mini_batch_size = floor(0.7 * len(features))

    mini_batch_features = []
    mini_batch_class_list = []

    # Generate a random index between 0 and the length of the list of features, which gets smaller with every iteration
    for i in range(mini_batch_size):
        random_index = np.random.randint(0, (len(features) - 1))

        # Add the values at the random index to the mini batch lists
        mini_batch_features.append(features[random_index])
        mini_batch_class_list.append(class_list[random_index])

        # Remove the values at that index from the full lists to create the validation lists
        features = np.delete(features, random_index, axis=0)
        class_list = np.delete(class_list, random_index, axis=0)

    mini_batch_features = np.array(mini_batch_features)
    mini_batch_class_list = np.array(mini_batch_class_list)


    return mini_batch_features, mini_batch_class_list, features, class_list

def main():

    if sys.argv[1] == "train":
        # Get the feature matrix and class list from a .dat file
        features, class_list = generate_data("iris_train.dat")

        mini_features, mini_class_list, validation_features, validation_class_list = create_mini_batch(features,
                                                                                                       class_list)

        # Calculate the number of different classes in the class list
        num_classes = get_num_classes(class_list)
        num_nodes = 6

        # Initialize the weight matrix and bias vector based on the number of features per sample, and the number of
        # different classes
        weights_input = np.array(np.random.rand(features.shape[1], num_nodes))
        bias_input = np.array(np.random.rand(1, num_nodes))

        weights_hidden = np.array(np.random.rand(num_nodes, num_classes))
        bias_hidden = np.array(np.random.rand(1, num_classes))

        # Calculate one hot encoded matrix
        one_hot_encoded = one_hot_encoding(mini_class_list, num_classes)

        # Use gradient descent to find the optimal weights, bias, and cost histories using 2 methods
        optimal_weights, optimal_bias, optimal_weights_hidden, optimal_bias_hidden, cost_history, cost_estimates, \
        weight_estimate, bias_estimate, hidden_weight_estimate, hidden_bias_estimate = \
            gradient_descent(mini_features, one_hot_encoded, weights_input, bias_input, weights_hidden, bias_hidden, 0.01,
                             60000)

        # Check if optimal weights match with weights estimated using secant approximation
        optimal_vs_estimates = abs(optimal_weights - weight_estimate)
        bias_comparison = abs(optimal_bias - bias_estimate)

        check_matrix = [[], [], [], []]
        bias_check = []

        for i in range(len(optimal_vs_estimates)):
            for j in range(len(optimal_vs_estimates[i])):
                if optimal_vs_estimates[i][j] <= 0.0001:
                    check_matrix[i].append("CORRECT")
                else:
                    check_matrix[i].append("WRONG")

        for i in range(len(bias_comparison[0])):
            if bias_comparison[0][i] <= 0.0001:
                bias_check.append("CORRECT")
            else:
                bias_check.append("WRONG")

        print("Optimal Input weights vs Estimated Input weights:\n ", check_matrix)

        print("Optimal Input Bias vs Estimated Input Bias:\n ", bias_check)

        hidden_optimal_vs_estimates = abs(optimal_weights_hidden - hidden_weight_estimate)
        hidden_bias_comparison = abs(optimal_bias_hidden - hidden_bias_estimate)

        check_matrix = [[], [], [], [], [], []]
        bias_check = []

        for i in range(len(hidden_optimal_vs_estimates)):
            for j in range(len(hidden_optimal_vs_estimates[i])):
                if hidden_optimal_vs_estimates[i][j] <= 0.0001:
                    check_matrix[i].append("CORRECT")
                else:
                    check_matrix[i].append("WRONG")

        for i in range(len(hidden_bias_comparison[0])):
            if hidden_bias_comparison[0][i] <= 0.0001:
                bias_check.append("CORRECT")
            else:
                bias_check.append("WRONG")

        print("Optimal Hidden weights vs Estimated Hidden weights:\n ", check_matrix)

        print("Optimal Hidden Bias vs Hidden Estimated Bias:\n ", bias_check)

        model = Optimal_Parameters(optimal_weights, optimal_bias, optimal_weights_hidden, optimal_bias_hidden)

        pickle.dump(model, open("iris_model_grading", "wb"))

        # Plot the cost histories obtained from both methods
        plt.figure(1)
        plt.plot(range(len(cost_history)), cost_history)
        plt.xlabel("Iterations")
        plt.ylabel("Cost")
        plt.title("Cost History")

        plt.figure(2)
        plt.plot(range(len(cost_estimates)), cost_estimates)
        plt.xlabel("Iterations")
        plt.ylabel("Cost")
        plt.title("Cost History Using Secant Approximation")

        # Calculate logits and probability matrix of the validation set
        net_inputs_mini = logit_score_matrix(validation_features, optimal_weights, optimal_bias)
        sigmoid_validation = sigmoid(net_inputs_mini)
        smax_validation = softmax(logit_score_matrix(sigmoid_validation, optimal_weights_hidden, optimal_bias_hidden))

        one_hot_encoded = one_hot_encoding(validation_class_list, num_classes)

        optimal_weights, optimal_bias, optimal_weights_hidden, optimal_bias_hidden, cost_history, cost_estimates, \
        weight_estimate, bias_estimate, hidden_weight_estimate, hidden_bias_estimate = \
            gradient_descent(validation_features, one_hot_encoded, weights_input, bias_input, weights_hidden, bias_hidden,
                             0.01,
                             60000)

        # Plot the cost histories obtained from both methods
        plt.figure(1)
        plt.plot(range(len(cost_history)), cost_history, "r")
        plt.xlabel("Iterations")
        plt.ylabel("Cost")
        plt.title("Cost History")
        plt.legend(["Training Loss", "Validation Loss"])

        plt.figure(2)
        plt.plot(range(len(cost_estimates)), cost_estimates, "r")
        plt.xlabel("Iterations")
        plt.ylabel("Cost")
        plt.title("Cost History Using Secant Approximation")
        plt.legend(["Training Loss", "Validation Loss"])
        plt.show()

        # Use the validation set to make predictions
        validation_predictions = get_predictions(smax_validation)
        print("Validation Predictions: ", validation_predictions)

        # Calculate and display the validation accuracy
        print("Validation Accuracy: ", calculate_accuracy(validation_predictions, validation_class_list) * 100, "%")

    elif sys.argv[1] == "predict":
        # Read features and corresponding class values from .dat file
        features, class_list = generate_data("iris_test.dat")

        # Load model from directory
        model = pickle.load(open("iris_model", "rb"))

        # Get optimal weights and bias from trained model
        input_optimal_weights = model.input_weights
        input_optimal_bias = model.input_bias
        hidden_optimal_weights = model.hidden_weights
        hidden_optimal_bias = model.hidden_bias

        # Calculate the logits from the optimal weights and bias, and the probability matrix from that
        net_inputs_optimal = logit_score_matrix(features, input_optimal_weights, input_optimal_bias)
        sigmoid_optimal = sigmoid(net_inputs_optimal)
        smax_optimal = softmax(logit_score_matrix(sigmoid_optimal, hidden_optimal_weights, hidden_optimal_bias))

        # Get the array of predictions
        predictions = get_predictions(smax_optimal)
        print("Predictions: ", predictions)

        # Calculate and show accuracy of trained model
        print("Accuracy: ", calculate_accuracy(predictions, class_list) * 100, "%")


if __name__ == "__main__":
    main()