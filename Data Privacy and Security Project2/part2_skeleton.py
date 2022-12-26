import matplotlib.pyplot as plt
import numpy as np
import statistics
import pandas as pd
import math
import collections
from numpy.random import default_rng
import random

""" 
    Helper functions
    (You can define your helper functions here.)
"""


def read_dataset(filename):
    """
        Reads the dataset with given filename.
    """

    df = pd.read_csv(filename, sep=',', header=0)
    return df


### HELPERS END ###


''' Functions to implement '''


# TODO: Implement this function!
def get_histogram(dataset, chosen_anime_id="199"):
    from collections import OrderedDict

    # dealing with null dataset case and null anime id case and empty dataset case###########
    if dataset is None:
        print("The dataset is null !!!")
        raise Exception("The dataset is null !!!")
    if chosen_anime_id is None:
        print("The choosen anime id is None !!!!!")
        raise Exception("The choosen anime id is None !!!!!")
    if (dataset.shape[0] <= 0) or (dataset.shape[1] <= 0):
        print("The dataset parameter is empty or invalid !!!")
        raise Exception("The dataset parameter is empty or invalid !!!")

    column_under_chosen_anime_id = dataset[chosen_anime_id]

    total_count_of_values = column_under_chosen_anime_id.value_counts()

    all_keys = total_count_of_values.keys()

    all_values = total_count_of_values.values

    list_of_ratings = list(all_keys)

    list_of_counts = list(all_values)

    print("User Ratings: " + str(list_of_ratings) + "")

    print("Counts of user ratings for anime id = 199: " + str(list_of_counts) + "")

    # IMPORTANT NOTE !!
    # Note: For individually observing the output of the get_dp_histogram() function
    # (i.e., the noisy histogram) , please comment the plt.bar() and plt.show() parts
    # in the get_histogram() function. Moreover, please remove the # symbols from the
    # beginnings of all the lines in the get_dp_histogram() function. In other words,
    # remove all commentings from the source code lines which are inside the get_dp_histogram()
    # function and make all the code inside the get_dp_histogram() function execute.
    # IMPORTANT NOTE !!

    # IMPORTANT NOTE !!!
    # After you observe the original histogram, you should close the appearing GUI screen in order
    # to see the results of the errors and the results of the exponential mechanism in the second part.
    # After you close the appearing GUI screen, you will be able to see the results of the errors and the
    # results of the exponential mechanism in your terminal.
    # IMPORTANT NOTE !!!

    # IMPORTANT NOTE !!!
    # In the second part, for only observing the results of the errors and the results of the
    # exponential mechanism; you can comment out all source code lines related to plotting,
    # drawing, and showing histograms/graphs (i.e., you can comment out all source code lines
    # based on the matplotlib library (the source code lines which have plt prefixes)).
    # IMPORTANT NOTE !!!

    plt.bar(list_of_ratings, list_of_counts, width=1)

    plt.xlabel("User Ratings")

    plt.ylabel('Counts')

    plt.title("Rating Counts for Anime id = 199")

    list_of_ticks = [-1, 0, 2, 4, 6, 8, 10]
    list_of_labels = ["-1", "0", "2", "4", "6", "8", "10"]
    plt.xticks(ticks=list_of_ticks, labels=list_of_labels)

    plt.show()

    return list_of_counts


def get_dp_histogram(counts, epsilon: float):
    from numpy.random import default_rng

    if (counts is None) or \
            (len(counts) < 0) or \
            (len(counts) == 0):
        print("The list of counts is None or Empty or Invalid !!!")
        raise Exception("The list of counts is None or Empty or Invalid !!!")

    if (epsilon < 0) or (epsilon is None):
        print("Invalid epsilon parameter is encountered now !!!!!!")
        raise Exception("Invalid epsilon parameter is encountered now !!!!!!")

    # since the neighboring dataset is defined by applying modification on
    # the original dataset, we should take the sensitivity as 2 here.
    value_of_scale = float(2) / epsilon

    counts_with_noise = list()

    for count_elem in counts:
        laplace_smp = default_rng().laplace(scale=value_of_scale)
        countPlusNoise = count_elem + laplace_smp
        counts_with_noise.append(countPlusNoise)

    hist_with_noise = counts_with_noise

    all_user_ratings = [10.0, 9.0, 8.0, -1.0, 7.0, 6.0, 5.0, 4.0, 1.0, 2.0, 3.0]

    all_keys = list(all_user_ratings)

    plt.bar(all_keys, hist_with_noise)

    plt.xlabel("User ratings")

    plt.ylabel("Counts")

    plt.title("User rating counts for different epsilon values")

    list_of_ticks = [-1, 0, 2, 4, 6, 8, 10]

    list_of_labels = ["-1", "0", "2", "4", "6", "8", "10"]

    plt.xticks(ticks=list_of_ticks, wrap=True, labels=list_of_labels)

    # IMPORTANT NOTE !!
    # NOTE: For individually observing the output of the get_histogram() function
    # (i.e., the original histogram) please comment the plt.show() function which
    # is below this note. Moreover, please remove all the # symbols from the
    # beginnings of the source code lines inside the get_histogram() function. In other words,
    # remove all the commentings from the source code lines which are inside the get_histogram()
    # function and make all the code in the get_histogram() function execute.
    # After you see the original histogram, you should close the appearing GUI screen in order
    # to see the results of the errors and the results of the exponential mechanism in the second part.
    # IMPORTANT NOTE !!

    # plt.show()

    # IMPORTANT NOTE !!!
    # In the second part, for only observing the results of the errors and the results of the
    # exponential mechanism; you can comment out all source code lines related to plotting,
    # drawing, and showing histograms/graphs (i.e., you can comment out all source code lines
    # based on the matplotlib library (the source code lines which have plt prefixes)).
    # IMPORTANT NOTE !!!

    return hist_with_noise


# TODO: Implement this function!
def calculate_average_error(actual_hist, noisy_hist):
    if actual_hist is None or (len(actual_hist) < 0) or (len(actual_hist) == 0):
        print("The actual histogram is None or Empty!")
    if (noisy_hist is None) or (len(noisy_hist) < 0) or (len(noisy_hist) == 0):
        print("The noisy histogram is None or Empty!")
    value_of_total_err = float(0)
    length_of_actual_histogram = len(actual_hist)
    my_custom_range = range(0, length_of_actual_histogram)
    for w in my_custom_range:
        individual_error_value = noisy_hist[w] - actual_hist[w]
        if individual_error_value < 0:
            individual_error_value = -1 * individual_error_value
        elif individual_error_value == 0:
            individual_error_value = +1 * individual_error_value
        elif individual_error_value > 0:
            individual_error_value = +1 * individual_error_value
        value_of_total_err = value_of_total_err + individual_error_value

    average_err_val = value_of_total_err / length_of_actual_histogram
    return average_err_val


# TODO: Implement this function!
def calculate_mean_squared_error(actual_hist, noisy_hist):
    if actual_hist is None or (len(actual_hist) < 0) or (len(actual_hist) == 0):
        print("The actual histogram is None or Empty!")
        raise Exception("The actual histogram is None or Empty!")
    elif (noisy_hist is None) or (len(noisy_hist) < 0) or (len(noisy_hist) == 0):
        print("The noisy histogram is None or Empty!")
        raise Exception("The noisy histogram is None or Empty!")
    tot_value_holder = list()
    length_of_actual_histogram = len(actual_hist)

    my_custom_range_for_mean_squared_error = range(0, length_of_actual_histogram)

    for q in my_custom_range_for_mean_squared_error:
        individ_error_val = noisy_hist[q] - actual_hist[q]
        squared_individual_error = math.pow(individ_error_val, 2)
        tot_value_holder.append(squared_individual_error)

    sum_of_all = sum(tot_value_holder)
    value_of_mean_squared_err = sum_of_all / length_of_actual_histogram
    return value_of_mean_squared_err


# TODO: Implement this function!
def epsilon_experiment(counts, eps_values: list):
    if (counts is None) \
            or \
            (eps_values is None) \
            or \
            (len(eps_values) < 0) or \
            (len(eps_values) == 0):
        print("Invalid list of counts or invalid list of epsilon values is encountered !!!")
        raise Exception("Invalid list of counts or invalid list of epsilon values is encountered !!!")
    all_avg_errors = list()
    all_mse_errors = list()
    for eps in eps_values:
        first_error_list = list()
        second_error_list = list()
        for k in range(0, 40):
            dp_histogram = get_dp_histogram(counts, eps)
            average_error_value = calculate_average_error(counts, dp_histogram)
            mean_squared_error = calculate_mean_squared_error(counts, dp_histogram)
            first_error_list.append(average_error_value)
            second_error_list.append(mean_squared_error)
        average_of_avg_errors = sum(first_error_list) / len(first_error_list)
        all_avg_errors.append(average_of_avg_errors)
        average_of_mean_squared_errors = sum(second_error_list) / len(second_error_list)
        all_mse_errors.append(average_of_mean_squared_errors)

    return all_avg_errors, all_mse_errors


# FUNCTIONS FOR LAPLACE END #
# FUNCTIONS FOR EXPONENTIAL START #


# TODO: Implement this function!
def most_10rated_exponential(dataset, epsilon):
    if (dataset.shape[0] <= 0) or (dataset.shape[1] <= 0):
        print("The dataset is empty or invalid !!!")
        raise Exception("The dataset is empty or invalid !!!")
    if (epsilon < 0) or (dataset is None):
        print("The epsilon parameter or dataset parameter is invalid !")
        raise Exception("The epsilon parameter or dataset parameter is invalid !")
    denominatorOfExponentialProb = 0
    key_list = list(dataset.head(1).keys())
    max_ten_rating_count = 0
    ten_count_lst = list()
    returned_anime_id = str()
    animal_ids = list()
    for i in key_list:
        if i != "user_id":
            animal_ids.append(i)
            my_key_list = list(dataset[i].value_counts().keys())
            my_value_list = list(dataset[i].value_counts().values)
            for j in my_key_list:
                if j == 10.0:
                    ten_index = my_key_list.index(j)
                    current_ten_count = my_value_list[ten_index]
                    ten_count_lst.append(current_ten_count)
                    denominatorOfExponentialProb += math.exp((epsilon * current_ten_count) / 2)
                    max_ten_rating_count = max(max_ten_rating_count, current_ten_count)

    for i in range(0, len(ten_count_lst)):
        if ten_count_lst[i] == max_ten_rating_count:
            returned_anime_id = animal_ids[i]

    numeratorOfExponentialProb = math.exp((epsilon * max_ten_rating_count) / 2)
    exponentialProb = numeratorOfExponentialProb / denominatorOfExponentialProb
    random_number = random.random()
    if random_number <= exponentialProb:
        return returned_anime_id

    return 1


# TODO: Implement this function!
def exponential_experiment(dataset, eps_values: list):
    if (dataset.shape[0] <= 0) or (dataset.shape[1] <= 0):
        print("The dataset is empty or invalid !!!")
        raise Exception("The dataset is empty or invalid !!!")
    if dataset is None:
        print("The dataset parameter is None !")
    if eps_values is None:
        print("The eps_values parameter is None !")
    if len(eps_values) < 0:
        print("The length of the list of epsilon values is invalid !")
    if len(eps_values) == 0:
        print("The length of the list of epsilon values is zero !!")

    my_list = list()
    for epsilon_value in eps_values:
        true_answer_count = 0
        false_answer_count = 0
        for w in range(0, 1000):
            most10 = most_10rated_exponential(dataset, epsilon_value)
            if most10 == 1:  # if we are not in the range of exponential mechanism probability, I have returned 1 in most_10rated_exponential function.
                false_answer_count = false_answer_count + 1  # increment false answer count
            else:
                true_answer_count = true_answer_count + 1  # increment true answer count
        all_count = (true_answer_count + false_answer_count)
        accuracy_percentage = true_answer_count / all_count
        my_list.append(accuracy_percentage)

    return my_list


# FUNCTIONS TO IMPLEMENT END #

def main():
    filename = "anime-dp.csv"
    dataset = read_dataset(filename)

    counts = get_histogram(dataset)

    print("**** LAPLACE EXPERIMENT RESULTS ****")
    eps_values = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 1.0]
    error_avg, error_mse = epsilon_experiment(counts, eps_values)
    print("**** AVERAGE ERROR ****")
    for i in range(len(eps_values)):
        print("eps = ", eps_values[i], " error = ", error_avg[i])
    print("**** MEAN SQUARED ERROR ****")
    for i in range(len(eps_values)):
        print("eps = ", eps_values[i], " error = ", error_mse[i])

    print("**** EXPONENTIAL EXPERIMENT RESULTS ****")
    eps_values = [0.001, 0.005, 0.01, 0.03, 0.05, 0.1]
    exponential_experiment_result = exponential_experiment(dataset, eps_values)
    for i in range(len(eps_values)):
        print("eps = ", eps_values[i], " accuracy = ", exponential_experiment_result[i])


if __name__ == "__main__":
    main()
