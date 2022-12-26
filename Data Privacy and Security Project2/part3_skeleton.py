import math

import numpy

""" Globals """

DOMAIN = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]

""" Helpers """


def read_dataset(filename):
    """
        Reads the dataset with given filename.
    """

    result = []
    with open(filename, "r") as f:
        for line in f:
            result.append(int(line))
    return result


# You can define your own helper functions here. #
def avg_err_calc(actual_hist, noisy_hist):
    if (noisy_hist is None) or (actual_hist is None):
        print("The noisy histogram or the actual histogram is None !!!")
    total_err_value_keeper = list()
    actual_length = len(actual_hist)
    for w in range(0, actual_length):
        difference_value = actual_hist[w] - noisy_hist[w]
        if difference_value < 0 or difference_value == 0:
            difference_value = -1 * difference_value
        else:
            pass
        total_err_value_keeper.append(difference_value)

    sum_of_all_errors = sum(total_err_value_keeper)
    average_error_value = sum_of_all_errors / actual_length
    return average_error_value


### HELPERS END ###

""" Functions to implement """


# GRR

# TODO: Implement this function!
def perturb_grr(val, epsilon):
    if (val < 0) or \
            (val is None) or \
            (epsilon < 0) or \
            (epsilon is None):
        print("The val parameter or epsilon parameter is invalid !!!")
        raise Exception("The val parameter or epsilon parameter is invalid !!!")

    if (DOMAIN is None) or \
            (len(DOMAIN) <= 0):
        print("Invalid domain or empty domain is encountered !!!")
        raise Exception("Invalid domain or empty domain is encountered !!!")
    from random import random
    from random import sample
    from math import exp
    first_index = 0
    single_element = 1
    length_of_whole_domain = len(DOMAIN)
    numerator = exp(epsilon)
    denominator = numerator - 1 + length_of_whole_domain
    probab_real = numerator / denominator
    random_num = random()
    if probab_real < random_num:
        whole_without_value_param = list()
        my_range = range(0, length_of_whole_domain)
        for v in my_range:
            domain_element = DOMAIN[v]
            whole_without_value_param.append(domain_element)
        whole_without_value_param.remove(val)
        my_sample_list = sample(whole_without_value_param, single_element)
        my_sample_value = my_sample_list[first_index]
        my_sample_value = int(my_sample_value)
        return my_sample_value

    elif probab_real == random_num:
        val = int(val)
        return val
    elif random_num < probab_real:
        val = int(val)
        return val
    else:
        print("Invalid Situation Happened !")
        raise Exception("Invalid Situation Happened !")


# TODO: Implement this function!
def estimate_grr(perturbed_values, epsilon):
    if len(perturbed_values) == 0:
        print("There are no perturbed values !")
        raise Exception("There are no perturbed values !")
    if len(perturbed_values) < 0:
        print("Invalid length of perturbed values !")
        raise Exception("Invalid length of perturbed values !")
    if epsilon < 0:
        print("Epsilon cannot be negative !")
        raise Exception("Epsilon cannot be negative !")
    if epsilon is None:
        print("Null epsilon parameter is detected !")
        raise Exception("Epsilon cannot be negative !")
    if DOMAIN is None:
        print("The domain is none !")
        raise Exception("The domain is null !!")
    if len(DOMAIN) <= 0:
        print("The domain length is invalid or the domain is empty !")
        raise Exception("The domain length is invalid or the domain is empty !")

    length_of_perturbed_values = len(perturbed_values)
    length_of_whole_domain = len(DOMAIN)
    values_and_counts = dict()
    for ptrb_value in perturbed_values:
        default_value = 0
        dictionary_value = values_and_counts.get(ptrb_value, default_value)
        values_and_counts[ptrb_value] = dictionary_value + 1

    from math import exp
    numerator_of_true = exp(epsilon)
    denominator_of_true = numerator_of_true + length_of_whole_domain - 1
    true_prob = numerator_of_true / denominator_of_true

    numerator_of_false = 1 - true_prob
    denominator_of_false = (length_of_whole_domain - 1)
    false_prob = numerator_of_false / denominator_of_false

    all_cv_values = list()
    for domain_element in DOMAIN:
        count_value = values_and_counts[domain_element]
        if (count_value is None) or (count_value < 0):
            value_of_nv = 0
        elif count_value >= 0 and (not (count_value is None)):
            value_of_nv = values_and_counts[domain_element]
        else:
            print("An invalid situation has happened !!!")

        true_case = true_prob * value_of_nv
        false_case = (length_of_perturbed_values - value_of_nv) * false_prob
        value_of_iv = true_case + false_case

        diff_value = true_prob - false_prob
        weighted_length = length_of_perturbed_values * false_prob
        value_of_cv = (value_of_iv - weighted_length) / diff_value
        all_cv_values.append(value_of_cv)
    return all_cv_values


# TODO: Implement this function!
def grr_experiment(dataset, epsilon):
    if (dataset is None) or \
            (epsilon < 0) or \
            (epsilon is None) or \
            (DOMAIN is None) or \
            (len(DOMAIN) <= 0):
        print("The dataset parameter is None or the epsilon parameter is None !!!!")
        raise Exception("The dataset parameter is None or the epsilon parameter is None !!!!")

    real_list = list()
    dict_of_values_and_counts = dict()
    for ptrb_value in dataset:
        default_value = 0
        dictionary_value = dict_of_values_and_counts.get(ptrb_value, default_value)
        dict_of_values_and_counts[ptrb_value] = dictionary_value + 1

    for domain_val in DOMAIN:
        value_of_count = dict_of_values_and_counts[domain_val]
        real_list.append(value_of_count)

    list_of_data_with_randomized_changes = list()
    for value_of_data in dataset:
        perturbation_outcome = perturb_grr(value_of_data, epsilon)
        list_of_data_with_randomized_changes.append(perturbation_outcome)

    list_of_grr_estimations = estimate_grr(list_of_data_with_randomized_changes, epsilon)

    average_error_val = avg_err_calc(real_list, list_of_grr_estimations)
    average_error_val = float(average_error_val)
    return average_error_val


# RAPPOR

# TODO: Implement this function!
def encode_rappor(val):
    """
           Encodes the given value into a bit vector.
           Args:
               val (int): The user's true value.
           Returns:
               The encoded bit vector as a list: [0, 1, ..., 0]
       """
    if val is None:
        print("The value parameter is None !!!")
        raise Exception("The value parameter is None !!!")
    if (DOMAIN is None) or (len(DOMAIN) <= 0):
        print("Invalid or Empty Domain Found !")
        raise Exception("Invalid or Empty Domain Found !")

    list_of_bits = list()
    length_of_whole_dom = len(DOMAIN)
    for j in range(0, length_of_whole_dom):
        list_of_bits.append(0)

    val_loc_in_domain = DOMAIN.index(val)
    list_of_bits[val_loc_in_domain] = 1
    return list_of_bits


# TODO: Implement this function!
def perturb_rappor(encoded_val, epsilon):
    """
           Perturbs the given bit vector using RAPPOR protocol.
           Args:
               encoded_val (list) : User's encoded value
               epsilon (float): Privacy parameter
           Returns:
               Perturbed bit vector that the user reports to the server as a list: [1, 1, ..., 0]
       """
    if (len(encoded_val) <= 0) or \
            (encoded_val is None) or \
            (epsilon is None) or \
            (epsilon < 0) or (DOMAIN is None) or (len(DOMAIN) <= 0):
        print("Encoded val parameter or epsilon parameter or domain is invalid or none !")
        raise Exception("Encoded val parameter or epsilon parameter or domain is invalid or none !")

    index_of_current_bit = 0
    from random import random

    probability_of_preservation = (math.exp(epsilon / 2)) / ((math.exp(epsilon / 2)) + 1)
    probability_of_flipping = 1 / ((math.exp(epsilon / 2)) + 1)

    for binary_val in encoded_val:
        randomization_out = random()
        if (randomization_out >= 0) and (
                randomization_out < probability_of_flipping or randomization_out == probability_of_flipping) and (
                randomization_out <= 1):
            negation = not binary_val
            encoded_val[index_of_current_bit] = int(negation)
            index_of_current_bit = index_of_current_bit + 1
        elif (randomization_out >= 0) and (
                randomization_out == probability_of_preservation or randomization_out < probability_of_preservation) and (
                randomization_out <= 1):
            index_of_current_bit = index_of_current_bit + 1
            continue
        elif randomization_out > probability_of_preservation:
            pass
        elif randomization_out < 0 or randomization_out > 1:
            print("An imaginary situation happened now !!!")
            raise Exception("An imaginary situation happened now !!!")

    return encoded_val


def increase_steps(k: int):
    k = k + 1
    return k


# TODO: Implement this function!
def estimate_rappor(perturbed_values, epsilon):
    """
            Estimates the histogram given RAPPOR perturbed values of the users.
            Args:
                perturbed_values (list of lists): Perturbed bit vectors of all users
                epsilon (float): Privacy parameter
            Returns:
                Estimated histogram as a list: [1.5, 6.7, ..., 1061.0]
                for each hour in the domain [0, 1, ..., 24] respectively.
        """

    if (len(perturbed_values) < 0) \
            or \
            (len(perturbed_values) == 0) \
            or \
            (perturbed_values is None) \
            or \
            (epsilon < 0) \
            or \
            (epsilon is None) \
            or \
            (DOMAIN is None) \
            or \
            (len(DOMAIN) < 0) \
            or \
            (len(DOMAIN) == 0):
        print("The parameter of perturbed values or the epsilon parameter is invalid !")
        raise Exception("The parameter of perturbed values or the epsilon parameter is invalid !")
    length_of_perturbed_values = len(perturbed_values)
    count_keeper_for_all_bits = dict()
    # for data in dataset:
    # default_value = 0
    # dictionary_val = element_count_holder.get(data, default_value)
    # element_count_holder[data] = dictionary_val + 1

    for individual_prsn in perturbed_values:
        if len(individual_prsn) < 0:
            pass
        elif len(individual_prsn) == 0:
            pass
        elif individual_prsn is None:
            pass
        else:
            traverse_amount = 1
            for binary_part in individual_prsn:
                check_equality_with_zero = (binary_part == 0)
                check_equality_with_one = (binary_part == 1)
                other_cases = (
                    not (
                            (check_equality_with_zero == True) or
                            (check_equality_with_one == True)
                    )
                )
                if other_cases:
                    print("Invalid situation happened !!!")
                    raise Exception("Invalid situation happened !!!")
                elif check_equality_with_zero == True:
                    traverse_amount = increase_steps(traverse_amount)
                elif check_equality_with_one == True:
                    value_default = 0
                    count_value = count_keeper_for_all_bits.get(traverse_amount, value_default)
                    count_keeper_for_all_bits[traverse_amount] = count_value + 1
                    traverse_amount = increase_steps(traverse_amount)

    from math import exp

    value_exp = exp(epsilon / 2)

    true_numerator = value_exp
    true_denominator = (value_exp + 1)
    true_prob = true_numerator / true_denominator

    false_numerator = 1
    false_denominator = (value_exp + 1)
    false_prob = false_numerator / false_denominator

    keeper_of_all_cv_values = list()

    if (len(DOMAIN) < 0) \
            or \
            (DOMAIN is None) or (len(DOMAIN) == 0):
        print("Invalid or Empty Domain is found !")
    for domain_component in DOMAIN:
        check_zero = (count_keeper_for_all_bits[domain_component] == 0)
        check_non_zero = (
                (count_keeper_for_all_bits[domain_component] > 0) or
                (count_keeper_for_all_bits[domain_component] < 0)
        )
        if check_zero == True:
            value_of_nv = 0
        elif check_non_zero == True:
            value_of_nv = count_keeper_for_all_bits[domain_component]
        else:
            print("An invalid situation happened !")
            raise Exception("An invalid situation happened now !!!")

        true_case = true_prob * value_of_nv
        remaining_perturbed = length_of_perturbed_values - value_of_nv
        false_case = remaining_perturbed * false_prob

        value_of_iv = true_case + false_case

        true_minus_false = true_prob - false_prob
        length_with_weight = length_of_perturbed_values * false_prob
        iv_minus_weighted_length = (value_of_iv - length_with_weight)
        value_of_cv = iv_minus_weighted_length / true_minus_false

        keeper_of_all_cv_values.append(value_of_cv)
    return keeper_of_all_cv_values


# TODO: Implement this function!
def rappor_experiment(dataset, epsilon):
    """
            Conducts the data collection experiment for RAPPOR.
            Args:
                dataset (list): The daily_time dataset
                epsilon (float): Privacy parameter
            Returns:
                Error of the estimated histogram (float) -> Ex: 330.78
        """
    # dealing with some base cases, null cases, and imaginary cases.
    if (dataset is None) \
            or \
            (epsilon is None) \
            or \
            (epsilon < 0) or (DOMAIN is None) or (len(DOMAIN) <= 0):
        print("The dataset parameter or epsilon parameter or domain is None !")
    actual_list_of_all_counts = list()
    element_count_holder = dict()
    for data in dataset:
        default_value = 0
        dictionary_val = element_count_holder.get(data, default_value)
        element_count_holder[data] = dictionary_val + 1
    for domain_element in DOMAIN:
        actual_count = element_count_holder[domain_element]
        actual_list_of_all_counts.append(actual_count)
    ds_with_perturbation = list()
    for value_prs in dataset:  # for all dataset elements
        outcome_of_encoding_process = encode_rappor(value_prs)  # firstly, encode
        perturbation_outcome = perturb_rappor(
            outcome_of_encoding_process,
            epsilon
        )  # then, perturb
        ds_with_perturbation.append(
            perturbation_outcome)  # append the perturbation outcome the list keeping datasets which include perturbations.
    list_of_estimated_counts = estimate_rappor(
        ds_with_perturbation,
        epsilon
    )  # estimate the results
    value_of_average_err = avg_err_calc(
        actual_list_of_all_counts,
        list_of_estimated_counts
    )  # calculation of the average error value
    value_of_average_err = float(value_of_average_err)  # convert to float
    return value_of_average_err  # return the average error value


# OUE

# TODO: Implement this function!
def encode_oue(val):
    if val is None:  # dealing with none val case
        print("The value parameter is None !!!")
        raise Exception("The value parameter is None !!!")
    if (DOMAIN is None) or (len(DOMAIN) == 0):  # dealing with null domain and empty domain cases
        print("Invalid or Empty Domain Found !")
        raise Exception("Invalid or Empty Domain Found !")
    keeper_of_bits = list()  # initializing the list of bits
    length_of_whole_dom = len(DOMAIN)  # assigning length of the domain to a variable
    for j in range(0, length_of_whole_dom):  # construction of initial list with length equal to domain length.
        keeper_of_bits.append(0)  # filling the list with zeros initially
    location_of_value = DOMAIN.index(val)  # finding the location of value in the domain
    keeper_of_bits[location_of_value] = 1  # changing the value of bit at the found location to 1
    return keeper_of_bits  # returning the encoded list of bits


def increment_repetitions(k: int):
    k = k + 1
    return k


# TODO: Implement this function!
def perturb_oue(encoded_val, epsilon):
    if len(encoded_val) <= 0:
        print("Empty or invalid encoded val found !")
        raise Exception("Empty or invalid encoded val found !")
    if epsilon < 0:
        print("Epsilon cannot be negative !")
        raise Exception("Epsilon cannot be negative !")
    if encoded_val is None:
        print("Encoded val is None !!!")
        raise Exception("Encoded val is None !!!")

    index_of_current_bit = 0
    from random import random

    bit_preservation_probability = 0
    flip_bit_probability = 0

    for value_binary in encoded_val:
        if value_binary == 1:
            bit_preservation_probability = 1 / 2
            flip_bit_probability = 1 / 2
        elif value_binary == 0:
            bit_preservation_probability = (math.exp(epsilon)) / ((math.exp(epsilon)) + 1)
            flip_bit_probability = 1 / ((math.exp(epsilon)) + 1)

        randomization_outcome = random()
        if randomization_outcome < 0 or randomization_outcome > 1:
            print("An invalid situation has happened !")
            raise Exception("An invalid situation has happened !")
        elif (randomization_outcome >= 0) and (
                randomization_outcome < flip_bit_probability or randomization_outcome == flip_bit_probability) and (
                randomization_outcome <= 1):
            negation = not value_binary
            encoded_val[index_of_current_bit] = int(negation)
            index_of_current_bit = index_of_current_bit + 1
        elif randomization_outcome is None:
            print("Randomized outcome is None !!!")
            raise Exception("Randomized outcome is None !!!")
        else:
            index_of_current_bit = increment_repetitions(index_of_current_bit)
            continue

    return encoded_val


# TODO: Implement this function!
def estimate_oue(perturbed_values, epsilon):
    import math
    if (len(perturbed_values) < 0) \
            or \
            (len(perturbed_values) == 0) \
            or \
            (perturbed_values is None) \
            or \
            (epsilon < 0) \
            or \
            (epsilon is None) \
            or \
            (DOMAIN is None) \
            or \
            (len(DOMAIN) < 0) \
            or \
            (len(DOMAIN) == 0):
        print("The parameter of perturbed values or the epsilon parameter is invalid !")
        raise Exception("The parameter of perturbed values or the epsilon parameter is invalid !")
    length_of_perturbed_values = len(perturbed_values)
    count_keeper_for_all_bits = dict()

    for i in range(0, length_of_perturbed_values):
        x = 0
        if len(perturbed_values[i]) < 0:
            pass
        elif len(perturbed_values[i]) == 0:
            pass
        elif perturbed_values[i] is None:
            pass
        else:
            traverse_amount = 1
            for j in range(0, len(perturbed_values[i])):
                check_equality_with_zero = (perturbed_values[i][j] == 0)
                check_equality_with_one = (perturbed_values[i][j] == 1)
                other_cases = (
                    not (
                            (check_equality_with_zero == True) or
                            (check_equality_with_one == True)
                    )
                )
                if other_cases:
                    print("Invalid situation happened !!!")
                elif check_equality_with_zero:
                    traverse_amount = increase_steps(traverse_amount)
                elif check_equality_with_one:
                    value_default = 0
                    count_value = count_keeper_for_all_bits.get(traverse_amount, value_default)
                    count_keeper_for_all_bits[traverse_amount] = count_value + 1
                    traverse_amount = increase_steps(traverse_amount)

    keeper_of_cv_values = list()

    if (len(DOMAIN) <= 0) \
            or \
            (DOMAIN is None):
        print("Invalid or Empty Domain is found !")

    s = 0
    number_of_ones = 0
    for w in range(0, len(DOMAIN)):
        check_zero = (count_keeper_for_all_bits[DOMAIN[w]] == 0)
        check_non_zero = (
                (count_keeper_for_all_bits[DOMAIN[w]] > 0) or
                (count_keeper_for_all_bits[DOMAIN[w]] < 0)
        )
        check_other_conditions = (
            not (
                    (count_keeper_for_all_bits[DOMAIN[w]] > 0) or
                    (count_keeper_for_all_bits[DOMAIN[w]] < 0) or
                    (count_keeper_for_all_bits[DOMAIN[w]] == 0)
            )
        )

        if check_zero:
            number_of_ones = 0
        elif check_non_zero:
            number_of_ones = count_keeper_for_all_bits[DOMAIN[w]]
        elif check_other_conditions:
            print("An invalid situation happened !")
            raise Exception("An invalid situation happened !")

        exponent_expression = (math.exp(epsilon) + 1)
        oue_numerator = (exponent_expression * number_of_ones - length_of_perturbed_values)
        denominator_oue = math.exp(epsilon) - 1
        value_of_cv = (2 * oue_numerator) / denominator_oue
        keeper_of_cv_values.append(value_of_cv)

    return keeper_of_cv_values


# TODO: Implement this function!
def oue_experiment(dataset, epsilon):
    if (dataset is None) \
            or \
            (epsilon is None) \
            or \
            (epsilon < 0) or (DOMAIN is None) or (len(DOMAIN) <= 0):
        print("The dataset parameter or epsilon parameter or domain is None !")
    actual_list_of_all_counts = list()
    element_count_holder = dict()
    for data in dataset:
        default_value = 0
        dictionary_val = element_count_holder.get(data, default_value)
        element_count_holder[data] = dictionary_val + 1
    for domain_element in DOMAIN:
        actual_value_of_count = element_count_holder[domain_element]
        actual_list_of_all_counts.append(actual_value_of_count)

    ds_including_perturbation = list()
    for value_prs in dataset:  # go along all dataset elements
        outcome_of_encoding_process = encode_oue(value_prs)  # initially, encode
        perturbation_outcome = perturb_oue(
            outcome_of_encoding_process,
            epsilon
        )  # then, perturb
        ds_including_perturbation.append(
            perturbation_outcome)  # append the perturbation outcome the list keeping datasets which include perturbations.
    list_of_estimated_counts = estimate_oue(
        ds_including_perturbation,
        epsilon
    )  # estimate the results

    value_of_average_err = avg_err_calc(
        actual_list_of_all_counts,
        list_of_estimated_counts
    )  # calculation of the average error value
    value_of_average_err = float(value_of_average_err)  # convert to float
    return value_of_average_err  # return the average error value


def main():
    dataset = read_dataset("msnbc-short-ldp.txt")

    print("GRR EXPERIMENT")
    for epsilon in [0.1, 0.5, 1.0, 2.0, 4.0, 6.0]:
        error = grr_experiment(dataset, epsilon)
        print("e={}, Error: {:.2f}".format(epsilon, error))

    print("*" * 50)

    print("RAPPOR EXPERIMENT")
    for epsilon in [0.1, 0.5, 1.0, 2.0, 4.0, 6.0]:
        error = rappor_experiment(dataset, epsilon)
        print("e={}, Error: {:.2f}".format(epsilon, error))

    print("*" * 50)

    print("OUE EXPERIMENT")
    for epsilon in [0.1, 0.5, 1.0, 2.0, 4.0, 6.0]:
        error = oue_experiment(dataset, epsilon)
        print("e={}, Error: {:.2f}".format(epsilon, error))


if __name__ == "__main__":
    main()
