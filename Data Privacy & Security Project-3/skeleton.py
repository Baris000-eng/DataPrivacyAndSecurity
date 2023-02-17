import sys
import random

import numpy as np
import pandas as pd
import copy

from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


###############################################################################
############################### Label Flipping ################################
###############################################################################
def machine_learning_model_accuracy_calculation_process(tst_x, trn_x, tst_y, dpcp_trn_y, md_typ):
    if tst_x is None:
        print("x test data is None !")
        raise Exception("x test data is None !!")
    if trn_x is None:
        print("x train data is None !")
        raise Exception("x train data is none !")
    if tst_y is None:
        print("y test data is None !")
        raise Exception("y test data is None !")
    if dpcp_trn_y is None:
        print("y train data is None !")
        raise Exception("y train data is None !")

    svc_equality = (md_typ.lower() == "svc")
    dt_equality = (md_typ.lower() == "dt")
    lr_equality = (md_typ.lower() == "lr")
    none_of_provided_ml_models = not (svc_equality or dt_equality or lr_equality)

    if md_typ is None:
        print("The Model Type is None !")
        raise Exception("The Model Type is None !")
    elif svc_equality:
        my_model_of_support_vector_classification = SVC(C=0.5, kernel='poly', random_state=0, probability=True)
        my_model_of_support_vector_classification.fit(trn_x, dpcp_trn_y)
        prediction_including_the_poison_val = my_model_of_support_vector_classification.predict(tst_x)
        return accuracy_score(tst_y, prediction_including_the_poison_val)
    elif dt_equality:
        my_decision_tree_classification_model = DecisionTreeClassifier(max_depth=5, random_state=0)
        my_decision_tree_classification_model.fit(trn_x, dpcp_trn_y)
        val_of_guessing = my_decision_tree_classification_model.predict(tst_x)
        return accuracy_score(tst_y, val_of_guessing)
    elif lr_equality:
        my_model_of_logistic_regression = LogisticRegression(penalty='l2', tol=0.001, C=0.1, max_iter=1000)
        my_model_of_logistic_regression.fit(trn_x, dpcp_trn_y)
        my_value_of_predicting = my_model_of_logistic_regression.predict(tst_x)
        return accuracy_score(tst_y, my_value_of_predicting)
    elif none_of_provided_ml_models:
        pass


def attack_label_flipping(X_train, X_test, y_train, y_test, model_type, n):
    # TODO: You need to implement this function!
    # You may want to use copy.deepcopy() if you will modify data
    if X_test is None:
        print("Test data for x values is None !")
        raise Exception("Test data for x values is None !")
    if X_train is None:
        print("Train data for x values is None !")
        raise Exception("Train data for x values is None !")
    if y_train is None:
        print("The train data for y values is None !")
        raise Exception("The train data for y values is None !")
    if y_test is None:
        print("The test data for the y values is None !")
        raise Exception("The test data for the y values is None !")
    if model_type is None:
        print("The model type is None !")
        raise Exception("The model type is None !")
    if n < 0:
        print("Invalid percentage of training data !")
        raise Exception("Invalid percentage of training data !")

    accuracy_summation_keeper_structure = list()
    from copy import deepcopy
    length_value_of_y_train_data = len(y_train)
    length_y_train_times_n = n * length_value_of_y_train_data  # get the specific percentage of y train data
    total_sample_number = int(length_y_train_times_n)
    my_custom_range = range(0, 100)  # iterate 100 times
    from random import choices
    for range_element in my_custom_range:
        deepcopied_version_of_y_train_data = deepcopy(y_train)
        length_of_deepcopied_y_train_data = len(deepcopied_version_of_y_train_data)
        my_custom_sec_rng = range(0, length_of_deepcopied_y_train_data)
        my_custom_lst_of_all_indices = choices(
            my_custom_sec_rng,
            k=total_sample_number
        )
        for cus_ind_elm in my_custom_lst_of_all_indices:
            single_data_of_y_train = deepcopied_version_of_y_train_data[cus_ind_elm]
            flipped_index_value = 0
            if single_data_of_y_train == 0:  # if original element of the y train data is 0
                flipped_index_value = 1  # get the flipped version as 1
            elif single_data_of_y_train == 1:  # if original element of the y train data is 1
                flipped_index_value = 0  # get the flipped version as 0
            deepcopied_version_of_y_train_data[cus_ind_elm] = int(flipped_index_value)
            # convert the label to int
            # then assign the label as an element of the deepcopied y train data.

        individ_accuracy_val = machine_learning_model_accuracy_calculation_process(
            X_test,
            X_train,
            y_test,
            deepcopied_version_of_y_train_data,
            model_type
        )
        accuracy_summation_keeper_structure.append(individ_accuracy_val)

    value_of_coefficient = (1 / 10) * (1 / 10)  # get the coefficient to be multiplied with accuracy sum
    # the final goal is to find the average of accuracies after multiplying the coefficient value with the accuracy summation
    summation_of_all_accuracies = sum(accuracy_summation_keeper_structure)
    averaged_accuracy_outcome = value_of_coefficient * summation_of_all_accuracies  # get the average
    return averaged_accuracy_outcome


###############################################################################
############################## Inference ########################################
###############################################################################

def inference_attack(trained_model, samples, t):
    if t < 0:
        print("Invalid threshold value is encountered now !")
        raise Exception("Invalid threshold value is encountered now !")
    if trained_model is None:
        print("The training model is None !")
        raise Exception("The training model is None !")
    if samples is None:
        print("Samples is None !")
        raise Exception("Samples is None !")
    if len(samples) <= 0:
        print("Empty samples or samples with invalid length is encountered !")
        raise Exception("Empty samples or samples with invalid length is encountered !")

    tp_v = 0
    fn_v = 0
    for sample in samples:
        sample = sample.reshape(+1, -1)
        possible_scores_of_confidence = trained_model.predict_proba(sample)
        for a in possible_scores_of_confidence:
            if a[0] > t or a[1] > t:
                tp_v = tp_v + 1
            elif a[0] <= t or a[1] <= t:
                fn_v = fn_v + 1

    tp_plus_fn = tp_v + fn_v
    recall_value = (tp_v / tp_plus_fn)
    return recall_value


###############################################################################
################################## Backdoor ###################################
###############################################################################

def backdoor_attack(X_train, y_train, model_type, num_samples):
    # TODO: You need to implement this function!
    # You may want to use copy.deepcopy() if you will modify data
    if (X_train is None) or (y_train is None) or (model_type is None) or (num_samples is None):
        print(
            "Train data of x coordinates, train data of y coordinates, the model type, or the number of samples is None !")
        raise Exception(
            "Train data of x coordinates, train data of y coordinates, the model type, or the number of samples is None !")
    return -999


###############################################################################
############################## Evasion ########################################
###############################################################################

def evade_model(trained_model, actual_example):
    # TODO: You need to implement this function!
    if trained_model is None:
        print("The trained model parameter is None !")
        raise Exception("The trained model parameter is None !")
    if actual_example is None:
        print("The actual example parameter is None !")
        raise Exception("The actual example parameter is None !")

    actual_class = trained_model.predict([actual_example])[0]  # actual class
    modified_example = copy.deepcopy(actual_example)  # deepcopying the actual class
    first_index = 0
    length_of_modified_example = len(modified_example)
    value_of_perturbation_proc = 75 * ((1 / 10) * (1 / 10) * (1 / 10))  # initializing the perturbation value to 0.075
    pred_class = actual_class
    actual_and_pred_are_identical = (pred_class == actual_class)
    while actual_and_pred_are_identical:  # while predicted class and actual class are identical
        my_custom_rng = range(0, length_of_modified_example)
        for location_val in my_custom_rng:
            modified_example[location_val] = value_of_perturbation_proc + modified_example[
                location_val]  # add perturbation to every element of modified example
            list_version_of_modified_example = [modified_example]
            modified_prediction_list = trained_model.predict(
                list_version_of_modified_example)  # predict the modified example
            pred_class = modified_prediction_list[first_index]
            pred_actual_not_eq = (pred_class != actual_class)
            pred_actual_equal = (pred_class == actual_class)
            if pred_actual_equal:
                subtracted_perturbation_amount = 4 * value_of_perturbation_proc
                modified_example[location_val] = modified_example[location_val] - subtracted_perturbation_amount
                lst_vers_of_modified_example = list()
                lst_vers_of_modified_example.append(modified_example)
                initial_index = 0
                modified_pred_lst = trained_model.predict(lst_vers_of_modified_example)
                pred_class = modified_pred_lst[initial_index]
                act_pred_are_equal = (
                        pred_class == actual_class
                )  # boolean checking whether the actual class is equal to the predicted class
                act_pred_not_equal = (
                        pred_class != actual_class
                )  # checking if the actual class is equal to the predicted class
                if act_pred_are_equal:
                    added_perturbation_amount = value_of_perturbation_proc * 3
                    modified_example[location_val] = modified_example[location_val] + added_perturbation_amount
                elif act_pred_not_equal:
                    return modified_example

            elif pred_actual_not_eq:
                return modified_example

        value_of_perturbation_proc = value_of_perturbation_proc + 0.075

    modified_example[0] = -2.0
    return modified_example


def calc_perturbation(actual_example, adversarial_example):
    # You do not need to modify this function.
    if len(actual_example) != len(adversarial_example):
        print("Number of features is different, cannot calculate perturbation amount.")
        return -999
    else:
        tot = 0.0
        for i in range(len(actual_example)):
            tot = tot + abs(actual_example[i] - adversarial_example[i])
        return tot / len(actual_example)


###############################################################################
############################## Transferability ################################
###############################################################################
def increase_index_loc(my_var: int):
    my_var = my_var + 1
    return my_var


def evaluate_transferability(DTmodel, LRmodel, SVCmodel, actual_examples):
    # TODO: You need to implement this function!
    if DTmodel is None:
        print("The decision tree classification model parameter is None !")
        raise Exception("The decision tree classification model parameter is None !")
    if SVCmodel is None:
        print("Support vector classification model parameter is None !")
        raise Exception("Support vector classification model parameter is None !")
    if LRmodel is None:
        print("Logistic regression model parameter is None !")
        raise Exception("Logistic regression model is None !")
    if len(actual_examples) <= 0:
        print("There are no actual examples or the list of actual examples has an invalid length !")
        raise Exception("There are no actual examples or the list of actual examples has an invalid length !")
    if actual_examples is None:
        print("The actual examples is None !")
        raise Exception("The actual examples is None !")

    all_provided_models_of_machine_learning = list()
    all_provided_models_of_machine_learning.append(LRmodel)
    all_provided_models_of_machine_learning.append(SVCmodel)
    all_provided_models_of_machine_learning.append(DTmodel)

    respective_names_of_machine_learning_models = list()
    respective_names_of_machine_learning_models.append("Logistic Regression (LR)")
    respective_names_of_machine_learning_models.append("Support Vector Classifier (SVC)")
    respective_names_of_machine_learning_models.append("Decision Tree (DT)")

    common_range = range(0, 3)
    logistic_regression_dictionary = dict()
    for w in common_range:
        logistic_regression_dictionary[w] = 0  # 0 refers to LR, 1 refers to SVC, 2 refers to DT

    support_vector_classifier_dictionary = dict()
    support_vector_classifier_dictionary[0] = 0  # refers to LR
    support_vector_classifier_dictionary[1] = 0  # refers to SVC
    support_vector_classifier_dictionary[2] = 0  # refers to DT

    decision_tree_classifier_dictionary = dict()
    for q in common_range:
        decision_tree_classifier_dictionary[q] = 0  # 0 refers to LR, 1 refers to SVC, 2 refers to DT

    location_over_ml_models = 0
    all_dictionaries_of_machine_learning_models = list()
    all_dictionaries_of_machine_learning_models.append(logistic_regression_dictionary)
    all_dictionaries_of_machine_learning_models.append(support_vector_classifier_dictionary)
    all_dictionaries_of_machine_learning_models.append(decision_tree_classifier_dictionary)

    fir_ind = 0

    for mdl_machine_learning in all_provided_models_of_machine_learning:  # traversing over all ml models
        location_in_dict = 0
        for machine_lrng_mdl in all_provided_models_of_machine_learning:  # traversing for identifying the ml models to be transferred to
            for smp_real in actual_examples:
                real_smp_lst = list()
                real_smp_lst.append(smp_real)
                prediction_outcome_of_real_sample = mdl_machine_learning.predict(real_smp_lst)
                cla_real = prediction_outcome_of_real_sample[fir_ind]

                instance_with_alteration = evade_model(
                    mdl_machine_learning,
                    smp_real
                )

                list_of_instance_with_alteration = list()
                list_of_instance_with_alteration.append(instance_with_alteration)
                prediction_outcome_of_changed_sample = machine_lrng_mdl.predict(list_of_instance_with_alteration)
                cla_enemy = prediction_outcome_of_changed_sample[fir_ind]

                check_real_enemy_equality = (cla_real == cla_enemy)
                check_real_enemy_inequality = (cla_real != cla_enemy)

                if not (check_real_enemy_equality or check_real_enemy_inequality):
                    pass
                elif check_real_enemy_equality:
                    pass
                elif check_real_enemy_inequality:  # check if the adversarial class is equal to the real or actual class.
                    all_dictionaries_of_machine_learning_models[location_over_ml_models][location_in_dict] = \
                        all_dictionaries_of_machine_learning_models[location_over_ml_models].get(location_in_dict,
                                                                                                 0) + 1

            location_in_dict = increase_index_loc(location_in_dict)
        location_over_ml_models = increase_index_loc(location_over_ml_models)
    loc_val_of_dictionary_struct = 0
    for dictionary_str in all_dictionaries_of_machine_learning_models:
        all_set_of_dictionary_elements = dictionary_str.items()
        print("---------------------------------------------------------------------------------------")
        for tup in all_set_of_dictionary_elements:
            type_of_the_machine_learning_model, transferred_amount = tup
            ml_from_model = respective_names_of_machine_learning_models[loc_val_of_dictionary_struct]
            ml_to_model = respective_names_of_machine_learning_models[type_of_the_machine_learning_model]
            print()
            print(
                str(transferred_amount) + " adversarial examples out of 40 adversarial examples are transferred from the " + str(
                    ml_from_model) + " to the " + str(ml_to_model) + "")
            outcome_of_accuracy = (1 / 40) * transferred_amount
            print("Accuracy of the transfer from the " + str(ml_from_model) + " to the " + str(
                ml_to_model) + " is " + str(outcome_of_accuracy) + "")
            print()
        print("---------------------------------------------------------------------------------------")
        loc_val_of_dictionary_struct = increase_index_loc(loc_val_of_dictionary_struct)


###############################################################################
########################## Model Stealing #####################################
###############################################################################

def steal_model(remote_model, model_type, examples):
    # TODO: You need to implement this function!
    # This function should return the STOLEN model, but currently it returns the remote model
    # You should change the return value once you have implemented your model stealing attack
    if remote_model is None:
        print("The remote model parameter is None !")
        raise Exception("The remote model parameter is None !")
    if (model_type is None) or (examples is None):
        print("The model type parameter or the examples parameter is None !")
        raise Exception("The model type parameter or the examples parameter is None !")
    if len(examples) <= 0:
        print("The parameter of examples has no length or invalid length !")
        raise Exception("The parameter of examples has no length or invalid length !")

    mdl_of_thief = None
    check_equality_with_svc = (
            model_type.lower() == "svc")  # checks whether the model is support vector classification regardless of case
    check_equality_with_lr = (
            model_type.lower() == "lr")  # checks whether the model is logistic regression regardless of case
    check_equality_with_dt = (
            model_type.lower() == "dt")  # checks whether the model is decision tree classifier regardless of case

    none_of_given_machine_lrng_models = not (
            check_equality_with_svc or check_equality_with_dt or check_equality_with_lr)

    if check_equality_with_svc:
        mdl_of_thief = SVC(
            C=0.5,
            kernel='poly',
            random_state=0,
            probability=True
        )
    elif check_equality_with_dt:
        mdl_of_thief = DecisionTreeClassifier(
            max_depth=5,
            random_state=0
        )
    elif check_equality_with_lr:
        mdl_of_thief = LogisticRegression(
            penalty='l2',
            tol=0.001,
            C=0.1,
            max_iter=1000
        )
    elif none_of_given_machine_lrng_models:
        pass

    lst_of_all_lbl = remote_model.predict(examples)
    mdl_of_thief.fit(examples, lst_of_all_lbl)
    return mdl_of_thief


###############################################################################
############################### Main ##########################################
###############################################################################

## DO NOT MODIFY CODE BELOW THIS LINE. FEATURES, TRAIN/TEST SPLIT SIZES, ETC. SHOULD STAY THIS WAY. ## 
## JUST COMMENT OR UNCOMMENT PARTS YOU NEED. ##

def main():
    data_filename = "forest_fires.csv"
    features = ["Temperature", "RH", "Ws", "Rain", "FFMC", "DMC", "DC", "ISI", "BUI", "FWI"]

    df = pd.read_csv(data_filename)
    df = df.dropna(axis=0, how='any')
    df["DC"] = df["DC"].astype('float64')
    y = df["class"].values
    y = LabelEncoder().fit_transform(y)
    X = df[features].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

    # Model 1: Decision Tree
    myDEC = DecisionTreeClassifier(max_depth=5, random_state=0)
    myDEC.fit(X_train, y_train)
    DEC_predict = myDEC.predict(X_test)
    print('Accuracy of decision tree: ' + str(accuracy_score(y_test, DEC_predict)))

    # Model 2: Logistic Regression
    myLR = LogisticRegression(penalty='l2', tol=0.001, C=0.1, max_iter=1000)
    myLR.fit(X_train, y_train)
    LR_predict = myLR.predict(X_test)
    print('Accuracy of logistic regression: ' + str(accuracy_score(y_test, LR_predict)))

    # Model 3: Support Vector Classifier
    mySVC = SVC(C=0.5, kernel='poly', random_state=0, probability=True)
    mySVC.fit(X_train, y_train)
    SVC_predict = mySVC.predict(X_test)
    print('Accuracy of SVC: ' + str(accuracy_score(y_test, SVC_predict)))

    # Label flipping attack executions:
    model_types = ["DT", "LR", "SVC"]
    n_vals = [0.05, 0.10, 0.20, 0.40]
    for model_type in model_types:
        for n in n_vals:
            acc = attack_label_flipping(X_train, X_test, y_train, y_test, model_type, n)
            print("Accuracy of poisoned", model_type, str(n), ":", acc)

    # Inference attacks:
    samples = X_train[0:100]
    t_values = [0.99, 0.98, 0.96, 0.8, 0.7, 0.5]
    for t in t_values:
        print("Recall of inference attack", str(t), ":", inference_attack(mySVC, samples, t))

    # Backdoor attack executions:
    counts = [0, 1, 3, 5, 10]
    for model_type in model_types:
        for num_samples in counts:
            success_rate = backdoor_attack(X_train, y_train, model_type, num_samples)
            print("Success rate of backdoor:", success_rate, "model_type:", model_type, "num_samples:", num_samples)

    # Evasion attack executions:
    trained_models = [myDEC, myLR, mySVC]
    model_types = ["DT", "LR", "SVC"]
    num_examples = 40
    for a, trained_model in enumerate(trained_models):
        total_perturb = 0.0
        for i in range(num_examples):
            actual_example = X_test[i]
            adversarial_example = evade_model(trained_model, actual_example)
            if trained_model.predict([actual_example])[0] == trained_model.predict([adversarial_example])[0]:
                print("Evasion attack not successful! Check function: evade_model.")
            perturbation_amount = calc_perturbation(actual_example, adversarial_example)
            total_perturb = total_perturb + perturbation_amount
        print("Avg perturbation for evasion attack using", model_types[a], ":", total_perturb / num_examples)

    # Transferability of evasion attacks:
    trained_models = [myDEC, myLR, mySVC]
    num_examples = 40
    evaluate_transferability(myDEC, myLR, mySVC, X_test[0:num_examples])

    # Model stealing:
    budgets = [8, 12, 16, 20, 24]
    for n in budgets:
        print("******************************")
        print("Number of queries used in model stealing attack:", n)
        stolen_DT = steal_model(myDEC, "DT", X_test[0:n])
        stolen_predict = stolen_DT.predict(X_test)
        print('Accuracy of stolen DT: ' + str(accuracy_score(y_test, stolen_predict)))
        stolen_LR = steal_model(myLR, "LR", X_test[0:n])
        stolen_predict = stolen_LR.predict(X_test)
        print('Accuracy of stolen LR: ' + str(accuracy_score(y_test, stolen_predict)))
        stolen_SVC = steal_model(mySVC, "SVC", X_test[0:n])
        stolen_predict = stolen_SVC.predict(X_test)
        print('Accuracy of stolen SVC: ' + str(accuracy_score(y_test, stolen_predict)))


if __name__ == "__main__":
    main()
