##############################################################################
# This skeleton was created by Efehan Guner  (efehanguner21@ku.edu.tr)       #
# Note: requires Python 3.5+                                                 #
##############################################################################


# Necessary import statements############################################################################################################################################################################################################################################
import csv
import operator
from enum import unique
import glob
import os
import treelib
import sys
from copy import deepcopy
import numpy as np
import datetime
import _collections
import random
from random import randint
from random import randrange
import collections
import math


domain_generalization_hierarchy_list = list()
# Necessary import statements###########################################################################################################################################################################################################################################


########The variables I defined ###########################################################
tree_struct_lst = list()
########The variables I defined ###########################################################
from collections import defaultdict

if sys.version_info[0] < 3 or sys.version_info[1] < 5:
    sys.stdout.write("Requires Python 3.x.\n")
    sys.exit(1)

##############################################################################
# Helper Functions                                                           #
# These functions are provided to you as starting points. They may help your #
# code remain structured and organized. But you are not required to use      #
# them. You can modify them or implement your own helper functions.          #
##############################################################################


DGHs = dict()
#########MY HELPER FUNCTIONS###########################################################################################################


def obtain_adj(my_custom_lst: list, my_tr):
    difference_value = 0
    z = 0
    len_of_my_custom_list = len(my_custom_lst)
    my_custom_range = range(0, len_of_my_custom_list)
    already_trav = list()
    difference_value = z - len_of_my_custom_list
    w = 0
    while (z - len_of_my_custom_list) < 0:
        while (w - len_of_my_custom_list) < 0:
            if z == w:
                pass
            else:
                my_tuple_structure = (z, w)
                if my_tuple_structure in already_trav:
                    pass
                else:
                    notequ = (
                        not (
                            (check_equal(my_custom_lst[z], my_custom_lst[w])
                            )
                        )
                    )
                    if notequ == True:
                        diff_val = my_custom_lst[z].data - my_custom_lst[w].data
                        if diff_val < 0:
                            get_upper_1(my_custom_lst, z, w, my_tr)
                        elif diff_val > 0:
                            get_upper_2(my_custom_lst, z, w, my_tr)
                        my_condition = (not (
                            (check_equal(my_custom_lst[z], my_custom_lst[w])
                             )
                        )
                                        )
                        if my_condition == True:
                            while (not
                            (
                                    check_equal(my_custom_lst[z], my_custom_lst[w])
                            )      ) == True:
                                my_ini_mrk = my_custom_lst[z].identifier
                                my_subseq_mrk = my_custom_lst[w].identifier
                                my_custom_lst[z] = my_tr.parent(my_ini_mrk)
                                my_custom_lst[w] = my_tr.parent(my_subseq_mrk)
                            my_custom_location = my_custom_lst[z]
                        else:
                            return my_custom_lst[z]
                    my_custom_location = my_custom_lst[z]
                    perform_final_appending_process(already_trav, z, w)
            w = increment_repetition_number(w)
        z = increment_repetition_number(z)
    return my_custom_location

def prop_ksec(mycl, kval, domain_gen_hierarch):
    if mycl is None:
        print("none class parameter !")
        raise Exception("none class parameter !")
    if kval is None:
        print("none k value parameter !")
        raise Exception("none k value parameter  !")
    if domain_gen_hierarch is None:
        print("none parameter of domain generalization hierarchies !")
        raise Exception("none parameter of domain generalization hierarchies !")
    if kval <= 0:
        print("invalid value for the k anonymity parameter !!!!!!!!!!!!")
        raise Exception("*** !!!!!invalid k anonymity parameter found !!!!!!!****")
    all_of_the_elements = domain_gen_hierarch.items()
    my_dict_object = collections.Counter()
    keeper = list()
    all_of_the_keys = domain_gen_hierarch.keys()
    all_of_the_elements = domain_gen_hierarch.items()

    for element in mycl:
        for key in all_of_the_keys:
            second_tuple_element = element[key]
            tuple_structure = (key, second_tuple_element)
            my_dict_object[tuple_structure] = increment_repetition_number(my_dict_object[tuple_structure])
    for e in all_of_the_elements:
        e1, e2 = e
        for part in mycl:
            keeper.append(e2.get_node(part[e1]))
    rep_num = 1
    for x in all_of_the_elements:
        first, second = x
        minusOne = rep_num - 1
        low_bound_value = minusOne * kval
        up_bound_value = kval * rep_num
        sliced_part = keeper[low_bound_value:up_bound_value]
        if sliced_part is None:
            print("the sliced list part is None !!!!!!!!!!!!!")
            raise Exception("the sliced list part is None !!!!!!!!!!!!!")
        location = obtain_adj(sliced_part, second)
        for subpart in mycl:
            subpart[first] = location.tag
        rep_num = increment_repetition_number(rep_num)

    return mycl

def deletion_operation(f: collections.Counter, g, h):
    if f[(g, h)] == 0:
        del f[(g, h)]


def increment_repetition_number(rep):
    rep = rep + 1
    return rep


def obtain_summation_of_the_list(v):
    l = len(v)
    value_of_summation = 0
    for f in range(0, l):
        value_of_summation = value_of_summation + v[f]
    return value_of_summation


def my_custom_posit_func(h):
    if h <= 0:
        h = -1 * h
    else:
        h = +1 * h
    return h


def my_custom_file_reading_process(my_file, my_reading_char, my_backslash_tab_char, outcome):
    if my_file is None or my_reading_char is None or my_backslash_tab_char is None:
        print("Cannot read the given file !")
        print("reading char, file, or tab char is null!")
        raise Exception("Cannot read the given file. Reading char, file, or tab char is null!")

    if outcome is None:
        print("the parameter of the outcome is None !")
        raise Exception("the parameter of the outcome is None !")

    with open(my_file, my_reading_char) as my_custom_file:
        for my_custom_line in csv.reader(my_custom_file, delimiter=my_backslash_tab_char):
            outcome.append(my_custom_line)


def handleNullCases(my_custom_tree_data_structure, spacing, repetition, ancestor, initial, outcome):
    if (my_custom_tree_data_structure is None) or \
            (spacing is None) or \
            (repetition is None) or \
            (ancestor is None) or \
            (initial is None) or \
            (outcome is None):
        print("At least one of the parameters of this function is null !")
        raise Exception("At least one of the parameters of this function is null !")


def handle_impossible_cases(repetition):
    if repetition < 0:
        print("Repetition number cannot be smaller than 0 !")
        raise Exception("Negative repetition number !")


def my_custom_iterations(outcome, spacing, repetition, anc, initial, my_custom_tree_data_structure):
    if (outcome is None) or (spacing is None) or (repetition is None) or (anc is None) or (my_custom_tree_data_structure is None):
        print("At least one of the function parameters is None !!!!!!!!!!!!!!!!!!!")
        raise Exception("At least one of the function parameters is None !!!!!!!!!!!!!!!!!!!")

    blank = str()
    cond = anc.is_root()
    handleNullCases(
        my_custom_tree_data_structure,
        spacing,
        repetition,
        anc,
        initial,
        outcome
    )
    handle_impossible_cases(repetition)
    for element in outcome:
        if len(element) < 0:
            print("Invalid or impossible length value !")
        elif len(element) > 1:
            value = 1
            for sub in element:
                if sub == blank:
                    termination_condition = False
                else:
                    spacing = value
                    termination_condition = True
                if termination_condition:
                    break
                value = increment_repetition_number(value)

        if repetition < 0:
            print("invalid repetition number !")
        if repetition == 0:
            pass



        max_spac = spacing - 1


        if repetition > 0:
            my_condition_keeper = (anc.is_root())
            if my_condition_keeper == False:
                diff_value = spacing - anc.data
                if diff_value == 0 or diff_value < 0:
                    if anc.data > 0:
                        while (spacing - anc.data) == 0 or (spacing - anc.data) < 0:
                            anc = my_custom_tree_data_structure.parent(anc.identifier)
                        anc = my_custom_tree_data_structure.create_node(
                            outcome
                            [
                                repetition
                            ]
                            [
                                max_spac
                            ],
                            outcome
                            [
                                repetition
                            ]
                            [
                            max_spac
                            ],
                            anc,
                            spacing
                        )
                else:
                    anc = my_custom_tree_data_structure.create_node(
                        outcome
                        [
                            repetition
                        ]
                        [
                            max_spac
                        ],
                        outcome
                        [
                            repetition
                        ]
                        [
                            max_spac
                        ],
                        anc,
                        spacing
                    )
            else:
                value_of_one = 1
                anc = my_custom_tree_data_structure.create_node(
                    outcome[repetition][value_of_one],
                    outcome[repetition][value_of_one],
                    initial,
                    spacing
                )
        repetition = increment_repetition_number(repetition)


def decrement_reps(a):
    a = a - 1
    return a


def perform_appending_process(val1, val2, val3):
    for y, h in val1.items():
        val2.append(y)
        val3.append(h)


def init_md(rd_cnt, ad_cnt, fir, sec):
    rd_cnt = collections.Counter()
    ad_cnt = collections.Counter()
    for fir_elem in fir:
        set_all = fir_elem.items()
        for y, r in set_all:
            tup = (y, r)
            rd_cnt[tup] = increment_repetition_number(rd_cnt[tup])

    for sec_elem in sec:
        set_all_2 = sec_elem.items()
        for t, u in set_all_2:
            tup2 = (t, u)
            ad_cnt[tup2] = increment_repetition_number(
                ad_cnt[tup2])


def obtain_dict_differences(d1c, d2c):
    d2c.subtract(d1c)


def initialize_dicts(a, b):
    a = collections.Counter()
    b = collections.Counter()


def find_difference(x, y):
    x.subtract(y)


def fill_dicts(c, d, e, f):
    for first_elem in c:
        for a in first_elem.items():
            d[a] = increment_repetition_number(d[a])
    for second_elem in e:
        for b in second_elem.items():
            f[b] = increment_repetition_number(f[b])


def check_whether_lengths_are_zero(r, t):
    return len(r) == 0 and len(t) == 0




def perform_final_appending_process(my_lst: list, a: int, b: int):
    tuple_var_1 = (a, b)
    my_lst.append(tuple_var_1)
    tuple_var_2 = (b, a)
    my_lst.append(tuple_var_2)


def get_upper_1(lst: list, a: int, b: int, tre):
    while lst[a].data != lst[b].data:
        sec_mark = lst[b].identifier
        lst[b] = tre.parent(sec_mark)

def check_equal(x, y):
    return x == y

def get_upper_2(lst: list, a: int, b: int, tre):
    while lst[a].data != lst[b].data:
        sec_mark = lst[a].identifier
        lst[a] = tre.parent(sec_mark)


#########MY HELPER FUNCTIONS###########################################################################################################################

def read_dataset(dataset_file: str):
    """ Read a dataset into a list and return.

    Args:
        dataset_file (str): path to the dataset file.

    Returns:
        list[dict]: a list of dataset rows.
    """
    result = []
    with open(dataset_file) as f:
        records = csv.DictReader(f)
        for row in records:
            result.append(row)
    return result


def write_dataset(dataset, dataset_file: str) -> bool:
    """ Writes a dataset to a csv file.

    Args:
        dataset: the data in list[dict] format
        dataset_file: str, the path to the csv file

    Returns:
        bool: True if succeeds.
    """
    assert len(dataset) > 0, "The anonymized dataset is empty."
    keys = dataset[0].keys()
    with open(dataset_file, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(dataset)
    return True

def display_tree_structure(x):
    x.show()

def read_DGH(DGH_file: str):
    """ Reads one DGH file and returns in desired format.

    Args:
        DGH_file (str): the path to DGH file.
    """
    # TODO: complete this code so that a DGH file is read and returned
    # in your own desired format.
    outcome = []
    my_custom_file_reading_process(DGH_file, "r", "\t", outcome)
    my_custom_tree_data_structure = treelib.Tree()
    first_parameter = outcome[0][0].strip()
    second_parameter = 'Any'
    third_parameter = 1
    ancestor = my_custom_tree_data_structure.create_node(first_parameter, second_parameter, data=third_parameter)
    initial = ancestor
    repetition = 0
    spacing = 0
    my_custom_iterations(outcome, spacing, repetition, ancestor, initial, my_custom_tree_data_structure)
    outcome = my_custom_tree_data_structure
    return outcome



def read_DGHs(DGH_folder: str) -> dict:
    """ Read all DGH files from a directory and put them into a dictionary.

    Args:
        DGH_folder (str): the path to the directory containing DGH files.

    Returns:
        dict: a dictionary where each key is attribute name and values
            are DGHs in your desired format.
    """
    DGHs = {}
    for DGH_file in glob.glob(DGH_folder + "/*.txt"):
        attribute_name = os.path.basename(DGH_file)[:-4]
        DGHs[attribute_name] = read_DGH(DGH_file)

    return DGHs


##############################################################################
# Mandatory Functions                                                        #
# You need to complete these functions without changing their parameters.    #
##############################################################################


######################PART1#####################################################################################################################################
def cost_MD(raw_dataset_file: str, anonymized_dataset_file: str,
            DGH_folder: str) -> float:
    global DGHs
    """Calculate Distortion Metric (MD) cost between two datasets.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        anonymized_dataset_file (str): the path to the anonymized dataset file.
        DGH_folder (str): the path to the DGH directory.

    Returns:
        float: the calculated cost.
    """
    raw_dataset = read_dataset(raw_dataset_file)
    anonymized_dataset = read_dataset(anonymized_dataset_file)

    empty_len = 0
    condition = (len(DGHs) == empty_len)

    if condition == True:
        assert (len(raw_dataset) > 0 and len(raw_dataset) == len(anonymized_dataset)
                and len(raw_dataset[0]) == len(anonymized_dataset[0]))
        DGHs = read_DGHs(DGH_folder)

    # TODO: complete this function.

    return evaluate_value_of_distortion_metric(
        raw_dataset,
        anonymized_dataset,
        domain_generalization_hierarchy_list,
        tree_struct_lst,
        DGHs
    )


def perform_deletion(a, b, c: collections.Counter()):
    if c[(a, b)] == 0:
        del c[(a, b)]


def give_penalty_or_reward(d, e, f):
    if d[(e, f)] <= 0:
        d[(e, f)] = increment_repetition_number(
            d[(e, f)])
    else:
        d[(e, f)] = decrement_reps(d[(e, f)])


def add_cost_to_total_cost(o, g):
    o = o + g
    return o


def has_elements(wu):
    return len(wu) > 0


def my_custom_function(g, h, u):
    if g >= 0:
        return u - h
    return h - u


def take_element_out(cn: collections.Counter):
    cn.popitem()


def obtain_diff(x, y):
    x.subtract(y)


def has_more_elements(my_list):
    return len(my_list) > 0


def appending_operation(w, q, u):
    for m, o in u.items():
        w.append(m)
        q.append(o)


def provide_penalty_or_reward(dataset, u):
    if dataset[u] <= 0:
        dataset[u] = increment_repetition_number(
            dataset[u])
    else:
        dataset[u] = decrement_reps(
            dataset[u])


def append_tags(d: treelib.Tree, ls, df):
    for q in d.expand_tree(mode=df):
        tag_value = d[q].tag
        ls.append(tag_value)


#############################MD COST CALCULATION FUNCTION ####################################################
def evaluate_value_of_distortion_metric(first_equivalence_class, second_equivalence_class, ldgh, ltr, all_dghs):
    dpth_md = treelib.Tree.DEPTH
    if (first_equivalence_class is None) or (second_equivalence_class is None):
        print("The first equivalence class or the second equivalence class is None !")
        raise Exception("The first equivalence class or the second equivalence class is None !")
    if (ldgh is None) or (ltr is None) or (all_dghs is None):
        print("One of the List of domain generalization hierarchies, all dghs or list of tree is Null !!!")
        raise Exception("One of the List of domain generalization hierarchies, all dghs or list of tree is Null !!!")
    my_custom_raw_ds_counter = collections.Counter()
    terminate_bool = False
    my_custom_anonymized_ds_counter = collections.Counter()
    fill_dicts(first_equivalence_class, my_custom_raw_ds_counter, second_equivalence_class,
               my_custom_anonymized_ds_counter)
    my_custom_anonymized_ds_counter.subtract(my_custom_raw_ds_counter)
    if my_custom_raw_ds_counter is None:
        print("Null raw dataset dictionary")
        raise Exception("Null raw dataset dictionary")
    if my_custom_anonymized_ds_counter is None:
        print("Null anonymized dataset dictionary")
        raise Exception("Null anonymized dataset dictionary")
    if check_whether_lengths_are_zero(ldgh, ltr) == True:
        appending_operation(ldgh, ltr, all_dghs)
    my_custom_tree = treelib.Tree()
    if my_custom_tree is None:
        print("Null tree instance is detected !")
        raise Exception("Null tree !!!")
    summation_val = 0
    while has_elements(my_custom_anonymized_ds_counter):
        (a, w), t = my_custom_anonymized_ds_counter.popitem()
        pstv_t = my_custom_posit_func(t)
        if t > 0 or t < 0:
            if a in DGHs:
                my_custom_tree = ltr[ldgh.index(a)]
                first_value = my_custom_tree.get_node(w).data
                lst = []
                append_tags(my_custom_tree, lst, dpth_md)
                if lst is None:
                    print("list called lst is Null !")
                    raise Exception("List called lst is Null !")
                for v in range(0, pstv_t):
                    for ls_elm in lst:
                        tup = (a, ls_elm)
                        if tup in my_custom_anonymized_ds_counter:
                            second_value = my_custom_tree.get_node(ls_elm).data
                            if t >= 0:
                                diff_value = second_value - first_value
                                my_ini_arr = []
                                my_ini_arr.append(1)
                                my_ini_arr.append(diff_value)
                                individ = math.prod(my_ini_arr)
                            else:
                                df_value = second_value - first_value
                                my_sec_arr = []
                                my_sec_arr.append(-1)
                                my_sec_arr.append(df_value)
                                individ = math.prod(my_sec_arr)
                            provide_penalty_or_reward(my_custom_anonymized_ds_counter, tup)
                            deletion_operation(my_custom_anonymized_ds_counter, a, ls_elm)
                            summation_val = summation_val + individ
                            terminate_bool = True
                            if terminate_bool:
                                break

    if summation_val < 0 or summation_val == 0:
        array_ret = []
        array_ret.append(-1)
        array_ret.append(summation_val)
        return math.prod(array_ret)
    second_array_ret = []
    second_array_ret.append(+1)
    second_array_ret.append(summation_val)
    return math.prod(second_array_ret)

#############################MD COST CALCULATION FUNCTION ####################################################


################################################LM#######################################
def perform_appending_for_lm(w, e, y):
    for d, t in y:
        w.append(d)
        e.append(t)

def cost_LM(raw_dataset_file: str, anonymized_dataset_file: str,
            DGH_folder: str) -> float:
    """Calculate Loss Metric (LM) cost between two datasets.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        anonymized_dataset_file (str): the path to the anonymized dataset file.
        DGH_folder (str): the path to the DGH directory.

    Returns:
        float: the calculated cost.
    """
    raw_dataset = read_dataset(raw_dataset_file)
    anonymized_dataset = read_dataset(anonymized_dataset_file)


    assert (len(raw_dataset) > 0 and len(raw_dataset) == len(anonymized_dataset)
        and len(raw_dataset[0]) == len(anonymized_dataset[0]))
    DGHs = read_DGHs(DGH_folder)

    elements = DGHs.items()


    if check_whether_lengths_are_zero(tree_struct_lst, domain_generalization_hierarchy_list) == True:
        perform_appending_for_lm(domain_generalization_hierarchy_list, tree_struct_lst, elements)
    difference_value = evaluate_total_value_of_the_loss_metric_cost(
        raw_dataset, tree_struct_lst,
        domain_generalization_hierarchy_list) - evaluate_total_value_of_the_loss_metric_cost(anonymized_dataset,
                                                                                             tree_struct_lst,
                                                                                              domain_generalization_hierarchy_list)


    if len(raw_dataset_file) <= 0 \
            or \
            len(anonymized_dataset_file) <= 0 \
            or \
            len(DGH_folder) <= 0 \
            or \
            raw_dataset_file is None \
            or \
            anonymized_dataset_file is None \
            or DGH_folder is None \
            or \
            raw_dataset is None \
            or \
            anonymized_dataset is None \
            or \
            DGHs is None \
            or \
            elements is None:
        print("invalid parameter of cost lm function is detected !")
        raise Exception("invalid parameter of cost lm function is detected !")



    if len(anonymized_dataset_file) <= 0:
        print("invalid length value for the anonymized dataset file is found !")
        raise Exception("invalid length value for the anonymized dataset file is found !")

    if len(DGH_folder) <= 0:
        print("invalid length value for the DGH folder is found !")
        raise Exception("invalid length value for the DGH folder is found !")





    import math
    if difference_value < 0 or difference_value == 0:
        arr1 = []
        arr1.append(-1)
        arr1.append(difference_value)
        difference_value = math.prod(
            arr1)
    else:
        arr2 = []
        arr2.append(+1)
        arr2.append(difference_value)
        difference_value = math.prod(arr2)
    return difference_value


#################################LM COST CALCULATION FUNCTION################################################################################
def evaluate_total_value_of_the_loss_metric_cost(my_custom_ds, lst, lsd):
    if my_custom_ds is None or lst is None or lsd is None:
        print("invalid parameter of the loss metric evaluation function is detected !")
        raise Exception("invalid parameter of the loss metric evaluation function is detected !")
    cost_keeper = list()
    for element in my_custom_ds:
        record_level_cost = 0
        my_custom_set = element.items()
        for a in my_custom_set:
            b, c = a
            if b not in lsd:
                pass
            else:
                any_string = 'Any'
                my_custom_tree_datastructure = lst[lsd.index(b)]
                my_custom_nodal_location = my_custom_tree_datastructure.get_node(c)
                below = len(
                    my_custom_tree_datastructure.subtree(my_custom_nodal_location.identifier).leaves(
                    my_custom_nodal_location.identifier)
                )
                belowMinOne = below - 1
                tot = len(
                    my_custom_tree_datastructure.leaves(any_string)
                )
                totMinOne = tot - 1
                cellular = belowMinOne / totMinOne
                record_level_cost = record_level_cost + ( float(cellular) / len(lsd) )
        cost_keeper.append(record_level_cost)
    return obtain_summation_of_the_list(cost_keeper)



#################################LM COST CALCULATION FUNCTION################################################################################

################################################LM#######################################


######################PART1##########################################################################################################################################################################





##################################PART2######################################################################################################################

def perform_randomization_algo(all_domain_gen_hie, k_value: int, rds):
    if all_domain_gen_hie is None or k_value is None or rds is None:
        print("at least one of the parameters of randomization function is none !")
        raise Exception("at least one of the parameters of randomization function is none !")

    if k_value == 0 or k_value < 0:
        print("******invalid value of the k parameter is detected right now****")
        raise Exception("******invalid value of the k parameter is detected right now****")

    import random
    all_clusters = dict(list)
    from random import randrange
    if all_domain_gen_hie is None:
        print("null domain generalization hierarchies !")
        raise Exception("null domain generalization hierarchies !")
    if k_value <= 0 or k_value is None:
        print("invalid k anonymity parameter !!!!!!!!!!!!!")
        raise Exception("invalid k anonymity parameter !")
    length_value_of_original_dataset = len(rds)
    group_amount = length_value_of_original_dataset / k_value
    converted_group_amount = int(group_amount)

    value_of_left = length_value_of_original_dataset % k_value
    converted_value_of_left = int(value_of_left)


    my_array_structure = [-1 , converted_value_of_left]
    from math import prod
    element_to_be_subtracted = prod(my_array_structure)
    converted_version_of_prod = int(element_to_be_subtracted)


    custom_diff_val = (-1 * converted_value_of_left) + length_value_of_original_dataset
    converted_lrg = int(custom_diff_val)
    starting_pos = 0
    my_amount_keeper_var = 0
    ending_pos = converted_group_amount
    ending = False

    for divide_and_conqer_elem in rds:
        from random import randrange
        outcome_of_rand_range = randrange(starting_pos, converted_group_amount)
        my_custom_group_elem = all_clusters.get(outcome_of_rand_range)
        len_of_my_custom_group_elem = len(my_custom_group_elem)
        if len_of_my_custom_group_elem > 0:
            my_extracted_element = all_clusters.get(outcome_of_rand_range)
            len_of_my_extracted_element = len(my_extracted_element)
            subtracted_part = int(math.prod([-1, k_value]))
            my_difference_value = (-1 * k_value) + len_of_my_extracted_element
            my_custom_cnd = (my_difference_value == 0)
            while my_custom_cnd == True:
                dff_val = my_amount_keeper_var - converted_lrg
                condition_bool = (dff_val == 0)
                if condition_bool == True:
                    ending = True
                if ending == True:
                    break

                from random import randrange
                outcome_of_rand_range = randrange(starting_pos, converted_group_amount)


                extracted_element = all_clusters.get(outcome_of_rand_range)
                length_of_extracted_element = len(extracted_element)
                if length_of_extracted_element == 0 or length_of_extracted_element < 0:
                    all_clusters[outcome_of_rand_range] = list()


            part = all_clusters.get(outcome_of_rand_range)
            part.append(divide_and_conqer_elem)
            if part is None or len(part) < 0 or len(part) == 0:
                print("the cluster part has no elements or invalid length !")
                raise Exception("the cluster part has no elements or invalid length !")
        else:
            random_outcome = all_clusters[outcome_of_rand_range]
            random_outcome.append(divide_and_conqer_elem)
        my_amount_keeper_var = increment_repetition_number(my_amount_keeper_var)


        all_set_of_keys = all_domain_gen_hie.keys()
        my_custom_keeper = list()
        variable_of_iteration = 0
        if all_set_of_keys is None:
            print("none set of keys !")
            raise Exception("none set of keys !")


        for domain_generalization_hier in all_set_of_keys:
            my_custom_keeper[variable_of_iteration] = domain_generalization_hier
            variable_of_iteration = increment_repetition_number(variable_of_iteration)
        return all_clusters


def random_anonymizer(raw_dataset_file: str, DGH_folder: str, k: int,
                      output_file: str, s: int):
    """ K-anonymization a dataset, given a set of DGHs and a k-anonymity param.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        DGH_folder (str): the path to the DGH directory.
        k (int): k-anonymity parameter.
        output_file (str): the path to the output dataset file.
        s (int): seed of the randomization function
    """
    raw_dataset = read_dataset(raw_dataset_file)
    DGHs = read_DGHs(DGH_folder)

    # print("hello111")

    for i in range(len(raw_dataset)):  ##set indexing to not lose original places of records
        raw_dataset[i]['index'] = i

    raw_dataset = np.array(raw_dataset)
    np.random.seed(s)  ## to ensure consistency between runs
    np.random.shuffle(raw_dataset)  ##shuffle the dataset to randomize

    clusters = []

    D = len(raw_dataset)

    # TODO: START WRITING YOUR CODE HERE. Do not modify code in this function above this line.



    ##################for deadling with invalid situations ##########
    if k == 0 or k < 0:
        print("invalid kanonymity parameter value found !")
        raise Exception("invalid kanonymity parameter value found !")
    if D == 0 or D < 0:
        print("invalid dataset length value found !")
        raise Exception("invalid dataset length value found !")
    ##################for deadling with invalid situations ##########


    division = D / k #obtaining cluster number
    amount_of_equivalence_classes = int(division)
    total_repetition_counter = 1

    left_value = D % k
    #obtaining the remainder of d with k in case if d is not a multiple
    # of k and we have to deal with the size of the last equivalence class
    converted_left_value = int(left_value)

    if converted_left_value > k or converted_left_value < 0:
        print("invalid remainder value detected !!!!!")
        raise Exception("invalid remainder value detected !!!!!")

    cluster_keeper = perform_randomization_algo(DGHs, k, raw_dataset)
    set_of_all_vals = cluster_keeper.values()
    if set_of_all_vals is None:
        print("null return value set found !")
        raise Exception("null return value set found !")


    q = k



    for my_custom_group in set_of_all_vals:
        element_num_in_cluster = len(my_custom_group)
        subtracted_el = (-1 * q)
        diff = (-1 * q) + (element_num_in_cluster)
        condition = (diff == 0)
        holding = (condition == True)
        if holding:
            print("multiple of k")
            k = q
        else:
            print("not multiple of k")
            remainderValueAdded = k + converted_left_value
            k = remainderValueAdded
        my_custom_group = prop_ksec(my_custom_group, k, DGHs)
        if my_custom_group is None:
            print("return value of the k security function is found as null !!!!!!")
            raise Exception("return value of the k security function is found as null !!!!!!")
        clusters.append(my_custom_group) #appending to the clusters list
        if clusters is None:
            print("clusters list is none !")
        if len(clusters) < 0 or len(clusters) == 0:
            print("invalid length value for clusters list or there is no element inside the clusters list!")
        for component in my_custom_group:
            my_custom_group.append(component)
        total_repetition_counter = increment_repetition_number(total_repetition_counter) #incrementing the repetition number variable


    #############dealing with null case, or 0-length case , or invalid length case##########
    if clusters is None or len(clusters) == 0 or len(clusters) < 0:
        print("invalid cluster keeper !")
        raise Exception("invalid cluster keeper !")
    #############dealing with null case, or 0-length case , or invalid length case##########

    # Store your results in the list named "clusters".
    # Order of the clusters is important. First cluster should be the first EC, second cluster second EC, ...

    # END OF STUDENT'S CODE. Do not modify code in this function below this line.



    anonymized_dataset = [None] * D

    for cluster in clusters:  # restructure according to previous indexes
        for item in cluster:
            anonymized_dataset[item['index']] = item
            del item['index']

    write_dataset(anonymized_dataset, output_file)


##################################PART2###########################################################################

#################################PART3#####################################################################################
def obtain_summation_of_odd_elems(lst):
    sm = 0
    for w in range(0, len(lst)):
        if w % 2 == 1:
            sm = sm + lst[w]
    return sm

def obt_lst_min(my_lst):
    fir = my_lst[0]
    for h in range(0, len(my_lst)):
        if my_lst[h] <= fir:
            fir = my_lst[h]
    return fir

def obtain_summation_of_even_elems(lst):
    sm = 0
    for w in range(0, len(lst)):
        if w % 2 == 0:
            sm = sm + lst[w]
    return sm

def obt_lst_max(my_lst):
    fir = my_lst[0]
    for h in range(0, len(my_lst)):
        if my_lst[h] >= fir:
            fir = my_lst[h]
    return fir


def extract_s(rds, lst1: list, lst2: list, dghf: str, kval: int):
    if(rds is None):
        print("the original dataset parameter is none !")
        raise Exception("the original dataset parameter is none !")
    if len(lst1) <= 0 or len(lst2) <= 0 or len(dghf) <= 0:
        print("lst1, lst2, or dghf has an invalid or no length !")
        raise Exception("lst1, lst2, or dghf has an invalid or no length !")
    if (lst1 is None) or (lst2 is None) or (dghf is None):
        print("one of the lst1, lst2, or dghf parameters is none !")
        raise Exception("one of the lst1, lst2, or dghf parameters is none !")

    if k < 0 or k == 0:
        print("invalid k anonymity parameter !")
        raise Exception("invalid k anonymity parameter !")


    my_custom_loc_ls = list()
    iteration_number = 0
    equivalence_class_dictionary_structure = dict()

    for element in lst1:
        if element > 0 or element < 0:
            pass
        else:
            my_custom_loc_ls.append(iteration_number)
        iteration_number = increment_repetition_number(iteration_number)



    my_var = 0
    crml = lst1[:]
    for location in my_custom_loc_ls:
        equivalence_class_dictionary_structure[location] = lst2[:]
        equivalence_class_dictionary_structure[location].append(rds[location])
        my_var = increment_repetition_number(my_var)
    distance_keeper = []
    subsequent_location = next(iter(equivalence_class_dictionary_structure))
    lst_version = list(equivalence_class_dictionary_structure)
    last_location = len(lst_version) - 1
    last_element = lst_version[last_location]
    all_dictionary = equivalence_class_dictionary_structure.items()
    if all_dictionary is None:
        print("dictionary is none !")
        print("dictionary is none !")


    for j in all_dictionary:
        loc, initial_equiv_cla = j
        if not subsequent_location.__eq__(last_element):
            pass
        else:
            break
        subsequent_location = next(
            iter(equivalence_class_dictionary_structure)
        )




        second_equiv_cla = equivalence_class_dictionary_structure[subsequent_location]









    my_tuple = (
        loc,
         evaluate_total_value_of_the_loss_metric_cost(
           rds,
           lst=tree_struct_lst,
           lsd=domain_generalization_hierarchy_list
         )

   )

    distance_keeper.append(my_tuple)
    if len(distance_keeper) > 0:
        sorted_vers = sorted(
            distance_keeper,
            key=operator.itemgetter(1)
        )
        first_of_sorted = sorted_vers[0]
        smallest = first_of_sorted
    elif len(distance_keeper) < 0:
        print("invalid length !")
    elif len(distance_keeper) == 0:
        smallest = (loc, 0)

    lst2 = []
    first_element_of_smallest = smallest[0]
    first_part = equivalence_class_dictionary_structure[first_element_of_smallest]
    for my_element in first_part:
        lst2.append(my_element)
    crml[first_element_of_smallest] = 1
    lst1 = crml

    my_return_tuple = (lst2, lst1)
    return my_return_tuple


def clustering_anonymizer(raw_dataset_file: str, DGH_folder: str, k: int,
                          output_file: str):
    """ Clustering-based anonymization a dataset, given a set of DGHs.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        DGH_folder (str): the path to the DGH directory.
        k (int): k-anonymity parameter.
        output_file (str): the path to the output dataset file.
    """
    raw_dataset = read_dataset(raw_dataset_file)
    DGHs = read_DGHs(DGH_folder)

    # TODO: complete this function.

    upper_bound = k - 1

    element_amount_keeper = 0
    one_value = 1
    my_number_of_reps = 0
    bnd = k - 1
    lower = 0


    ########dealing with none cases and invalid cases
    if upper_bound <= 0 or upper_bound is None:
        print("invalid upper bound !")
        raise Exception("invalid upper bound found !")
    if raw_dataset is None:
        print("the raw dataset parameter is none !!!!!!!")
        raise Exception("the raw dataset parameter is none !!!!!!!")
    if raw_dataset_file is None:
        print("the raw dataset file parameter is none !!!!!!!")
        raise Exception("the raw dataset file parameter is none !!!!!!!")
    if DGH_folder is None:
        print("the folder containing domain generalization hierarchies is none !!!")
        raise Exception("the folder containing domain generalization hierarchies is none !!!")
    if k is None:
        print("the k parameter is none !!!!!")
        raise Exception("the k parameter is none !!!!!")
    if output_file is None:
        print("the output file parameter is none !!!!")
        raise Exception("the output file parameter is none !!!!")
    if DGHs is None:
        print("the parameter of domain generalization hierarchies is none !!")
        raise Exception("the parameter of domain generalization hierarchies is none !!")

    ###for handling none cases of the parameters ########################
    if raw_dataset is None \
            or raw_dataset_file is None \
            or DGH_folder is None \
            or k is None \
            or output_file is None \
            or raw_dataset is None \
            or DGHs is None:
        print("At least one of the function parameters is detected as none !!!!!!!!!!!!!!!!!!!!")
        raise Exception("At least one of the function parameters is detected as none !!!!!!!!!!!!!!!!!!!!")

    if len(raw_dataset_file) <= 0 or len(DGH_folder) <= 0 or len(output_file) <= 0:
        print("the length value for at least one of the string type parameters is not valid")
        raise Exception("the length value for at least one of the string type parameters is not valid")
    ###for handling none cases of the parameters ########################


    if k <= 0:
        print("the value of the k anonymity parameter is not valid !")
        raise Exception("the value of the k anonymity parameter is not valid !")

    anonymized_dataset = []
    len_val = len(raw_dataset)
    my_rem = len_val % k
    if my_rem >= k:
        print("invalid value of the remainder")
        raise Exception("invalid value of the remainder !!!!")
    elif my_rem < 0:
        print("invalid value of the remainder")
        raise Exception("invalid value of the remainder !!!!")

    int_version = int(my_rem)




    unit_zero_array = []
    unit_zero_array.append(0)
    signature = unit_zero_array * len_val
    if(signature.count(0)-k) < 0:
        print("amount of zeros inside the signature list is less than k!")


    zero_amount = signature.count(0)
    diff = zero_amount - k
    while ((signature.count(0) - k)) == 0 or ((signature.count(0) - k)) > 0:
        signature_length = len(signature)
        if signature_length < 0:
            print("invalid length value detected for the signature list !!!!!")
        elif signature_length == 0:
            print("no element detected inside the signature list !!!!")
        location = signature.index(0, 0, signature_length)
        if location < 0:
            print("0 not found inside signature list !!!!!")
        signature[location] = one_value
        groupk = [raw_dataset[location]]
        element_amount_keeper = increment_repetition_number(element_amount_keeper)
        w = 0
        while (w-bnd) < 0:
            groupk, signature = extract_s(
                raw_dataset,
                signature,
                groupk,
                DGH_folder,
                k
            )
            w = increment_repetition_number(w)



        groupk = prop_ksec(
            groupk,
            k,
            DGHs
        )
        if groupk is None:
            print("the group is detected as None !!!!!")
            raise Exception("the group is detected as None !!!!!")
        for component in groupk:
            anonymized_dataset.append(component)
        my_number_of_reps = increment_repetition_number(my_number_of_reps)


    zero_amount = signature.count(0)
    if zero_amount < 0:
        print("invalid zero amount !")
        raise Exception("invalid zero amount !")
    while not (zero_amount < 0 or zero_amount == 0): #if the zero amount is not 0 or negative
            signature_array_length = len(signature)
            if signature_array_length < 0:
                print("invalid length value detected for the signature list !!!!!")
            elif signature_array_length == 0:
                print("no element detected inside the signature list !!!!")
            starting_point = 0
            index_to_be_found = 0
            ending_point = signature_array_length
            location = signature.index(
                index_to_be_found,
                starting_point,
                ending_point
            )
            signature[location] = one_value #change the location value of 0 with 1
            groupk.append(raw_dataset[location]) #append the data of original dataset at the found location to the groupk
            element_amount_keeper = increment_repetition_number(element_amount_keeper) #increment the iteration numbers


    summation_array = []
    summation_array.append(k)
    summation_array.append(int_version)
    added = sum(summation_array)
    added = int(added)
    my_array = []
    my_array.append(k)
    my_array.append(-1)
    from math import prod
    product_value = prod(my_array)
    upper = int(product_value)
    sliced = anonymized_dataset[lower:upper]
    groupk = prop_ksec(
        groupk,
        added,
        DGHs
    )
    for element in groupk:
        sliced.append(element)

    anonymized_dataset = sliced[:]

    write_dataset(anonymized_dataset, output_file)


#################################PART3#####################################################################################







#######################################PART4#################################################################################################################
def bottomup_anonymizer(raw_dataset_file: str, DGH_folder: str, k: int,
                        output_file: str):
    """ Bottom up-based anonymization a dataset, given a set of DGHs.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        DGH_folder (str): the path to the DGH directory.
        k (int): k-anonymity parameter.
        output_file (str): the path to the output dataset file.
    """

    raw_dataset = read_dataset(raw_dataset_file)
    DGHs = read_DGHs(DGH_folder)

    # TODO: complete this function.
    my_custom_struc = dict()
    value_of_element_amount = len(raw_dataset) #obtaining the length of the raw dataset
    my_string_of_root = "root" #root string
    my_any_string = "Any" #any string
    my_first_blank_string = str()
    my_second_blank_string = ""
    my_tree_data_struct = treelib.Tree() #creating the tree data structure by using treelib module of python


    ############################Process of assignment and filling with any #################################
    for domain_generalization in DGHs: #going along all domain generalization hierarchies
        my_custom_struc[domain_generalization] = my_any_string
    ############################Process of assignment and filling with any #################################

    all_of_leaves = my_tree_data_struct.leaves() ###obtaining the leaves of the tree

    noParent = None #for specifying no parent situation

    ###########creation of the node at the root ###################################################################
    my_tree_data_struct.create_node(tag=my_custom_struc, identifier=my_root_string, parent=noParent, data=value_of_element_amount)
    ###########creation of the node at the root ###########################################################################



    my_ini_location = my_tree_data_struct.get_node(my_string_of_root) #obtaining the location of root by using get_node function of treelib module.


    # Finally, write dataset to a file
    write_dataset(anonymized_dataset, output_file)


#######################################PART4######################################################################################################################################################################


########################################TESTING PART WHICH IS ALREADY PROVIDED IN THE SKELETON.PY##########################################################################################

# Command line argument handling and calling of respective anonymizer:
if len(sys.argv) < 6:
    print(f"Usage: python3 {sys.argv[0]} algorithm DGH-folder raw-dataset.csv anonymized.csv k seed(for random only)")
    print(f"\tWhere algorithm is one of [clustering, random, bottomup]")
    sys.exit(1)

algorithm = sys.argv[1]
if algorithm not in ['clustering', 'random', 'bottomup']:
    print("Invalid algorithm.")
    sys.exit(2)

start_time = datetime.datetime.now()  ##
print(start_time)  ##

dgh_path = sys.argv[2]
raw_file = sys.argv[3]
anonymized_file = sys.argv[4]
k = int(sys.argv[5])

function = eval(f"{algorithm}_anonymizer");
if function == random_anonymizer:
    if len(sys.argv) < 7:
        print(
            f"Usage: python3 {sys.argv[0]} algorithm DGH-folder raw-dataset.csv anonymized.csv k seed(for random only)")
        print(f"\tWhere algorithm is one of [clustering, random, bottomup]")
        sys.exit(1)

    seed = int(sys.argv[6])
    function(raw_file, dgh_path, k, anonymized_file, seed)
else:
    function(raw_file, dgh_path, k, anonymized_file)

cost_md = cost_MD(raw_file, anonymized_file, dgh_path)
cost_lm = cost_LM(raw_file, anonymized_file, dgh_path)
print(f"Results of {k}-anonimity:\n\tCost_MD: {cost_md}\n\tCost_LM: {cost_lm}\n")

end_time = datetime.datetime.now()  ##
print("End time is " + str(end_time))  ##
print("Duration is " + str(end_time - start_time))  ##

########################################TESTING PART WHICH IS ALREADY PROVIDED IN THE SKELETON.PY##########################################################################################


# Sample usage:
# python3 skeleton.py clustering DGHs adult-hw1.csv ffeq.csv 2 2