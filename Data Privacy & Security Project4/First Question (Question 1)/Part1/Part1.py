# necessary imports
import csv  # for reading txt files
import pandas as pnd  # for reading txt files
from io import open
import hashlib  # for hashing


def dict_att_with_no_salt_option_and_no_key_streching():
    # Reading the txt file which contains all the provided passwords
    all_poss_passwd_choices = pnd.read_table("rockyou.txt", header=None, names=["Provided Password Tokens"])
    feasible_pswd_options = list(all_poss_passwd_choices["Provided Password Tokens"])  # convert pandas series to a list
    stripped_feasible_pswd_options = list()  # creation of a list which will keep the stripped passwords
    for feasible_pss in feasible_pswd_options:  # going along all possible password options
        feasible_pss = feasible_pss.strip()  # removing all whitespaces at the end and at the beginning of each password
        stripped_feasible_pswd_options.append(feasible_pss)  # append this stripped password to the list keeping that

    # initialization of an empty list which will store the hashes of the password options
    all_hash_versions_of_the_existing_pswd = list()
    for single_pwd in stripped_feasible_pswd_options:
        string_of_utf_8_encoding = 'utf-8'  # specification of encoding style
        encoded_version_of_pass = single_pwd.encode(string_of_utf_8_encoding)  # encoding passwords to utf8
        hashing_process_sha512 = hashlib.sha512(encoded_version_of_pass)  # apply sha512 hashing
        digest_str_with_hexadecimal_digits = hashing_process_sha512.hexdigest()  # convert the digest string to the hexadecimal digits
        all_hash_versions_of_the_existing_pswd.append(
            digest_str_with_hexadecimal_digits)  # append the hash made of hexadecimals

    mode_of_writing_to_csv = 'w+'  # for specifying the writing mode
    # open the attack table csv file with the write mode
    # write the header part to this attack table csv file
    # write the rows which contain the password and the corresponding hash of this password to this attack table csv file
    with open("attack_table.csv", mode_of_writing_to_csv) as attack_table_csv_file:
        obj_wrt_csv = csv.writer(attack_table_csv_file)
        lst_for_row_of_column_headers = list()  # creation of an empty list which will keep the header row
        lst_for_row_of_column_headers.append("Password String Token")  # constructing the header row
        lst_for_row_of_column_headers.append("Corresponding Hash of Password")  # constructing the header row
        obj_wrt_csv.writerow(lst_for_row_of_column_headers)  # constructing the header row
        version_zipped = zip(stripped_feasible_pswd_options, all_hash_versions_of_the_existing_pswd)
        for custom_pwd_hash_tuple_structure in version_zipped:
            obj_wrt_csv.writerow(custom_pwd_hash_tuple_structure)  # write the remaining rows of csv file

    dictionary_for_conversion_of_hash_to_its_corresponding_pass = dict()  # creation of an empty dict which will keep the hash and its corresponding pass
    zipped_hashes_and_passwords = zip(all_hash_versions_of_the_existing_pswd, stripped_feasible_pswd_options)
    for hash_and_password_tuple_structure in zipped_hashes_and_passwords:
        my_custom_hs, pwd_str = hash_and_password_tuple_structure
        dictionary_for_conversion_of_hash_to_its_corresponding_pass[
            my_custom_hs
        ] = pwd_str  # storing hashes as keys and passwords as values

    # reading the digital corp txt file containing the tuples each of which contains a username and a hash of the password of that username.
    with open("digitalcorp.txt") as digital_corp_file_txt:
        seperator_ch = ','
        digital_corp_file_rdr = csv.reader(digital_corp_file_txt, delimiter=seperator_ch)
        next(digital_corp_file_rdr)

        for username_and_hash_tuple_structure in digital_corp_file_rdr:  # going along all username and hash tuples in digital corp txt file
            unm, hsh_of_unm = username_and_hash_tuple_structure  # extracting the username and hash of this username
            password_token = dictionary_for_conversion_of_hash_to_its_corresponding_pass[hsh_of_unm]  # extracting the password
            print("User named " + str(unm) + " has the password " + str(
                password_token) + " and the hash of this password is " + str(hsh_of_unm) + "")
            print("User: " + str(unm) + " , Password: " + str(password_token) + "  , Hash: " + str(hsh_of_unm) + "")
            print()  # for readability and convenience
            print()  # for readability and convenience


if __name__ == "__main__":
    print("The dictionary attack with no salt usage and no key streching usage starts running now !!!")
    print()
    dict_att_with_no_salt_option_and_no_key_streching()
    print()
    print("The dictionary attack with no salt usage and no key streching usage ends running now !!!")
