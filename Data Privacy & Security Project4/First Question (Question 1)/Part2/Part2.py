import csv
import hashlib


def dct_attack_with_salt_option_but_no_key_streching():
    with open("salty-digitalcorp.txt") as salty_file:  # opening the txt file
        sep_char = ','  # adjusting the seperator as comma
        obj_rd_csv = csv.reader(salty_file, delimiter=sep_char)  # reading the salty txt file
        next(obj_rd_csv)  # reading next lines
        for triple_tuple_structure in obj_rd_csv:  # going along the reader object
            str_unm, str_salt_val, str_hsh = triple_tuple_structure  # extracting username, salt, hash outcome
            with open("rockyou.txt") as rock_you_fl_txt:  # reading the rockyou txt file
                rock_you_fl_txt_lines = rock_you_fl_txt.readlines()  # reading the rockyou txt file by using readlines() method
                all_feasible_tokens_of_pass = list()  # initializing a list which will keep the stripped passwords
                for single_rock_ln_pss in rock_you_fl_txt_lines:  # going along all lines of the rock you txt file
                    single_rock_ln_pss = single_rock_ln_pss.strip()  # removing the whitespaces at the end and at the beginning of the password
                    all_feasible_tokens_of_pass.append(
                        single_rock_ln_pss)  # add this password to the list which will keep it

            custom_dictionary_structure_of_the_attack = dict()  # creation of an empty dictionary of attacker
            for sng_pss in all_feasible_tokens_of_pass:  # going along all possible passwords
                password_string_with_salt = sng_pss + str_salt_val  # concatenating the salt to the password
                encoding_style_utf_string = 'utf-8'  # specifying the encoding style
                bytes_of_encoded_password_with_salt = password_string_with_salt.encode(encoding_style_utf_string)  # encoding process
                hashed_with_sha_hashing = hashlib.sha512(bytes_of_encoded_password_with_salt)  # applying sha512 hashing
                hash_outcome_with_hexadecimals = hashed_with_sha_hashing.hexdigest()  # convert the hash to its version made of hexadecimals
                custom_dictionary_structure_of_the_attack[
                    hash_outcome_with_hexadecimals
                ] = sng_pss  # adding hash and password as key value pair to dictionary

            my_custom_username_string = str_unm  # username
            my_custom_password_string = custom_dictionary_structure_of_the_attack[str_hsh]  # corresponding password of the username
            my_custom_hash_string = str_hsh  # corresponding hash of the password
            print("The user named " + str(my_custom_username_string) + " has the following password " + str(
                my_custom_password_string) + " which has a following found hash " + str(my_custom_hash_string) + "")
            print("User: " + str(my_custom_username_string) + ", Password: " + str(my_custom_password_string) + ", Hash: " + str(
                my_custom_hash_string) + "")
            print()  # for the readability and convenience purposes
            print()  # for the readability and convenience purposes


if __name__ == "__main__":
    print()     # for the readability and convenience purposes
    print("The dictionary attack with salt usage and no key streching usage starts running now !!!")
    print()     # for the readability and convenience purposes
    dct_attack_with_salt_option_but_no_key_streching()  # calling the function in main
    print()     # for the readability and convenience purposes
    print("The dictionary attack with salt usage and no key streching usage terminates running now !!!")
    print()     # for the readability and convenience purposes
