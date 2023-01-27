import hashlib
import csv


def key_streching_and_salting_function():
    custom_key1 = str()
    list_of_custom_key1 = list()

    custom_key2 = str()
    list_of_custom_key2 = list()

    custom_key3 = str()
    list_of_custom_key3 = list()

    custom_key4 = str()
    list_of_custom_key4 = list()

    custom_key5 = str()
    list_of_custom_key5 = list()

    custom_key6 = str()
    list_of_custom_key6 = list()

    with open("keystreching-digitalcorp.txt") as my_key_csv:
        custom_reader_object = csv.reader(my_key_csv, delimiter=',')
        next(custom_reader_object)
        for unm_string, salt_string, hsh_string in custom_reader_object:
            with open("rockyou.txt") as my_custom_file:
                all_lines_of_rock_you = my_custom_file.readlines()
                for single_pass_line in all_lines_of_rock_you: # going along all possible passwords
                    stripped_password = single_pass_line.strip()
                    for iteration_numbers in range(0, 2000):    # going along all possible number of key streching iterations
                        list_of_custom_key1.append(custom_key1)
                        list_of_custom_key1[iteration_numbers] = hashlib.sha512(
                            (list_of_custom_key1[
                                 iteration_numbers - 1] + stripped_password + salt_string).encode()).hexdigest()
                        if list_of_custom_key1[iteration_numbers] == hsh_string:
                            print("User " + str(unm_string) + " having the password " + str(
                                stripped_password) + " having the hash " + str(hsh_string) + "")
                            print("The number of iterations is " + str(iteration_numbers) + "")
                            print()
                            break
                        list_of_custom_key2.append(custom_key2)
                        list_of_custom_key2[iteration_numbers] = hashlib.sha512(
                            (stripped_password + list_of_custom_key2[
                                iteration_numbers - 1] + salt_string).encode()).hexdigest()
                        if list_of_custom_key2[iteration_numbers] == hsh_string:
                            print("User " + str(unm_string) + " having the password " + str(
                                stripped_password) + " having the hash " + str(hsh_string) + "")
                            print("The number of iterations is " + str(iteration_numbers) + "")
                            print()
                            break
                        list_of_custom_key3.append(custom_key3)
                        list_of_custom_key3[iteration_numbers] = hashlib.sha512(
                            (salt_string + stripped_password + list_of_custom_key3[
                                iteration_numbers - 1]).encode()).hexdigest()
                        if list_of_custom_key3[iteration_numbers] == hsh_string:
                            print("User " + str(unm_string) + " having the password " + str(
                                stripped_password) + " having the hash " + str(hsh_string) + "")
                            print("The number of iterations is " + str(iteration_numbers) + "")
                            print()
                            break
                        list_of_custom_key4.append(custom_key4)
                        list_of_custom_key4[iteration_numbers] = hashlib.sha512(
                            (salt_string + list_of_custom_key4[
                                iteration_numbers - 1] + stripped_password).encode()).hexdigest()
                        if list_of_custom_key4[iteration_numbers] == hsh_string:
                            print("User " + str(unm_string) + " having the password " + str(
                                stripped_password) + " having the hash " + str(hsh_string) + "")
                            print("The number of iterations is " + str(iteration_numbers) + "")
                            print()
                            break
                        list_of_custom_key5.append(custom_key5)
                        list_of_custom_key5[iteration_numbers] = hashlib.sha512(
                            (list_of_custom_key5[
                                 iteration_numbers - 1] + salt_string + stripped_password).encode()).hexdigest()
                        if list_of_custom_key5[iteration_numbers] == hsh_string:
                            print("User " + str(unm_string) + " having the password " + str(
                                stripped_password) + " having the hash " + str(hsh_string) + "")
                            print("The number of iterations is " + str(iteration_numbers) + "")
                            print()
                            break
                        list_of_custom_key6.append(custom_key6)
                        list_of_custom_key6[iteration_numbers] = hashlib.sha512(
                            (stripped_password + salt_string + list_of_custom_key6[
                                iteration_numbers - 1]).encode()).hexdigest()
                        if list_of_custom_key6[iteration_numbers] == hsh_string:
                            print("User " + str(unm_string) + " having the password " + str(
                                stripped_password) + " having the hash " + str(hsh_string) + "")
                            print("The number of iterations is " + str(iteration_numbers) + "")
                            print()
                            break


if __name__ == "__main__":
    print()  # for the readability and convenience purposes
    print("The dictionary attack with salt usage and no key streching usage starts running now !!!")
    print()  # for the readability and convenience purposes
    key_streching_and_salting_function()  # calling the function in main
    print()  # for the readability and convenience purposes
    print("The dictionary attack with salt usage and no key streching usage terminates running now !!!")
    print()  # for the readability and convenience purposes
