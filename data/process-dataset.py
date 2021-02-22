import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
import random

def preprocess_kingbase():
    print("Now processing kingbase-ftfy.txt")

    write_folder = "./data/datasets-cleaned/"
    if not os.path.exists(write_folder):
        os.makedirs(write_folder)

    # check if this file has already been preprocessed
    if os.path.exists("./data/datasets-cleaned/kingbase_cleaned.txt"):
        response = ""
        while response.lower() != "y" and response.lower() != "n":
            response = input("It appears that the kingbase file has already been preprocessed; reprocess?  (Y/N):  ")

            if response.lower() == "n":
                return

        os.remove("./data/datasets-cleaned/kingbase_cleaned.txt")
    
    unprocessed_kingbase_lines = open("./data/datasets/kingbase-ftfy.txt", "r").readlines()

    processed_kingbase_lines = open("./data/datasets-cleaned/kingbase_cleaned.txt", "w")

    line_length = []
    for line in tqdm.tqdm(unprocessed_kingbase_lines):
        split_line = line.split()
        output_line = " ".join(split_line[6:-1]) + "\n"
        if len(output_line) <= 1024:
            processed_kingbase_lines.writelines(output_line)
            line_length.append(len(output_line))

    x = np.array(line_length)

    plt.hist(x, density=True, bins=100)  # density=False would make counts
    plt.ylabel('Relative Frequency')
    plt.xlabel('Sequence Length')
    plt.show()

def preprocess_kaggle():
    print("Now preprocessing all_with_filtered_anotations_since1998.txt")

    write_folder = "./data/datasets-cleaned/"
    if not os.path.exists(write_folder):
        os.makedirs(write_folder)

    # check if this file has already been preprocessed
    if os.path.exists("./data/datasets-cleaned/kaggle_cleaned.txt"):
        response = ""
        while response.lower() != "y" and response.lower() != "n":
            response = input("It appears that the kaggle file has already been preprocessed; reprocess?  (Y/N):  ")

            if response.lower() == "n":
                return

        os.remove("./data/datasets-cleaned/kaggle_cleaned.txt")

    unprocessed_kaggle_lines = open("./data/datasets/all_with_filtered_anotations_since1998.txt", "r").readlines()[5:]

    processed_kaggle_lines = open("./data/datasets-cleaned/kaggle_cleaned.txt", "w")

    line_length = []
    for line in tqdm.tqdm(unprocessed_kaggle_lines):
        split_line = line.split()
        for index, token in enumerate(split_line):
            if index % 2 == 0:
                split_line[index] = token[3:]
            else:
                split_line[index] = token[1:]
        output_line = " ".join(split_line[17:]) + "\n"
        
        if len(output_line) <= 1024:
            processed_kaggle_lines.writelines(output_line)
            line_length.append(len(output_line))

    x = np.array(line_length)

    plt.hist(x, density=True, bins=100)  # density=False would make counts
    plt.ylabel('Relative Frequency')
    plt.xlabel('Sequence Length')
    plt.show()

def test_chess_move_dataset_approach():
    BLOCK_SIZE = 1024

    processed_kingbase_lines = open("./data/datasets-cleaned/kingbase_cleaned.txt", "r").readlines()

    game = processed_kingbase_lines[0]

    moves = [move for move in game.split()]
    print(moves)
    index = random.randint(0, len(moves) - 1)
    masked_move = moves[index]
    print(masked_move)
    if index % 2 == 0:
        masked_move = masked_move[2:]
        print(masked_move)
        moves[index] = moves[index][:2]
    else:
        moves[index] = ""

    # now, we have masked_move as the actual masked move, and the moves list has been left as if only the move itself were removed

    prefix = " ".join(moves[:index + 1])
    suffix = " ".join(moves[index + 1:])
    print(prefix)
    print(suffix)

    MASK_CHAR = u"\u2047"

    output = prefix + MASK_CHAR + " " + suffix + MASK_CHAR + masked_move

    print(output)


def main():
    print("Welcome to ChePT Data Preprocessor")
    test_chess_move_dataset_approach()
    return
    preprocess_kingbase()
    preprocess_kaggle()

if __name__ == '__main__':
    main()

