import tqdm
import os

print("Welcome to ChePT Data Preprocessor")

print("Now processing kingbase-ftfy.txt")

unprocessed_kingbase_lines = open("./data/datasets/kingbase-ftfy.txt", "r").readlines()
write_folder = "./data/datasets-cleaned/"
write_file = os.path.join(write_folder, 'kingbase_cleaned.txt')

if not os.path.exists(write_folder):
    os.makedirs(write_folder)

processed_kingbase_lines = open(write_file, "w")

line_length = []
for line in tqdm.tqdm(unprocessed_kingbase_lines):
    split_line = line.split()
    output_line = " ".join(split_line[6:]) + "\n"
    if len(output_line) <= 1024:
        processed_kingbase_lines.writelines(output_line)
        line_length.append(len(output_line))
    

import matplotlib.pyplot as plt
import numpy as np

x = np.array(line_length)

plt.hist(x, density=True, bins=100)  # density=False would make counts
plt.ylabel('Probability')
plt.xlabel('Data')
plt.show()