ranks = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
files = ['1', '2', '3', '4', '5', '6', '7', '8']



pieces               = ['K', 'Q', 'R', 'B', 'N']
move_modifiers       = ['+', '#', 'x']
castles              = ['O-O', 'O-O-O']
space                = [' ']
possible_coordinates = [rank + chosen_file for rank in ranks for chosen_file in files]