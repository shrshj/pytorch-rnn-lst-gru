import string
import glob
import torch
import random

LETTERS = string.ascii_letters

def load_data():
    category_lines = {}
    all_categories = []
    
    for filename in glob.glob('data/names/*.txt'):
        category = filename.split('/')[-1].split('.')[0]
        all_categories.append(category)
        
        with open(filename, 'r') as f:
            lines = f.read().strip().split('\n')
        f.close()
        
        category_lines[category] = lines
        
    return category_lines, all_categories


# Turn a line into a [line_length , 1 , n_letters] one-hot tensor,
def line_to_tensor(line):
    tensor = torch.zeros(len(line), 1, len(LETTERS))
    for i, letter in enumerate(line):
        tensor[i][0][LETTERS.find(letter)] = 1
    return tensor


def random_training_sample(category_lines, all_categories):
    category = random.choice(all_categories)
    line = random.choice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = line_to_tensor(line)
    return category, line, category_tensor, line_tensor

def category_from_output(output,all_categories):
    category_idx = torch.argmax(output).item()
    return all_categories[category_idx]

if __name__ == '__main__':
    print(LETTERS)
    category_lines, all_categories = load_data()
    print(category_lines['English'][:5])
    print(line_to_tensor('Jones').size()) # [5, 1, 52]