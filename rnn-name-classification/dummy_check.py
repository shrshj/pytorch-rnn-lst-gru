from model import RNN
from utils import LETTERS
from utils import load_data, line_to_tensor, category_from_output


# Loading data and defining model hyperparameters
category_lines, all_categories = load_data()
n_input = len(LETTERS)
n_hidden = 128
n_classes = len(all_categories)

# Defining the model, intial input and, initial hidden
model = RNN(n_input, n_hidden, n_classes)
hidden = model.init_hidden()
dummy_input = line_to_tensor('Touma')[0] # ONLY first char

# Applying the model on the input (only the first char)
output, next_hidden = model(dummy_input, hidden)
print(output.shape, next_hidden.shape)

# Output class (category) name
output_class = category_from_output(output, all_categories)
print(output_class)
