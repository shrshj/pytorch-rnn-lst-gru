import torch
from utils import LETTERS
from utils import load_data, line_to_tensor, category_from_output


# Predict method
def predict(model, input:str,all_categories):
    with torch.inference_mode():
        model.eval()

        # Prepare input
        input = line_to_tensor(input)
        hidden = model.init_hidden()

        # Forward
        for i in range(len(input)):
            output, hidden = model(input[i],hidden)

        # Report predicted value
        result = category_from_output(output=output,all_categories=all_categories)
        print(result)




# Load data to use in predict method
_, all_categories = load_data()


# Load the model
model = torch.load("models/rnn-model.pt")
model.eval()

# Inference
while True:
    x = input("Insert a name for inference or press 'Q' to quit...\n")
    if x=='Q':
        break
    predict(model=model, input=x, all_categories=all_categories)
