import torch
import torch.nn as nn
from model import RNN
import matplotlib.pyplot as plt
from utils import LETTERS
from utils import load_data, random_training_sample

# --------------------------------------------------Defining train method--------------------------------------------------------

# Training method
def train(model, criterion, optimizer, epochs ,category_lines, all_categories,report_step = 100):

    cumulative_loss = 0
    cumulative_losses_dict = {}

    accuracy = 0
    corrects = 0
    accuracies_dict = {}

    for epoch in range(epochs):

        # Prepare model for train
        model.train()

        # Data
        category, line, category_tensor, line_tensor = random_training_sample(category_lines, all_categories)
        hidden = model.init_hidden()

        # 1 - Forward (until the ned of the sequence)
        for i in range(len(line)):
            output, hidden = model(line_tensor[i], hidden)

        # 2 - Loss
        loss = criterion(output,category_tensor)
        cumulative_loss+=loss
        
        # Accuracy
        pred = torch.argmax(output,dim = 1)
        corrects += (pred==category_tensor)

        # 3 - Zero grad
        optimizer.zero_grad()

        # 4 - Loss backward
        loss.backward()

        # 5 - Step
        optimizer.step()

        # Report
        if (epoch+1)%report_step == 0: # is it like batch??? 
            average_cumulative_loss = cumulative_loss.item()/report_step
            cumulative_losses_dict[epoch + 1] = average_cumulative_loss
            cumulative_loss = 0

            accuracy = corrects.item()/report_step
            accuracies_dict[epoch + 1] = accuracy
            corrects = 0

            print(f"Epoch : {epoch + 1}/{epochs} --> Train loss : {average_cumulative_loss:.5f} | Train acc : {accuracy}")
    

    return cumulative_losses_dict, accuracies_dict


# -------------------------------------------------Train the model and report the results-------------------------------------------------------
# Loading data
category_lines, all_categories = load_data()

# Define model hyperparameters
n_input = len(LETTERS)
n_hidden = 128
n_classes = len(all_categories)

# Defining the model
model = RNN(n_input, n_hidden, n_classes)

# Set train hyperparameters
learning_rate = 0.005
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
criterion = nn.NLLLoss()
epochs = 30000

# Train
loss_dict, acc_list = train(model, criterion, optimizer, epochs ,category_lines, all_categories)
print("Train is finished.")

# Save the model
torch.save(obj = model, f = "models/rnn-model.pt")

# Plot
plt.figure()
plt.plot(loss_dict.keys(), loss_dict.values(), label = "Loss")
plt.plot(acc_list.keys(), acc_list.values(), label = "Accuracy")
plt.title("<Train Loss and Accuracy>")
plt.xlabel("Epoch")
plt.ylabel("Loss & Accuracy")
plt.legend()
plt.show()
