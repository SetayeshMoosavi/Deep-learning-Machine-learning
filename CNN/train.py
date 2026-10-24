import torch
import torch.optim as opt
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from data import get_datasets
from model import SudokuNet, count_trainable_parameters

train_data , validation_data , test_data = get_datasets("./data")

BATCH_SIZE = 64
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
validation_loader = DataLoader(validation_data, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SudokuNet().to(device)

total_number_parameter = count_trainable_parameters(model)
print(f'total number of parameter of this network is {total_number_parameter}')

loss_digit_f = nn.CrossEntropyLoss()
loss_order_f = nn.CrossEntropyLoss()
loss_sum_f = nn.L1Loss() 
optimizer = opt.Adam(model.parameters(), lr=0.001)

history = {
    'train_loss': [],
    'validation_loss': [],
    'train_digit_accuracy': [],
    'validation_digit_accuracy': [],
    'train_order_accuracy': [],
    'validation_order_accuracy': [],
    'train_sum_MAE': [],
    'validation_sum_MAE': []
}

def accuracy_for_classification(data_x, data_y):

    if data_x.dim() == 3:
        prediction = torch.argmax(data_x, dim=2)
        
    else:
        prediction = torch.argmax(data_x, dim=1)

    correct = (prediction == data_y).sum().item()

    return correct


def evaluate (data_load):
    running_loss = 0.0
    MAE_loss = 0.0
    accuracy_digit = 0.0
    accuracy_order = 0.0
    correct_digit= 0.0
    correct_order = 0.0

    model.eval() 
    with torch.no_grad():
        for item in data_load:

            data_x = item ['image'].to(device)
            data_y_digit = item ['missing_digit'].to(device)
            data_y_order = item ['sorted_labels'].to(device)
            data_y_sum = item ['sum_labels'].to(device)

            logit_digit_head , logit_order_head , logit_sum_head = model(data_x)

            loss_digit = loss_digit_f(logit_digit_head,data_y_digit)
            loss_order = loss_order_f(logit_order_head.view(-1, 3), data_y_order.view(-1)) 
            loss_sum = loss_sum_f(logit_sum_head,data_y_sum)

            total_loss = loss_digit + loss_order + loss_sum

            running_loss += total_loss.item() * data_x.size(0)
            MAE_loss += loss_sum.item() * data_x.size(0)

            correct_digit += accuracy_for_classification(logit_digit_head,data_y_digit)
            correct_order += accuracy_for_classification(logit_order_head,data_y_order)

    validation_loss = running_loss / len(data_load.dataset)
    accuracy_digit = correct_digit / len(data_load.dataset)
    accuracy_order = correct_order / (len(data_load.dataset) * 6)
    validation_sum_MAE = MAE_loss / len(data_load.dataset)

    history['validation_loss'].append(validation_loss)
    history['validation_digit_accuracy'].append(accuracy_digit)
    history['validation_order_accuracy'].append(accuracy_order)
    history['validation_sum_MAE'].append(validation_sum_MAE)


EPOCHS = 17

for epoch in range (EPOCHS):

    model.train()
    running_loss = 0.0
    train_sum_MAE = 0.0
    accuracy_digit = 0.0
    accuracy_order = 0.0
    correct_digit= 0.0
    correct_order = 0.0
    MAE_loss = 0.0
    
    for item in train_loader:

        optimizer.zero_grad()

        train_data_x = item ['image'].to(device)
        train_data_y_digit = item ['missing_digit'].to(device)
        train_data_y_order = item ['sorted_labels'].to(device)
        train_data_y_sum = item ['sum_labels'].to(device)

        logit_digit_head , logit_order_head , logit_sum_head = model(train_data_x)

        loss_digit = loss_digit_f(logit_digit_head,train_data_y_digit)
        loss_order = loss_order_f(logit_order_head.view(-1, 3), train_data_y_order.view(-1)) 
        loss_sum = loss_sum_f(logit_sum_head,train_data_y_sum)

        total_loss = loss_digit + loss_order + loss_sum

        total_loss.backward()

        optimizer.step()
        running_loss += total_loss.item() * train_data_x.size(0)
        MAE_loss += loss_sum.item() * train_data_x.size(0)

        correct_digit += accuracy_for_classification(logit_digit_head,train_data_y_digit)
        correct_order += accuracy_for_classification(logit_order_head,train_data_y_order)

    epoch_loss = running_loss / len(train_loader.dataset)
    train_sum_MAE = MAE_loss / len(train_loader.dataset)

    print(f"Epoch {epoch+1}/{EPOCHS} complete. Training Loss: {epoch_loss:.4f}")


    accuracy_digit = correct_digit / len(train_loader.dataset)
    accuracy_order = correct_order / (len(train_loader.dataset) * 6)

    history['train_loss'].append(epoch_loss)
    history['train_digit_accuracy'].append(accuracy_digit)
    history['train_order_accuracy'].append(accuracy_order)
    history['train_sum_MAE'].append(train_sum_MAE)

    evaluate(validation_loader)



def test (data_load):
    running_loss = 0.0
    MAE_loss = 0.0
    test_accuracy_digit = 0.0
    test_accuracy_order = 0.0
    correct_digit= 0.0
    correct_order = 0.0

    model.eval() 
    with torch.no_grad():
        for item in data_load:

            data_x = item ['image'].to(device)
            data_y_digit = item ['missing_digit'].to(device)
            data_y_order = item ['sorted_labels'].to(device)
            data_y_sum = item ['sum_labels'].to(device)

            logit_digit_head , logit_order_head , logit_sum_head = model(data_x)

            loss_digit = loss_digit_f(logit_digit_head,data_y_digit)
            loss_order = loss_order_f(logit_order_head.view(-1, 3), data_y_order.view(-1)) 
            loss_sum = loss_sum_f(logit_sum_head,data_y_sum)

            total_loss = loss_digit + loss_order + loss_sum

            running_loss += total_loss.item() * data_x.size(0)
            MAE_loss += loss_sum.item() * data_x.size(0)

            correct_digit += accuracy_for_classification(logit_digit_head,data_y_digit)
            correct_order += accuracy_for_classification(logit_order_head,data_y_order)

    test_loss = running_loss / len(data_load.dataset)
    test_accuracy_digit = correct_digit / len(data_load.dataset)
    test_accuracy_order = correct_order / (len(data_load.dataset) * 6)
    test_sum_MAE = MAE_loss / len(data_load.dataset)

    return test_loss , test_accuracy_digit , test_accuracy_order , test_sum_MAE
   
loss , accuracy_digit, accuracy_order , sum_MAE = test(test_loader)

final_test_matrix = {
    'total_loss': loss,
    'digit_acc': accuracy_digit,
    'order_acc': accuracy_order,
    'sum_mae': sum_MAE
}

def plot_training_history(history, test_metrics=None):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(12, 10))
    
    # 1. Total Loss Plot
    plt.subplot(2, 2, 1)
    plt.plot(epochs, history['train_loss'], 'bo-', label='Train Total Loss')
    plt.plot(epochs, history['validation_loss'], 'rs-', label='Validation Total Loss')
    if test_metrics:
        plt.axhline(y=test_metrics['total_loss'], color='purple', linestyle='--', label=f"Test Loss: {test_metrics['total_loss']:.4f}")
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Missing Digit Plot
    plt.subplot(2, 2, 2)
    plt.plot(epochs, history['train_digit_accuracy'], 'g^-', label='Train Acc')
    plt.plot(epochs, history['validation_digit_accuracy'], 'd-', color='orange', label='Val Acc')
    if test_metrics:
        plt.axhline(y=test_metrics['digit_acc'], color='green', linestyle='--', label=f"Test Acc: {test_metrics['digit_acc']:.4f}")
    plt.axhline(y=0.95, color='gray', linestyle=':', label='Target Threshold (95%)')
    plt.title('Missing Digit Task')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Sort Order Plot
    plt.subplot(2, 2, 3)
    plt.plot(epochs, history['train_order_accuracy'], 'g^-', label='Train Acc')
    plt.plot(epochs, history['validation_order_accuracy'], 'd-', color='orange', label='Val Acc')
    if test_metrics:
        plt.axhline(y=test_metrics['order_acc'], color='green', linestyle='--', label=f"Test Acc: {test_metrics['order_acc']:.4f}")
    plt.axhline(y=0.93, color='gray', linestyle=':', label='Target Threshold (93%)')
    plt.title('Sort Order Task')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Sum Prediction Plot
    plt.subplot(2, 2, 4)
    plt.plot(epochs, history['train_sum_MAE'], 'g^-', label='Train MAE')
    plt.plot(epochs, history['validation_sum_MAE'], 'd-', color='orange', label='Val MAE')
    if test_metrics:
        plt.axhline(y=test_metrics['sum_mae'], color='green', linestyle='--', label=f"Test MAE: {test_metrics['sum_mae']:.4f}")
    plt.axhline(y=0.06, color='gray', linestyle=':', label='Target Threshold (0.06)')
    plt.title('Sum Prediction Task')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

plot_training_history(history, test_metrics=final_test_matrix)