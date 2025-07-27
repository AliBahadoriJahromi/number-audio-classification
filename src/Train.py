import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import pickle
import os
import argparse

from Model import SimpleCNN
from AudioDataset import AudioDataset

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    parser = argparse.ArgumentParser(description='Train a model.')
    # Add the `epochs` argument
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs for training')
    args = parser.parse_args()

    root_dir = "src/data/Train"  # Path to your audio dataset folder
    batch_size = 16
    epochs = args.epochs
    learning_rate = 0.001
    target_length = 20000 

    dataset = AudioDataset(root_dir=root_dir, target_length=target_length)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size   

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = SimpleCNN().to(device)# Move model to GPU if available
    if os.path.exists("src/weights.pth"):
        model.load_state_dict(torch.load("src/weights.pth"))
        print("Loading weights from {src/weights.pth}")
    
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if os.path.exists("src/training_results.pkl"):
        with open("src/training_results.pkl", 'rb') as f:
            train_losses, eval_losses, train_corrects, eval_corrects, best_accuracy = pickle.load(f)
    else:
        train_losses = []
        eval_losses = []
        train_corrects = []
        eval_corrects = []
        best_accuracy = 0.0  # Track the best accuracy

    best_model_state = model.state_dict()  # Store the state of the best model
    
    for epoch in range(epochs):

        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in tqdm(train_loader, total=len(train_loader), desc="Training Data"):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted = torch.max(outputs, 1)[1]
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        train_losses.append(train_loss)
        train_corrects.append(correct)

        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, total=len(test_loader), desc="Evaluating Data"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item()
                predicted = torch.max(outputs.data, 1)[1]
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            eval_loss = running_loss / len(test_loader)
            eval_losses.append(eval_loss)
            eval_corrects.append(correct)

            accuracy = correct / total

             # Save the model if the accuracy improved
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_state = model.state_dict()  # Save the state of the best model
                
        print(f"Epoch [{epoch+1}/{epochs}],Train Loss: {train_loss:.4f},Train Accuracy: {train_accuracy:.2f}%, Test Loss: {eval_loss:.4f},Test Accuracy: {accuracy:.2f}%")

    if best_model_state:
        torch.save(best_model_state, 'src/weights.pth')
        print("Best model saved with accuracy:", best_accuracy)

    with open('src/training_results.pkl', 'wb') as f:
        pickle.dump([train_losses, eval_losses, train_corrects, eval_corrects, best_accuracy], f)

        
