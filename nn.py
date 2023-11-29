import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import re
import matplotlib.pyplot as plt
import seaborn as sns

# Define your hyperparameters sets
'''
Each of these sets of hyperparameters will result in a different model. 
The best set of hyperparameters usually depends on the specific task and data, 
and is often found through experimentation and hyperparameter tuning.
'''
hyperparameters_sets = [
    (25, 50, 100),
    (30, 55, 150),
    (30, 60, 100),
    (40, 65, 250),
    (45, 70, 200),
    (50, 75, 350),
    (55, 80, 200),
    (60, 85, 350),
    (65, 90, 200),
    (70, 95, 350)
]

test_accuracies = []
validation_accuracies = []


# Tokenize and pad/truncate
def tokenize(text, max_length):
    tokens = re.findall(r'\w+', text.lower())
    if len(tokens) > max_length:
        return tokens[:max_length]
    return tokens + ['<PAD>'] * (max_length - len(tokens))

def load_data(file_path, max_length):
    texts, labels = [], []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            text, label = line.strip().split(';')
            texts.append(tokenize(text, max_length))
            labels.append(label)
    return texts, labels

# Function to evaluate model
def evaluate_model(model, data_loader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for texts, labels in data_loader:
            outputs = model(texts)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

class EmotionDataset(Dataset):
    def __init__(self, texts, labels, word_to_idx):
        self.texts = [[word_to_idx.get(word, word_to_idx['<UNK>']) for word in text] for text in texts]
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return torch.tensor(self.texts[idx]), torch.tensor(self.labels[idx])

class EmotionClassifier(nn.Module):
    def __init__(self, vocab_size, EMBED_DIM, num_classes):
        super(EmotionClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM)
        self.fc = nn.Linear(EMBED_DIM, num_classes)
    
    def forward(self, x):
        x = self.embedding(x)
        x = torch.mean(x, dim=1)
        return self.fc(x)


for MAX_SEQ_LENGTH, EPOCHS, EMBED_DIM in hyperparameters_sets:

    # Load and process data
    train_texts, train_labels = load_data('data/train.txt', MAX_SEQ_LENGTH)
    test_texts, test_labels = load_data('data/test.txt', MAX_SEQ_LENGTH)

    # Encode labels
    label_encoder = LabelEncoder()
    train_labels = label_encoder.fit_transform(train_labels)
    test_labels = label_encoder.transform(test_labels)

    # Build vocabulary
    all_words = [word for text in train_texts for word in text]
    word_counts = Counter(all_words)
    vocab = ['<PAD>', '<UNK>'] + [word for word, count in word_counts.items() if count > 1]
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}

    # Model parameters
    vocab_size = len(vocab)

    num_classes = len(label_encoder.classes_)

    # Model, Loss, and Optimizer
    model = EmotionClassifier(vocab_size, EMBED_DIM, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Data loaders
    train_dataset = EmotionDataset(train_texts, train_labels, word_to_idx)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    test_dataset = EmotionDataset(test_texts, test_labels, word_to_idx)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Training loop
    for epoch in range(EPOCHS):
        for texts, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for texts, labels in test_loader:
            outputs = model(texts)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    test_accuracies.append(accuracy)
    print(f'Accuracy: {accuracy*100:.2f}%')

    # Save the model to disk
    torch.save(model.state_dict(), 'emotion_classifier_model.pth')

    # Load the model from disk
    loaded_model = EmotionClassifier(vocab_size, EMBED_DIM, num_classes)
    loaded_model.load_state_dict(torch.load('emotion_classifier_model.pth'))

    # Load and process validation data
    val_texts, val_labels = load_data('data/val.txt', MAX_SEQ_LENGTH)
    val_labels = label_encoder.transform(val_labels)
    val_dataset = EmotionDataset(val_texts, val_labels, word_to_idx)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Evaluate the loaded model on validation data
    accuracy = evaluate_model(loaded_model, val_loader)
    print(f'Validation Accuracy: {accuracy*100:.2f}%')

    # Store accuracies
    validation_accuracies.append(accuracy)

# Set a theme
sns.set_theme()

# Plot test accuracies
plt.figure(figsize=(10, 6))
plt.plot([str(params) for params in hyperparameters_sets], test_accuracies, marker='o', linestyle='-', color='b')
plt.xlabel('Hyperparameters', fontsize=14)
plt.ylabel('Model Accuracy', fontsize=14)
plt.title('Model Accuracy for different hyperparameters', fontsize=16)
plt.xticks(rotation=90, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
plt.show()

# Plot validation accuracies
plt.figure(figsize=(10, 6))
plt.plot([str(params) for params in hyperparameters_sets], validation_accuracies, marker='o', linestyle='-', color='r')
plt.xlabel('Hyperparameters', fontsize=14)
plt.ylabel('Validation Accuracy', fontsize=14)
plt.title('Validation Accuracy for different hyperparameters', fontsize=16)
plt.xticks(rotation=90, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
plt.show()

# Print test and validation accuracies
print("Test accuracies:", test_accuracies)
print("Validation accuracies:", validation_accuracies)

'''
Test accuracies: [0.8095, 0.8285, 0.828, 0.8345, 0.8295, 0.8315, 0.8355, 0.831, 0.829, 0.822]
Validation accuracies: [
    0.8415, 
    0.855, 
    0.8525, 
    0.8495, 
    0.8535, 
    0.8465, 
    0.8535, 
    0.8495, 
    0.847, 
    0.839
    ]
'''