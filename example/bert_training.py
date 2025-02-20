import torch
from transformers import BertTokenizer, BertForTokenClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForTokenClassification, AdamW
from torch.nn.utils.rnn import pad_sequence
import random
import pandas as pd
import numpy as np
import os 

seed = 64*400 
random.seed(seed)
torch.manual_seed(seed)

relative_path = os.path.abspath(__file__)

relative_path = relative_path.split('Text-To-Structure-Hypernetworks')[0]


def split_array_with_ratio(input_array, ratios):
    total_length = len(input_array)
    split_points = [int(ratio * total_length) for ratio in ratios]
    split_arrays = []

    start = 0
    for split_point in split_points:
        split_arrays.append(input_array[start:start + split_point])
        start += split_point

    return split_arrays

# Load Data
file_path = relative_path + r"Text-To-Structure-Hypernetworks\example\data\example_descriptions.txt"

with open(file_path, 'r') as file:
    descriptions = file.read().splitlines()

ratios = [0.8, 0.1, 0.1]
All_descriptions  = split_array_with_ratio(descriptions, ratios)
training_descriptions = All_descriptions[0]
validation_descriptions = All_descriptions[1]
test_descriptions = All_descriptions[2]

#####################
#   TRAINING DATA   #
#####################

# Load Training Data
train_dt = pd.read_json( relative_path + r"Text-To-Structure-Hypernetworks\example\data\example_bert_data\example_train.json")

coord_a = list(train_dt['Coord_A'])
coord_b = list(train_dt['Coord_B'])

train_x = []
train_y = []
for i,c in enumerate(coord_a):
    rand = random.choice(training_descriptions)
    rand = rand.replace('X',str(c) + ' X').replace('Y',str(coord_b[i]) + ' Y')
    train_x.append(rand)
    train_y.append([c,coord_b[i]])

################
#   Val DATA   #
################
    
# Load val Data
val_dt = pd.read_json(relative_path +r"Text-To-Structure-Hypernetworks\example\data\example_bert_data\example_val.json")

val_a = list(val_dt['Coord_A'])
val_b = list(val_dt['Coord_B'])

val_x = []
val_y = []
for i,c in enumerate(val_a):
    rand = random.choice(validation_descriptions)

    rand = rand.replace('X',str(c) + ' X ').replace('Y',str(val_b[i]) + ' Y')

    val_x.append(rand)
    val_y.append([c,val_b[i]])

#################
#   TEST DATA   #
#################
    
# Load test Data
Test_dt = pd.read_json(relative_path + r"Text-To-Structure-Hypernetworks\example\data\example_bert_data\example_test.json")

test_a = list(Test_dt['Coord_A'])
test_b = list(Test_dt['Coord_B'])

test_x = []
test_y = []
for i,c in enumerate(test_a):
    rand = random.choice(test_descriptions)

    rand = rand.replace('X',str(c) + ' X').replace('Y',str(test_b[i]) + ' Y')

    test_x.append(rand)
    test_y.append([c,test_b[i]])
    
class CustomNERDataset(Dataset):
    def __init__(self, input_, labels_):
        self.descriptions = input_
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.labels = labels_

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, idx):
        description = self.descriptions[idx]

        # Tokenize 
        tokens = self.tokenizer.encode(description, add_special_tokens=True)

        labels = [0] * len(tokens)  
        first_tokens = self.tokenizer.encode(str(self.labels[idx][0]), add_special_tokens=False)
        second_tokens = self.tokenizer.encode(str(self.labels[idx][1]), add_special_tokens=False)

        for i in range(len(tokens) - len(first_tokens) + 1):
            if tokens[i:i+len(first_tokens)] == first_tokens:
                labels[i:i+len(first_tokens)] = [1] * len(first_tokens)

        for i in range(len(tokens) - len(second_tokens) + 1):
            if tokens[i:i+len(second_tokens)] == second_tokens:
                labels[i:i+len(second_tokens)] = [2] * len(second_tokens)

        return {'tokens': torch.tensor(tokens), 'labels': torch.tensor(labels)}

# Padding function
def pad_sequence(batch):
    max_len = max(len(entry['tokens']) for entry in batch)
    padded_tokens = [entry['tokens'].tolist() + [0] * (max_len - len(entry['tokens'])) for entry in batch]
    padded_labels = [entry['labels'].tolist() + [0] * (max_len - len(entry['labels'])) for entry in batch]
    return {'tokens': torch.tensor(padded_tokens), 'labels': torch.tensor(padded_labels)}

# Create custom dataset and dataloaders
custom_dataset = CustomNERDataset(train_x,train_y)
train_dataloader = DataLoader(custom_dataset, batch_size=20, shuffle=True, collate_fn=pad_sequence)

model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=3) 
criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)  # Ignore padding tokens
epochs = 3
learning_rate = 2e-5

optimizer = AdamW(model.parameters(), lr=learning_rate)

# Example usage 
print("Training Loop")
for epoch in range(epochs):
    model.train()
    print(epoch)
    for batch in train_dataloader:
        tokens, labels = batch['tokens'], batch['labels']
        optimizer.zero_grad()
        outputs = model(tokens, labels=labels)

        loss = criterion(outputs.logits.view(-1, 3), labels.view(-1))
        loss.backward()
        optimizer.step()
        print(loss)

model.eval()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#########################
#   Training Results    #
#########################
training_loss =  0
training_accuracy =  0 

with torch.no_grad():
    for i,description in enumerate(train_x):

        # Tokenize 
        tokens = tokenizer.encode(description, add_special_tokens=True)
        inputs = {'input_ids': torch.tensor(tokens).unsqueeze(0)} 

        # Make prediction
        outputs = model(**inputs)

        # Label 
        labels = [0] * len(tokens) 
        first_tokens = tokenizer.encode(str(train_y[i][0]), add_special_tokens=False)
        second_tokens = tokenizer.encode(str(train_y[i][1]), add_special_tokens=False)

        for j in range(len(tokens) - len(first_tokens) + 1):
            if tokens[j:j+len(first_tokens)] == first_tokens:
                labels[j:j+len(first_tokens)] = [1] * len(first_tokens)

        for j in range(len(tokens) - len(second_tokens) + 1):
            if tokens[j:j+len(second_tokens)] == second_tokens:
                labels[j:j+len(second_tokens)] = [2] * len(second_tokens)

        loss = criterion(outputs.logits.view(-1, 3), torch.tensor(labels).view(-1)) 

        training_loss += loss

        predicted_labels = torch.argmax(outputs.logits, dim=2).squeeze().tolist()

        predicted_one = [tokens[i] for i, label in enumerate(predicted_labels) if label == 1]
        predicted_two = [tokens[i] for i, label in enumerate(predicted_labels) if label == 2]

        predicted_one_ = tokenizer.decode(predicted_one)
        predicted_two_ = tokenizer.decode(predicted_two)

        score = 0

        try:
            if float(predicted_one_) == float(train_y[i][0]):
                score += 1
        except:
            pass

        try:
            if float(predicted_two_) == float(train_y[i][1]):
                score += 1     
        except:
            pass

        training_accuracy += score

print('________________')

print('Training Results')
training_accuracy = training_accuracy / (len(train_x) * 2)
training_loss = np.array(training_loss) / (len(train_x) )
print(training_loss)
print(training_accuracy)

print('______________')

###########################
#   Validation Results    #
###########################
val_loss =  0
val_accuracy =  0 


with torch.no_grad():
    for i,description in enumerate(val_x):

        # Tokenize
        tokens = tokenizer.encode(description, add_special_tokens=True)
        inputs = {'input_ids': torch.tensor(tokens).unsqueeze(0)}  # Add batch dimension

        # Make prediction
        outputs = model(**inputs)

        # Label
        labels = [0] * len(tokens) 
        first_tokens = tokenizer.encode(str(val_y[i][0]), add_special_tokens=False)
        second_tokens = tokenizer.encode(str(val_y[i][1]), add_special_tokens=False)

        for j in range(len(tokens) - len(first_tokens) + 1):
            if tokens[j:j+len(first_tokens)] == first_tokens:
                labels[j:j+len(first_tokens)] = [1] * len(first_tokens)

        for j in range(len(tokens) - len(second_tokens) + 1):
            if tokens[j:j+len(second_tokens)] == second_tokens:
                labels[j:j+len(second_tokens)] = [2] * len(second_tokens)

        loss = criterion(outputs.logits.view(-1, 3), torch.tensor(labels).view(-1)) 

        val_loss += loss
    
        predicted_labels = torch.argmax(outputs.logits, dim=2).squeeze().tolist()

        predicted_one = [tokens[i] for i, label in enumerate(predicted_labels) if label == 1]
        predicted_two = [tokens[i] for i, label in enumerate(predicted_labels) if label == 2]

        predicted_one_ = tokenizer.decode(predicted_one)
        predicted_two_ = tokenizer.decode(predicted_two)

        score = 0
        try:
            if float(predicted_one_) == float(val_y[i][0]):
                score += 1
        except:
            pass
        try:
            if float(predicted_two_) == float(val_y[i][1]):
                score += 1     
        except:
            pass
        val_accuracy += score

print('________________')

print('Validation Results')
val_accuracy = val_accuracy / (len(val_x) * 2)
val_loss = np.array(val_loss) / (len(val_x) )
print(val_loss)
print(val_accuracy)

print('______________')

#####################
#   Test Results    #
#####################
test_loss =  0
test_accuracy =  0 

with torch.no_grad():
    for i,description in enumerate(test_x):

        # Tokenize 
        tokens = tokenizer.encode(description, add_special_tokens=True)
        inputs = {'input_ids': torch.tensor(tokens).unsqueeze(0)} 

        outputs = model(**inputs)

        # Label 
        labels = [0] * len(tokens)  
        first_tokens = tokenizer.encode(str(test_y[i][0]), add_special_tokens=False)
        second_tokens = tokenizer.encode(str(test_y[i][1]), add_special_tokens=False)

        for j in range(len(tokens) - len(first_tokens) + 1):
            if tokens[j:j+len(first_tokens)] == first_tokens:
                labels[j:j+len(first_tokens)] = [1] * len(first_tokens)

        for j in range(len(tokens) - len(second_tokens) + 1):
            if tokens[j:j+len(second_tokens)] == second_tokens:
                labels[j:j+len(second_tokens)] = [2] * len(second_tokens)


        loss = criterion(outputs.logits.view(-1, 3), torch.tensor(labels).view(-1)) 

        test_loss += loss
        predicted_labels = torch.argmax(outputs.logits, dim=2).squeeze().tolist()

        predicted_one = [tokens[i] for i, label in enumerate(predicted_labels) if label == 1]
        predicted_two = [tokens[i] for i, label in enumerate(predicted_labels) if label == 2]

        predicted_one_ = tokenizer.decode(predicted_one)
        predicted_two_ = tokenizer.decode(predicted_two)

        score = 0

        print('~~~~~~~~~~~~~~~~')
        print(f"Input String: {description}")
        print(f"Predicted Params: [{predicted_one_},{predicted_two_}]")
        print(f"Target Params: {str(test_y[i])}")
        print('~~~~~~~~~~~~~~~~')

        try:
            if float(predicted_one_) == float(test_y[i][0]):
                score += 1
        except:
            pass

        try:
            if float(predicted_two_) == float(test_y[i][1]):
                score += 1     
        except:
            pass

        test_accuracy += score

print('________________')

print('Test Results')
test_accuracy = test_accuracy / (len(test_x) * 2)
test_loss = np.array(test_loss) / (len(test_x) )
print(f'Loss:  {test_loss}')
print(f'Acc: {test_accuracy}')

print('______________')
