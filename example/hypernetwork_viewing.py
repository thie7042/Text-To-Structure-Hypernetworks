import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# Set random seed
random_seed = 26
torch.manual_seed(random_seed)
random.seed(random_seed)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Define the gpu if available 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

relative_path = os.path.abspath(__file__)

# Get the directory
relative_path = relative_path.split('Text-To-Structure-Hypernetworks')[0]

#######################
#   Core Functions    #
#######################

# Calculate the Binary Crossentropy Loss. This can be weighted if needed
class WeightedBCELoss(nn.Module):
    def __init__(self, weight_factor):
        super(WeightedBCELoss, self).__init__()
        self.weight_factor = weight_factor

    def forward(self, input, target):

        bce_loss = nn.functional.binary_cross_entropy(input, target, reduction='none')
        weighted_bce_loss = torch.where(target == 1, self.weight_factor * bce_loss, bce_loss)
        mean_loss = torch.mean(weighted_bce_loss)
        
        return mean_loss

# Custom data loader 
class CustomDataset(Dataset):
    def __init__(self):

        self.image_base = relative_path +  r"Text-To-Structure-Hypernetworks\example\data\example_images\\"

        data_path = relative_path +  r"Text-To-Structure-Hypernetworks\example\data\example_set.json"
        self.list_of_images = pd.read_json(data_path)['ID'].to_list()

        file_path = relative_path + r"Text-To-Structure-Hypernetworks\example\data\example_descriptions.txt"
        
        with open(file_path, 'r') as file:
            self.description_template  = file.read().splitlines()


    def __len__(self):
        return len(self.list_of_images)

    def __getitem__(self, idx):

        image_name = self.list_of_images[idx]

        rand_desc = random.choice(self.description_template)

        # Load image 
        image = Image.open(self.image_base + image_name).convert('L')

        newsize = (369, 369)
        image = image.resize(newsize)

        image_array = np.array(image)
        height = 369
        width = 369

        x_coords = np.arange(width)
        y_coords = np.arange(height)

        xx, yy = np.meshgrid(x_coords, y_coords)
        coordinates_tensor = np.dstack((xx, yy))

        # Normalize coordinates_tensor 
        coordinates_tensor = coordinates_tensor / [width - 1, height - 1]

        in_out_tensor = (image_array != 255).astype(int)

        coordinates_tensor = torch.tensor(coordinates_tensor, dtype=torch.float32)
        in_out_tensor =  torch.tensor(in_out_tensor, dtype=torch.float32)

        in_out_tensor = in_out_tensor.reshape(in_out_tensor.size()[0]**2,1)
        coordinates_tensor = coordinates_tensor.reshape(coordinates_tensor.size()[0]**2,2)

        pos = image_name.replace('.png','').split('_')

        # To convert inches -> cm, multiply by 2.54
        rand_desc = rand_desc.replace('X',str(pos[0]) + ' X').replace('Y',str(pos[1]) + ' Y')

        # Note: Here we are simply taking the labels from the ground truth dataset
        # Practical implementations will include the trained BERT model here (NER extaction of design parameters)
        x_in = int(pos[0]) / 300
        y_in = int(pos[1]) / 300

        input_ = torch.tensor([x_in,y_in ], dtype=torch.float32)

        return {'input': input_, 'points': coordinates_tensor, 'labels': in_out_tensor, 'strings': rand_desc, 'targets': np.array(image) } 


class HyperNetwork(nn.Module):
    def __init__(self, num_classes, bert_output_size=2, fc_hidden_size=1024):
        super(HyperNetwork, self).__init__()
        self.num_classes = num_classes

        # Classifier layers
        self.fc1 = nn.Linear(bert_output_size, fc_hidden_size)
        self.fc2 = nn.Linear(fc_hidden_size, fc_hidden_size)
        self.fc3 = nn.Linear(fc_hidden_size, fc_hidden_size)
        self.fc4 = nn.Linear(fc_hidden_size, fc_hidden_size)
        self.fc5 = nn.Linear(fc_hidden_size, num_classes)

    def forward(self, input_):

        x = self.fc1(input_)
        x = nn.ReLU()(x)

        x = self.fc2(x)
        x = nn.ReLU()(x)

        x = self.fc3(x)
        x = nn.ReLU()(x)

        x = self.fc4(x)
        x = nn.ReLU()(x)

        x = self.fc5(x)
        x = nn.Tanh()(x)  

        return x

# Define the primary network
class PrimaryNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(PrimaryNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, 1)

    def forward(self, x, weights):
        x = torch.relu(nn.functional.linear(x, weights[0], weights[1]))
        x = torch.relu(nn.functional.linear(x, weights[2], weights[3]))
        x = torch.relu(nn.functional.linear(x, weights[4], weights[5]))
        x = torch.relu(nn.functional.linear(x, weights[6], weights[7]))
        x = torch.sigmoid(nn.functional.linear(x, weights[8], weights[9]))
        return x

###################
#   Parameters    #
###################

# Number of design criteria:
input_dim_hyper = 2 

# Problem dimension (2D/3D)
input_dim_prim = 2 

hidden_dim_primary = 128
hidden_dim_hyper = 256

# We only need one pass through to visualise output
batch_size = 1
num_epochs = 1

print("Creating networks")

primary_net = PrimaryNetwork(input_dim_prim, hidden_dim = 32)
primary_net.to(device)

# Count the number of trainable parameters in the primary network
total_params = sum(p.numel() for p in primary_net.parameters() if p.requires_grad)

# Instantiate the model
hypernet = HyperNetwork(total_params).to(device)

# Define the loss function
criterion_hypernet = WeightedBCELoss(1)

print("Creating datasets")

dataset = CustomDataset()
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122) 

losses = []
val_losses = []

torch.autograd.set_detect_anomaly(True)  

# Generate 2D XY data
x = np.linspace(0, 1, 369)
y = np.linspace(0, 1, 369)
X, Y = np.meshgrid(x, y)

grid_points = np.column_stack((X.flatten(), Y.flatten()))
grid_points = torch.tensor(grid_points, dtype = torch.float32).to(device)

save_path = relative_path + r"Text-To-Structure-Hypernetworks\example\weights\hypernetwork_DE_2D.pth"

# Load state 
hypernet.load_state_dict(torch.load(save_path))

num_batches = dataloader.__len__()

print("Generating example visuals")

for epoch in range(num_epochs):

    accumulated_loss = 0

    for step,batch in enumerate(dataloader):

        rec_losses = []
    
        points = batch['points'].to(device)
        labels = batch['labels'].to(device)
        input_= batch['input'].to(device)
        strings = batch['strings']
        target_images = batch['targets']

        weights = hypernet(input_)

        for j in range(len(weights)):

            weights_ = weights[j]
            weights_ = weights_.view(1, -1)

            points_ = points[j]
            targets = labels[j]

            primary_net_weights = []

            weight_index = 0
            for name, param in primary_net.named_parameters():
                if 'weight' in name:
                    primary_net_weights.append(
                        weights_[:, weight_index:weight_index + param.numel()].reshape(param.size()))
                    weight_index += param.numel()
                elif 'bias' in name:
                    primary_net_weights.append(weights_[:, weight_index:weight_index + param.numel()])
                    weight_index += param.numel()

            # Forward pass through the primary network
            output = primary_net(points_, primary_net_weights)
            heatmap_data = output.cpu().reshape(369, 369).detach().numpy() 

            fig.suptitle(strings[j], fontsize=9)

            # Create the prediction heatmap 
            ax1.cla()
            ax1.set_title('Hypernetwork Output')
            ax1.imshow(heatmap_data, cmap='gray', aspect='auto')  

            ax2.cla()
            ax2.set_title('Target Output')
            ax2.imshow(target_images[j], cmap='gray', aspect='auto')  
           
            ax1.set_aspect('equal', adjustable='box')
            ax2.set_aspect('equal', adjustable='box')

            # Animate
            plt.pause(2)

print("Finished visualisation")