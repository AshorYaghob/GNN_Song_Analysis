import torch
import pandas as pd
import numpy as np
import os
import json
from pathlib import Path as Data_Path
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv, to_hetero
import torch.nn.functional as F #for activation function
from torch import Tensor
from torch_geometric.loader import LinkNeighborLoader
import tqdm
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score



#loads your json files (playlist data and returns the data as a list)
#TODO
#Potentially remove the "info" key from each folder before merging, Not info we need
# Function to load a specified number of JSON files and extract only the playlists
def load_json_files(folder_path, file_names, num_files):
    all_playlists = []
    files_to_load = file_names[:num_files]  # Get the first 'num_files' file names

    for file_name in files_to_load:
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
            all_playlists.extend(data['playlists'])  # Append only the playlists to the list
    
    return all_playlists


folder_path ='data'
folder_contents = os.listdir(folder_path)

DATA_PATH = Data_Path(folder_path)

#List of all the names of the files in our Data Folder
file_names = [f for f in os.listdir(DATA_PATH) if os.path.isfile(os.path.join(DATA_PATH, f)) and f.endswith('.json')]

# Specify the number of files to load
num_files_to_load = 5

# Load the specified number of JSON files
loaded_playlists = load_json_files(folder_path, file_names, num_files_to_load)

# Example of accessing loaded data
print(f"Total number of playlists loaded: {len(loaded_playlists)}")

# Example of accessing a playlist from the loaded data
"""
if loaded_playlists:
    print("Example playlist:", loaded_playlists[0])
"""

#Now we are creating a Dataframe consisting of playlist id's with corresponding track id's.
data_list = []

# Loop through playlists and tracks to create the data list
for playlist in loaded_playlists:
    for track in playlist['tracks']:
        data_list.append([playlist['pid'], track['track_uri']])

# Create a DataFrame
complete_df = pd.DataFrame(data_list, columns=['pid', 'track_uri'])

# Create a mapping from unique playlist id to range [0, num_playlists)
unique_playlist_id = complete_df['pid'].unique()
unique_playlist_id = pd.DataFrame(data={'playlistID': unique_playlist_id, 'mappedID': pd.RangeIndex(len(unique_playlist_id))})

# Create a mapping from unique track id to range [0, num_tracks)
unique_track_id = complete_df['track_uri'].unique()
unique_track_id = pd.DataFrame(data={'trackID': unique_track_id, 'mappedID': pd.RangeIndex(len(unique_track_id))})

# Merge the playlist and track dataframes with our mapped values and make the mapped column a tensor
complete_pid = pd.merge(complete_df['pid'], unique_playlist_id, left_on='pid', right_on='playlistID', how='left')
complete_pid = torch.from_numpy(complete_pid['mappedID'].values)
complete_tid = pd.merge(complete_df['track_uri'], unique_track_id, left_on='track_uri', right_on='trackID', how='left')
complete_tid = torch.from_numpy(complete_tid['mappedID'].values)

# Construct our `edge_index` in COO format following PyG semantics
edge_index_playlist_to_track = torch.stack([complete_pid, complete_tid], dim=0)
print(edge_index_playlist_to_track.shape)
print(edge_index_playlist_to_track)

# Initialize Heterogeneous Data Structure
data = HeteroData()

# Save node indices
data["playlist"].node_id = torch.arange(len(unique_playlist_id))
data["track"].node_id = torch.arange(len(unique_track_id))

# Add edges
edge_index_playlist_to_track = torch.stack([complete_pid, complete_tid], dim=0)
data["playlist", "has", "track"].edge_index = edge_index_playlist_to_track

# Convert to undirected graph
data = T.ToUndirected()(data)

# Split edges into training, validation, and testing sets
transform = T.RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    disjoint_train_ratio=0.3,
    neg_sampling_ratio=2.0,
    add_negative_train_samples=False,
    edge_types=("playlist", "has", "track"),
    rev_edge_types=("track", "rev_has", "playlist"),
)

train_data, val_data, test_data = transform(data)
#TODO
#see if we should apple RELU to the second layer as well.
#Can perform testing to see if that improves performance.
#If we want more layers we can also just put the conv line in a while loop
class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x
# Our final classifier applies the dot-product between source and destination
# node embeddings to derive edge-level predictions:
class Classifier(torch.nn.Module):
    def forward(self, x_playlist: Tensor, x_track: Tensor, edge_label_index: Tensor) -> Tensor:
        # Convert node embeddings to edge-level representations:
        edge_feat_playlist = x_playlist[edge_label_index[0]]
        edge_feat_track = x_track[edge_label_index[1]]
        # Apply dot-product to get a prediction per supervision edge:
        return (edge_feat_playlist * edge_feat_track).sum(dim=-1)

class Model(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        # Since the dataset does not come with rich features, we also learn two
        # embedding matrices for users and tracks:
        self.track_lin = torch.nn.Linear(20, hidden_channels)
        self.playlist_emb = torch.nn.Embedding(data["playlist"].num_nodes, hidden_channels)
        self.track_emb = torch.nn.Embedding(data["track"].num_nodes, hidden_channels)
        # Instantiate homogeneous GNN:
        self.gnn = GNN(hidden_channels)
        print(self.gnn)
        # Convert GNN model into a heterogeneous variant:
        self.gnn = to_hetero(self.gnn, metadata=data.metadata())
        self.classifier = Classifier()
    def forward(self, data: HeteroData) -> Tensor:
        x_dict = {
          "playlist": self.playlist_emb(data["playlist"].node_id),
          "track": self.track_emb(data["track"].node_id),
          # self.track_lin(data["track"].x) +
        }
        # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        pred = self.classifier(
            x_dict["playlist"],
            x_dict["track"],
            data["playlist", "has", "track"].edge_label_index,
        )
        return pred

model = Model(hidden_channels=64)

# Define seed edges:
edge_label_index = train_data["playlist", "has", "track"].edge_label_index
edge_label = train_data["playlist", "has", "track"].edge_label

# TODO: we can play around with how we sample neighbours
# num_neighbours, neg_sampling_ratio
# In the first hop, we sample at most 20 neighbors.
# In the second hop, we sample at most 10 neighbors.
# In addition, during training, we want to sample negative edges on-the-fly with
# a ratio of 2:1.
# We can make use of the `loader.LinkNeighborLoader` from PyG:
train_loader = LinkNeighborLoader(
    data=train_data,
    num_neighbors=[10, 5],
    neg_sampling_ratio=2.0,
    edge_label_index=(("playlist", "has", "track"), edge_label_index),
    edge_label=edge_label,
    batch_size=512,
    shuffle=True,
)

# Inspect a sample:
sampled_data = next(iter(train_loader))
print(sampled_data)

criterion = torch.nn.BCEWithLogitsLoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: '{device}'")
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# best_vloss = 1_000_000
#TODO
#Consider a metric to stop the epochs at an optimal value for loss
#we dont want to underfit or overfit our model
#But just performing 4 loops cant be optimal
for epoch in range(1, 9):
    total_loss = total_examples = 0
    for sampled_data in tqdm.tqdm(train_loader):
        optimizer.zero_grad()
        sampled_data.to(device)
        pred = model(sampled_data)
        ground_truth = sampled_data["playlist", "has", "track"].edge_label
        # loss = F.binary_cross_entropy_with_logits(pred, ground_truth) #same as criterion func
        loss = criterion(pred, ground_truth)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * pred.numel()
        total_examples += pred.numel()
    print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")


# Define the validation seed edges:
edge_label_index = val_data["playlist", "has", "track"].edge_label_index
edge_label = val_data["playlist", "has", "track"].edge_label

val_loader = LinkNeighborLoader(
    data=val_data,
    num_neighbors=[10, 5],
    # num_neighbors=[20, 10],
    edge_label_index=(("playlist", "has", "track"), edge_label_index),
    edge_label=edge_label,
    batch_size=512,
    # batch_size=3 * 128,
    shuffle=False,
)

sampled_data = next(iter(val_loader))

assert sampled_data["playlist", "has", "track"].edge_label_index.size(1) == 512
assert sampled_data["playlist", "has", "track"].edge_label.min() >= 0
assert sampled_data["playlist", "has", "track"].edge_label.max() <= 1


preds = []
ground_truths = []
for sampled_data in tqdm.tqdm(val_loader):
    with torch.no_grad():
        sampled_data.to(device)
        preds.append(model(sampled_data))
        ground_truths.append(sampled_data["playlist", "has", "track"].edge_label)

pred = torch.cat(preds, dim=0).cpu().numpy()
ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
auc = roc_auc_score(ground_truth, pred)
print()
print(f"Validation AUC: {auc:.4f}")

# Define the test seed edges:
edge_label_index = test_data["playlist", "has", "track"].edge_label_index
edge_label = test_data["playlist", "has", "track"].edge_label

test_loader = LinkNeighborLoader(
    data=test_data,
    num_neighbors=[10, 5],
    edge_label_index=(("playlist", "has", "track"), edge_label_index),
    edge_label=edge_label,
    batch_size=512,
    shuffle=False,
)

# Evaluate on test data
test_preds = []
test_ground_truths = []
for sampled_data in tqdm.tqdm(test_loader):
    with torch.no_grad():
        sampled_data.to(device)
        test_preds.append(model(sampled_data))
        test_ground_truths.append(sampled_data["playlist", "has", "track"].edge_label)

test_pred = torch.cat(test_preds, dim=0).cpu().numpy()
test_ground_truth = torch.cat(test_ground_truths, dim=0).cpu().numpy()
test_auc = roc_auc_score(test_ground_truth, test_pred)
print()
print(f"Test AUC: {test_auc:.4f}")

