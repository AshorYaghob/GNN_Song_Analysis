# Song Analysis Using a Graph Neural Network

## Description
We use graph neural networks and data from spotifies Million Playlist Dataset to train a model that will predict songs for playlists. The dataset can be found here: https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge. Included is a few slices of that dataset, 5 slices of 1000 playlists each. If you have a powerful computer we advise you to use as much data as you can to train the best model. 

## Requirements
- Python 3.8 to 3.11

## Setup

### Creating and Activating a Virtual Environment

#### On macOS/Linux
1. Open your terminal.
2. Navigate to your project directory:
   cd path/to/your/project
3. python3 -m venv venv
4. source venv/bin/activate

#### On Windows
1. Open your terminal.
2. Navigate to your project directory:
3. cd path\to\your\project
4. python -m venv venv
5. venv\Scripts\activate

### Installing Dependencies
Once your virtual environment is activated, type the following command to install the dependencies.
1. pip install -r requirements.txt

## Contributing
1. As of now we only use the track_id and playlist_id to make predictions, we don't use any attributes of the playlists and songs due to potential legal issues from spotify. If you would like to try you can add on to the code by using attributes of the songs and playlist as extra features for their embeddings. This current version creates random embeddings for the track and playlist nodes and learns them through the training process. 
2. If you would like you can also create a sample of data and remove a few edges from each playlist, then you can feed that to the model and see if it predicts those edges you removed.
3. Play around with the code and have fun. 


