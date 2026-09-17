import numpy as np
import torch

def predict_tracks(models, sequence_one_hot, slices):
    predicted_tracks = []
    for fold_ix in range(len(models)):
        with torch.no_grad():
            yh = models[fold_ix](sequence_one_hot[None, ...])[:, None, ...].numpy(force = True)[:,:,slices]
        predicted_tracks.append(yh)

    predicted_tracks = np.concatenate(predicted_tracks,axis=1).swapaxes(3,2)

    return predicted_tracks