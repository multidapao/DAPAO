import numpy as np
from sklearn.cluster import KMeans
from torch import nn
import torchvision


def project(param_data, backup, epsilon):
    return backup + epsilon * (param_data - backup)


class DenseLoss(nn.Module):
    def __init__(self):
        super(DenseLoss, self).__init__()
        dense = torchvision.models.densenet169(pretrained=True)
        self.dense_loss = nn.Sequential(
            dense.features.conv0, dense.features.norm0, dense.features.relu0,
        )

    def forward(self, image):
        return self.dense_loss(image)


def cluster_channels(feature_map, k=4):
    feature_map = feature_map.detach().cpu()
    channels = feature_map.squeeze(0).view(feature_map.shape[0], -1).numpy()
    return KMeans(n_clusters=k, random_state=0, init="k-means++").fit_predict(channels)


def select_channel_weights(feature_map, clusters=4, selected_clusters=2):
    """Select the least-populated channel clusters from one donor's features."""
    if feature_map.ndim != 3:
        raise ValueError("Expected a single CxHxW feature map")
    channel_count = feature_map.shape[0]
    if not 1 <= selected_clusters < clusters <= channel_count:
        raise ValueError("Require 1 <= selected-clusters < clusters <= channel count")
    labels = cluster_channels(feature_map, k=clusters)
    unique_labels, counts = np.unique(labels, return_counts=True)
    if len(unique_labels) <= selected_clusters:
        raise ValueError("Too few distinct feature clusters for a nonempty selection and complement")
    count_by_label = dict(zip(unique_labels, counts))
    selected_labels = sorted(count_by_label, key=count_by_label.get)[:selected_clusters]
    indices = np.flatnonzero(np.isin(labels, selected_labels))
    weights = feature_map.new_zeros(channel_count)
    weights[indices] = 1.0
    return weights.view(1, -1, 1, 1), indices
