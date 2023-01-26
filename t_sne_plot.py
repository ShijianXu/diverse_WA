import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import numpy as np



# with open('feature_usps_2_mnist_before_adapt.npy', 'rb') as f:
with open('feature_m2s_single_adapted.npy', 'rb') as f:
    features = np.load(f)
with open('label_m2s_single_adapted.npy', 'rb') as f:
    labels = np.load(f)

print(features.shape)
print(labels.shape)

# Perform t-SNE
print("Peforming t-SNE")
tsne_features = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=20).fit_transform(features)

print(tsne_features.shape)

# Plot the results using matplotlib
plt.scatter(tsne_features[:,0], tsne_features[:,1], c=labels)
# plt.savefig('./figures/tsne_svhn_resnet18_imagenet_before_adaptation.png')
plt.savefig('./figures/tsne_m2s_single_adapted.png')