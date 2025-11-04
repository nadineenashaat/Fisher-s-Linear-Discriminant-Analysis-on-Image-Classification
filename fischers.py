import numpy as np
from PIL import Image
import matplotlib.image as image
import cv2
import os
from os import listdir
from numpy import asarray
import seaborn as sns
import patoolib

np.set_printoptions(threshold=np.inf)

# Install and extract dataset
!pip install patool
patoolib.extract_archive('/content/Data.zip')

# Load and flatten training images
X = []
for i in range(1, 2401):
    img = image.imread(f'/content/Data/Train/{i}.jpg')
    X.append(img.flatten())
X = np.array(X)

# Initialize class and non-class containers
classes = [[] for _ in range(10)]
notclasses = [[] for _ in range(10)]

# Split data into 10 classes (240 images each) 
for i in range(2400):
    label = i // 240
    for j in range(10):
        if j == label:
            classes[j].append(X[i])
        else:
            notclasses[j].append(X[i])

# Convert lists to NumPy arrays
classes = [np.array(c) for c in classes]
notclasses = [np.array(nc) for nc in notclasses]

# Compute class means and non-class means
means2 = [np.mean(c, axis=0) for c in classes]    
means1 = [np.mean(nc, axis=0) for nc in notclasses]  

# Compute within-class scatter matrices for each class
SWj1 = []
for i in range(10):
    scatter = np.zeros((784, 784))
    for x in classes[i]:
        diff = (x - means2[i]).reshape(784, 1)
        scatter += diff @ diff.T
    SWj1.append(scatter)

# Compute non-class scatter matrices
SWj2 = []
for i in range(10):
    scatter = np.zeros((784, 784))
    for x in notclasses[i]:
        diff = (x - means1[i]).reshape(784, 1)
        scatter += diff @ diff.T
    SWj2.append(scatter)

# Total within-class scatter matrix
SW = [SWj1[i] + SWj2[i] for i in range(10)]

# Compute inverse scatter matrices
SWINV = [np.linalg.pinv(sw) for sw in SW]

# Compute Fisher weights for each class
weights = [(means1[i] - means2[i]) @ SWINV[i] for i in range(10)]

# Load and flatten test images
Test = []
for i in range(1, 201):
    img = image.imread(f'/content/Data/Test/{i}.jpg')
    Test.append(img.flatten())
Test = np.array(Test)

# Predict class for each test image using Fisher's criterion
result = np.ones(200)
for i in range(200):
    scores = [Test[i] @ w for w in weights]
    result[i] = np.argmin(scores)

# Load true labels
labels = np.genfromtxt("./Data/Test/Test Labels.txt")

# Evaluate model
from sklearn import metrics
import matplotlib.pyplot as plt

confusionmat = metrics.confusion_matrix(labels, result)
acc = metrics.accuracy_score(labels, result)

# Save confusion matrix plot
plt.savefig('./ConfusionNoBias.jpg')

# Display results
display(confusionmat, acc)

# Compute bias terms for decision boundaries
weights2 = []
for i in range(10):
    w = -weights[i].reshape(1, 784)
    midpoint = (means2[i] + means1[i]) / 2
    bias = w @ midpoint
    weights2.append(bias)
