import random
import matplotlib.pyplot as plt

from matplotlib.pyplot import figure

def visualize_image(dataset):
    samples = random.sample(range(1, len(dataset)), 1)

    for i in samples:
        figure(figsize=(20, 15), dpi=80)

        image, mask = dataset[i]
        image = image.permute(1, 2, 0)
        mask = mask.permute(1, 2, 0)

        plt.subplot(1, 2, 1)
        plt.imshow(image, vmin=-.2, vmax=.2, interpolation='nearest',)

        plt.subplot(1, 2, 2)
        plt.imshow(mask)