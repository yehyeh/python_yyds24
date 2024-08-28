import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt


def demo(path):
    dataset = Dataset()
    dataset.load_from_path(path)

    print(f"Shape of dataset.data: {dataset.data.shape}")
    print(f"Length of the dataset object: {len(dataset)}")
    print(dataset)

    two_features = [0, 3]

    # Generate plot scatter:
    dataset.plot_scatter('plot1.png', two_features)

    # Run snippet:
    for i, (observation, label) in enumerate(dataset):
        print(observation, label)
        if i == 3:
            break

    # Filter by label, and generate plot scatter by class
    for label in ["Iris-Error", "Iris-setosa"]:
        print(f"filter by label: {label}")

        try:
            sub_dataset_tuple = dataset.filter_by_label(label)
            sub_dataset = Dataset(sub_dataset_tuple[0], sub_dataset_tuple[1], sub_dataset_tuple[2])
            print(sub_dataset)
            sub_dataset.plot_scatter(f"plot_{label}.png", two_features)
        except KeyError:
            print(f"no such label")


def generate_random_color():
    r = round(random.uniform(0.0, 1.0), 1)
    g = round(random.uniform(0.0, 1.0), 1)
    b = round(random.uniform(0.0, 1.0), 1)
    return r, g, b


class Dataset:
    def __init__(self, data=None, labels=None, column_names=None):
        self.data = data
        self.labels = labels
        self.column_names = column_names
        if self.labels is not None:
            self.num_classes = len(set(self.labels))
        else:
            self.num_classes = 0

    def load_from_path(self, path):
        df = pd.read_csv(path)
        self.column_names = df.columns[1:].tolist()
        # Ignore the label
        self.data = df.iloc[:, 1:len(self.column_names)].values
        self.labels = df.iloc[:, -1].values
        self.num_classes = len(set(self.labels))

    def __len__(self):
        return len(self.data)

    def __str__(self):
        return (f"A dataset object with {len(self.data)} observations,"
                f"{len(self.column_names)} features and {self.num_classes} classes")

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index < len(self.data):
            observation = self.data[self.index]
            label = self.labels[self.index]
            self.index += 1
            return observation, label
        raise StopIteration

    def filter_by_label(self, value):
        indices = np.where(self.labels == value)[0]
        print("indices: ", indices)
        if len(indices) == 0:
            raise KeyError(f"Label '{value}' not found in the dataset")
        print("self.labels[indices]: ", self.labels[indices])
        return self.data[indices], self.labels[indices], self.column_names

    def plot_scatter(self, path, feature_indices):
        x = self.data[:, feature_indices[0]]
        y = self.data[:, feature_indices[1]]
        colors = self.colors_by_label

        plt.scatter(x, y, c=colors)
        plt.xlabel(self.column_names[feature_indices[0]])
        plt.ylabel(self.column_names[feature_indices[1]])
        plt.savefig(path)
        plt.clf()

    @property
    def colors_by_label(self):
        unique_labels = set(self.labels)

        colors_dict = {}
        for label in unique_labels:
            colors_dict[label] = generate_random_color()

        print("colors_dict: ", colors_dict)

        colors_list = [colors_dict[label] for label in self.labels]
        print("colors_list: ", colors_list)

        return colors_list
