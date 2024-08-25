import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def demo(path):

    dataset = Dataset()
    dataset.load_from_path(path)

    print(f"Shape of dataset.data: {dataset.data.shape}")
    print(f"Length of the dataset object: {len(dataset)}")
    print(dataset.column_names)
    print(dataset.labels)
    print(dataset)

    two_features = [0, 3]
    dataset.plot_scatter('plot1.png', two_features)

    for i, (observation, label) in enumerate(dataset):
        print(observation, label)
        if i == 3:
            break

    for label in ["Iris-Error", "Iris-setosa"]:
        print(f"filter by label: {label}")

        try:
            sub_dataset = dataset.filter_by_label(label)
            print(sub_dataset)
            sub_dataset.plot_scatter(f"plot_{label}.png", two_features)
        except KeyError:
            print(f"no such label")



class Dataset:
    def __init__(self, data=None, labels=None, column_names=None):
        self.data = data
        self.labels = labels
        self.column_names = column_names

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
        return f"A dataset object with {len(self.data)} observations, {len(self.column_names)} features and {self.num_classes} classes"

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
        print(f"filter by label: {value}")
        indices = np.where(self.labels == value)[0]
        print(f"filter by label: {indices}")
        if len(indices) == 0:
            raise KeyError(f"Label '{value}' not found in the dataset")
        return Dataset(self.data[indices], self.labels[indices], self.column_names)

    def plot_scatter(self, path, feature_indices):
        x = self.data[:, feature_indices[0]]
        y = self.data[:, feature_indices[1]]
        colors = np.random.rand(self.num_classes, 3)
        plt.scatter(x, y, c=colors)
        plt.xlabel(self.column_names[feature_indices[0]])
        plt.ylabel(self.column_names[feature_indices[1]])
        plt.savefig(path)