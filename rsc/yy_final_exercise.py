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
    features = [0, 3]
    dataset.plot_scatter('plot1.png', features)
    for i, (observation, label) in enumerate(dataset):
        print(observation, label)
        if i == 3:
            break

    for label in ["Iris-Error", "Iris-setosa"]:
        print(f"filter by label: {label}")

        try:
            sub_dataset = dataset.filter_by_label(label)
            print(sub_dataset)
            sub_dataset.plot_scatter(f"plot_{label}.png", features)
        except KeyError:
            print(f"no such label")



class Dataset:
    def __init__(self, data=None, labels=None, column_names=None):
        self.data = data
        self.labels = labels
        self.column_names = column_names

    def load_from_path(self, path):
        df = pd.read_csv(path)

        self.data = df.iloc[:, 1:].values
        self.labels = df.iloc[:, 0].values
        self.column_names = df.columns[1:].tolist()

    def __len__(self):
        return len(self.data)

    def __str__(self):
        num_classes = len(set(self.labels))
        return f"A dataset object with {len(self.data)} observations, {len(self.column_names)} features and {num_classes} classes"

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

    def filter_by_label(self, label):
        if label not in self.labels:
            raise KeyError(f"Label '{label}' not found in the dataset")
        indices = np.where(self.labels == label)[0]
        return Dataset(self.data[indices], self.labels[indices], self.column_names)

    def plot_scatter(self, path, feature_indices):
        x = self.data[:, feature_indices[0]]
        y = self.data[:, feature_indices[1]]
        plt.scatter(x, y, c=self.labels)
        plt.xlabel(self.column_names[feature_indices[0]])
        plt.ylabel(self.column_names[feature_indices[1]])
        plt.savefig(path)