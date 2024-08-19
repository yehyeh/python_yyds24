# Press ⌃R to execute
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import csv

def create_df_from_csv(filename):
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)

csv_name = "Iris.csv"

df = create_df_from_csv(csv_name)
