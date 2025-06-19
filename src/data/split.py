import splitfolders
import os

# base_dir = os.path.dirname(__file__)
# base_dir = os.getcwd()
# raw_path = os.path.abspath(os.path.join(base_dir, "..", "..", "data", "raw", "Berries_Fruit-262"))
# processed_path = os.path.abspath(os.path.join(base_dir, "..", "..", "data", "processed", "Berries_Fruit-262"))
#
# print("RAW PATH:", raw_path)
# print("PROCESSED PATH:", processed_path)
# print("Содержимое папки RAW:", os.listdir(raw_path))


base_dir = os.path.dirname(__file__)
raw_path = os.path.join(base_dir, "..", "..", "data", "raw", "Berries_Fruit-262")

splitfolders.ratio(raw_path, output=os.path.join(base_dir, "..", "..", "data", "processed", "Berries_Fruit-262"), seed=42, ratio=(.7, .2, .1))

