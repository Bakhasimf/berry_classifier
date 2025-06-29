import splitfolders
import os

base_dir = os.path.dirname(__file__)
raw_path = os.path.join(base_dir, "..", "..", "data", "raw", "Berries_Fruit-262")

splitfolders.ratio(raw_path, output=os.path.join(base_dir, "..", "..", "data", "processed", "split_Berries_Fruit-262"), seed=42, ratio=(.7, .2, .1))

