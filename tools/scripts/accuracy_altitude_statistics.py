#!/usr/bin/env python3
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import DatabaseHelper
import Predictor
import seaborn as sns
import numpy as np
import math
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd


def quantiles():
    d = DatabaseHelper.DatabaseHelper()
    result = d.run_specific_query("SELECT accuracy FROM location")
    accuracies = [row[0] for row in result]

    df = pd.DataFrame({'accuracy': accuracies})
    q = df.quantile([0.0, 0.25, 0.5, 0.75, 1.0])
    means = df.mean()
    std = df.std()
    print(q)
    print("mean: {}".format(means))
    print("std: {}".format(std))


if __name__ == '__main__':
    quantiles()
    