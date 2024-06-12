from collections import defaultdict
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.interpolate import interp2d, RectBivariateSpline, griddata
import matplotlib.pyplot as plt
from scipy.stats import norm

import yfinance as yf
