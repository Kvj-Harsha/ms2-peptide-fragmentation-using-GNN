"""Reference baselines: Global mean-profile and Bag-of-AA MLP."""
from .global_mean import GlobalMeanBaseline
from .bag_of_aa import BagOfAAMLP

__all__ = ["GlobalMeanBaseline", "BagOfAAMLP"]
