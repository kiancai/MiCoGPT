import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from MiCoGPT.utils.mgm_evaluator import Evaluator
import os
import pickle
#from MiCoGPT.utils.corpus import MiCoGPTokenizer, MicroCorpus
from tqdm import tqdm

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'MiCoGPTokenizer':
            from MiCoGPT.utils.corpus import MiCoGPTokenizer
            return MiCoGPTokenizer
        return super().find_class(module, name)