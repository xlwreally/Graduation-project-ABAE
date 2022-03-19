# -*- coding: utf-8 -*-
import csv
import logging

import hydra
import numpy as np
import torch
import os

from model import ABAE
from reader import get_centroids, get_w2v, read_data_tensors

logger = logging.getLogger(__name__)
@hydra.main("configs", "config")
def main(cfg):
    w2v_model = get_w2v(os.path.join(hydra.utils.get_original_cwd(), cfg.embeddings.path))
    model=torch.load(r"C:\Users\xlw\Desktop\毕业设计\Graduation-project-ABAE\Phone\outputs\2022-03-13\23-02-13\abae_0.01_014000.bin")
    all=model.get_aspect_words(w2v_model, logger,topn=-1)
    for i in all:
        print(i)

if __name__ == "__main__":
     main()



