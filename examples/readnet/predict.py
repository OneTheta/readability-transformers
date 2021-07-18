import os
import pickle
import argparse
import shutil
import argparse
import pandas as pd
import torch
from torch import nn
from torch.nn import MSELoss
from easydict import EasyDict as edict
from typing import List, Union

from readability_transformers import ReadNet, models, readers, dataset, losses
from readability_transformers.features import SentenceLingFeatExtractor, DocumentLingFeatExtractor

def RMSELoss(yhat,y):
    return torch.sqrt(torch.mean((yhat-y)**2))

def readnet():
    device = "cuda"
    double = True
    count = 2
    batch_size = 64
    lr=1e-5
    gradient_accumulation=2

    # 0. Initialize ReadabilityTransformers
    model = ReadNet(
        model_name="checkpoints/readnet/prediction_2",
        device=device,
        double=double
    )

    passages = [
        "This model is a distilled version of the RoBERTa-base model. It follows the same training procedure as DistilBERT. The code for the distillation process can be found here. This model is case-sensitive: it makes a difference between english and English.",
        "BERT is a transformers model pretrained on a large corpus of English data in a self-supervised fashion. This means it was pretrained on the raw texts only, with no humans labelling them in any way (which is why it can use lots of publicly available data) with an automatic process to generate inputs and labels from those texts. ",
        "BERT is a transformers model pretrained on a large corpus of English data in a self-supervised fashion. This means it was pretrained on the raw texts only, with no humans labelling them in any way (which is why it can use lots of publicly available data) with an automatic process to generate inputs and labels from those texts. ",
        "BERT is a transformers model pretrained on a large corpus of English data in a self-supervised fashion. This means it was pretrained on the raw texts only, with no humans labelling them in any way (which is why it can use lots of publicly available data) with an automatic process to generate inputs and labels from those texts. ",
    ]

    predictions = model.predict(passages, batch_size=2)
    print(predictions)

      

    

if __name__ == "__main__":
    readnet()