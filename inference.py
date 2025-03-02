from jiwer import wer
from transformers import Wav2Vec2Processor, Wav2Vec2FeatureExtractor, HubertConfig, HubertModel
import torch, json, os, librosa, transformers, gc
import torch.nn as nn
import torch.nn.functional as F
from pyctcdecode import build_ctcdecoder
import pandas as pd
from tqdm import tqdm
import warnings
import numpy as np
from models import Hubert
import time

feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, padding_side='right', do_normalize=True, return_attention_mask=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

gc.collect()
df_dev = pd.read_csv("./skd-ctc/dataset/test.csv")

save_path = "./skd-ctc/result.csv"
model = Hubert.from_pretrained(
    'facebook/hubert-base-ls960',
)
model.load_state_dict(torch.load("./model.pth/model-data/comet-torch-model.pth"))


# for param in model.feature_extractor.parameters():
#     param.requires_grad = False

model.freeze()
model = model.to(device)

PATH = []
TRANSCRIPT = []
PREDICT = []

phoneme_list = [
    "ae", "m", "k", "eh*", "n", "aw", "ao*", "iy", "er*", "z*", 
    "uw*", "f", "p", "d*", "ao", "l*", "uw", "hh*", "t", "ah*", 
    "y*", "n*", "th", "hh", "err", "uh*", "p*", "zh", "k*", "eh", 
    "ow*", "ay", "w", "ey", "aw*", "l", "zh*", "ih", "v", "oy", 
    "aa*", "t*", "jh", "b*", "w*", "ow", "ng", "b", "ch", "dh*", 
    "y", "er", "v*", "ah", "sh", "aa", "g", "d", "dh", "r*", "ae*", 
    "ey*", "uh", "r", "g*", "s", "z", "jh*", " "
]
decoder_ctc = build_ctcdecoder(
                              labels = phoneme_list,
                              )

time_start = time.time()
with torch.no_grad():
  model.eval().to(device)
  worderrorrate = []
  for point in tqdm(range(len(df_dev))):
    acoustic, _ = librosa.load(df_dev['Path'][point], sr=16000)
    acoustic = feature_extractor(acoustic, sampling_rate = 16000)
    acoustic = torch.tensor(acoustic.input_values, device=device)
    transcript = df_dev['Transcript'][point]

    logits, _ = model(acoustic)
    logits = F.log_softmax(logits.squeeze(0), dim=1)
    x = logits.detach().cpu().numpy()
    hypothesis = decoder_ctc.decode(x).strip()
    PREDICT.append(hypothesis.strip())
    PATH.append(df_dev['Path'][point])
    TRANSCRIPT.append(df_dev['Transcript'][point])
time_end = time.time()

print(time_end-time_start)

train = pd.DataFrame([PATH, TRANSCRIPT, PREDICT])
train = train.transpose()
train.columns=['Path', 'Transcript', 'Predict'] 
train.to_csv(save_path)

# Currently: THIS IS ASR (currently using text (character)) ->  (to the Phoneme)
# Suggestion: Print the Dim to understanf the behavior
# THE GOAL - Using the data
# Deadline : Monday 
# Discussion with Khanh 

"""
https://github.com/huutuongtu/Refining-Linguistic-Information-Utilization-MDD - the Code
https://drive.google.com/file/d/1UnXgktrlF8ORT-rMey_bitPOlbaP1ond/view?usp=sharing - The Dataset
Using HuBert (the skd-ctc)

Dataset: 1 (L2-) &  TIMIT (Check the CSV)

=> Paper: https://arxiv.org/pdf/2110.07274

"""