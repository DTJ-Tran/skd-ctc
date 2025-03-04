import torch
from torch.utils.data import Dataset
import numpy as np
import json
import librosa
import os

file_path = "./skd-ctc/vocab.json"
dict_vocab = dict()
if not os.path.exists(file_path):
  print(f"Error: File '{file_path}' not found.")
else:
  with open(file_path) as f:
      dict_vocab = json.load(f)
key_list = list(dict_vocab.keys())
val_list = list(dict_vocab.values())

def text_to_tensor(string_text):
    text = string_text
    if not isinstance(string_text, str):  
        raise TypeError(f"Expected string, but got {type(string_text)} and got {string_text}")
    text_list = []
    # Split text into phonemes based on space
    phonemes = string_text.split(" ") 
    for phoneme in phonemes:
      if phoneme in dict_vocab:
        text_list.append(dict_vocab[phoneme])
      else:
        raise KeyError(f"Phoneme '{phoneme}' not found in dict_vocab")
    return text_list

class ASR_Dataset(Dataset):

    def __init__(self, data):
        self.len_data           = len(data)
        self.path               = list(data['Path'])
        self.transcript         = list(data['Transcript'])

    def __getitem__(self, index): # of 1 datapoint
        waveform, _ = librosa.load(self.path[index], sr=16000)
        transcript  = text_to_tensor(self.transcript[index])
        return waveform, transcript

    def __len__(self):
        return self.len_data
    
