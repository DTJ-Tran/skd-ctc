from jiwer import wer
from transformers import Wav2Vec2Processor, Wav2Vec2FeatureExtractor, HubertConfig, HubertModel
import torch, json, os, librosa, transformers, gc
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from pyctcdecode import build_ctcdecoder
import pandas as pd
from tqdm import tqdm
import warnings
from torch.utils.data import Dataset
import numpy as np
from dataloader import ASR_Dataset
from dataloader import text_to_tensor
from models import Hubert
from torch.optim.lr_scheduler import CosineAnnealingLR
from comet_ml import Experiment
from comet_ml import start
from comet_ml.integration.pytorch import log_model

feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, padding_side='right', do_normalize=True, return_attention_mask=False)
min_wer = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# torch.autograd.set_detect_anomaly(True)
def scheduling_func(e, E=200, t=0.3):
    return min(max((e-1)/(E-1), t), 1-t)

def collate_fn(batch):
    with torch.no_grad():
        sr = 16000
        max_col = [-1] * 2  # Random Initialization (it should be the index of the longest audio file)
        target_length = []
        for row in batch:
          if row[0].shape[0] > max_col[0]: 
              max_col[0] = row[0].shape[0]
          if len(row[1]) > max_col[1]:
              max_col[1] = len(row[1])
        cols = {'waveform':[], 'transcript':[], 'outputlengths':[]}
        padding_element = ''
        for row in batch: # Padding & Getting multiple Waves
            pad_wav = np.concatenate([row[0], np.full( (max_col[0] - row[0].shape[0]), padding_element) ])  # Padding = the_longest - current-audio-size
            cols['waveform'].append(pad_wav)
            cols['outputlengths'].append(len(row[1]))
            row[1].extend([0] * (max_col[1] - len(row[1])))
            cols['transcript'].append(row[1])
        
        inputs = feature_extractor(cols['waveform'], sampling_rate = 16000)
        input_values = torch.tensor(inputs.input_values, device=device)
        cols['transcript'] = torch.tensor(cols['transcript'], dtype=torch.long, device=device) # Padd the silence to the Transcript
        cols['outputlengths'] = torch.tensor(cols['outputlengths'], dtype=torch.long, device=device)
    
    return input_values, cols['transcript'], cols['outputlengths']

#dataset should contain 2 cols, 1 is Path contain absolute path of audio and 1 is Transcript contain text transcript of audio 
current_dir = os.getcwd()
print("current_dir: ", current_dir)
target_train = os.path.join(current_dir, "skd-ctc", "dataset", "train.csv")
target_dev = os.path.join(current_dir, "skd-ctc", "dataset", "dev.csv")
df_train = pd.read_csv(target_train)
df_dev = pd.read_csv(target_dev)



from comet_ml import start
from comet_ml.integration.pytorch import log_model

experiment = start(
  api_key="Raze72byBUZdn5cWyLf3IFZ4h",
  project_name="mdd-experiment-with-hubert",
  workspace="dtj-tran"
)


train_dataset = ASR_Dataset(df_train)
batch_size = 8
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
# Grouping 2 data-point 
# Pytorch: https://www.youtube.com/watch?v=tHL5STNJKag
model = Hubert.from_pretrained(
    'facebook/hubert-base-ls960',
)



model = model.to(device)

phoneme_list = ['ae ', 'm ', 'k ', 'eh* ', 'n ', 'aw ', 'ao* ', 'iy ', 'er* ', 'z* ', 'uw* ', 'f ', 'p ', 'd* ', 'ao ', 'l* ', 'uw ', 'hh* ', 't ', 'ah* ', 'y* ', 'n* ', 'th ', 'hh ', 'err ', 'uh* ', 'p* ', 'zh ', 'k* ', 'eh ', 'ow* ', 'ay ', 'w ', 'ey ', 'aw* ', 'l ', 'zh* ', 'ih ', 'v ', 'oy ', 'aa* ', 't* ', 'jh ', 'b* ', 'w* ', 'ow ', 'ng ', 'b ', 'ch ', 'dh* ', 'y ', 'er ', 'v* ', 'ah ', 'sh ', 'aa ', 'g ', 'd ', 'dh ', 'r* ', 'ae* ', 'ey* ', 'uh ', 'r ', 'g* ', 's ', 'z ', 'jh* ', '']
decoder_ctc = build_ctcdecoder(
                              labels = phoneme_list,
                              )

num_epoch=20 # initial 200
temperature = 1
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
warmup_steps = num_epoch//10
scheduler = CosineAnnealingLR(optimizer, T_max=num_epoch - warmup_steps)
ctc_loss = nn.CTCLoss(blank = 68)

hyper_params = {
  "batch_size" : batch_size,
  "num_epoch" : num_epoch,
  "learning_rate -lr ": 3e-5,
  "temperature" : temperature,
  "warmup_steps" : warmup_steps,
  "min_wer" : min_wer
}

experiment.log_parameters(hyper_params)

for epoch in range(num_epoch):
  #freeze model first 12.5% of the steps except linear
  if epoch < num_epoch//8:
    model.freeze()
  else:
    model.unfreeze()
  model.train().to(device)
  experiment.set_epoch(epoch)  # Track the current epoch
  running_loss = []
  print(f'EPOCH {epoch}:')
  for i, data in tqdm(enumerate(train_loader)):
    acoustic, labels, target_lengths = data # Collate_fn ref
    i_logits, logits= model(acoustic)
    
    # Apply clamping to prevent extreme values that cause NaNs
    logits = torch.clamp(logits, min=-10, max=10)
    i_logits = torch.clamp(i_logits, min=-10, max=10)
    

    #skd
    teacher_logits_detached = logits.clone().detach() #Stop gradients for the teacher's logits
    l_skd = F.kl_div(F.log_softmax(i_logits/temperature, dim=2), F.softmax(teacher_logits_detached/temperature, dim=2))

    logits = logits.transpose(0,1)
    i_logits = i_logits.transpose(0,1)
    input_lengths = torch.full(size=(logits.shape[1],), fill_value=logits.shape[0], dtype=torch.long, device=device)
    logits = F.log_softmax(logits, dim=2)
    i_logits = F.log_softmax(i_logits, dim=2)
   
    #ctc and ictc
    l_ctc = ctc_loss(logits, labels, input_lengths, target_lengths)
    l_ictc = ctc_loss(i_logits, labels, input_lengths, target_lengths)

    #alpha and total loss
    alpha = scheduling_func(e=epoch+1, E=num_epoch, t=0.3)
    loss = (1-alpha)*l_ctc + alpha*(l_ictc + l_skd)

    running_loss.append(l_ictc.item())
    loss.backward() 
    optimizer.step() 

    # Log loss after every batch
    experiment.log_metric("batch_loss", loss.item())
    # linear warmup lr
    if epoch < warmup_steps:
      lr = 3e-5 * (epoch + 1) / warmup_steps
      for param_group in optimizer.param_groups:
          param_group['lr'] = lr  
    else:
      scheduler.step()
    optimizer.zero_grad()


  # Log average training loss for the epoch
  avg_loss = sum(running_loss) / len(running_loss)
  experiment.log_metric("epoch_loss", avg_loss)

  print(f"Training loss: {sum(running_loss) / len(running_loss)}")
  if sum(running_loss) / len(running_loss)<=1: #ensure decode fast
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
        # print(hypothesis)
        error = wer(transcript, hypothesis)
        worderrorrate.append(error)
      epoch_wer = sum(worderrorrate)/len(worderrorrate)

      # Log WER to Comet
      experiment.log_metric("WER", epoch_wer) # Word Err Rate

      if (epoch_wer < min_wer):
        print("save_checkpoint...")
        min_wer = epoch_wer
        # Save model locally
        checkpoint_path = 'checkpoint/checkpoint.pth'
        torch.save(model.state_dict(), checkpoint_path)

        experiment.log_model(
          name="Best_MDD_Model",
          file_or_folder=checkpoint_path,
          metadata={"WER": min_wer, "epoch": epoch}
        )

      print("wer checkpoint " + str(epoch) + ": " + str(epoch_wer))
      print("min_wer: " + str(min_wer))

log_model(experiment, model=model, model_name="MDD_MODEL")
# End the experiment
experiment.end()
