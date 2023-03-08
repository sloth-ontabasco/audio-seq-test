import torch
import torchaudio 
import torchaudio.transforms as T
import torch.nn.functional as F
from torch import nn
from torchsummary import summary
from torch.utils.data import DataLoader , Dataset
from tqdm import tqdm
import numpy as np
import pandas as pd
import csv
import os
import time as t
import matplotlib.pyplot as plt
import matplotlib 

LOSS_VALUES = []
BATCH_SIZE = 32
EPOCHS  = 1
LR = 0.001
ANNOTATIONS_FILE = "C:\\Users\\nagav\\OneDrive\\Documents\\UrbanSound8K.tar\\UrbanSound8K\\metadata\\UrbanSound8K.csv"
AUDIO_DIR = "C:\\Users\\nagav\\OneDrive\\Documents\\UrbanSound8K.tar\\UrbanSound8K\\audio"
SAMPLE_RATE = 22050*4 
NUM_SAMPLES = 22050*4

INP = 64*173     #INPUTS 
HID = BATCH_SIZE     #NUMBER OF HIDDEN LAYERS 
NL  = 16     #NUMBER OF LAYERS 
NU  = 2     #NUMBER OF UNITS 
NCLS = 10   #NUMBER OF CLASSES
NH = 16

PT_FILE = "rnnV2.pth"


#creating a custom datasets
#getdatasets
#create data loader
#build a model
#train
#save trained model

class UrbanSoundDataset(Dataset) :
    def __init__(self , annotation_file , audio_dir , transformation , target_sample_rate , num_samples , device = "cpu") :
        self.annotation = pd.read_csv(annotation_file)
        self.audio_dir = audio_dir
        self.device = device 
        self.transformation = transformation.to(self.device)#apply this mel-spectrogram to audio file that we are loading -> modifying the getitem method
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        
        
    def __len__(self) :
        return len(self.annotation)

    def __getitem__(self,index):
                                                        # lst[idx] == lst.__getitem__(idx)
                                                        #what do we really want here
                                                        #getting and loading the waveform of the audio sample associated to the certain index
                                                        #at the same time .., return the label associated with it
                                                        #getting the path

        audio_sample_path = self._get_audio_sample_path(index)  #this is a private method
        label = self._get_audio_sample_label(index)             #loading audio files

                                                        #use the load functionallity of the torchaudio
                                                        #this function returns signal -> waveform or time series and sample rate

        signal , sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        #this signal is Pytorch tensor(num_channels , samples) -> (2 , 16000) ->(1 , 16000)
        signal = self._resampleIfNecessary(signal, sr)
                                                        #We are unifying the process because the dataset contains mono audio samples they may be stero samples so they have two channels or they may have multiple channels
                                                        #we are not intrested in more than one channel
                                                        #take the initial signal which is loaded --> then mix it down to mono
        
        signal = self._mixDownIfNecessary(signal)
        #print(f"Before {signal.shape}")
        signal = self._cutIfNecessary(signal)
        signal = self._rightPadIfNecessary(signal)
        #print(f"After {signal.shape}")
                                                        #converting the signal to mel-spectrogram
        signal = self.transformation(signal)

                                                        #Changing the signal to fixed length before making transformations
                                                        #we have different durations in length the problem is most deep learning architectures have data fixed in its shape
                                                        #We need to ensure that the signal we load before we produce and process and extract the mal-spectorgram is consistent
                                                        #The number of samples we need to have in a process 
        
        return signal , label

    def _cutIfNecessary(self , signal) :
        if signal.shape[1] > self.num_samples :
            signal = signal[:,:self.num_samples]
        return signal

    def _rightPadIfNecessary(self , signal) :
        length_signal = signal.shape[1]
        if length_signal < self.num_samples :
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0 , num_missing_samples)
            signal = F.pad(signal , last_dim_padding)
        return signal 
            
        
    def _resampleIfNecessary(self, signal, sr) :
        if sr != self.target_sample_rate:
            resampler = T.Resample(sr , self.target_sample_rate).to(self.device)
            signal = resampler(signal)
        return signal

    def _mixDownIfNecessary(self,signal):
        if signal.shape[0] > 1 :                                                # if not mono signal then mix it down 
            signal = torch.mean(signal , dim = 0 , keepdim = True)
        return signal 
        


    def _get_audio_sample_path(self,index: int) :
        
        fold = f"fold{self.annotation.iloc[index ,5]}"                          #identify the fold
        path = os.path.join(self.audio_dir , fold , self.annotation.iloc[index , 0])
        return path

    def _get_audio_sample_label(self, index : int) :
        return self.annotation.iloc[index , 6]

class RNN_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_units , num_classes, num_heads):
        super(RNN_LSTM, self).__init__()
        self.num_layers = num_layers
        self.num_units = num_units
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        # Create multiple LSTM units
        self.flatten = nn.Flatten() 
        self.lstm_units = nn.ModuleList()
        self.lstm_units.append(nn.LSTM(input_size, hidden_size, num_layers, batch_first=True))
        for i in range(num_units-1):
            self.lstm_units.append(nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True))
        # Multi-Head Attention
        self.mha = nn.MultiheadAttention(hidden_size, num_heads=num_heads)
        # Fully connected layer
        self.fc1 = nn.Linear(hidden_size, 16)
        self.fc2 = nn.Linear(16, num_classes)
        
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.flatten(x)
        x = x.view(batch_size, -1, INP)
        # LSTM Layers
        output, (hn, cn) = self.lstm_units[0](x)
        for i in range(1, self.num_units):
            output, (hn, cn) = self.lstm_units[i](output)
        # Multi-Head Attention
        output, attn = self.mha(output, output, output)
        # Fully Connected Layers
        output = output.mean(dim=1) # Average the sequence length dimension
        output = self.fc1(output)
        output = F.relu(output)
        output = self.fc2(output)
        output = F.log_softmax(output, dim=1)
        return output


def create_data_loader(trData , bs) :
    return DataLoader(trData , batch_size = bs,shuffle = True,drop_last=True)

def train_single_epoch(model , data_loader , loss_fun , optimiser , device , scheduler) :
    for inp , tar in data_loader :
        inp , tar = inp.to(device) , tar.to(device)
        #calculate the loss
        #It looks like the input shape is (batch_size, sequence_length, input_size_1, input_size_2) = (128, 1, 64, 44).
        pred = model(inp.view(-1,64,173))
        loss = loss_fun(pred , tar)

        #propagate backwards
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        #scheduler.step(loss)
    LOSS_VALUES.append(loss.item())
    print(f"loss :{loss.item()} lr = {optimizer.param_groups[0]['lr']}")

def train(model , data_loader , loss_fun , optimiser , device , epochs , scheduler):

    for i in tqdm(range(epochs) , desc = "Training...,"):
        train_single_epoch(model , data_loader , loss_fun , optimiser , device ,scheduler)
    print("finished training...")




if __name__  == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device using :{device}")
    #RNN_LSTM__ = RNN_LSTM(INP, HID , NL , NU , NCLS , NH)
    #summary(RNN_LSTM__.to(device) , (INP, HID , NL , NU) )

    #instantiating our data set objects
    
                                                        
    mel_spectrogram = T.MelSpectrogram(
        sample_rate = SAMPLE_RATE,
        n_fft = 1024  ,                             
        hop_length = 512 ,                          
        n_mels = 64 
        )                                           
    
    usd = UrbanSoundDataset(
        ANNOTATIONS_FILE ,
        AUDIO_DIR ,
        mel_spectrogram ,
        SAMPLE_RATE ,
        NUM_SAMPLES ,
        device
        )

    trDataLoader = create_data_loader(usd, BATCH_SIZE)

    #construct model and assign it to device
    model = RNN_LSTM(input_size = INP, hidden_size = HID , num_layers = NL , num_units = NU , num_classes = NCLS , num_heads = NH).to(device)
    model.load_state_dict(torch.load("C:\\Users\\nagav\\OneDrive\\Desktop\\ML\\"+PT_FILE))

    print(f"current model:{model}")

    loss_fun = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters() , lr = LR)#ASGD , SGD
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)
    #train our model
    #Exploring better optimizer and shedulers
    train(model , trDataLoader , loss_fun , optimizer ,  device , EPOCHS , scheduler)

    torch.save(model.state_dict() , PT_FILE)
    print("trained RNN saved as "+PT_FILE)

    plt.plot(np.arange(len(LOSS_VALUES)),np.array(LOSS_VALUES))
    plt.show()
    fieldnames = ['Loss values']
    with open("C:\\Users\\nagav\\OneDrive\\Desktop\\ML\\lossValues.csv","a+",newline = "") as f :
        dict_writer = csv.writer(f)
        for d in LOSS_VALUES:
            dict_writer.writerow({str(d)})
        
        

    


        
