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
import os
import time as t



BATCH_SIZE = 128
EPOCHS  = 20
LR = 0.01
ANNOTATIONS_FILE = "C:\\Users\\nagav\\OneDrive\\Documents\\UrbanSound8K.tar\\UrbanSound8K\\metadata\\UrbanSound8K.csv"
AUDIO_DIR = "C:\\Users\\nagav\\OneDrive\\Documents\\UrbanSound8K.tar\\UrbanSound8K\\audio"
SAMPLE_RATE = 22050 
NUM_SAMPLES = 22050

INP = 64*44     #INPUTS 
HID = 8     #NUMBER OF HIDDEN LAYERS 
NL  = 8     #NUMBER OF LAYERS 
NU  = 2     #NUMBER OF UNITS 
NCLS = 10   #NUMBER OF CLASSES
NH = 8

PT_FILE = "rnnV1.pth"


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

import torch.nn.functional as F

class RNN_GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_units, num_classes, num_heads):
        super(RNN_GRU, self).__init__()
        self.num_layers = num_layers
        self.num_units = num_units
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        # Create multiple GRU units
        self.fc0 = nn.Linear(44 , 44*64)
        self.gru = nn.ModuleList([nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True) for i in range(num_units)])
        self.fc1 = nn.Linear(hidden_size*2*num_heads, num_units)
        self.fc2 = nn.Linear(num_units, num_classes)
        
        # Define multi-head attention mechanism
        self.query_layer = nn.Linear(hidden_size*2, hidden_size)
        self.key_layer = nn.Linear(hidden_size*2, hidden_size)
        self.value_layer = nn.Linear(hidden_size*2, hidden_size)
        self.final_layer = nn.Linear(hidden_size*num_heads, hidden_size*2)
        
    def forward(self, x):
        # Pass the input through the GRU layers
        x = self.fc0(x)
        outputs = []
        for i in range(self.num_units):
            output, _ = self.gru[i](x)
            outputs.append(output)
        
        # Apply multi-head attention mechanism
        attn_outputs = []
        for i in range(self.num_units):
            query = self.query_layer(outputs[i])
            key = self.key_layer(torch.cat(outputs, dim=0))
            value = self.value_layer(torch.cat(outputs, dim=0))
            
            attn_weights = F.softmax(torch.matmul(query, key.transpose(2, 1)) / np.sqrt(self.hidden_size*2), dim=-1)
            attn_output = torch.matmul(attn_weights, value)
            attn_outputs.append(attn_output)
        
        # Concatenate the multi-head attention outputs
        output = torch.cat(attn_outputs, dim=-1)
        
        # Reshape the output for the fully connected layer
        output = output.reshape(output.shape[0], -1)
        # Pass the output through the fully connected layers
        output = self.fc1(output)
        output = F.relu(output)
        output = self.fc2(output)
        return output



def create_data_loader(trData , bs) :
    return DataLoader(trData , batch_size = bs,shuffle = True,drop_last=True)

def train_single_epoch(model , data_loader , loss_fun , optimiser , device , scheduler) :
    for inp , tar in data_loader :
        inp , tar = inp.to(device) , tar.to(device)
        #calculate the loss

        #It looks like the input shape is (batch_size, sequence_length, input_size_1, input_size_2) = (128, 1, 64, 44).
        pred = model(inp.view(-1,64,44))
        loss = loss_fun(pred , tar)

        #propagate backwards
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        #scheduler.step(loss)
    print(f'\nloss :{loss.item()} lr :{optimizer.param_groups[0]["lr"]}')

def train(model , data_loader , loss_fun , optimiser , device , epochs , scheduler):

    for i in tqdm(range(epochs) , desc = "Training...,"):
        train_single_epoch(model , data_loader , loss_fun , optimiser , device ,scheduler)
    print("finished training...")




if __name__  == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"The device :{device}")
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
    model = RNN_GRU(input_size = INP, hidden_size = HID , num_layers = NL , num_units = NU , num_classes = NCLS , num_heads = NH).to(device)
    print(f"current model:{model}")

    loss_fun = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.ASGD(model.parameters() , lr = LR)#ASGD , SGD
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)
    #train our model
    #Exploring better optimizer and shedulers
    train(model , trDataLoader , loss_fun , optimizer ,  device , EPOCHS , scheduler)

    torch.save(model.state_dict() , PT_FILE)
    print("trained RNN saved as "+PT_FILE)


    


        
