###################################################################################################
#
# Calibrator.py
#
# Copyright (C) by Andreas Zoglauer & Shreya Sareen
# All rights reserved.
#  
###################################################################################################


  
###################################################################################################


#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.pyplot as plt

import random 

import signal
import sys
import time
import math
import csv
import os
import argparse
from datetime import datetime
from functools import reduce


print("\nA machine-learning based COSI calibrator")
print("========================+++++===========\n")



###################################################################################################
# Step 1: Input parameters
###################################################################################################


# Default parameters

NumberOfComptonEvents = 10000


OutputDirectory = "Output"


# Parse command line:

print("\nParsing the command line (if there is any)\n")

parser = argparse.ArgumentParser(description='Perform training and/or testing for gamma-ray burst localization')
parser.add_argument('-m', '--mode', default='toymodel', help='Choose an input data more: toymodel or simulations')
parser.add_argument('-t', '--toymodeloptions', default='10000:510.99:511.00', help='The toy-model options: source_events:energy_min:energy_max"')
parser.add_argument('-s', '--simulationoptions', default='', help='')
parser.add_argument('-b', '--batchsize', default='256', help='The number of GRBs in one training batch (default: 256 corresponsing to 5 degree grid resolution (64 for 3 degrees))')
parser.add_argument('-o', '--outputdirectory', default='Output', help='Name of the output directory. If it exists, the current data and time will be appended.')
parser

args = parser.parse_args()

  
Mode = (args.mode).lower()
if Mode != 'toymodel' and Mode != 'simulation':
  print("Error: The mode must be either \'toymodel\' or \'simulation\'")
  sys.exit(0)


if Mode == 'toymodel':
  print("CMD-Line: Using toy model")

  ToyModelOptions = args.toymodeloptions.split(":")
  if len(ToyModelOptions) != 3:
    print("Error: You need to give 3 toy model options. You gave {}. Options: {}".format(len(ToyModelOptions), ToyModelOptions))
    sys.exit(0)
  
  NumberOfComptonEvents = int(ToyModelOptions[0])
  if NumberOfComptonEvents <= 10:
    print("Error: You need at least 10 source events and not {}".format(NumberOfComptonEvents))
    sys.exit(0)       
  print("CMD-Line: Toy model: Using {} source events per GRB".format(NumberOfComptonEvents))

  MinimumEnergy = float(ToyModelOptions[1])
  if MinimumEnergy < 0:
    print("Error: You need a non-negative number for the energy, and not {}".format(MinimumEnergy))
    sys.exit(0)
  print("CMD-Line: Toy model: Using {} keV as minimum energy".format(MinimumEnergy))

  MaximumEnergy = float(ToyModelOptions[2])
  if MaximumEnergy < 0:
    print("Error: You need a non-negative number for the energy, and not {}".format(MinimumEnergy))
    sys.exit(0)
  print("CMD-Line: Toy model: Using {} keV as minimum energy".format(MinimumEnergy))

  '''
  NumberOfTrainingBatches = int(ToyModelOptions[3])
  if NumberOfTrainingBatches < 1:
    print("Error: You need a positive number for the number of traing batches and not {}".format(NumberOfTrainingBatches))
    sys.exit(0)
  print("CMD-Line: Toy model: Using {} training batches".format(NumberOfTrainingBatches))

  NumberOfTestingBatches = int(ToyModelOptions[4])
  if NumberOfTestingBatches < 1:
    print("Error: You need a positive number for the number of testing batches and not {}".format(NumberOfTestingBatches))
    sys.exit(0)
  print("CMD-Line: Toy model: Using {} testing batches".format(NumberOfTestingBatches))
  '''

elif Mode == 'simulation':
  print("Error: The simulation mode has not yet implemented")
  sys.exit(0)  

  
MaxBatchSize = int(args.batchsize)
if MaxBatchSize < 1 or MaxBatchSize > 1024:
  print("Error: The batch size must be between 1 && 1024")
  sys.exit(0)  
print("CMD-Line: Using {} as batch size".format(MaxBatchSize))
  
OutputDirectory = args.outputdirectory
# TODO: Add checks
print("CMD-Line: Using \"{}\" as output directory".format(OutputDirectory))

print("\n\n")


# Determine derived parameters


NumberOfTrainingEvents = int(0.8*NumberOfComptonEvents)
NumberOfTestingEvents = NumberOfComptonEvents - NumberOfTrainingEvents


'''
if os.path.exists(OutputDirectory):
  Now = datetime.now()
  OutputDirectory += Now.strftime("_%Y%m%d_%H%M%S")
    
os.makedirs(OutputDirectory)
'''


###################################################################################################
# Step 2: Global functions
###################################################################################################


# Take care of Ctrl-C
Interrupted = False
NInterrupts = 0
def signal_handler(signal, frame):
  global Interrupted
  Interrupted = True        
  global NInterrupts
  NInterrupts += 1
  if NInterrupts >= 2:
    print("Aborting!")
    sys.exit(0)
  print("You pressed Ctrl+C - waiting for graceful abort, or press  Ctrl-C again, for quick exit.")
signal.signal(signal.SIGINT, signal_handler)


# Everything ROOT related can only be loaded here otherwise it interferes with the argparse
#from CalibrationCreatorToyModel import CalibrationCreatorToyModel

# Load MEGAlib into ROOT so that it is usable
import ROOT as M
M.gSystem.Load("$(MEGALIB)/lib/libMEGAlib.so")
M.PyConfig.IgnoreCommandLineOptions = True



###################################################################################################
# Step 3: Create some training, test & verification data sets
###################################################################################################


print("Info: Creating {} Compton events".format(NumberOfComptonEvents))

from CalibrationData import CalibrationData

def generateOneDataSet(_):
  DataSet = CalibrationData()
  DataSet.create()
  return DataSet
  

# Create data sets
TimerCreation = time.time()

TrainingDataSets = []
for i in range(NumberOfTrainingEvents):
    result = generateOneDataSet(i)
    TrainingDataSets.append(generateOneDataSet(i))
print("Info: Created {:,} training data sets. ".format(NumberOfTrainingEvents))

TestingDataSets = []
for i in range(NumberOfTestingEvents):
    result = generateOneDataSet(i)
    TestingDataSets.append(generateOneDataSet(i))
print("Info: Created {:,} testing data sets. ".format(NumberOfTestingEvents))



TimeCreation = time.time() - TimerCreation
print("Info: Total time to create data sets: {:.1f} seconds (= {:,.0f} events/second)".format(TimeCreation, (NumberOfTrainingEvents + NumberOfTestingEvents) / TimeCreation))



###################################################################################################
# Step 4: Setting up the neural network
###################################################################################################
def dataset_to_tensors(datasets):

    X = []
    y = []

    for data in datasets:

        sh1 = data.StripHits[0]
        sh2 = data.StripHits[1]

        features = [
            sh1.DetectorID,
            sh1.StripID,
            sh1.IsHV,
            sh1.ADC,
            sh1.TAC,
            sh2.DetectorID,
            sh2.StripID,
            sh2.IsHV,
            sh2.ADC,
            sh2.TAC
        ]

        hit = data.Hits[0]

        target = [
            hit.X,
            hit.Y,
            hit.Z,
            hit.Energy
        ]

        X.append(features)
        y.append(target)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    return X, y


# Neural neutral

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# -------------------------------
# Convert datasets to tensors
# -------------------------------
train_X, train_y = dataset_to_tensors(TrainingDataSets)
test_X, test_y   = dataset_to_tensors(TestingDataSets)

# -------------------------------
# Feature engineering 
# -------------------------------
train_X = torch.cat([train_X, train_X**2], dim=1)
test_X  = torch.cat([test_X, test_X**2], dim=1)

# -------------------------------
# Standardize inputs
# -------------------------------
X_mean = train_X.mean(dim=0)
X_std  = train_X.std(dim=0) + 1e-8

train_X_norm = (train_X - X_mean) / X_std
test_X_norm  = (test_X  - X_mean) / X_std

# -------------------------------
# Standardize outputs
# -------------------------------
y_mean = train_y.mean(dim=0)
y_std  = train_y.std(dim=0) + 1e-8

train_y_norm = (train_y - y_mean) / y_std
test_y_norm  = (test_y  - y_mean) / y_std

# -------------------------------
# Linear Model
# -------------------------------
class LinearNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, 4)

    def forward(self, x):
        return self.linear(x)

input_size = train_X.shape[1]
model = LinearNN(input_size)

# -------------------------------
# Optimizer
# -------------------------------
optimizer = optim.SGD(model.parameters(), lr=0.01)

# -------------------------------
# Weighted loss
# -------------------------------
weights = torch.tensor([1.0, 1.0, 1.0, 8.0])

def weighted_mse(pred, target):
    return ((pred - target)**2 * weights).mean()

# -------------------------------
# Training
# -------------------------------
epochs = 1200

train_loss_history = []
test_loss_history = []

for epoch in range(epochs):

    model.train()
    optimizer.zero_grad()

    output = model(train_X_norm)
    loss = weighted_mse(output, train_y_norm)

    loss.backward()
    optimizer.step()

    train_loss_history.append(loss.item())

    model.eval()
    with torch.no_grad():
        test_output = model(test_X_norm)
        test_loss = weighted_mse(test_output, test_y_norm)

    test_loss_history.append(test_loss.item())

    if (epoch+1) % 200 == 0:
        print(f"Epoch {epoch+1}: Train Loss = {loss.item():.6f}, Test Loss = {test_loss.item():.6f}")

# -------------------------------
# Evaluate
# -------------------------------
model.eval()
with torch.no_grad():
    pred_norm = model(test_X_norm)
    pred = pred_norm * y_std + y_mean

output_columns = ['X','Y','Z','Energy']

rmse = torch.sqrt(((pred - test_y)**2).mean(dim=0))

print("\nRMSE per output component:")
for i,name in enumerate(output_columns):
    print(f"{name} = {rmse[i]:.3f}")

overall_loss = ((pred - test_y)**2).mean()
print(f"\nFinal Test Loss: {overall_loss:.6f}")

# -------------------------------
# Plot loss
# -------------------------------
plt.figure(figsize=(8,5))
plt.plot(train_loss_history,label="Training Loss")
plt.plot(test_loss_history,label="Test Loss")

plt.xlabel("Epoch")
plt.ylabel("Weighted MSE Loss")
plt.title("Training vs Test Loss")

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
