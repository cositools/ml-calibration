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

# Neural neutral


#input("Press [enter] to EXIT")
sys.exit(0)

