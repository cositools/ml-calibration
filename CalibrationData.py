###################################################################################################
#
# CalibrationData.py
#
# Copyright (C) by Andreas Zoglauer & Shreya Sareen
# All rights reserved.
#
###################################################################################################




###################################################################################################


import random 
import numpy as np
import ROOT as M
M.gSystem.Load("$(MEGALIB)/lib/libMEGAlib.so")


###################################################################################################


class StripHit:
    def __init__(self, DetectorID, StripID, IsHV, ADC, TAC):
        self.DetectorID = DetectorID
        self.StripID = StripID
        self.IsHV = IsHV
        self.ADC = ADC
        self.TAC = TAC

    def __str__(self):
        return f"Strip D:{self.DetectorID}, S:{self.StripID}, HV:{self.StripID}, ADCs:{self.ADC}, TAC:{self.TAC}"


###################################################################################################

###################################################################################################


class Hit:
    def __init__(self, X, Y, Z, Energy):
        self.X = X
        self.Y = Y
        self.Z = Z
        self.Energy = Energy

    def __str__(self):
        return f"Hit x:{self.X}cm, y:{self.Y}cm, z:{self.Z}cm, E:{self.Energy}keV"



###################################################################################################

###################################################################################################



class CalibrationData:
  """
  This class represents a single calibration data set

  """


###################################################################################################


  def __init__(self):
    """
    The default constructor for class EventClustering

    Attributes
    ----------
    FileName : string
      Data file name (something like: X.maxhits2.eventclusterizer.root)
    OutputPrefix: string
      Output filename prefix as well as outout directory name
    Algorithms: string
      The algorithms used during training. Seperate multiples by commma (e.g. "MLP,DNNCPU")
    MaxEvents: integer
      The maximum amount of events to use

    """

    # The measured data
    self.StripHits: list[StripHit] = []

    # The real data
    self.Hits: list[Hit] = []


###################################################################################################


  def create(self):
    """
    Create a single event
    """

    # Random start hit:
    x = random.uniform(-4, 4)
    y = random.uniform(-4, 4)
    z = random.uniform(-0.75, 0.75)
    e = random.uniform(15, 1000) # keV

    hit1 = Hit(x, y, z, e)
    self.Hits.append(hit1)

    striphitLV1 = StripHit(0, self.get_strip_id(x), 0, 2000+4*e, 2000+500*(x+0.75))
    self.StripHits.append(striphitLV1)
    striphitHV1 = StripHit(0, self.get_strip_id(y), 1, 2000+4*e, 2000+500*(x+0.75))
    self.StripHits.append(striphitHV1)

    '''
    # Random second hit
    x = random.uniform(-4, 4)
    y = random.uniform(-4, 4)
    z = random.uniform(-0.75, 0.75)
    e = random.uniform(15, 1000)

    hit2 = Hit(x, y, z, e)
    self.Hits.append(hit2)

    striphitLV2 = StripHit(0, self.get_strip_id(x), 0, 2000+10*e, 2000+500*(x+0.75))
    self.StripHits.append(striphitLV2)
    striphitHV2 = StripHit(0, get_strip_id(y), 1, 2000+10*e, 2000+500*(x+0.75))
    self.StripHits.append(striphitHV2)
    '''


  def get_strip_id(self, V):

    # Discretized into 64 strips (0-63)
    num_strips = 64
    min_x = -4
    max_x = 4

    # Ensure X is within bounds to avoid errors
    clamped_x = max(min_x, min(max_x, V))

    # Mapping formula
    strip_index = int(((clamped_x - min_x) / (max_x - min_x)) * num_strips)

    # Edge case: if X is exactly 4.0, it might try to return ID 64
    return min(strip_index, num_strips - 1)

