# -*- coding: utf-8 -*-
"""
Copyright (C) 2020 - Event-driven Perception for Robotics
Authors: Sim Bamford
This program is free software: you can redistribute it and/or modify it under 
the terms of the GNU General Public License as published by the Free Software 
Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY 
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A 
PARTICULAR PURPOSE.  See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with 
this program. If not, see <https://www.gnu.org/licenses/>.

This script is intended to be used in 'scientific mode' i.e. executed one block 
at a time, with temporary results in the global workspace
"""

#%% Import data in a format 

from bimvee.importAe import importAe

filePathOrName = "C:/repos/extNeuromorphic/dvs-reconstruction/data/desk_swing.aer2"
container = importAe(filePathOrName=filePathOrName)

# Take data from the first channel
dvs = container['data'][list(container['data'].keys())[0]]['dvs']
# Alternatively, manually choose which channel the data comes from 
#dvs = container['data']['name_of_channel']['dvs']

#%% Optionally, if you want to choose a section of data to work with:

from bimvee.split import cropTime

dvs = cropTime(dvs, startTime=1, stopTime=4.5)

#%% Run the reconstruction

from reinbacher import reinbacher

framesReconstructed = reinbacher(dvs, iterations = 5, stopTime = 1.0, tau=1.0)

#%% Start a 'mustard' visualiser, from https://github.com/robotology/event-driven/tree/master/python

import sys
import threading

# Start a visualizer
sys.path.append('C:/repos/event-driven/python')
import visualizer.ntupleviz
visualizerApp = visualizer.ntupleviz.Ntupleviz()
thread = threading.Thread(target=visualizerApp.run)
thread.daemon = True
thread.start()

#%% Visualise
    
containerVis = {'chDvs': {'dvs': dvs},
                'frame': {'frame': framesReconstructed},
               }
    
visualizerApp.root.data_controller.data_dict = {}
visualizerApp.root.data_controller.data_dict = containerVis

