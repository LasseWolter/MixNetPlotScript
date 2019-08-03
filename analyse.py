import datetime
import collections
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Gets input args with default value none
argNames = ["command", "logDir","show"]
args = dict(zip(argNames, sys.argv))
ArgList = collections.namedtuple("ArgList", argNames)
args = ArgList(*(args.get(arg, None) for arg in argNames))

timeStampFormat = '%b %d %H:%M:%S.%f'  # Timestamp format used in logFiles - later used for parsing

# Read in python array into pandas
columnNames=['time', 'queueLength']

# Take out the log files which log queueLengths
logFiles=os.listdir(args.logDir)
nodes = [x for x in logFiles if x.startswith('ql_')]

strpData = {} # The data each stipped to period between first and last nonzero queueLength
data = {}     # This dictonary will contain a dataframe for each logFile - during one common experiment period
for node in nodes:
    path = os.path.join(args.logDir, node)
    data[node] = pd.read_csv(path, names=['time', 'queueLength'])
    # Turn time string into datetime object 
    data[node]['time'] = data[node]['time'].apply(lambda x: datetime.datetime.strptime(x, timeStampFormat))
    fromInd = data[node].queueLength.to_numpy().nonzero()[0][0]-1 # finds the first nonzero queueLength (-1 to include the last 0)
    toInd=data[node].queueLength.to_numpy().nonzero()[0][-1]+1    # simlar find the last nonzero queueLength
    strpData[node] = data[node][fromInd:toInd]                  # strip out zero values at the beginning and end
    data[node].columns=['time', node] # rename such that different files have different names for the join below

# Find experiment start and end time by finding earliest/latest time stamp out of all log files
startTime= min([x['time'].iat[0] for x in strpData.values()])
endTime  = max([x['time'].iat[-1] for x in strpData.values()])

# Cut Data that all include the same times - this ensures that all dfs have the same shape for the concat later on
for node in nodes:
    data[node] = data[node][(data[node].time >= startTime) & (data[node].time <= endTime)]

# Calculate experiment Duration in seconds
expDuration = (endTime - startTime).total_seconds()

# Add column for relative time to all dataframes
for node in nodes:
    data[node]['relTime']=data[node]['time']-startTime
