import collections
import datetime
import sys

import matplotlib.pyplot as plt
import numpy as np

# Gets input args with default value none
argNames = ["command", "logDir"]
args = dict(zip(argNames, sys.argv))
ArgList = collections.namedtuple("ArgList", argNames)
args = ArgList(*(args.get(arg, None) for arg in argNames))

timeStampFormat = '%b %d %H:%M:%S.%f'  # Timestamp format used in logFiles - later used for parsing
zips = {}
nodes = ["ql_mix1", "ql_provider", "ql_serviceprovider"]
nodes = list(map(lambda a: args.logDir + "/" + a, nodes))  # prepends the lodDir to all files

# These arrays will contain the lengths and their corresponding time for the different mixnet-nodes
times = {}
lengths = {}

# Try parsing the log files and exit if directory/file doesn't exist
try:
    for node in nodes:
        times[node] = []
        lengths[node] = []
        with open(node) as f:
            data = f.readlines()
            for line in data:
                words = line.rstrip().split(",")
                time = datetime.datetime.strptime(words[0], timeStampFormat)
                lengths[node].append(int(words[1]))  # length string converted to int
                times[node].append(time)

except IOError as err:
    print(err)
    print("Please check if the directory exists.\nExiting...")
    sys.exit(0)

# Find experiment start by finding earliest time stamp
startTime = datetime.datetime.now()
for nodeTimes in times.values():
    if nodeTimes[0] < startTime:  # looking at the first log entry from each file
        startTime = nodeTimes[0]

# Print figures
f1 = plt.figure()
plt.title("original")
for node in nodes:
    plt.plot(times[node], lengths[node])

f3 = plt.figure()
plt.title("convolution")
for node in nodes:
    y = np.array(lengths[node])
    x = np.linspace(0, 500, num=len(y))
    winSize = 20  # Size of the convolution window
    f = np.convolve(y, np.ones(winSize), mode='same') / winSize
    plt.plot(times[node], f)

f1.show()
f3.show()
input()  # Just used to keep the figures alive
