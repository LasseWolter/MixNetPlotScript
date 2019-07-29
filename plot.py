import collections
import datetime
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

# Strips the directory and 'ql_' from the entries in node array
def stripDir(s):
    return s.split("_")[1]

def stripExpName(s):
    # strip tailing slash if it exists
    if s[-1] == '/':
        s = s[:-1]
    return s.split("/")[-1]

# Gets input args with default value none
argNames = ["command", "logDir","show"]
args = dict(zip(argNames, sys.argv))
ArgList = collections.namedtuple("ArgList", argNames)
args = ArgList(*(args.get(arg, None) for arg in argNames))

timeStampFormat = '%b %d %H:%M:%S.%f'  # Timestamp format used in logFiles - later used for parsing
zips = {}
nodes = ["ql_mix1", "ql_provider", "ql_serviceprovider"]
nodes = list(map(lambda a: args.logDir + "/" + a, nodes))  # prepends the lodDir to all files
expName = stripExpName(args.logDir)  # Returns the experiment directory without it's full path

#These arrays will contain the lengths and their corresponding time for the different mixnet-nodes
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

# Strip out the zero values before the experiment started - during setup phase
for node in nodes:
    firstNonZeroInd = np.nonzero(lengths[node])[0][0] - 1  # -1 to include the last 0
    lengths[node] = lengths[node][firstNonZeroInd:]
    times[node] = times[node][firstNonZeroInd:]

# Find experiment start by finding earliest time stamp
startTime = datetime.datetime.now()
for nodeTimes in times.values():
    if nodeTimes[0] < startTime:  # looking at the first log entry from each file
        startTime = nodeTimes[0]

# Find experiment end time following the same principle
endTime = startTime
for nodeTimes in times.values():
    if nodeTimes[-1] > startTime:  # looking at the first log entry from each file
        endTime = nodeTimes[-1]

# Calculate experiment Duration in seconds
expDuration = (endTime - startTime).total_seconds()

# Make times relative to start time
for node in times.keys():
    for i in range (0, len(times[node])):
        times[node][i] = (times[node][i] - startTime).total_seconds()  # time relative to start in seconds 

# Calculate Mean and standard deviation
means = {}
stds  = {}
zeroFrac = {}  # Contains the fraction of 0 queueLength records over all recorded queueLength records 
for node in nodes:
    means[node] = round(np.mean(lengths[node]), 2)
    stds[node] = round(np.std(lengths[node]), 2)
    zeroFrac[node] = lengths[node].count(0) / len(lengths[node])

# Print figures 
fSize=(18,10) # Size of figures
minNum = int(expDuration/60) + 1 # How many full minutes happened during the experiment (+1 because of 0)
minTicks = [x*60 for x in range(minNum)]
winSize = 20  # Size of the convolution window
textstr = ""

# Original
f1 = plt.figure(figsize=fSize)
plt.subplots_adjust(left=0.25, hspace = 0.5)
ax1 = f1.add_subplot(211)
ax1.set_xlim(-10, expDuration +10)  # +10 to give some space
ax1.set_title("Original")
ax1.set_xlabel("Time [s]")
ax1.set_ylabel("Queue Length")

# Add seconds axis showing full minutes
ax2 = ax1.twiny()
ax2.set_xlim(ax1.get_xlim())
ax2.set_xticks(minTicks)
ax2.set_xticklabels(list(range(minNum))) # list of numbers from 0..minNum
ax2.set_xlabel("Time [min]")
for node in nodes:
    textstr += '\n'.join((
        r'%s $\mu=%.2f$' % (stripDir(node),means[node]),
        r'%s $\sigma=%.2f$' % (stripDir(node),stds[node]),
        r'%s zeroFrac=%.2f' % (stripDir(node),zeroFrac[node]),
        '',''))
    ax1.plot(times[node], lengths[node])

ax1.text(-0.25, 0.2, textstr, transform=ax1.transAxes)


# Apply convolution with given window size
ax3 = f1.add_subplot(212)
ax3.set_xlim(-10, expDuration +10)  # +10 to give some space
ax3.set_title("Convolution (winSize: %d)" % winSize)
ax3.set_xlabel("Time [s]")
ax3.set_ylabel("Queue Length")
# Add seconds axis showing full minutes
ax4 = ax3.twiny()
ax4.set_xlim(ax3.get_xlim())
ax4.set_xticks(minTicks)
ax4.set_xticklabels(list(range(minNum))) # list of numbers from 0..minNum
ax4.set_xlabel("Time [min]")
for node in nodes:
    y = np.array(lengths[node])
    x = np.linspace(0, 500, num=len(y))
    f = np.convolve(y, np.ones(winSize), mode='same') / winSize
    ax3.plot(times[node], f)

f1.savefig("%s/%s.pdf" % (args.logDir, stripExpName(expName)))
f1.savefig("%s/%s.png" % (args.logDir, stripExpName(expName)))

# Print stats to file
for node in nodes:
    fName = os.path.join(args.logDir, stripDir(node) + "_stats")
    with open(fName, 'w+') as f:
        f.write("mean,%.2f\n"     % means[node])
        f.write("std,%.2f\n"      % stds[node])
        f.write("zeroFrac,%.2f\n" % zeroFrac[node])

# In case show is set, display the figures
if args.show == "show":
    f1.show()
    input()  # Just used to keep the figures alive
