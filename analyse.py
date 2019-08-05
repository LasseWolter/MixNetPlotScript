import datetime
import collections
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Calculate stats for this particular logDir and returns the stats in a DataFrame
def calcStats(logDir):

    timeStampFormat = '%b %d %H:%M:%S.%f'  # Timestamp format used in logFiles - later used for parsing

    # Take out the log files which log queueLengths
    logFiles=os.listdir(args.logDir)
    nodes = [x for x in logFiles if x.startswith('ql_')]

    rawData = {} # The data each stipped to period between first and last nonzero queueLength
    data = {}     # This dictonary will contain a dataframe for each logFile - during one common experiment period
    for node in nodes:
        path = os.path.join(args.logDir, node)
        data[node] = pd.read_csv(path, names=['time', 'queueLength'])
        # Turn time string into datetime object 
        data[node]['time'] = data[node]['time'].apply(lambda x: datetime.datetime.strptime(x, timeStampFormat))
        fromInd = data[node].queueLength.to_numpy().nonzero()[0][0]-1 # finds the first nonzero queueLength (-1 to include the last 0)
        toInd=data[node].queueLength.to_numpy().nonzero()[0][-1]+1    # simlar find the last nonzero queueLength
        rawData[node] = data[node][fromInd:toInd]                  # strip out zero values at the beginning and end
        #data[node].columns=['time', node] # rename such that different files have different names for the join below


    winSize=8000
    print(rawData[nodes[0]])
    print(rawData[nodes[0]].shape)

    conv = np.convolve(rawData[nodes[0]].queueLength, np.ones(winSize), mode='same') / winSize
    diff = pd.Series(conv).diff()
    startInd = diff.lt(0).idxmax()
    stopInd =  diff[::-1].gt(0).idxmax()
    # Find experiment start and end time by finding earliest/latest time stamp out of all log files
    startTime= min([x['time'].iat[0] for x in rawData.values()])
    endTime  = max([x['time'].iat[-1] for x in rawData.values()])
    # Calculate experiment Duration in seconds
    expDuration = (endTime - startTime).total_seconds()

    colNames = np.array([['mean_'+x, 'std_'+x, 'zeroFreq_'+x] for x in nodes]).flatten()
    df = pd.DataFrame(columns=colNames)
    df.loc[0]=np.zeros(df.shape[1])


    stats = [] # array 
    # Cut Data that all include the same times - this ensures that all dfs have the same shape for the concat later on
    for node in nodes:
        # Only take data recorded during experiment time
        data[node] = data[node][(data[node].time >= startTime) & (data[node].time <= endTime)]

        # Add column for relative time to all dataframes
        data[node]['relTime']=data[node]['time']-startTime
        data[node]['relTime']=data[node]['relTime'].apply(lambda x: x.total_seconds())

        #Calculate statistics
        qLs = data[node].queueLength # panda series containing the queue Lengths 
        df.loc[0]['mean_'+node]=qLs.mean()
        df.loc[0]['std_'+node] =qLs.std()
        df.loc[0]['zeroFreq_'+node] = qLs.value_counts(0).size/qLs.size


    # Print figures 
    fSize=(18,10) # Size of figures
    minNum = int(expDuration/60) + 1 # How many full minutes happened during the experiment (+1 because of 0)
    minTicks = [x*60 for x in range(minNum)]
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
        print(data[node].queueLength)
        print(data[node].relTime)
        ax1.plot(data[node].relTime[startInd:stopInd], data[node].queueLength[startInd:stopInd])

   # ax1.text(-0.25, 0.2, textstr, transform=ax1.transAxes)


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
        y = np.array(data[node].queueLength)
        x = np.linspace(0, 500, num=len(y))
        f = np.convolve(y, np.ones(winSize), mode='same') / winSize
        ax3.plot(data[node].relTime, f)

    f1.show()
    input()

    return df


# Gets input args with default value none
argNames = ["command", "logDir","show"]
args = dict(zip(argNames, sys.argv))
ArgList = collections.namedtuple("ArgList", argNames)
args = ArgList(*(args.get(arg, None) for arg in argNames))


df = calcStats(args.logDir)
# Put nodes into dataframe
print (df)
exit(0)

# Print figures 
fSize=(18,10) # Size of figures
minNum = int(expDuration/60) + 1 # How many full minutes happened during the experiment (+1 because of 0)
minTicks = [x*60 for x in range(minNum)]
winSize = 3000  # Size of the convolution window
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

# Put queueLogs of all files in one dataframe
#dfs = [x for x in data.values()]
#df = pd.concat(dfs)
