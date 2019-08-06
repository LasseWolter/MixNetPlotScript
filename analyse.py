"""
Script to analyse logData from the mixnet
Author: Lasse Wolter
"""
import datetime
import collections
import sys
import os
import toml
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Calculate stats for this particular logDir and returns the stats in a DataFrame


def calcStats(logDir):

    # Timestamp format used in log_files - later used for parsing
    time_stamp_fmt = '%b %d %H:%M:%S.%f'

    # Take out the log files which log queueLengths
    log_files = os.listdir(logDir)
    nodes = [x for x in log_files if x.startswith('ql_')]

    raw_data = {
    }  # The data each stipped to period between first and last nonzero queueLength
    data = {
    }  # This dictonary will contain a dataframe for each logFile - during one common experiment period
    for node in nodes:
        path = os.path.join(ARGS.logDir, node)
        data[node] = pd.read_csv(path, names=['time', 'queueLength'])
        # Turn time string into datetime object
        data[node]['time'] = data[node]['time'].apply(
            lambda x: datetime.datetime.strptime(x, time_stamp_fmt))
        # finds the first nonzero queueLength (-1 to include the last 0)
        from_ind = data[node].queueLength.to_numpy().nonzero()[0][0] - 1
        # simlar find the last nonzero queueLength
        to_ind = data[node].queueLength.to_numpy().nonzero()[0][-1] + 1
        # strip out zero values at the beginning and end
        raw_data[node] = data[node][from_ind:to_ind]
        # data[node].columns=['time', node] # rename such that different files have different names for the join below

    win_size = 8000
    print(raw_data[nodes[0]])
    print(raw_data[nodes[0]].shape)

    conv = np.convolve(raw_data[nodes[0]].queueLength,
                       np.ones(win_size),
                       mode='same') / win_size
    diff = pd.Series(conv).diff()
    start_ind = diff.lt(0).idxmax()
    stop_ind = diff[::-1].gt(0).idxmax()
    # Find experiment start and end time by finding earliest/latest time stamp out of all log files
    start_time = min([x['time'].iat[0] for x in raw_data.values()])
    end_time = max([x['time'].iat[-1] for x in raw_data.values()])

    # Calculate experiment Duration in seconds
    exp_duration = (end_time - start_time).total_seconds()
    col_names = np.array([['mean_' + x, 'std_' + x, 'zeroFreq_' + x]
                          for x in nodes]).flatten()
    df = pd.DataFrame(columns=col_names)
    df.loc[0] = np.zeros(df.shape[1])

    # Cut Data that all include the same times - this ensures that all dfs have the same shape for the concat later on
    for node in nodes:
        # Only take data recorded during experiment time
        data[node] = data[node][(data[node].time >= start_time)
                                & (data[node].time <= end_time)]

        # Add column for relative time to all dataframes
        data[node]['relTime'] = data[node]['time'] - start_time
        data[node]['relTime'] = data[node]['relTime'].apply(
            lambda x: x.total_seconds())

        # Calculate statistics
        # panda series containing the queue Lengths
        q_length = data[node].queueLength
        df.loc[0]['mean_' + node] = q_length.mean()
        df.loc[0]['std_' + node] = q_length.std()
        df.loc[0]['zeroFreq_' +
                  node] = q_length.value_counts(0).size / q_length.size

    # Print figures
    f_size = (18, 10)  # Size of figures
    # How many full minutes happened during the experiment (+1 because of 0)
    min_num = int(exp_duration / 60) + 1
    min_ticks = [x * 60 for x in range(min_num)]
    text_str = ""

    # Original
    fig = plt.figure(figsize=f_size)
    plt.subplots_adjust(left=0.25, hspace=0.5)
    ax1 = fig.add_subplot(211)
    ax1.set_xlim(-10, exp_duration + 10)  # +10 to give some space
    ax1.set_title("Original")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Queue Length")

    # Add seconds axis showing full minutes
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(min_ticks)
    # list of numbers from 0..min_num
    ax2.set_xticklabels(list(range(min_num)))
    ax2.set_xlabel("Time [min]")
    for node in nodes:
        ax1.plot(data[node].relTime[start_ind:stop_ind],
                 data[node].queueLength[start_ind:stop_ind])
        text_str += '\n'.join(
            (node + ':', r'mean=%.2f' % (df.loc[0]['mean_' + node]),
             r'$\sigma=%.2f$' % (df.loc[0]['std_' + node]),
             r'zeroFrac=%.2f' % (df.loc[0]['zeroFreq_' + node]), '', ''))

    text_str += '\n'.join(('Parameters:', r'$\lambda P=%g$' % (LAMBDA_P),
                           r'Mu=%g' % (MU), '', ''))
    ax1.text(-0.25, 0.2, text_str, transform=ax1.transAxes)

    # Apply convolution with given window size
    ax3 = fig.add_subplot(212)
    ax3.set_xlim(-10, exp_duration + 10)  # +10 to give some space
    ax3.set_title("Convolution (win_size: %d)" % win_size)
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Queue Length")
    # Add seconds axis showing full minutes
    ax4 = ax3.twiny()
    ax4.set_xlim(ax3.get_xlim())
    ax4.set_xticks(min_ticks)

    # list of numbers from 0..min_num
    ax4.set_xticklabels(list(range(min_num)))
    ax4.set_xlabel("Time [min]")
    for node in nodes:
        qls = np.array(data[node].queueLength)
        x = np.linspace(0, 500, num=len(qls))
        conv = np.convolve(qls, np.ones(win_size), mode='same') / win_size
        ax3.plot(data[node].relTime, conv)

    fig.show()
    input()

    return df


def parse_conf(conf):
    """
    Parses the config file and returns LAMBDA_P and MU
    """
    with open(conf, 'r') as f:
        raw_conf = f.read()

    parsed_conf = toml.loads(raw_conf)
    return (parsed_conf["Experiment"]["LambdaP"],
            parsed_conf["Experiment"]["Mu"])


# Gets input args with default value none
ARG_NAMES = ["command", "logDir", "config"]
ARGS = dict(zip(ARG_NAMES, sys.argv))
if len(sys.argv) < len(ARG_NAMES):
    usage = 'Usage: ' + sys.argv[0]
    for arg in ARG_NAMES[1:]:  # exclude the command itself
        usage += " <%s>" % arg
    print(usage)
    exit(0)

ArgList = collections.namedtuple("ArgList", ARG_NAMES)
ARGS = ArgList(*(ARGS.get(arg, None) for arg in ARG_NAMES))

LAMBDA_P, MU = parse_conf(ARGS.config)
data_frame = calcStats(ARGS.logDir)
# Put nodes into dataframe
print(data_frame)
exit(0)
# Put queueLogs of all files in one dataframe
# dfs = [x for x in data.values()]
# df = pd.concat(dfs)
