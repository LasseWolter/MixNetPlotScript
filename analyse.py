"""
Script to analyse logData from the mixnet
Author: Lasse Wolter
"""
import datetime
import collections
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import toml


def calcStats(logDir, start_sec=0, end_sec=0, onlyStable=False):
    """
    Calculates the statistics for all log files in the directory given with logDir
    - start_sec, end_sec: define the period of the experiment in seconds for which the stats should be calculated
                            if not set the whole duration of the experiment will be taken (regardless of the default value being 0)
    - onlyStable: (False by default) start_time and end_time are chosen automatically to exclude the setup and teardown phase
                  thus only looking at the "steady state phase"
    """
    # Timestamp format used in log_files - later used for parsing
    time_stamp_fmt = '%b %d %H:%M:%S.%f'

    # Take out the log files which log queueLengths
    log_files = os.listdir(logDir)
    nodes = [x for x in log_files if x.startswith('ql_')]

    data = {}  # contains a DataFrame for each logFile
    start_ind = sys.maxsize
    stop_ind = 0
    for node in nodes:
        path = os.path.join(ARGS.logDir, node)
        data[node] = pd.read_csv(path, names=['time', 'queueLength'])
        # Turn time string into datetime object
        data[node]['time'] = data[node]['time'].apply(
            lambda x: datetime.datetime.strptime(x, time_stamp_fmt))
        # Finds the first and last nonzero queueLength (-+1 to include the last 0)
        # And sets start/stop index for experiment by taking extremes of all values
        from_ind = data[node].queueLength.to_numpy().nonzero()[0][0] - 1
        start_ind = from_ind if from_ind < start_ind else start_ind
        to_ind = data[node].queueLength.to_numpy().nonzero()[0][-1] + 1
        stop_ind = to_ind if to_ind > stop_ind else stop_ind

    # Strip DataFrames to common length only containing relevant data
    data = {node: df[start_ind:stop_ind + 1] for (node, df) in data.items()}

    # Find experiment start and end time by finding earliest/latest time stamp out of all log files
    start_time = min([x['time'].iat[0] for x in data.values()])
    end_time = max([x['time'].iat[-1] for x in data.values()])

    # Calculate experiment Duration in seconds
    exp_duration = (end_time - start_time).total_seconds()

    # Find start and end of the steady state of nodes[0] which is not necessarily the first node
    # in the mixnet but suffices to get the rough time for all steady states
    win_size = 8000
    conv = np.convolve(
        data[nodes[0]].queueLength, np.ones(win_size), mode='same') / win_size
    diff = pd.Series(conv).diff()
    start_ind = diff.lt(0).idxmax()
    stop_ind = diff[::-1].gt(0).idxmax()
    steady_start_time = data[nodes[0]]['time'].iloc[start_ind]
    steady_end_time = data[nodes[0]]['time'].iloc[stop_ind]

    # Apply costum start and end time if given
    if onlyStable:
        start_time = steady_start_time
        end_time = steady_end_time
        exp_duration = (end_time - start_time).total_seconds()
    elif start_sec >= 0 and end_sec <= exp_duration:
        end_time = start_time + datetime.timedelta(seconds=end_sec)
        start_time = start_time + datetime.timedelta(seconds=start_sec)
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

    # Calculate the steady state period to show it on the plot
    stead_start = (steady_start_time - start_time).total_seconds()
    stead_end = (steady_end_time - start_time).total_seconds()

    plotFig(data, nodes, exp_duration, df, win_size, stead_start, stead_end)

    return df


def plotFig(data, nodes, exp_duration, stats_df, win_size, steady_start_time,
            steady_end_time):
    """
    Plot two plots in one figure
        - the original data
        - data with convolution applied to it
    """

    # General figure parameters
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
        ax1.plot(data[node].relTime, data[node].queueLength)
        # Add text showing stats and experiment parameters
        text_str += '\n'.join(
            (node + ':', r'mean=%.2f' % (stats_df.loc[0]['mean_' + node]),
             r'$\sigma=%.2f$' % (stats_df.loc[0]['std_' + node]),
             r'zeroFrac=%.2f' % (stats_df.loc[0]['zeroFreq_' + node]), '', ''))

    text_str += '\n'.join(('Parameters:', r'$\lambda P=%g$' % (LAMBDA_P),
                           r'Mu=%g' % (MU), '', ''))
    ax1.text(-0.25, 0.2, text_str, transform=ax1.transAxes)
    # Plot vertical lines to mark the steady state period
    ax1.axvline(x=steady_start_time, c='r')
    ax1.axvline(x=steady_end_time, c='r')

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
        # win_size can max be the number of samples
        win_size = len(qls) if win_size > len(qls) else win_size
        conv = np.convolve(qls, np.ones(win_size), mode='same') / win_size
        ax3.plot(data[node].relTime, conv)

    if ARGS.show == "show":
        fig.show()
        print(stats_df)
        input()  # just to block exec und keep fig open


def parse_conf(conf):
    """
    Parses the config file and returns LAMBDA_P and MU
    """
    with open(conf, 'r') as f:
        raw_conf = f.read()

    parsed_conf = toml.loads(raw_conf)
    return (parsed_conf["Experiment"]["LambdaP"],
            parsed_conf["Experiment"]["Mu"])


def parse_args():
    """
    Parses input arguments and returns a dict containing them mapped to their arg_names
    """
    # Gets input args with default value none
    arg_names = ["command", "logDir", "config", "show"]
    args = dict(zip(arg_names, sys.argv))
    if len(sys.argv) < len(arg_names):
        usage = 'Usage: ' + sys.argv[0]
        for arg in arg_names[1:]:  # exclude the command itself
            usage += " <%s>" % arg
        print(usage)
        exit(0)

    ArgList = collections.namedtuple("ArgList", arg_names)
    args = ArgList(*(args.get(arg, None) for arg in arg_names))
    return args


def main():
    """
    main function
    """
    data_frame = calcStats(ARGS.logDir, end_sec=220, start_sec=200)


# Constants
ARGS = parse_args()
LAMBDA_P, MU = parse_conf(ARGS.config)
main()

# Put nodes into dataframe
# Put queueLogs of all files in one dataframe
# dfs = [x for x in data.values()]
# df = pd.concat(dfs)
