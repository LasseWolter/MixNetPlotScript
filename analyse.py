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


def calcStats(log_dir, start_sec=0, end_sec=0, only_steady=False):
    """
    Calculates the statistics for all log files in the directory given with log_dir
    - start_sec, end_sec: define the period of the experiment in seconds for which the stats should be calculated
                            if not set the whole duration of the experiment will be taken (regardless of the default value being 0)
    - only_steady: (False by default) start_time and end_time are chosen automatically to exclude the setup and teardown phase
                  thus only looking at the "steady state phase"
    """
    # Timestamp format used in log_files - later used for parsing
    time_stamp_fmt = '%b %d %H:%M:%S.%f'

    # Take out the log files which log queueLengths
    log_files = os.listdir(log_dir)
    nodes = [x for x in log_files if x.startswith('ql_')]

    data = {}  # contains a DataFrame for each logFile
    start_ind = sys.maxsize
    stop_ind = 0
    for node in nodes:
        path = os.path.join(log_dir, node)
        data[node] = pd.read_csv(path, names=['time', 'queueLength'])
        # Turn time string into datetime object
        data[node]['time'] = data[node]['time'].apply(
            lambda x: datetime.datetime.strptime(x, time_stamp_fmt))
        # If one of the logFiles just contains zeros, skip index calculation for stipping zeros
        # this only works if one of the log files is not all zeros which is a reasonable assumption
        if data[node].queueLength.to_numpy().nonzero()[0].size <= 0:
            continue
        # Finds the first and last nonzero queueLength (-+1 to include the last 0)
        # And sets start/stop index for experiment by taking extremes of all values
        from_ind = data[node].queueLength.to_numpy().nonzero()[0][0] - 1
        start_ind = from_ind if from_ind < start_ind else start_ind
        to_ind = data[node].queueLength.to_numpy().nonzero()[0][-1] + 1
        stop_ind = to_ind if to_ind > stop_ind else stop_ind

    if start_ind == sys.maxsize:
        print(
            "All logs contain only queueLengths of value 0, cannot analyse such data."
            + "\nReturning empty DataFrame for this experiment run...")
        return pd.DataFrame()

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
    if only_steady:
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

    plotFig(data, nodes, exp_duration, df, win_size, stead_start, stead_end,
            log_dir)

    return df


def plotFig(data, nodes, exp_duration, stats_df, win_size, steady_start_time,
            steady_end_time, log_dir):
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
             r'zeroFreq=%.2f' % (stats_df.loc[0]['zeroFreq_' + node]), '', ''))

    text_str += '\n'.join(
        ('parameters:', r'$\lambda p=%g$' % (LAMBDA_P), r'mu=%g' % (MU),
         r'$\frac{\lambda p}{mu}=%g$' % (LAMBDA_P / MU), ''))
    ax1.text(-0.25, 0.2, text_str, transform=ax1.transAxes)
    # plot vertical lines to mark the steady state period
    ax1.axvline(x=steady_start_time, c='r')
    ax1.axvline(x=steady_end_time, c='r')

    # apply convolution with given window size
    ax3 = fig.add_subplot(212)
    ax3.set_xlim(-10, exp_duration + 10)  # +10 to give some space
    ax3.set_title("convolution (win_size: %d)" % win_size)
    ax3.set_xlabel("time [s]")
    ax3.set_ylabel("queue length")

    # add seconds axis showing full minutes
    ax4 = ax3.twiny()
    ax4.set_xlim(ax3.get_xlim())
    ax4.set_xticks(min_ticks)

    # list of numbers from 0..min_num
    ax4.set_xticklabels(list(range(min_num)))
    ax4.set_xlabel("time [min]")
    for node in nodes:
        qls = np.array(data[node].queueLength)
        # win_size can max be the number of samples
        win_size = len(qls) if win_size > len(qls) else win_size
        conv = np.convolve(qls, np.ones(win_size), mode='same') / win_size
        ax3.plot(data[node].relTime, conv)

    # plot vertical lines to mark the steady state period
    ax3.axvline(x=steady_start_time, c='r')
    ax3.axvline(x=steady_end_time, c='r')

    fig.legend(labels=nodes + ['steady_period_edge'])
    path = os.path.join(log_dir, str(int(exp_duration)) + "s.png")
    fig.savefig(path)

    if ARGS.show == "show":
        fig.show()
        print(stats_df)
        input()  # just to block exec und keep fig open


def parse_conf(conf):
    """
    parses the config file and returns lambda_p and mu
    """
    with open(conf, 'r') as f:
        raw_conf = f.read()

    parsed_conf = toml.loads(raw_conf)
    return (parsed_conf["Experiment"]["LambdaP"],
            parsed_conf["Experiment"]["Mu"])


def make_mom_plot(df, ax, title=''):
    """
    calculates the stats for a bunch of expriment and returns a corresponding axis containing a mean of means plot
        - df: dataframe containing statistics for each experiment
    """
    mean_cols = [x for x in df.columns if 'mean' in x]
    # strip the nodes name
    std_cols = [x for x in df.columns if 'std' in x]
    zeroFreq_cols = [x for x in df.columns if 'zeroFreq' in x]

    mean_of_means = df[mean_cols].mean(axis=0)
    mean_of_stds = df[std_cols].mean(axis=0)
    mean_of_zeroFreq = df[zeroFreq_cols].mean(axis=0)

    # Each label contains stats about mean, std and zeroFreq
    labels = []
    for i in range(len(mean_cols)):
        labels.append('\n'.join(
            (mean_cols[i].split('_')[2],
             r'mean  =%.2f' % (mean_of_means[mean_cols[i]]),
             r'sigma =%.2f' % (mean_of_stds[std_cols[i]]),
             r'0-freq =%.2f' % (mean_of_zeroFreq[zeroFreq_cols[i]]))))

    x_pos = np.arange(len(labels))
    ax.bar(x_pos, mean_of_means, yerr=mean_of_stds, capsize=10, alpha=0.5)
    ax.set_ylabel('Queue Length')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.yaxis.grid(True)


def plot_mean_of_means(df_dict):
    """
    Plot mean of means of all given DataFrames in one figure and save them to file
        -dfs: dict with keys being the plot title and values being the corresponding DataFrames
    """
    print("Plotting Mean of Means for %d DataFrames..." % (len(df_dict)))
    # Create a figure which size and subplot layout adjusts according to the number of DataFrames given
    fig, axs = plt.subplots(figsize=(5 * len(df_dict), 6),
                            ncols=len(df_dict),
                            constrained_layout=True,
                            sharey=True)
    fig.suptitle("Server Msg Queue Length\nMean of Means")

    # Print general parameters to the bottom of the plot
    params_str = ' | '.join(
        ('Parameters:', r'$\lambda P=%g$' % (LAMBDA_P), r'Mu=%g' % (MU),
         r'$\frac{\lambda P}{Mu}=%g$' % (LAMBDA_P / MU), ''))
    fig.subplots_adjust(bottom=0.2)
    fig.text(0.5, 0.01, params_str, ha='center')

    # For each dataFrame create a plot an plot them next to one another
    for (i, title) in enumerate(df_dict.keys()):
        make_mom_plot(df_dict[title], axs[i], title)
        file_title = ''.join(title.split(' '))
        path = os.path.join(ARGS.exp_dir, file_title + '.csv')
        # Save stats to file
        with open(path, 'w+') as f:
            df_dict[title].to_csv(f)
        print('Saved .csv file for \"%s\": to %s' % (title, path))

    if ARGS.show == "show":
        fig.show()
        input()  # only to keep the plot open

    path = os.path.join(ARGS.exp_dir, 'mean_of_means.png')
    fig.savefig(path)
    print('Successfully saved plot to %s' % path)


def print_header(title):
    print('\n' + '-' * 80)
    print(title)
    print('-' * 80 + '\n')


def calc_all_stats(start_sec=-1, end_sec=-1, only_steady=False):
    """
    Calculates stats for all log files and returns them in one common DataFrame
    """
    exp_df = pd.DataFrame()  # Will contain data from all experiments

    # Create list of files/folders in exp_dir
    files = [os.path.join(ARGS.exp_dir, x) for x in os.listdir(ARGS.exp_dir)]
    log_dirs = list(filter(os.path.isdir, files))
    print_header(
        "Calculating stats for %d experiment runs in experiment folder..." %
        len(log_dirs))
    for log_dir in log_dirs:
        print('Calculating stats for %s' % log_dir)
        df = calcStats(log_dir,
                       start_sec=start_sec,
                       end_sec=end_sec,
                       only_steady=only_steady)
        exp_df = exp_df.append(df)

    # Calculate stats for whole dataset and plot them
    print('Finished calculating stats for each experiment')
    exp_df = exp_df.reset_index()  # s.t. we have indices from 1 to n
    return exp_df


def parse_args():
    """
    Parses input arguments and returns a dict containing them mapped to their arg_names
    """
    # Gets input args with default value none
    arg_names = ["command", "exp_dir", "config", "from_disc", "show"]
    args = dict(zip(arg_names, sys.argv))
    # The first 2 arguments are compulsory
    if len(sys.argv) < 3:
        usage = 'Usage: ' + sys.argv[0]
        for i, arg in enumerate(arg_names[1:]):  # exclude the command itself
            if i >= 2:
                usage += " (<%s>)" % arg
            else:
                usage += " <%s>" % arg
        print(usage)
        exit(0)

    ArgList = collections.namedtuple("ArgList", arg_names)
    args = ArgList(*(args.get(arg, None) for arg in arg_names))
    return args


# Constants
ARGS = parse_args()
LAMBDA_P, MU = parse_conf(ARGS.config)

if __name__ == '__main__':
    dfs = {}
    if ARGS.from_disc == "from_disc":
        files = [x for x in os.listdir(ARGS.exp_dir) if x.endswith('.csv')]
        print("Reading files %d .csv-files from disc..." % len(files))
        for file in files:
            file_name = file.split('.')[0]  # strip the .csv ending
            file = ''.join(file.split(' '))  # strip out whitespace
            path = os.path.join(ARGS.exp_dir, file)  # absolut path
            dfs[file_name] = pd.read_csv(path)
            print("Created DataFrame from disc -> %s" % file)
    else:
        dfs["Whole duration"] = calc_all_stats()
        dfs["Only Steady Period"] = calc_all_stats(only_steady=True)

    plot_mean_of_means(dfs)
