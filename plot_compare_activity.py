'''
Function to plot and compare activity profile
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from datetime import datetime, timedelta, time

BACKGROUND_COLOR = '#d3d3d3' # lightgray

LABEL_COLOURS = {'sleep':'blue', 
                'sit-stand': 'red',
                'vehicle': 'darkorange',
                'walking': 'lightgreen',
                'mixed': 'lime',
                'bicycling': 'green',
                'tasks-light': 'darkorange',
                'moderate': 'green',
                'sedentary': 'red',
                'light': 'darkorange',
                'moderage-vigorous': 'green',}


def plot_compare_activity(t, y_true, y_pred, accel=None):
    ''' Plot and compare activity profiles '''

    convert_date = np.vectorize(
        lambda day, x: matplotlib.dates.date2num(datetime.combine(day, x)))

    data = pd.DataFrame(data={'time':t, 
                              'y_true': y_true, 
                              'y_pred': y_pred, 
                              'accel': accel})
    data.set_index('time', inplace=True)

    # Preprocess accel so that plot is nicer by median-smoothing and rescaling to [0,1] 
    data['accel'] = data['accel'].rolling(window=10, min_periods=1).median()
    data['accel'] = (data['accel'] - 1.0).abs()  # "movement" (because exact 1g is stationary)
    data['accel'] = data['accel'] / (data['accel'].max() - data['accel'].min())  # scale to [0,1]

    labels = np.unique(y_true)
    colors = [LABEL_COLOURS[label] for label in labels]
    data_by_days = data.groupby(data.index.date)
    ndays = len(data_by_days)
    nrows = 3*ndays + 1  # ndays x (prediction + annotation + spacing) + legend

    fig = plt.figure(figsize=(10,nrows), dpi=200)
    gs = fig.add_gridspec(nrows=nrows, ncols=1, height_ratios=[2, 2, 1]*ndays+[2])
    axes = [fig.add_subplot(gs[i]) for i in range(nrows) if (i+1)%3]  # every 3rd axes is a spacing

    for i, (day, daydata) in enumerate(data_by_days):

        ax_true, ax_pred = axes[2*i], axes[2*i+1]

        for _, chunk in splitby_timegap(daydata):
            _t = convert_date(day, chunk.index.time)
            _y_true_dummies = [(chunk['y_true']==label).astype('int') for label in labels]
            _y_pred_dummies = [(chunk['y_pred']==label).astype('int') for label in labels]
            ax_true.plot(_t, chunk['accel'], c='k')
            ax_pred.plot(_t, chunk['accel'], c='k')
            ax_true.stackplot(_t, _y_true_dummies, colors=colors, alpha=.5, edgecolor='none')
            ax_pred.stackplot(_t, _y_pred_dummies, colors=colors, alpha=.5, edgecolor='none')

            format_axes(ax_true, day)
            format_axes(ax_pred, day)

        ax_true.set_ylabel('true', fontsize='x-small')
        ax_pred.set_ylabel('predicted', fontsize='x-small')

        # add date to left side of axes
        ax_pred.set_title(
            day.strftime("%A,\n%d %B"), weight='bold',
            x=-.2, y=-.3,
            horizontalalignment='left',
            verticalalignment='bottom',
            rotation='horizontal',
            transform=ax_pred.transAxes,
            fontsize='medium',
            color='k'
        )

    # legends
    axes[-1].axis('off')
    legend_patches = []
    legend_patches.append(mlines.Line2D([], [], color='k', label='acceleration'))
    for label, color in zip(labels, colors):
        legend_patches.append(mpatches.Patch(facecolor=color, label=label, alpha=.5))
    axes[-1].legend(handles=legend_patches, bbox_to_anchor=(0., 0., 1., 1.),
        # loc='center', ncol=min(3,len(legend_patches)), mode="best",
        loc='center', ncol=len(legend_patches), mode="best",
        borderaxespad=0, framealpha=0.6, frameon=True, fancybox=True)
    axes[-1].spines['left'].set_visible(False)
    axes[-1].spines['right'].set_visible(False)
    axes[-1].spines['top'].set_visible(False)
    axes[-1].spines['bottom'].set_visible(False)

    # format x-axis to show hours
    fig.autofmt_xdate()
    hours = ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00', '24:00']
    axes[0].set_xticklabels(hours)
    axes[0].tick_params(labelbottom=False, labeltop=True, labelleft=False)

    fig.show()


def format_axes(ax, day):
    # run gridlines for each hour bar
    ax.get_xaxis().grid(True, which='major', color='grey', alpha=0.5)
    ax.get_xaxis().grid(True, which='minor', color='grey', alpha=0.25)
    # set x and y-axes
    ax.set_xlim((datetime.combine(day,time(0, 0, 0, 0)),
        datetime.combine(day + timedelta(days=1), time(0, 0, 0, 0))))
    ax.set_xticks(pd.date_range(start=datetime.combine(day,time(0, 0, 0, 0)),
        end=datetime.combine(day + timedelta(days=1), time(0, 0, 0, 0)),
        freq='4H'))
    ax.set_xticks(pd.date_range(start=datetime.combine(day,time(0, 0, 0, 0)),
        end=datetime.combine(day + timedelta(days=1), time(0, 0, 0, 0)),
        freq='1H'), minor=True)
    ax.get_yaxis().set_ticks([]) # hide y-axis lables
    # make border less harsh between subplots
    ax.spines['top'].set_color(BACKGROUND_COLOR)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # set background colour to lightgray
    ax.set_facecolor(BACKGROUND_COLOR)


def splitby_timegap(data, gap=30):
    split_id = (data.index.to_series().diff() > pd.Timedelta(gap, 'S')).cumsum()
    splits = data.groupby(by=split_id)
    return splits
