#!/usr/bin/env python3


import argparse
import csv
import datetime as dt

import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import numpy as np

__description__ = 'Gentoo emerge Heatmap Generator'


DEFAULT_CSV = 'emerge.csv'
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
FIGURE_PARAMS = None


def read_emerges(csvfile):

    emerges = []
    for emerge in csv.reader(csvfile):
        if emerge[0] == 'START':
            continue
        r = tuple(dt.datetime.fromtimestamp(float(ts)) for ts in emerge)
        emerges.append(r)

    return emerges


def bin_data(emerges):
    '''Putting seconds into minute/hour bins for later data aggregation

    Split am emerge time range (START, END) by minute or hour, e.g.,
    14:03:21 to 14:05:13.

    By minute, put
    - 39 seconds in Bin 14:03,
    - 60 seconds in Bin 14:04 and
    - 13 seconds in Bin 14:05

    By hour, put 112 seconds in Bin 14.

    Returns a dict of the data:
    - dts: list of dates (datetime.date)
    - by_minute: array of minute bins (DAYS, minute bins)
    - by_hour: array of hour bins (DAYS, hour bins)
    '''
    bins = {}

    BASE = emerges[0][0].replace(hour=0, minute=0, second=0)
    td = emerges[-1][1] - BASE
    DAYS = td.days + (td.seconds > 0)

    dts = [BASE.date() + dt.timedelta(days=i) for i in range(DAYS)]
    bins['dts'] = dts

    ###############
    # bins_minute #
    ###############
    # 24 * 60 = 1440 bins, bin width = 1 minute
    #         | 1st hour |     | last hour |
    # Day 1 [ 0 1 2 ... 59 ... 1380 ... 1439 ]
    # Day 2 [ .............................. ]
    #   :
    # DAYS  [ .............................. ]

    bins_minute = [x[:] for x in [[0] * 24 * 60] * DAYS]

    for emerge in emerges:
        fm = emerge[0]
        to = emerge[1]
        nt = fm.replace(second=0)
        while fm <= to:
            nt += dt.timedelta(minutes=1)
            if nt > to:
                n = (to - fm).seconds
            else:
                n = 60
            bins_minute[(fm - BASE).days][fm.hour * 60 + fm.minute] += n
            fm = nt
    bins['by_minute'] = bins_minute

    #############
    # bins_hour #
    #############
    # 24 bins, bin width = 1 hour
    # Day 1 [ 0 1 2 ... 23 ]
    # Day 2 [ ............ ]
    #   :
    # DAYS  [ ............ ]

    sum_hour = lambda mins: [sum(mins[i * 60:(i + 1) * 60]) for i in range(24)]
    bins_hour = [sum_hour(day_minutes) for day_minutes in bins_minute]
    bins['by_hour'] = bins_hour

    ############
    # bins_day #
    ############
    # 1 bin, bin width = 24 hour
    # [ Day1 Day2 ... DAYS ]

    bins['by_day'] = np.transpose(np.sum(bins_minute, axis=1))

    return bins


def agg_data(bins):
    '''Data aggregation

    Returns a dict of the following data:
    - dts: array of dates (datetime.date)
    - weekday_24hour: likelihood to merge of each minute in 24-hour by weekdays
    - year_days: historical emerges
    - yearly_total: yearly total emerging time in hours
    - yearly_avg_daily: yearly average emerging time in minutes
    - year_24hour: over 24-hour by years
    - year_weekday: over weekdays by years
    '''
    aggs = {}

    dts = np.array(bins['dts'])
    DAYS = dts.size
    bins_minute = np.array(bins['by_minute'])
    bins_day = np.array(bins['by_day'])

    ###################
    # weekday_24hour #
    ###################
    # Likelihood to merge of each minute in 24-hour by weekdays
    #
    # 24 * 60 = 1440 bins, bin width = 1 minute
    #       | 1st hour |     | last hour |
    # Mon [ 0 1 2 ... 59 ... 1380 ... 1439 ]
    # Tue [ .............................. ]
    #  :
    # Sun [ .............................. ]
    #
    # cbmax is the possible maximum of a minute = total weeks * 60.  It has an
    # error if DAYS % 7 != 0, but negligible if the number of weeks is high
    # enough.

    weekday_24hour = [x[:] for x in [[0] * 24 * 60] * 7]

    WD = dts[0].weekday()
    for i in range(len(bins_minute)):
        wd = (i + WD) % 7
        z = zip(weekday_24hour[wd], bins_minute[i])
        weekday_24hour[wd] = [x + y for x, y in z]

    aggs['weekday_24hour'] = {
        'data': weekday_24hour,
        'cbmax': np.ceil(DAYS / 7) * 60,
    }

    #############
    # year_days #
    #############
    # Historical emerges
    #
    # Note:
    # - February 29 is inserted to every year for alignments.
    #
    # Year 1 [ Jan1 Jan2 ... Feb29 ... Dec31 ]
    # Year 2 [ ............................. ]
    #   :
    # YEARS  [ ............................. ]

    fm = dts[0]
    to = dts[-1]
    YEARS = to.year - fm.year + 1
    YEAR_NUMBERS = range(fm.year, to.year + 1)
    aggs['YEAR_NUMBERS'] = YEAR_NUMBERS
    YEAR_LABELS = list(map(str, YEAR_NUMBERS))
    aggs['YEAR_LABELS'] = YEAR_LABELS

    b = np.array(list(d.month != 2 or d.day != 29 for d in dts))
    nb = np.logical_not(b)

    # taking leaf days out and reshape into (YEARS, 365)
    pad = (fm.replace(year=2001) - fm.replace(year=2001, month=1, day=1)).days
    noleaf = np.hstack((np.zeros(pad), bins_day[b]))
    noleaf.resize(YEARS, 365)

    # copy noleaf into final array in (YEAR, 366)
    # 59 = 31 + 28, 60 = 31 + 29
    Feb29 = np.zeros((YEARS, 1))
    year_days = np.hstack((noleaf[:, :59], Feb29, noleaf[:, 59:]))

    # putting leaf days back
    leaf_dts = np.array(dts)[nb]
    leaf_days = bins_day[nb]
    for d, day in zip(leaf_dts, leaf_days):
        year_days[d.year - fm.year, 59] = day

    aggs['year_days'] = {'data': year_days}

    ################
    # yearly_total #
    ################
    # yearly emerging time in hours

    yearly_total_seconds = np.sum(year_days, axis=1)
    yearly_total = yearly_total_seconds / 3600
    aggs['yearly_total'] = {
        'values': yearly_total,
    }

    ####################
    # yearly_avg_daily #
    ####################
    # yearly emerging time in minutes

    yearly_avg_daily = yearly_total_seconds / 60
    for y in range(YEARS):
        days = sum(1 for d in dts if d.year == YEAR_NUMBERS[y])
        yearly_avg_daily[y] /= days

    aggs['yearly_avg_daily'] = {
        'values': yearly_avg_daily,
    }

    ################
    # year_24hour #
    ################
    # Heatmap over 24-hour by years
    #
    # 24 * 60 = 1440 bins, bin width = 1 minute
    #          | 1st hour |     | last hour |
    # Year 1 [ 0 1 2 ... 59 ... 1380 ... 1439 ]
    # Year 2 [ .............................. ]
    #   :
    # YEARS  [ .............................. ]

    year_24hour = np.zeros((YEARS, 24 * 60))
    for y in range(YEARS):
        b = np.array(list(d.year == y + fm.year for d in dts))
        year_minutes = bins_minute[b]
        year_24hour[y] = np.sum(year_minutes.reshape(-1, 24 * 60), axis=0)

    aggs['year_24hour'] = {'data': year_24hour}

    ################
    # year_weekday #
    ################
    # Heatmap over weekdays by years
    #
    # 7 * 24 * 60 = 10,080 bins, bin width = 1 minute
    #          |  Monday  |     |   Sunday   |
    # Year 1 [ 0 1 ... 1439 ... 8640 ... 10079 ]
    # Year 2 [ ............................... ]
    #   :
    # YEARS  [ ............................... ]

    year_weekday = np.zeros((YEARS, 7 * 24 * 60))
    for y in range(YEARS):
        b = np.array(list(d.year == y + fm.year for d in dts))
        minutes = bins_minute[b]

        # padding to align
        start_weekday = dts[b][0].weekday()
        if start_weekday:
            minutes = np.vstack((np.zeros((start_weekday, 24 * 60)), minutes))
        minutes.resize((np.ceil(minutes.size / (7 * 24 * 60)), 7 * 24 * 60))
        year_weekday[y] = np.sum(minutes, axis=0)

    aggs['year_weekday'] = {'data': year_weekday}

    return aggs


def plot_heatmap(raw_data, title, ax_props=None, more_props=None):

    more_props = {} if more_props is None else more_props
    rows = len(ax_props['yticklabels'])
    cbmax = raw_data.get('cbmax', None)
    # normalized
    MAX = np.max(raw_data['data'])
    data = raw_data['data'] / MAX

    fig = plt.figure()
    fig.suptitle(title, fontsize=18)

    gskw = {'hspace': 0}
    if cbmax:
        gskw['bottom'] = 0.15
    gshmap = gs.GridSpec(rows, 1, **gskw)
    axes = list(fig.add_subplot(gs) for gs in gshmap)

    IMSHOW_OPTS = {
        'aspect': 'auto',
        'cmap': plt.get_cmap('Purples'),
        'interpolation': 'none',
    }
    for ax, row_data, label in zip(axes, data, ax_props['yticklabels']):
        plot_data = np.vstack((row_data, row_data))
        ax.imshow(plot_data, **IMSHOW_OPTS)
        pos = list(ax.get_position().bounds)
        x = pos[0] - 0.01
        y = pos[1] + pos[3] / 2
        fig.text(x, y, label, va='center', ha='right', fontsize=14)

    for i, ax in enumerate(axes):
        ax.set(xlim=0, yticks=[], **ax_props)
        prop_name = 'xminorticks'
        if prop_name in more_props:
            ax.xaxis.set_ticks(more_props[prop_name], minor=True)

        if i < rows - 1:
            for j, tick in enumerate(ax.xaxis.get_major_ticks()):
                tick.label1On = False
                tick.label2On = False

        ax.grid(which='major', linestyle='-')
        ax.grid(which='minor', linestyle=':')

    # draw the scale / colorbar
    if cbmax is not None:
        gscbar = gs.GridSpec(1, 1, top=0.10, bottom=0.05)
        ax = fig.add_subplot(gscbar[0])
        N = MAX / cbmax
        colorbar = np.linspace(0, 1, 256)
        colorbar = np.vstack((colorbar, colorbar))
        ax.imshow(colorbar, **IMSHOW_OPTS)
        ax.set_xlim(left=0)
        ax.yaxis.set_ticks([])
        xlabels = list('{:%}'.format(x) for x in np.arange(5) / 4 * N)
        ax.xaxis.set_ticks(np.hstack((np.arange(4) / 4 * 256, [255])))
        ax.xaxis.set_ticklabels(xlabels)

    return fig


def plot_barh(raw_data, title, ax_props=None):

    values = raw_data['values']

    fig, ax = plt.subplots()
    fig.suptitle(title, fontsize=18)

    ypos = np.arange(values.size)

    ax.barh(ypos, values, color='#54487a', edgecolor='#dddaec', align='center')
    ax.invert_yaxis()
    ax.set(yticks=ypos, **ax_props)
    ax.grid(which='major', axis='x')

    return fig


def init_figure_params():

    global FIGURE_PARAMS

    BASE_TITLE = 'Gentoo emerge'
    WKD_LABELS = ('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday',
                  'Saturday', 'Sunday')

    O24_LABELS = ('00:00', '06:00', '12:00', '18:00')
    O24_MAJORTICKS = range(0, 24 * 60, 6 * 60)
    O24_MINORTICKS = range(0, 24 * 60, 3 * 60)

    title = 'Likelihood to merge of each minute in 24-hour by weekdays'
    title = BASE_TITLE + ': ' + title

    MONTH_DAYS = (31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
    MONTH_LOCS = np.cumsum(np.array(MONTH_DAYS))
    MONTH_LOCS = np.hstack(([0], MONTH_LOCS[:-1]))
    MONTH_LABELS = ('January', 'February', 'March', 'April', 'May', 'June',
                    'July', 'August', 'September', 'October', 'November',
                    'December')

    WKD_MAJORTICKS = range(0, 7 * 24 * 60, 24 * 60)

    FIGURE_PARAMS = {
        'weekday_24hour': [
            plot_heatmap,
            BASE_TITLE + (': Likelihood to merge of each minute '
                          'in 24-hour by weekdays'),
            {
                'yticklabels': WKD_LABELS,
                'xticks': O24_MAJORTICKS,
                'xticklabels': O24_LABELS,
            },
            {
                'xminorticks': O24_MINORTICKS,
            },
        ],
        'year_days': [
            plot_heatmap,
            BASE_TITLE + ': historical emerges',
            {
                'xticks': MONTH_LOCS,
                'xticklabels': MONTH_LABELS,
            },
        ],
        'year_24hour': [
            plot_heatmap,
            BASE_TITLE + ': over 24-hour by years',
            {
                'xticks': O24_MAJORTICKS,
                'xticklabels': O24_LABELS,
            },
            {
                'xminorticks': O24_MINORTICKS,
            },
        ],
        'year_weekday': [
            plot_heatmap,
            BASE_TITLE + ': over weekdays by years',
            {
                'xticks': WKD_MAJORTICKS,
                'xticklabels': WKD_LABELS,
            },
        ],
        'yearly_total': [
            plot_barh,
            BASE_TITLE + ': emerging time by years',
            {
                'xlabel': 'Total emerging time (hour)',
            },
        ],
        'yearly_avg_daily': [
            plot_barh,
            BASE_TITLE + ': average daily emerging time by years',
            {
                'xlabel': 'Average daily emerging time (minute)',
            },
        ],
    }


def plot_graphs(aggs, args):

    YEAR_LABELS = aggs['YEAR_LABELS']
    for figure in ('year_days', 'year_24hour', 'year_weekday', 'yearly_total',
                   'yearly_avg_daily'):
        FIGURE_PARAMS[figure][2]['yticklabels'] = YEAR_LABELS

    for name in args.figures:
        item = FIGURE_PARAMS[name]
        plot_func = item[0]
        plot_args = item[1:]
        figure = plot_func(aggs[name], *plot_args)
        if args.figsave:
            figure.savefig('%s/%s.png' % (args.saveto, name))

    if not args.noshow:
        plt.show()


def main():

    init_figure_params()
    parser = argparse.ArgumentParser(description=__description__)
    parser.add_argument('-f', nargs='+', default=['all'], dest='figures',
                        choices=['all'] + list(FIGURE_PARAMS.keys()),
                        help='figures to generate (default %(default)s)')
    parser.add_argument('-W', '--width', type=int, default=DEFAULT_WIDTH,
                        help='width of figures (default: %(default)s)')
    parser.add_argument('-H', '--height', type=int, default=DEFAULT_HEIGHT,
                        help='height of figures (default: %(default)s)')
    parser.add_argument('-S', '--noshow', action='store_true',
                        help='do not show figures')
    parser.add_argument('-s', '--figsave', action='store_true',
                        help='save figures as images')
    parser.add_argument('-t', '--saveto', default='/tmp',
                        help='where to save images (default: %(default)s)')
    parser.add_argument('csvfile', nargs='?', type=open, default=DEFAULT_CSV)
    args = parser.parse_args()

    if 'all' in args.figures:
        args.figures = list(FIGURE_PARAMS.keys())
    args.saveto = args.saveto.rstrip('/')

    emerges = read_emerges(args.csvfile)
    bins = bin_data(emerges)
    aggs = agg_data(bins)

    dpi = plt.rcParams['figure.dpi']
    plt.rcParams['savefig.dpi'] = dpi
    plt.rcParams["figure.figsize"] = (args.width / dpi, args.height / dpi)

    plot_graphs(aggs, args)


if __name__ == '__main__':
    main()
