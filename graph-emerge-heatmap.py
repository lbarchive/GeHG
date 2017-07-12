#!/usr/bin/env python3


import argparse
import csv
import datetime as dt
import logging as log
import platform

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap as LSC

__description__ = 'Gentoo emerge Heatmap Generator'


DEFAULT_CSV = 'emerge.csv'
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
FIGURE_PARAMS = None
DEFAULT_RECT = [0.05, 0.1, 0.925, 0.825]  # L, B, W, H

# from webpage, #62548f = rgb(98, 84, 143)
GENTOO_PURPLE1 = '#62548f'
GENTOO_PURPLE2 = '#dddaec'
GENTOO_PURPLES = np.array([98, 84, 143]) / 255
GENTOO_PURPLES = [[1.0] * 3, GENTOO_PURPLES]
GENTOO_PURPLES = LSC.from_list('Gentoo-Purples', GENTOO_PURPLES)

LOG_FORMAT = ('%(asctime)s.%(msecs)03d %(levelname)8s '
              '%(funcName)s:%(lineno)d: %(message)s')
LOG_DATEFMT = '%H:%M:%S'
log.basicConfig(format=LOG_FORMAT, datefmt=LOG_DATEFMT, level='DEBUG')


# patch and insight for ax.xaxis and footer from:
#   https://stackoverflow.com/a/17562747/242583 by tacaswell
#   https://stackoverflow.com/a/30860027/242583 by eric
def _blit_draw(self, artists, bg_cache):
    # Handles blitted drawing, which renders only the artists given instead
    # of the entire figure.
    updated_ax = []
    for a in artists:
        # If we haven't cached the background for this axes object, do
        # so now. This might not always be reliable, but it's an attempt
        # to automate the process.
        if a.axes not in bg_cache:
            # get bbox from a.axes.figure instead of a.axes
            bbox = a.axes.figure.bbox
            bg_cache[a.axes] = a.figure.canvas.copy_from_bbox(bbox)
        a.axes.draw_artist(a)
        updated_ax.append(a.axes)

    # After rendering all the needed artists, blit each axes individually.
    for ax in set(updated_ax):
        # get bbox from ax.figure instead of ax
        ax.figure.canvas.blit(ax.figure.bbox)

# MONKEY PATCH!!
animation.Animation._blit_draw = _blit_draw


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
    log.info('binning emerge sessions...')
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


def agg_likelihood(dts, bins_minute):
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

    likelihood_by_weekday_timeofday = [x[:] for x in [[0] * 24 * 60] * 7]

    WD = dts[0].weekday()
    for i in range(len(bins_minute)):
        wd = (i + WD) % 7
        z = zip(likelihood_by_weekday_timeofday[wd], bins_minute[i])
        likelihood_by_weekday_timeofday[wd] = [x + y for x, y in z]

    return {
        'dts': dts,
        'range': (dts[0], dts[-1]),
        'data': likelihood_by_weekday_timeofday,
        'cbmax': np.ceil(dts.size / 7) * 60,
    }


def agg_animated_likelihood(aggs):

    window_size = 52
    interval = 1
    windows = []
    r1 = 0
    while r1 <= DAYS - window_size * 7:
        msg = 'generating animated_likelihood window at day index {}'
        log.info(msg.format(r1))
        r2 = r1 + window_size * 7
        windows.append(agg_likelihood(dts[r1:r2], bins_minute[r1:r2]))
        r1 += interval * 7
    # last window
    windows.append(agg_likelihood(dts[-window_size:],
                                  bins_minute[-window_size:]))
    return windows


def agg_data(bins, figures):
    '''Data aggregation

    Returns a dict of the following data:
    - dts: array of dates (datetime.date)
    - start and end: start and end of data range
    - see below
    '''
    log.info('aggregating data...')
    aggs = {}

    dts = np.array(bins['dts'])
    aggs['dts'] = dts
    fm = dts[0]
    to = dts[-1]
    agg_range = fm, to
    aggs['range'] = agg_range
    DAYS = dts.size
    bins_minute = np.array(bins['by_minute'])
    bins_day = np.array(bins['by_day'])

    ###################################
    # likelihood_by_weekday_timeofday #
    ###################################

    aggs['likelihood_by_weekday_timeofday'] = agg_likelihood(dts, bins_minute)

    #######################
    # animated_likelihood #
    #######################

    figure = 'animated_likelihood'
    if figure in figures:
        aggs[figure] = agg_animated_likelihood(aggs)

    ##############
    # historical #
    ##############
    # Note:
    # - February 29 is inserted to every year for alignments.
    #
    # Year 1 [ Jan1 Jan2 ... Feb29 ... Dec31 ]
    # Year 2 [ ............................. ]
    #   :
    # YEARS  [ ............................. ]

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
    historical = np.hstack((noleaf[:, :59], Feb29, noleaf[:, 59:]))

    # putting leaf days back
    leaf_dts = np.array(dts)[nb]
    leaf_days = bins_day[nb]
    for d, day in zip(leaf_dts, leaf_days):
        historical[d.year - fm.year, 59] = day

    aggs['historical'] = {
        'dts': dts,
        'range': agg_range,
        'data': historical,
    }

    ##########
    # yearly #
    ##########

    yearly_seconds = np.sum(historical, axis=1)
    yearly = yearly_seconds / 3600
    aggs['yearly'] = {
        'dts': dts,
        'range': agg_range,
        'data': yearly,
    }

    #########################
    # daily_average_by_year #
    #########################

    daily_average_by_year = yearly_seconds / 60
    for y in range(YEARS):
        days = sum(1 for d in dts if d.year == YEAR_NUMBERS[y])
        daily_average_by_year[y] /= days

    aggs['daily_average_by_year'] = {
        'dts': dts,
        'range': agg_range,
        'data': daily_average_by_year,
    }

    #####################
    # by_year_timeofday #
    #####################
    # 24 * 60 = 1440 bins, bin width = 1 minute
    #          | 1st hour |     | last hour |
    # Year 1 [ 0 1 2 ... 59 ... 1380 ... 1439 ]
    # Year 2 [ .............................. ]
    #   :
    # YEARS  [ .............................. ]

    by_year_timeofday = np.zeros((YEARS, 24 * 60))
    for y in range(YEARS):
        b = np.array(list(d.year == y + fm.year for d in dts))
        year_minutes = bins_minute[b]
        tsum = np.sum(year_minutes.reshape(-1, 24 * 60), axis=0)
        by_year_timeofday[y] = tsum

    aggs['by_year_timeofday'] = {
        'dts': dts,
        'range': agg_range,
        'data': by_year_timeofday,
    }

    ###################
    # by_year_weekday #
    ###################
    # Heatmap over weekdays by years
    #
    # 7 * 24 * 60 = 10,080 bins, bin width = 1 minute
    #          |  Monday  |     |   Sunday   |
    # Year 1 [ 0 1 ... 1439 ... 8640 ... 10079 ]
    # Year 2 [ ............................... ]
    #   :
    # YEARS  [ ............................... ]

    by_year_weekday = np.zeros((YEARS, 7 * 24 * 60))
    for y in range(YEARS):
        b = np.array(list(d.year == y + fm.year for d in dts))
        minutes = bins_minute[b]

        # padding to align
        start_weekday = dts[b][0].weekday()
        if start_weekday:
            minutes = np.vstack((np.zeros((start_weekday, 24 * 60)), minutes))
        minutes.resize((np.ceil(minutes.size / (7 * 24 * 60)), 7 * 24 * 60))
        by_year_weekday[y] = np.sum(minutes, axis=0)

    aggs['by_year_weekday'] = {
        'dts': dts,
        'range': agg_range,
        'data': by_year_weekday,
    }

    return aggs


def footer_text(name, start, end):

    diff = end - start
    days = diff.days + 1
    text = 'Range: {} to {} ({:,} days)'.format(start, end, days)
    if name:
        text += ' / Machine: {}'.format(name)
    text += ' / Generated by GeHG'
    return text


def plot_footer(ax, *args):

    text = footer_text(*args)
    footer = ax.text(0.9875, 0.025, text, color='gray',
                     transform=plt.gcf().transFigure,
                     horizontalalignment='right', verticalalignment='center')
    return footer


def plot_heatmap(raw_data, title, ax_props=None, more_props=None):

    more_props = {} if more_props is None else more_props
    rows = len(ax_props['yticklabels'])
    cbmax = raw_data.get('cbmax', None)
    # normalized
    MAX = np.max(raw_data['data'])
    data = raw_data['data'] / MAX

    fig = plt.figure()
    fig.suptitle(title, fontsize=18)

    rect = np.array(DEFAULT_RECT) + more_props['rect_adjust']
    if cbmax:
        rect += [0, 0.1, 0, -0.1]
    ax = plt.axes(rect)

    animated = more_props.get('animated', False)
    IMSHOW_OPTS = {
        'aspect': 'auto',
        'cmap': GENTOO_PURPLES,
        'interpolation': 'none',
        'animated': animated,
    }
    imdata = ax.imshow(data, **IMSHOW_OPTS)
    ax.xaxis.set_animated(animated)
    ax.yaxis.set_animated(animated)
    ax.set(xlim=0, yticks=np.arange(rows), **ax_props)
    # don't show yticks
    ax.tick_params(axis='y', length=0)

    prop_name = 'last-xlabel-right-align'
    if prop_name in more_props:
        xlabels = ax.xaxis.get_ticklabels()
        xlabels[-1].set_horizontalalignment('right')

    prop_name = 'xminorticks'
    if prop_name in more_props:
        ax.xaxis.set_ticks(more_props[prop_name], minor=True)

    ax.grid(axis='x', which='major', linestyle='-')
    ax.grid(axis='x', which='minor', linestyle=':')

    footer = plot_footer(ax, more_props['name'], *raw_data['range'])
    footer.set_animated(animated)

    if cbmax is None:
        return fig, imdata, footer

    # draw the scale / colorbar
    rect[1] = 0.1
    rect[3] = 0.05
    ax = plt.axes(rect)

    N = MAX / cbmax
    colorbar = np.linspace(0, 1, 256)
    colorbar = np.vstack((colorbar, colorbar))
    ax.imshow(colorbar, **IMSHOW_OPTS)
    ax.set_xlim(left=0, right=255.0)
    ax.set_ylim(top=-0.5, bottom=0.5)
    ax.yaxis.set_ticks([0])
    ax.yaxis.set_ticklabels([more_props['cbmax_label']])
    ax.tick_params(axis='y', length=0)
    xlabels = list('{:%}'.format(x) for x in np.arange(5) / 4 * N)
    ax.xaxis.set_ticks(np.hstack((np.arange(4) / 4 * 256, [255])))
    ax.xaxis.set_ticklabels(xlabels)
    ax.xaxis.set_animated(animated)
    ax.yaxis.set_animated(animated)

    prop_name = 'last-xlabel-right-align'
    if prop_name in more_props:
        xlabels = ax.xaxis.get_ticklabels()
        xlabels[-1].set_horizontalalignment('right')

    ax.grid(axis='x', which='major', linestyle='-')

    return fig, imdata, footer


def update_animated_heatmap(window, name, axes, im, footer, redraws):

    data = window['data']
    cbmax = window['cbmax']

    MAX = np.max(data)
    im.set_data(data / MAX)

    ax_im, ax = axes
    N = MAX / cbmax
    xlabels = list('{:%}'.format(x) for x in np.arange(5) / 4 * N)
    ax.xaxis.set_ticklabels(xlabels)

    footer.set_text(footer_text(name, *window['range']))

    return redraws


def plot_animated_heatmap(windows, title, ax_props=None, more_props=None):

    fig, im, footer = plot_heatmap(windows[0], title, ax_props, more_props)

    axes = fig.get_axes()
    ax1, ax2 = axes
    redraws = [im, footer, ax1.xaxis, ax1.yaxis, ax2.xaxis, ax2.yaxis]

    anim = animation.FuncAnimation(
        fig,
        update_animated_heatmap,
        frames=windows,
        fargs=(more_props['name'], axes, im, footer, redraws),
        interval=250,
        blit=True,
    )

    return anim,


def plot_barh(raw_data, title, ax_props=None, more_props=None):

    data = raw_data['data']

    fig = plt.figure()
    fig.suptitle(title, fontsize=18)
    rect = np.array(DEFAULT_RECT) + more_props['rect_adjust']
    ax = plt.axes(rect)

    ypos = np.arange(data.size)
    ax.barh(ypos, data, color=GENTOO_PURPLE1, edgecolor=GENTOO_PURPLE2,
            align='center')
    ax.invert_yaxis()
    ax.set(yticks=ypos, **ax_props)
    ax.grid(which='major', axis='x')

    plot_footer(ax, more_props['name'], *raw_data['range'])

    return fig,


def init_figure_params():

    global FIGURE_PARAMS

    WKD_LABELS = ('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday',
                  'Saturday', 'Sunday')

    O24_LABELS = ('00:00', '06:00', '12:00', '18:00', '23:59')
    O24_MAJORTICKS = list(range(0, 24 * 60, 6 * 60)) + [24 * 60 - 1]
    O24_MINORTICKS = range(0, 24 * 60, 3 * 60)

    MONTH_DAYS = (31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
    MONTH_LOCS = np.cumsum(np.array(MONTH_DAYS))
    MONTH_LOCS = np.hstack(([0], MONTH_LOCS[:-1]))
    MONTH_LABELS = ('January', 'February', 'March', 'April', 'May', 'June',
                    'July', 'August', 'September', 'October', 'November',
                    'December')

    WKD_MAJORTICKS = range(0, 7 * 24 * 60, 24 * 60)

    FIGURE_PARAMS = {
        'historical': [
            plot_heatmap,
            'Historical Gentoo emerge Running Time',
            {
                'xticks': MONTH_LOCS,
                'xticklabels': MONTH_LABELS,
            },
            {
                'rect_adjust': [0] * 4,
            },
        ],
        'likelihood_by_weekday_timeofday': [
            plot_heatmap,
            'Gentoo emerge Running Likelihood by Weekday and Time of Day',
            {
                'yticklabels': WKD_LABELS,
                'xticks': O24_MAJORTICKS,
                'xticklabels': O24_LABELS,
            },
            {
                'cbmax_label': 'Likelihood',
                'last-xlabel-right-align': True,
                'rect_adjust': [0.0375, 0, -0.0375, 0],
                'xminorticks': O24_MINORTICKS,
            },
        ],
        'animated_likelihood': [
            plot_animated_heatmap,
            ('Animated Gentoo emerge Running Likelihood by Weekday and '
             'Time of Day'),
            {
                'yticklabels': WKD_LABELS,
                'xticks': O24_MAJORTICKS,
                'xticklabels': O24_LABELS,
            },
            {
                'animated': True,
                'cbmax_label': 'Likelihood',
                'last-xlabel-right-align': True,
                'rect_adjust': [0.0375, 0, -0.0375, 0],
                'xminorticks': O24_MINORTICKS,
            },
        ],
        'by_year_timeofday': [
            plot_heatmap,
            'Gentoo emerge Running Time by Year and Time of Day',
            {
                'xticks': O24_MAJORTICKS,
                'xticklabels': O24_LABELS,
            },
            {
                'last-xlabel-right-align': True,
                'rect_adjust': [0] * 4,
                'xminorticks': O24_MINORTICKS,
            },
        ],
        'by_year_weekday': [
            plot_heatmap,
            'Gentoo emerge Running Time by Year and Weekday',
            {
                'xticks': WKD_MAJORTICKS,
                'xticklabels': WKD_LABELS,
            },
            {
                'rect_adjust': [0] * 4,
            },
        ],
        'yearly': [
            plot_barh,
            'Gentoo emerge Yearly Running Time',
            {
                'xlabel': 'Yearly Running Time (hour)',
            },
            {
                'rect_adjust': [0, 0.025, 0, -0.025],
            },
        ],
        'daily_average_by_year': [
            plot_barh,
            'Gentoo emerge Daily Average Running Time by Year',
            {
                'xlabel': 'Daily Average Running Time (minute)',
            },
            {
                'rect_adjust': [0, 0.025, 0, -0.025],
            },
        ],
    }


def plot_graphs(aggs, args):

    YEAR_LABELS = aggs['YEAR_LABELS']
    for figure in ('historical', 'by_year_timeofday', 'by_year_weekday',
                   'yearly', 'daily_average_by_year'):
        FIGURE_PARAMS[figure][2]['yticklabels'] = YEAR_LABELS

    for figure in FIGURE_PARAMS:
        rect = DEFAULT_RECT.copy()
        FIGURE_PARAMS[figure][3]['rect'] = rect
        FIGURE_PARAMS[figure][3]['name'] = args.name

    for name in args.figures:
        item = FIGURE_PARAMS[name]
        plot_func = item[0]
        plot_args = item[1:]
        log.info('plotting {}...'.format(name))
        figure = plot_func(aggs[name], *plot_args)[0]
        if args.figsave:
            log.info('saving {}...'.format(name))
            filename = '%s/GeHG-%s' % (args.saveto, name)
            if plot_args[2].get('animated', False):
                # figure.save(filename + '.gif', writer='imagemagick', fps=26)
                # For GIF, needs about 3G memory to run it comfortably, perhaps
                # using hours_bin would help a bit.
                figure.save(filename + '.mp4', fps=26, bitrate=3000)
            else:
                figure.savefig(filename + '.png')

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
    parser.add_argument('-n', '--name', default=platform.uname().node,
                        help=('name for identification '
                              '(default network name: %(default)s)'))
    parser.add_argument('csvfile', nargs='?', type=open, default=DEFAULT_CSV)
    args = parser.parse_args()

    if 'all' in args.figures:
        args.figures = list(FIGURE_PARAMS.keys())
    args.saveto = args.saveto.rstrip('/')

    emerges = read_emerges(args.csvfile)
    bins = bin_data(emerges)
    aggs = agg_data(bins, args.figures)

    dpi = plt.rcParams['figure.dpi']
    plt.rcParams['savefig.dpi'] = dpi
    plt.rcParams["figure.figsize"] = (args.width / dpi, args.height / dpi)

    plot_graphs(aggs, args)


if __name__ == '__main__':
    main()
