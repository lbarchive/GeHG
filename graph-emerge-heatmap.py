#!/usr/bin/env python3


import csv
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr


def read_emerges():

    emerges = []
    with open('emerge.csv') as f:
        for emerge in csv.reader(f):
            if emerge[0] == 'START':
                continue
            r = tuple(dt.datetime.fromtimestamp(float(ts)) for ts in emerge)
            emerges.append(r)

    return emerges


def bin_data(emerges):

    BASE = emerges[0][0]
    td = emerges[-1][1] - BASE
    DAYS = td.days + (td.seconds > 0)
    days = [x[:] for x in [[0] * 24] * DAYS]

    BASE = BASE.replace(hour=0, minute=0, second=0)
    dts = [BASE.date() + dt.timedelta(days=i) for i in range(DAYS)]

    for emerge in emerges:
        fm = emerge[0].replace()
        to = emerge[1]
        while fm <= to:
            nt = fm.replace(minute=0, second=0)
            nt += dt.timedelta(hours=1)
            if nt > to:
                td = to - fm
            else:
                td = nt - fm
            days[(fm - BASE).days][fm.hour] += td.seconds
            fm = nt

    print(days)
    return days, dts


def agg_data(bins):

    days, dts = bins

    wday_hr = [x[:] for x in [[0] * 24] * 7]

    WD = dts[0].weekday()
    for i in range(len(days)):
        wd = i % 7
        wday_hr[wd] = [x + y for x, y in zip(wday_hr[wd], days[i])]

    return {'wday_hr': wday_hr}


def plot_wday_hr(wday_hr):

    fig, axes = plt.subplots(nrows=7)
    fig.subplots_adjust(top=0.95, bottom=0.05, left=0.2, right=0.99, hspace=0)
    axes[0].set_title('Weekdays versus 24 hours', fontsize=14)

    # normalized
    wday_hr = np.array(wday_hr)
    wday_hr = wday_hr / np.max(wday_hr)

    day_names = ('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday')
    for ax, day, day_name in zip(axes, wday_hr, day_names):
        data = np.vstack((day, day))
        ax.imshow(data, aspect='auto', cmap=plt.get_cmap('Purples'))
        pos = list(ax.get_position().bounds)
        x_text = pos[0] - 0.01
        y_text = pos[1] + pos[3]/2.
        fig.text(x_text, y_text, day_name, va='center', ha='right', fontsize=10)

    for i, ax in enumerate(axes):
        ax.set_xlim(left=0)
        ax.yaxis.set_ticks([])
        ax.xaxis.set_ticks(range(0, 24, 6))
        ax.xaxis.set_ticks(range(0, 24, 3), minor=True)
        if i < 6:
            for j, tick in enumerate(ax.xaxis.get_major_ticks()):
                tick.label1On = False
                tick.label2On = False
                
        ax.grid(which='major', linestyle='-')
        ax.grid(which='minor', linestyle=':')

    plt.show()


def main():

    emerges = read_emerges()
    bins = bin_data(emerges)
    aggs = agg_data(bins)

    plot_wday_hr(aggs['wday_hr'])



if __name__ == '__main__':
    main()
