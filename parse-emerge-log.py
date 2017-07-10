#!/usr/bin/env python3
# parsing emerge.log from stdin
# Usage:
#   $ sudo cat /var/log/emerge.log | ./parse-emerge-log.py
#   $ parse-emerge-log.py <emerge.log

import logging as log
import sys

CSV_FILE = 'emerge.csv'
LOG_FORMAT = ('%(asctime)s.%(msecs)03d %(levelname)8s '
              '%(funcName)s:%(lineno)d: %(message)s')
LOG_DATEFMT = '%H:%M:%S'
log.basicConfig(format=LOG_FORMAT, datefmt=LOG_DATEFMT, level='DEBUG')


def STmap(line):

    if 'Started emerge' in line:
        return 'S'
    elif 'terminating.' in line:
        return 'T'


def update_data(data):
    '''update data (dict) using data['lines']

    lines is the raw lines of the log file.

    This function update keys of data:

    - STline: single line (str) with symbols S and T representing emerging
      starting and terminating lines.
    - STlns: mapping the index in STline to index in lines.
    '''
    lines = data['lines']

    g = ((ln, STmap(line)) for ln, line in enumerate(lines))
    g = ((ln, symbol) for ln, symbol in g if symbol)
    STlns, STline = zip(*g)

    STline = ''.join(STline)
    num_S = len(STline.replace('T', ''))
    msg = 'log ha {:,d} lines ({:,d} S + {:,d} T)'
    log.info(msg.format(len(lines), num_S, len(STline) - num_S))

    data['STline'] = STline
    data['STlns'] = STlns


def all_sub_STline(data, sub):

    STline = data['STline']
    STlns = data['STlns']
    lns = []
    pos = 0
    sublen = len(sub)
    try:
        while pos < len(STline):
            pos = STline.index(sub, pos)
            lns.append([STlns[p] for p in range(pos, pos + sublen)])
            pos += sublen
    except ValueError:
        pass

    return len(lns), lns


def fix_concurrent(data, sub):
    '''Fixing some of concurrent emerge runs in pattern like S(ST){1..#}T

    Since they are overlapping the outermost ST, the fix is simplely removing
    the inner-pairs.
    '''

    num, cc_lns = all_sub_STline(data, sub)
    if not num:
        return

    log.info('fixing {:d} {}...'.format(num, sub))
    lines = data['lines']
    for lns in cc_lns:
        # blank lines in (ln1, ln4)
        lines[lns[0] + 1:lns[-1]] = [None] * (lns[-1] - lns[0] - 1)

    data['lines'] = list(filter(None, lines))
    update_data(data)


def fix_noT_resume(data):
    '''When --resume is used, it's likely there would be no T before S, for
    examples:

        1239847830:  >>> emerge (2 of 5) app-editors/vim-core-7.2 to /
        1239848753: Started emerge on: Apr 16, 2009 02:25:53
        1239848753:  *** emerge --resume

        1257126182:  === (13 of 13) Compiling/Merging ([...])
        1257126844: Started emerge on: Nov 02, 2009 09:54:04
        1257126844:  *** emerge --quiet --resume

    Adding fake T lines using the timestamp from the line before S line.

    Normal cases would look like:

        1257199916:  *** exiting unsuccessfully with status '1'.
        1257199916:  *** terminating.
        1257200005: Started emerge on: Nov 03, 2009 06:13:25
        1257200005:  *** emerge --quiet --resume

        1267590749:  *** RESTARTING emerge via exec() after change of [...]
        1267590749:  *** terminating.
        1267590750: Started emerge on: Mar 03, 2010 12:32:30
        1267590750:  *** emerge --quiet --ignore-default-opts --resume [...]
    '''

    lines = data['lines']
    resume_lns = [ln for ln, line in enumerate(lines) if '--resume' in line]

    noT_lns = [ln for ln in resume_lns if 'terminating' not in lines[ln - 2]]
    num = len(noT_lns)
    if not num:
        return

    msg = 'fixing {:d} with no T line out of {:d} --resume'
    log.info(msg.format(len(noT_lns), len(resume_lns)))
    noT_lns.reverse()
    for ln in noT_lns:
        # inserting fake T line
        ts = lines[ln - 2].split(':')[0]
        lines[ln - 1:ln - 1] = [ts + ':  *** terminating.']

    update_data(data)


def fix_noT(data):
    '''Fixing no T line before S line'''

    num, SSlns = all_sub_STline(data, 'SS')
    if not num:
        return

    log.info('fixing {:d} SS lines...'.format(num))
    lines = data['lines']
    SSlns.reverse()
    for S1ln, S2ln in SSlns:
        # inserting fake T line
        ts = lines[S2ln - 1].split(':')[0]
        lines[S2ln:S2ln] = [ts + ':  *** terminating.']

    update_data(data)


def fix_TT(data):
    '''Strange doulbe terminating messages:
    1293318169:  *** terminating.
    1293318169:  *** terminating.
    '''

    num, TTlns = all_sub_STline(data, 'TT')
    if not num:
        return

    log.info('fixing {:d} TT lines...'.format(num))
    lines = data['lines']
    TTlns.reverse()
    for T1ln, T2ln in TTlns:
        if T1ln + 1 != T2ln or lines[T1ln] != lines[T2ln]:
            continue
        del lines[T2ln]

    update_data(data)


def main():

    data = {'lines': sys.stdin.readlines()}
    update_data(data)

    # fixing
    fix_noT_resume(data)
    for i in range(10, 0, -1):
        fix_concurrent(data, 'S{}T'.format('ST' * i))
    fix_noT(data)
    fix_TT(data)
    assert data['STline'].replace('ST', '') == ''

    lines = data['lines']
    STlns = data['STlns']
    num = len(STlns)
    with open(CSV_FILE, 'w') as f:
        f.write('"START","END"\n')
        for i in range(num // 2):
            Sts = lines[STlns[i * 2]].split(':')[0]
            Tts = lines[STlns[i * 2 + 1]].split(':')[0]
            f.write('{},{}\n'.format(Sts, Tts))

    log.info('{} emerge time ranges written to {}'.format(num // 2, CSV_FILE))


if __name__ == '__main__':
    main()
