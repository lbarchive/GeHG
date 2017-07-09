#!/bin/bash
# Usage:
#     $ sudo ./parse-emerge-log.sh > emerge.csv


main()
{
    echo '"START","END"'

    </var/log/emerge.log \
    egrep -B1 'Started emerge on|=== sync$|\*\*\* terminating\.' |
    while read line; do
        # skip syncs
        if [[ "$line" = *sync ]]; then
            START=
            while read line; do
                [[ "$line" = *Started* ]] && break
            done
        fi

        case "$line" in
        *Started*)
            if [[ ! -z "$START" ]]; then
                # abnormal exiting, using timestamp from previous line
                echo $START,${PREV%%:*}
            fi
            START=${line%%:*}
            ;;
        *terminating*)
            # FIXME two emerge processes overlapping
            [[ ! -z "$START" ]] && echo $START,${line%%:*} || echo "$line" >&2
            START=
            ;;
        esac
        PREV="$line"
    done
}


main
