#!/bin/bash
# Usage:
#     $ sudo ./parse-emerge-log.sh > emerge.csv


main()
{
    echo '"START","END"'

    </var/log/emerge.log \
    egrep 'Started emerge on|=== sync$|\*\*\* terminating\.' |
    while read line; do
        case "$line" in
        *Started*)
            START=${line%%:*}
            ISSYNC=
            ;;
        *sync)
            ISSYNC=1
            ;;
        *terminating*)
            if [[ ! $ISSYNC ]]; then
                echo $START,${line%%:*}
            fi
            ISSYNC=
            ;;
        esac
    done
}


main
