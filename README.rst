===============================
Gentoo emerge Heatmap Generator
===============================

GeHG parses ``emerge.log`` and generates some graphs.


Usage
=====

.. code:: sh

    # generating emerge.csv
    $ sudo cat /var/log/emerge.log | ./parse-emerge-log.py
    $ ./graph-emerge-heatmap.py  # use -h for help


Copyright
=========

Licensed under the MIT License.
