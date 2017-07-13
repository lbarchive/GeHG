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


Dependences
===========

* Python 3
* Matplotlib
* NumPy


Notes
=====

An *emerge run* is a continuous running of one or more emerge process, and in a *running duration* can have more than one emerge process running.  And a *running time* can be syncing, merging or unmerging.

GeHG only intends to look at any given time if emerge is running, it doesn't care of how many concurrent processes.


Copyright
=========

Licensed under the MIT License.
