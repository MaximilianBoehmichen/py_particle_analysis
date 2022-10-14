# -*- coding: utf-8 -*-
"""
Created 2022/06/20

@written by Kevin Maier (kevin.r.maier@tum.de)
    v0 writing of SMPS data to Excel for student internship
"""

import csv
import newSMPS_fileread_v4
import os

filename = newSMPS_fileread_v4.get_filename()
X, bar_width, Cn, time = newSMPS_fileread_v4.import_data(filename)

with open(f'{os.path.splitext(filename)[0]}.csv', 'w', encoding='UTF8', newline="") as f:
    writer = csv.writer(f)

    writer.writerow(X[0])
    # only writes one line of the particle sizes to the file, so when the measuring range is changed during the
    # measurements, this leads to errors

    for msmt in range(len(Cn)):
        writer.writerow(Cn[msmt])
