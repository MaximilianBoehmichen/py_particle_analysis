# -*- coding: utf-8 -*-
"""
Dist_Data_csv_convert.py

writing of SMPS data to Excel for student internship

Created 2022-06-20
@written by Kevin Maier (kevin.r.maier@tum.de)

2022-10-17: transferred to gitlab, old versioning was removed, so all referenced files ..._vX were renamed without
    version number
"""

import csv
import os
import pandas as pd
import Particle_analysis

data = Particle_analysis.get_data()

X, dX, dlogX, Cn, add_info, filename = data["X"], data["dX"], data["dlogX"], data["Cn"], data["add_info"], \
    data["filename"]

with open(f'{os.path.splitext(filename)[0]}.csv', 'w', encoding='UTF8', newline="") as f:
    writer = csv.writer(f)

    if ["Comment"] in add_info.columns.values:
        comment = "Comment"
    else:
        comment = "Scan Nr"

    for msmt in range((len(Cn))):
        scan_nr = msmt+1
        x_row = ["Scan Nr", "X", "nm"]
        [x_row.append(i) for i in X[msmt]]
        writer.writerow(x_row)
        dx_row= [scan_nr, "dX", "nm"]
        [dx_row.append(i) for i in dX[msmt]]
        writer.writerow(dx_row)
        dlogx_row = [comment, "dlogX", "nm"]
        [dlogx_row.append(i) for i in dlogX[msmt]]
        writer.writerow(dlogx_row)
        Cn_row = [f"{add_info[comment][msmt]}", "Conc.", "1/cm^3"]
        [Cn_row.append(i) for i in Cn[msmt]]
        writer.writerow(Cn_row)

