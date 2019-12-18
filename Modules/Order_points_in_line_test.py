import numpy as np


curr_corners = (np.array([ 184,  184,  184,  184,  184,  184,  184,  184,  184,  350,  350,
        350,  350,  350,  350,  350,  350,  350,  516,  516,  516,  516,
        516,  516,  516,  516,  516,  683,  683,  683,  683,  683,  683,
        683,  683,  683,  849,  849,  849,  849,  849,  849,  849,  849,
        849, 1015, 1015, 1015, 1015, 1015, 1015, 1015, 1015, 1015],
      dtype=np.int64), np.array([ 184,  350,  516,  683,  849, 1015, 1182, 1348, 1514,  184,  350,
        516,  683,  849, 1015, 1182, 1348, 1514,  184,  350,  516,  683,
        849, 1015, 1182, 1348, 1514,  184,  350,  516,  683,  849, 1015,
       1182, 1348, 1514,  184,  350,  516,  683,  849, 1015, 1182, 1348,
       1514,  184,  350,  516,  683,  849, 1015, 1182, 1348, 1514],
      dtype=np.int64))

sq_dists = (np.square(curr_corners[0][None, ] - curr_corners[0][:, None]) +
            np.square(curr_corners[1][None, ] - curr_corners[1][:, None]))



A, B, C, D = (0, 8, 45, 53)

hold = np.where(sq_dists > 0)
alpha = 0.8
dist_thresh = alpha*np.sqrt(np.min(sq_dists[hold]))

pt_1 = A
pt_2 = C
pt_1_y = curr_corners[0][pt_1]
pt_1_x = curr_corners[1][pt_1]
pt_2_y = curr_corners[0][pt_2]
pt_2_x = curr_corners[1][pt_2]

dy = pt_2_y - pt_1_y
dx = pt_2_x - pt_1_x
x2y1_x1y2 = pt_2_x*pt_1_y - pt_1_x*pt_2_y

dist_from_line = np.abs(dy*curr_corners[1]-dx*curr_corners[0]+x2y1_x1y2)/np.sqrt(dy*dy+dx*dx)
ind = np.where(dist_from_line<dist_thresh)
order = np.argsort(sq_dists[pt_1][ind])
ind[0][order]