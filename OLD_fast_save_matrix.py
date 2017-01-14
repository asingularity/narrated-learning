

import numpy as np

def savetxt(fname, data):
    f = open(fname, 'w')
    rshape = data.shape[0]
    cshape = data.shape[1]

    for row in range(rshape):
        for col in range(cshape):
            f.write("%f "% data[row, col])
        f.write('\n')
    f.close()
