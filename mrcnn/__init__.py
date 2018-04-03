import os
import logging

import matplotlib
import numpy as np

# in case you are running on machine without display, e.g. server
if os.environ.get('DISPLAY', '') == '' \
        and matplotlib.rcParams['backend'] != 'agg':
    logging.warning('No display found. Using non-interactive Agg backend')
    # https://matplotlib.org/faq/usage_faq.html
    matplotlib.use('Agg')

# parse the numpy versions
np_version = [int(i) for i in np.version.full_version.split('.')]
# comparing strings does not work for version lower 1.10
if np_version >= [1, 14]:
    np.set_printoptions(legacy='1.13')
