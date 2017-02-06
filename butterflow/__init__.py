# -*- coding: utf-8 -*-

import os
import settings
import ocl
import tempfile


__version__ = '0.2.4.dev0'

# need to do set up the environment before anything is run:

settings.default['version'] = __version__

# define temporary file locations
# since value of tempfile.tempdir is None python will search std list of
# dirs and will select the first one that the user can create a file in
# See: https://docs.python.org/2/library/tempfile.html#tempfile.tempdir
#
# butterflow will write renders to a temp file in tempdir and will move it
# to it's destination path when completed using shutil.move(). if the dest
# is on the current filesystem then os.rename() is used, otherwise the file
# is copied with shutil.copy2 then removed
settings.default['tempdir'] = os.path.join(tempfile.gettempdir(),
                                           'butterflow-{}'.format(__version__))

# where ocl cache files are stored:
settings.default['clbdir'] = os.path.join(settings.default['tempdir'], 'clb')

# make temporary directories
# and set the location of the clb cache
for x in [settings.default['clbdir'], settings.default['tempdir']]:
    if not os.path.exists(x):
        os.makedirs(x)
ocl.set_cache_path(settings.default['clbdir'] + os.sep)

ocl.set_num_threads(settings.default['ocv_threads'])
