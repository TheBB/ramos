import os
import subprocess
import sys

environ = dict(os.environ)
environ['LD_LIBRARY_PATH'] = '/usr/local/lib:{}'.format(
    environ.get('LD_LIBRARY_PATH', '')
)
environ['PYTHONPATH'] = '/usr/local/lib/python3.5/site-packages:{}'.format(
    environ.get('PYTHONPATH', '')
)

subprocess.run(['python3', '-m', 'gmesh'] + sys.argv[1:], env=environ)
