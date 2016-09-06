import argparse
import sys
from . import gui
from . import tools


parser = argparse.ArgumentParser()
parser.add_argument('--map', action='store_const', const='map', dest='mode')
parser.add_argument('--structure', action='store_const', const='structure', dest='mode')
parser.add_argument('files', type=str, nargs='*')
args = parser.parse_args()

if args.mode == 'map':
    gui.run()
elif args.mode == 'structure':
    tools.structure(args.files[0])
