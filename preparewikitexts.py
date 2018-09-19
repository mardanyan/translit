

from __future__ import print_function
import numpy as np
import theano
import codecs
import json
import argparse
import utils
from datetime import datetime

import shutil
import glob

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--sourcepath', default='/home/tigran/rau/translit/files/text/')
	parser.add_argument('--destpath', default='/home/tigran/rau/translit/translit/languages/hy-AM/data/all.txt')
	args = parser.parse_args()

	outfilename = args.destpath

	with open(outfilename, 'wb') as outfile:
		for filename in glob.glob('wiki__*'):
			if filename == outfilename:
			# don't want to copy the output into the output
				continue
			print(filename)
		#with open(filename, 'rb') as readfile:
		#	shutil.copyfileobj(readfile, outfile)






if __name__ == '__main__':
	main()