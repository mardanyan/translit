

from __future__ import print_function
import numpy as np
import theano
import codecs
import json
import argparse
#import utils
from datetime import datetime

import shutil
import glob

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--sourcepath', default='/home/tigran/rau/translit/files/text/*/wiki_*')
	parser.add_argument('--destpath', default='/home/tigran/rau/translit/translit/languages/hy-AM/data/all.txt')
	args = parser.parse_args()

	outfilenametmp = args.destpath + "_tmp"
	outfilename = args.destpath

	with open(outfilenametmp, 'wb') as outfile:
		for filename in glob.glob(args.sourcepath):			
			# print(filename)
			with open(filename, 'rb') as readfile:
				shutil.copyfileobj(readfile, outfile)


	output = open(outfilename, "wb")
	print(outfilename)
	print(outfilenametmp)

	for i, line in enumerate(outfilenametmp):
	 	print(i)
	# 	if i == 0:
	# 		output.write(line)
	# else:
	# 	if not line.startswith('<') :
	# 		output.write(line)



if __name__ == '__main__':
	main()