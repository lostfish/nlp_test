#! /usr/bin/python

import sys
import random

infile = sys.argv[1]
m = int(sys.argv[2])

n = 0
for line in file(infile):
    n += 1

a = set(random.sample(range(n), m))
i = 0
for line in file(infile):
    if i in a:
        print line,
    i += 1

