#!/usr/bin/env python

from dnns.data import dataSplit
import sys

def main():
    fname = sys.argv[1]
    test_pct = float(sys.argv[2])
    hash_on_key = sys.argv[3]

    dataSplit(fname, test_pct, hash_on_key)

if __name__ == '__main__':
    main()
