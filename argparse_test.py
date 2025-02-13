#!/usr/bin/python
import argparse
import os
parser = argparse.ArgumentParser(prog='PDB test', description='print pdb', epilog='')
parser.add_argument('-f', '--file', type=str, help='PDB filename')
parser.add_argument('-p', '--print', action='store_true', help='Print to STDOUT?')
args = parser.parse_args()

filename = args.file
if not filename:
    files = os.listdir()
    files = [f for f in files if len(f)>=4]
    pdbs = [f for f in files if f[-4:] == '.pdb']
    if len(pdbs)>0:
        print("No file given, defaulting to", pdbs[0])
    else:
        exit("No file given, and no pdbs in current dir")

lines = []
with open('1chc.pdb', 'r') as f:
	lines = f.readlines()
if (len(lines) == 0):
    exit("PDB contains no content")
print("Successfully read file")

if args.print:
    for line in lines:
        print(line[:-1])

