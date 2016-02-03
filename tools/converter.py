#!/usr/bin/env python3.5
import os
import sys
import subprocess

root_folder = sys.argv[1]
base_dir = os.path.dirname(os.path.realpath(__file__))
start_folder = os.path.join(base_dir,root_folder)

for (_,_,files) in os.walk(start_folder):
	for filename in files:
		print("converting {}".format(filename))
		name_without_extension = filename.split(".")[0]
		handle = open("{}.json".format(os.path.join(start_folder,name_without_extension)),"w+")
		subprocess.run(["java",
						 "-jar",
						  os.path.join(base_dir,"avro-tools-1.7.7.jar"),
						  "tojson",
						  "{}".format(os.path.join(start_folder,filename))], stdout=handle)
		handle.close()