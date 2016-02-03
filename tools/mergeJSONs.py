#!/usr/bin/env python3.5
import os
import sys
import subprocess


import json
import pprint

json_files_to_merge_with = []

print(len(sys.argv))
if len(sys.argv) == 1:
    raise NameError('Argument missing')
if len(sys.argv) == 2:
    root_folder = sys.argv[1]
if len(sys.argv) >= 3:
    root_folder = sys.argv[1]
    for arg in sys.argv[2:]:
        json_files_to_merge_with.append(arg)


data = []



def add_file_data_to_list(filename):
    with open(filename) as json_file:
        for line in json_file:
            data.append(json.loads(line))
    #print("Længde under: {0}".format(len(data)))
 
def save_json(data, filename="data.json"):
    with open(filename, 'w') as outfile:
        json.dump(data, outfile)


def get_ext_from_file(file_name, ch="."):
        ch_poses = [i for i, ltr in enumerate(file_name) if ltr == ch]  #gets the positions of all the dots in the filename
        if len(ch_poses) == 0:
            return ""
        ch_pos = max(ch_poses)+1   #gets the maximum position
        return file_name[ch_pos:]  #extract and return the file extention

def start_merging(start_folder, json_files):
    for (_,_,files) in os.walk(start_folder):
        for filename in files:
            if get_ext_from_file(filename) =="json":
                add_file_data_to_list(start_folder+"/"+filename)
    
    for filename in json_files:
        add_file_data_to_list(filename)


def create_filename(root_folder, json_files):
    filename = ""
    ch_poses = [i for i, ltr in enumerate(root_folder) if ltr == "/"]
    if len(ch_poses) == 0:
        filename = root_folder
    else:
        filename = "all_"+root_folder[(max(ch_poses)+1):]
    for s in json_files_to_merge_with:
        filename += "_"+s
    filename += ".json"
    return filename

base_dir = os.path.dirname(os.path.realpath(__file__))
start_folder = os.path.join(base_dir,root_folder)
print("Længde før: {0}".format(len(data)))
start_merging(start_folder, json_files_to_merge_with)
print("Længde efter: {0}".format(len(data)))

print("filename: {0}".format(create_filename(root_folder, json_files_to_merge_with)))
save_json(data, create_filename(root_folder, json_files_to_merge_with))
