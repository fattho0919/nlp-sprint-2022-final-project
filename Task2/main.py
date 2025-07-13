# -*- coding: utf-8 -*-
import os
import random

MED_list = os.listdir("./Task 2 Dataset/MED")
MED_files = ["./Task 2 Dataset/MED/"+ i for i in random.choices(MED_list, k=10)]

wf = open(f"MED_output", "w")
for MED_file in MED_files:
    f = open(MED_file, 'r')
    text = f.read()
    f.close()
    text = text.replace('\n', '')
    wf.write(text)
    wf.write("\n\n")
wf.close()