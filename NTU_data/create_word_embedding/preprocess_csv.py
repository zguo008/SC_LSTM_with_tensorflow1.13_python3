#task1: read csv file
#task2: get non-repeatitive keywords
#task3: generate sentence

ID_DICT = {"b827eb52c6ce":"248_Simei_St_1",
           "b827eb7680c5":"134_Simei_St_1",
           "b827ebc58e3c":"523C_Tampines_Central_7",
           "b827eb22d50a":"714_Tampines_St_71",
           "b827eba5f2c5":"723_Tampines_St_71",
           "b827eb02166a":"106_Tampines_St_11",
           "b827ebb954cd":"872B_Tampines_St_86",
           "b827eb0a63c9":"876_Tampines_St_84",
           "b827ebbc6e34":"381_Tampines_St_32",
           "b827eb2c0a4e":"373_Tampines_St_34",
           "b827eb712b1a":"YH_Basketball",
           "b827eb3e52b8":"YH_Carpark",
           "b827eb452ccb":"YH_Coffee_shop",
           "b827ebf3744c":"YH_Playground",
           "b827eb39963e":"YH_Saloon"}

import pandas as pd
import numpy as np 
data = pd.read_csv("../FYP_data.csv")



noise_class = []
sensor_id = []
laeq = []
time = []


location = []
noise_level = []
_time = []
for i in range(5000): #total 5000 files
    row = data.iloc[i]
    row = row[0].split(";")
    noise_class.append(row[0])
    sensor_id.append(row[1])
    laeq.append(row[2])
    time.append(row[3])

"""
process noise class 
"""
noise_class_save = noise_class
noise_class = set(noise_class)

"""
process sensor location
"""
sensor_id_save = sensor_id
sensor_id = set(sensor_id)
for idx in sensor_id:
    location.append(ID_DICT[idx])

"""
process sound level
"""

for l in laeq:
    level = float(l)
    if level>80:
        noise_level.append("completely_unacceptable")
    elif level<=80 and level>75:
        noise_level.append("usually_unacceptable")
    elif level<=75 and level>70:
        noise_level.append("less_acceptable")
    elif level<=70:
        noise_level.append("usually_acceptable")
laeq_save = noise_level
noise_level = set(noise_level)

"""
process time
"""

for t in time:
    t = t.split(" ")[1]
    hour = t[0:2]
    hour = int(hour)
    if hour<12:
        _time.append("morning")
    elif hour>=12 and hour<17:
        _time.append("afternoon")
    elif hour>=17 and hour<22:
        _time.append("evening")
_time_save = _time
_time = set(_time)

"""
write in all keywords
"""

with open("keywords.txt","a") as f:
    for n in noise_class:
        f.write(n+'\n')
    for l in location:
        f.write(l+'\n')
    for level in noise_level:
        f.write(level+'\n')
    for t in _time:
        f.write(t+'\n')
assert len(noise_class_save)==len(sensor_id_save)==len(laeq_save)==len(_time_save)

"""
using template to create sentence
"""

for i in range(len(_time_save)):
    sentence = "In the {} there is {} {} noise around {}".format(_time_save[i],laeq_save[i],noise_class_save[i],ID_DICT[sensor_id_save[i]])
    #print(sentence)
    keywords =  "{} {} {} {}".format(_time_save[i],laeq_save[i],noise_class_save[i],ID_DICT[sensor_id_save[i]])
    with open("TrainingData_text_NTU.txt","a") as f:
        f.write(sentence+"\n")
    with open("TrainingData_Keywords_NTU.txt","a") as f2:
        f2.write(keywords+'\n')
    
