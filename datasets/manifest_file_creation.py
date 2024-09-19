import os
import json
from tqdm import tqdm
import pandas as pd

ispfdv1_black = "< Path to data folder >"
parent_dir = dict()
for i in tqdm(os.listdir(ispfdv1_black)):
    for j in os.listdir(os.path.join(ispfdv1_black,i)):
        parent_dir[i+"_"+j] = dict()
        parent_dir[i+"_"+j]['Contactless'] = list()
        parent_dir[i+"_"+j]['Contactbased'] = list()
        for j1 in os.listdir(os.path.join(ispfdv1_black,i,j)):
            if j1.count(".") == 2:
                parent_dir[i+"_"+j]['Contactbased'].append(os.path.join(ispfdv1_black,i,j,j1))
            else:
                parent_dir[i+"_"+j]['Contactless'].append(os.path.join(ispfdv1_black,i,j,j1))

with open("manifest_jsons/ispfdv1_blackback.json",'w') as js:
    json.dump(parent_dir,js,indent=4)

ispfdv1_black = "< Path to data folder >"
parent_dir = dict()
for i in tqdm(os.listdir(ispfdv1_black)):
    for j in os.listdir(os.path.join(ispfdv1_black,i)):
        parent_dir[i+"_"+j] = dict()
        parent_dir[i+"_"+j]['Contactless'] = list()
        parent_dir[i+"_"+j]['Contactbased'] = list()
        for j1 in os.listdir(os.path.join(ispfdv1_black,i,j)):
            if j1.count(".") == 2:
                parent_dir[i+"_"+j]['Contactbased'].append(os.path.join(ispfdv1_black,i,j,j1))
            else:
                parent_dir[i+"_"+j]['Contactless'].append(os.path.join(ispfdv1_black,i,j,j1))

with open("manifest_jsons/ispfdv1_colorback.json",'w') as js:
    json.dump(parent_dir,js,indent=4)

ispfdv2_black = "< Path to data folder for BI>"
parent_dir = dict()
for i in tqdm(os.listdir(ispfdv2_black)):
    for j in os.listdir(os.path.join(ispfdv2_black,i)):
        parent_dir[i+"_"+j] = dict()
        parent_dir[i+"_"+j]['Contactless'] = list()
        parent_dir[i+"_"+j]['Contactbased'] = list()
        for j1 in os.listdir(os.path.join(ispfdv2_black,i,j)):
            if os.path.splitext(j1)[1] == '.png':
                parent_dir[i+"_"+j]['Contactbased'].append(os.path.join(ispfdv2_black,i,j,j1))
            else:
                parent_dir[i+"_"+j]['Contactless'].append(os.path.join(ispfdv2_black,i,j,j1))

ispfdv2_black = "< Path to data folder for RES >"
for i in tqdm(os.listdir(ispfdv2_black)):
    for j in os.listdir(os.path.join(ispfdv2_black,i)):
        for j1 in os.listdir(os.path.join(ispfdv2_black,i,j)):
            if os.path.splitext(j1)[1] == '.png':
                # parent_dir[i+"_"+j]['Contactbased'].append(os.path.join(ispfdv2_black,i,j,j1))
                pass
            else:
                parent_dir[i+"_"+j]['Contactless'].append(os.path.join(ispfdv2_black,i,j,j1))

with open("manifest_jsons/ispfdv2_blackback.json",'w') as js:
    json.dump(parent_dir,js,indent=4)

ispfdv2_color = "< Path to data folder for BI >"
parent_dir = dict()
for i in tqdm(os.listdir(ispfdv2_color)):
    for j in os.listdir(os.path.join(ispfdv2_color,i)):
        parent_dir[i+"_"+j] = dict()
        parent_dir[i+"_"+j]['Contactless'] = list()
        parent_dir[i+"_"+j]['Contactbased'] = list()
        for j1 in os.listdir(os.path.join(ispfdv2_color,i,j)):
            if os.path.splitext(j1)[1] == '.png':
                parent_dir[i+"_"+j]['Contactbased'].append(os.path.join(ispfdv2_color,i,j,j1))
            else:
                parent_dir[i+"_"+j]['Contactless'].append(os.path.join(ispfdv2_color,i,j,j1))

ispfdv2_color = "< Path to data folder for RES >"
for i in tqdm(os.listdir(ispfdv2_color)):
    for j in os.listdir(os.path.join(ispfdv2_color,i)):
        # parent_dir[i+"_"+j] = dict()
        # parent_dir[i+"_"+j]['Contactless'] = list()
        # parent_dir[i+"_"+j]['Contactbased'] = list()
        for j1 in os.listdir(os.path.join(ispfdv2_color,i,j)):
            if os.path.splitext(j1)[1] == '.png':
                # parent_dir[i+"_"+j]['Contactbased'].append(os.path.join(ispfdv2_color,i,j,j1))
                pass
            else:
                parent_dir[i+"_"+j]['Contactless'].append(os.path.join(ispfdv2_color,i,j,j1))

with open("manifest_jsons/ispfdv2_colorback.json",'w') as js:
    json.dump(parent_dir,js,indent=4)

ridgebase = "< Path to data folder >"
parent_dir = dict()
finger_lst = ['left_index','left_middle','left_ring','left_little','right_index','right_middle','right_ring','right_little']
for i in tqdm(os.listdir(ridgebase)):
    for j in finger_lst:
        parent_dir[i+"_"+j] = dict()
        parent_dir[i+"_"+j]['Contactless'] = list()
        parent_dir[i+"_"+j]['Contactbased'] = list()
        for j1 in os.listdir(os.path.join(ridgebase,i)):
            if j1.split("_")[2].lower() == j.split("_")[0].lower() and j1.split("_")[-1].lower() == j.split("_")[1].lower():
                for j2 in os.listdir(os.path.join(ridgebase,i,j1)):
                    if os.path.splitext(j2)[1] == '.png':
                        parent_dir[i+"_"+j]['Contactless'].append(os.path.join(ridgebase,i,j1,j2))
                    else:
                        parent_dir[i+"_"+j]['Contactbased'].append(os.path.join(ridgebase,i,j1,j2))

with open("manifest_jsons/ridgebase.json",'w') as js:
    json.dump(parent_dir,js,indent=4)

hkpoly = "< Path to data folder >"
parent_dir = dict()
for i in tqdm(os.listdir(hkpoly)):
    parent_dir[i] = dict()
    parent_dir[i]['Contactless'] = list()
    parent_dir[i]['Contactbased'] = list()
    for j in os.listdir(os.path.join(hkpoly,i)):
        for j1 in os.listdir(os.path.join(hkpoly,i,j)):
            if os.path.splitext(j1)[1] == '.png':
                parent_dir[i]['Contactless'].append(os.path.join(hkpoly,i,j,j1))
            else:
                parent_dir[i]['Contactbased'].append(os.path.join(hkpoly,i,j,j1))

with open("manifest_jsons/hkpolyu.json",'w') as js:
    json.dump(parent_dir,js,indent=4)

# testing hkpolyu
hkpoly = "< Path to data folder >"
parent_dir = dict()
for i in tqdm(os.listdir(hkpoly)):
    parent_dir[i] = dict()
    parent_dir[i]['Contactless'] = list()
    parent_dir[i]['Contactbased'] = list()
    for j in os.listdir(os.path.join(hkpoly,i)):
        for j1 in os.listdir(os.path.join(hkpoly,i,j)):
            if os.path.splitext(j1)[1] == '.png':
                parent_dir[i]['Contactless'].append(os.path.join(hkpoly,i,j,j1))
            else:
                parent_dir[i]['Contactbased'].append(os.path.join(hkpoly,i,j,j1))

with open("hkpolyu_test.json",'w') as js:
    json.dump(parent_dir,js,indent=4)

# validation hkpolyu
hkpoly = "< Path to data folder >"
parent_dir = dict()
for i in tqdm(os.listdir(hkpoly)):
    parent_dir[i] = dict()
    parent_dir[i]['Contactless'] = list()
    parent_dir[i]['Contactbased'] = list()
    for j in os.listdir(os.path.join(hkpoly,i)):
        for j1 in os.listdir(os.path.join(hkpoly,i,j)):
            if os.path.splitext(j1)[1] == '.png':
                parent_dir[i]['Contactless'].append(os.path.join(hkpoly,i,j,j1))
            else:
                parent_dir[i]['Contactbased'].append(os.path.join(hkpoly,i,j,j1))

with open("hkpolyu_val.json",'w') as js:
    json.dump(parent_dir,js,indent=4)
