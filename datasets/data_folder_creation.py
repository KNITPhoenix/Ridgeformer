import os
import pandas as pd
import shutil
from tqdm import tqdm

# <<---->> HKPolyU training
dstn_fold = "< Path to parent data folder >"

cb_df = pd.read_csv("< csv file for HKPolyU train dataset with CB paths and IDS >")
cl_df = pd.read_csv("< csv file for HKPolyU train dataset with CL paths and IDS >")

subject_id = cb_df['ids'].tolist()
files = cb_df['Filename'].tolist()
for i in tqdm(range(len(subject_id))):
    if not os.path.exists(os.path.join(dstn_fold,str(subject_id[i]))):
        os.mkdir(os.path.join(dstn_fold,str(subject_id[i])))
    finger_id = files[i].split('/')[-1].split('.')[0].split('_')[-1]
    if not os.path.exists(os.path.join(dstn_fold,str(subject_id[i]),finger_id)):
        os.mkdir(os.path.join(dstn_fold,str(subject_id[i]),finger_id))
    shutil.copy(files[i],os.path.join(dstn_fold,str(subject_id[i]),finger_id,'cb.jpg'))
count=0
subject_id = cl_df['ids'].tolist()
files = cl_df['Filename'].tolist()
for i in tqdm(range(len(subject_id))):
    if not os.path.exists(os.path.join(dstn_fold,str(subject_id[i]))):
        os.mkdir(os.path.join(dstn_fold,str(subject_id[i])))
    finger_id = files[i].split('/')[-1].split('.')[0].split('_')[-1]
    if not os.path.exists(os.path.join(dstn_fold,str(subject_id[i]),finger_id)):
        os.mkdir(os.path.join(dstn_fold,str(subject_id[i]),finger_id))
    shutil.copy(files[i].replace("hkpoly_data/cl", "hkpoly_data/cl_png/cl/").replace(".bmp", ".png"),os.path.join(dstn_fold,str(subject_id[i]),finger_id,'cl.png'))
    count+=1

# <<---->> Ridgebase training
dstn_fold = "< Path to parent data folder >"

cb_df = pd.read_csv("< csv file for RB train dataset with CB paths and IDS >")
cl_df = pd.read_csv("< csv file for RB train dataset with CL paths and IDS >")

files = cl_df['Filename'].tolist()
sub_lst = list()
for i in tqdm(files):
    sub_id = os.path.splitext(i.split("/")[-1])[0].split("_")[2]
    if not os.path.exists(os.path.join(dstn_fold,sub_id)):
        os.mkdir(os.path.join(dstn_fold,sub_id))
    sess = os.path.splitext(i.split("/")[-1])[0].split("_")[0]
    back = os.path.splitext(i.split("/")[-1])[0].split("_")[3]
    hand = os.path.splitext(i.split("/")[-1])[0].split("_")[4]
    fing = os.path.splitext(i.split("/")[-1])[0].split("_")[7]
    random_identifier = os.path.splitext(i.split("/")[-1])[0].split("_")[6]
    if fing == '0':
        fing = i.split("/")[-1].split(".")[0].split("_")[1]+"_"+'Index'
    elif fing == '1':
        fing = i.split("/")[-1].split(".")[0].split("_")[1]+"_"+'Middle'
    elif fing == '2':
        fing = i.split("/")[-1].split(".")[0].split("_")[1]+"_"+'Ring'
    elif fing == '3':
        fing = i.split("/")[-1].split(".")[0].split("_")[1]+"_"+'Little'
    if not os.path.exists(os.path.join(dstn_fold,sub_id,sess+"_"+back+"_"+hand+"_"+fing)):
        os.mkdir(os.path.join(dstn_fold,sub_id,sess+"_"+back+"_"+hand+"_"+fing))
    shutil.copy(i,os.path.join(dstn_fold,sub_id,sess+"_"+back+"_"+hand+"_"+fing,random_identifier+'_cl.png'))

f = 0
for i in os.listdir(dstn_fold):
    for j in os.listdir(os.path.join(dstn_fold,i)):
        for j1 in os.listdir(os.path.join(dstn_fold,i,j)):
            f+=1

f=0
for i in tqdm(cb_df['Filename'].tolist()):
    sub_id = os.path.splitext(i.split("/")[-1])[0].split("_")[-3]
    hand = os.path.splitext(i.split("/")[-1])[0].split("_")[-2]
    fing = os.path.splitext(i.split("/")[-1])[0].split("_")[-1]
    sess = os.path.splitext(i.split("/")[-1])[0].split("_")[0]
    ds = os.path.join(dstn_fold,sub_id)
    for j in os.listdir(ds):
        if os.path.exists(os.path.join(ds,j,'cb.png')):
            os.remove(os.path.join(ds,j,'cb.png'))
        if j.split("_")[0] == sess and j.split("_")[2].lower() == hand.lower() and j.split("_")[4] == fing:
            shutil.copy(i,os.path.join(ds,j,'cb.bmp'))
            f+=1

# << ISPFDv1 >>
dstn_fold = "< Path to parent data folder with Black back >"
# dstn_fold = "< Path to parent data folder with Color back >"

f=0
parent_fold = "< Path to cropped dataset >"
# parent_fold = "< Path to cropped dataset >"
# parent_fold = "< Path to cropped dataset >"
# parent_fold = "< Path to cropped dataset >"
for i in tqdm(os.listdir(parent_fold)):
    sub_id = i.split("_")[0]
    fing = i.split("_")[2]
    image_id = i.split("_")[1]
    if not os.path.exists(os.path.join(dstn_fold,sub_id)):
        os.mkdir(os.path.join(dstn_fold,sub_id))
    if not os.path.exists(os.path.join(dstn_fold,sub_id,fing)):
        os.mkdir(os.path.join(dstn_fold,sub_id,fing))
    shutil.copy(os.path.join(parent_fold,i),os.path.join(dstn_fold,sub_id,fing,i.split("_")[1]+i.split("_")[3]+i.split("_")[4]))
    f+=1
print(f)

cb_path = "< Path to Live Scans >"

for i in tqdm(os.listdir(cb_path)):
    sub_id = i.split("_")[0]
    fing = i.split("_")[1]
    sess = i.split("_")[2]
    shutil.copy(os.path.join(cb_path,i),os.path.join(dstn_fold,sub_id,fing,sess+".jpg"))

# <<ISPFD v2>>
dstn_fold = "< Path to parent data folder >"
parent_fold = "< Path to original data folder >"

for i in tqdm(os.listdir(parent_fold)):
    sub_id = i.split("_")[0]
    if not os.path.exists(os.path.join(dstn_fold,sub_id)):
        os.mkdir(os.path.join(dstn_fold,sub_id))
    fing = i.split("_")[1]
    if not os.path.exists(os.path.join(dstn_fold,sub_id,fing)):
        os.mkdir(os.path.join(dstn_fold,sub_id,fing))
    nam = "_".join(i.split("_")[2:])
    shutil.copy(os.path.join(parent_fold,i),os.path.join(dstn_fold,sub_id,fing,nam))

parent_fold = "< Path to original Live Scans folder >"
for i in tqdm(os.listdir(parent_fold)):
    sub_id = i.split("_")[0]
    fing = i.split("_")[1]
    nam = "_".join(i.split("_")[2:])
    shutil.copy(os.path.join(parent_fold,i),os.path.join(dstn_fold,sub_id,fing,nam))

# HKPolyU testing
dstn_fold = "< Path to parent data folder >"

cb_df = pd.read_csv("< csv file for HKPolyU train dataset with CB paths and IDS >")
cl_df = pd.read_csv("< csv file for HKPolyU train dataset with CL paths and IDS >")

subject_id = cb_df['ids'].tolist()
files = cb_df['Filename'].tolist()
for i in tqdm(range(len(subject_id))):
    if not os.path.exists(os.path.join(dstn_fold,str(subject_id[i]))):
        os.mkdir(os.path.join(dstn_fold,str(subject_id[i])))
    finger_id = files[i].split('/')[-1].split('.')[0].split('_')[-1]
    if not os.path.exists(os.path.join(dstn_fold,str(subject_id[i]),finger_id)):
        os.mkdir(os.path.join(dstn_fold,str(subject_id[i]),finger_id))
    shutil.copy(files[i],os.path.join(dstn_fold,str(subject_id[i]),finger_id,'cb.jpg'))

subject_id = cl_df['ids'].tolist()
files = cl_df['Filename'].tolist()
for i in tqdm(range(len(subject_id))):
    if not os.path.exists(os.path.join(dstn_fold,str(subject_id[i]))):
        os.mkdir(os.path.join(dstn_fold,str(subject_id[i])))
    finger_id = files[i].split('/')[-1].split('.')[0].split('_')[-1]
    if not os.path.exists(os.path.join(dstn_fold,str(subject_id[i]),finger_id)):
        os.mkdir(os.path.join(dstn_fold,str(subject_id[i]),finger_id))
    shutil.copy(files[i].replace("hkpoly_data/cl", "hkpoly_data/cl_png/cl/").replace(".bmp", ".png"),os.path.join(dstn_fold,str(subject_id[i]),finger_id,'cl.png'))

# HKPolyU validation
dstn_fold = "< Path to parent data folder >"

cb_df = pd.read_csv("< csv file for HKPolyU train dataset with CB paths and IDS >")
cl_df = pd.read_csv("< csv file for HKPolyU train dataset with CL paths and IDS >")

subject_id = cb_df['ids'].tolist()
files = cb_df['Filename'].tolist()
for i in tqdm(range(len(subject_id))):
    if not os.path.exists(os.path.join(dstn_fold,str(subject_id[i]))):
        os.mkdir(os.path.join(dstn_fold,str(subject_id[i])))
    finger_id = files[i].split('/')[-1].split('.')[0].split('_')[-1]
    if not os.path.exists(os.path.join(dstn_fold,str(subject_id[i]),finger_id)):
        os.mkdir(os.path.join(dstn_fold,str(subject_id[i]),finger_id))
    shutil.copy(files[i],os.path.join(dstn_fold,str(subject_id[i]),finger_id,'cb.jpg'))

subject_id = cl_df['ids'].tolist()
files = cl_df['Filename'].tolist()
for i in tqdm(range(len(subject_id))):
    if not os.path.exists(os.path.join(dstn_fold,str(subject_id[i]))):
        os.mkdir(os.path.join(dstn_fold,str(subject_id[i])))
    finger_id = files[i].split('/')[-1].split('.')[0].split('_')[-1]
    if not os.path.exists(os.path.join(dstn_fold,str(subject_id[i]),finger_id)):
        os.mkdir(os.path.join(dstn_fold,str(subject_id[i]),finger_id))
    shutil.copy(files[i].replace("hkpoly_data/cl", "hkpoly_data/cl_png/cl/").replace(".bmp", ".png"),os.path.join(dstn_fold,str(subject_id[i]),finger_id,'cl.png'))
