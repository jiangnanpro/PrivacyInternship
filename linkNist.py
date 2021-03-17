import os
import sys
import time
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle as pkl

myDIR = "/data/ajoe/nist/"

sessStartDT = datetime.now()
sessDTstring = sessStartDT.strftime('%Y%m%d_%H%M')
log_file = 'myLog_linkNist_' + sessDTstring + '.txt'
myfile = open(log_file, 'w', encoding='utf-8')

for hsf in [0,1,2,3,4,6,7]:
    
    # create dict - img : [full path, filename, label] + [idx] + [fpath]
    dict_byIMG = {}
    
    # create dict - fpath + fname : [fname, author] 
    #                             + [full path, filename, label] 
    #                             + [idx]
    dict_byPATH = {}
    
    print("\n ***  HSF_ "+str(hsf)+"  *** \n")
    print("\n ***  HSF_ "+str(hsf)+"  *** \n", file=myfile)
    
    
    for i in range(10):
        thispath = "by_field/hsf_"+str(hsf)+"/digit/3"+str(i)+"/"
        mypath = myDIR+thispath
        print("Scanning:  " + mypath)
        label = str(i)
        os.chdir(mypath)
        with os.scandir(mypath) as it:
            for entry in it:
                img = mpimg.imread(entry.name)
                imgk = tuple(img[:,:,0].astype(bool).flatten())
                fname = entry.name
                fpath = thispath + fname
                dict_byIMG[imgk] = [fpath,fname,label]

    N_IMG = len(dict_byIMG)
    print("Created dict_byIMG with "+str(N_IMG)+" entries\n")
    print("Created dict_byIMG with "+str(N_IMG)+" entries\n", file=myfile)
    images = np.zeros((N_IMG,2048),dtype=np.uint8)
    labels = np.zeros((N_IMG,),dtype=np.uint8)
    idx = 0
    dupes = []

    thispath = "by_write/hsf_"+str(hsf)+"/"
    mypath = myDIR+thispath
    
    for author in os.listdir(mypath):
        author_path = author+'/d'+author[1:]
        try:
            os.chdir(mypath+author_path)
            with os.scandir(mypath+author_path) as it:
                for entry in it:
                    img = mpimg.imread(entry.name)
                    imgk = tuple(img[:,:,0].astype(bool).flatten())
                    fname = entry.name
                    fpath = thispath + author_path + "/" + fname
                    
                    # find this image in the dictionary of images by author
                    if imgk in dict_byIMG:
                        this_entry = dict_byIMG[imgk]
                        this_len = len(this_entry)
                        
                        
                        if this_len == 3:
                            dict_byIMG[imgk] = this_entry + [idx] + [fpath]
                            dict_byPATH[fpath] = [fname,author] + this_entry + [idx]
                            images[idx,:] = np.packbits(imgk)
                            labels[idx] = this_entry[2]
                            idx += 1
                            if idx % 5000 == 0:
                                print("Added "+str(idx)+" images to dict_byPATH")
                        else:
                            dict_byIMG[imgk] = this_entry[0:4] + [[this_entry[4],fpath]]
                            dict_byPATH[fpath] = [fname,author] + this_entry[0:4]
                            dupes.append((this_entry[4],fpath))
                            
                    # image not matched in dictionary of images by authors 
                    else:
                        dict_byPATH[fpath] = [fname,author] + ["NO MATCH"]
                        
        except FileNotFoundError:
            pass

    print("Created dict_byPATH with "+str(idx)+" images\n")
    print("Created dict_byPATH with "+str(idx)+" images\n", file=myfile)
    if dupes:
        print("Found "+str(len(dupes))+" duplicates\n")
        print("Found "+str(len(dupes))+" duplicates\n", file=myfile)

    with open(myDIR+"HSF_"+str(hsf)+"_paths.pkl",'wb') as f:
        pkl.dump(dict_byPATH, f)

    with open(myDIR+"HSF_"+str(hsf)+"_images.npy",'wb') as f:
        np.save(f, images, allow_pickle=False, fix_imports=False)

    with open(myDIR+"HSF_"+str(hsf)+"_labels.npy",'wb') as f:
        np.save(f, labels, allow_pickle=False, fix_imports=False)
        
    with open(myDIR+"HSF_"+str(hsf)+"_byIMG.pkl",'wb') as f:
        pkl.dump(pd.DataFrame(dict_byIMG.values()), f)
        
    if dupes:
        with open(myDIR+"HSF_"+str(hsf)+"_dupes.pkl",'wb') as f:
            pkl.dump(dupes, f)
        
myfile.close()