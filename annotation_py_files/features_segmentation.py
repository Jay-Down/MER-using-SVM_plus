#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import re, os, sys
import matplotlib.pyplot as plt


# # 1.0 Features

# In[2]:


# read in dynamic features file as 260-dimension dataframe

path = '/Volumes/Seagate Expansion Drive/PMEmo dataset/PMEmo/PMEmo/'


# In[13]:


dynamic_features = pd.read_csv(path+'dynamic_features.csv')


# #### Arousal

# In[270]:


# load in arousal annotation changepoint locations

path = "/Users/jay/Documents/Jay's bits/Uni/Thesis/thesis-pipeline/data/interim/"

A_bkps = pd.read_pickle(path+'Arousal_breaks')


# In[260]:



def gen_ids(bkps):
    """generate list of filenames from successfully segmented File_IDs"""

    filenames = []

    for fid in bkps['File_ID']:   
        y=""
        x = list(fid)
        x=x[:-6]
        for i in x:
            y+=i
        filenames.append(y)
        
    return filenames


# In[264]:


# generate arousal filenames

filenames = gen_ids(A_bkps)


# In[278]:



def segment_features(bkps, features, filenames, dim):
   """use breakpoints as indexes to segment feature file, generate a list of Series entries"""
   
   split_features = []
   
   if dim=='A':
       ext='-A.csv'
   else:
       ext='-V.csv'
   
   for fid in filenames:
       data = features[features['musicId']==int(fid)].iloc[30:,:]
       break_id = str(fid)+ext
       breaks = bkps[bkps['File_ID']==break_id]
       
       if not breaks.empty:
           
           idxs = bkps[bkps['File_ID']==break_id]['bkps'].iloc[0]
       
           if len(idxs)==1:

               # split_features.append(pd.concat(str(fid), data.mean(axis=0).values))
               entry = [str(fid), data.mean(axis=0)]
#                entry = [str(fid), data.mean(axis=0).values]

               split_features.append(entry)

           else:
               idxs = idxs[:-1]
               feature_splits = np.split(data, idxs)
               suffix=1

               for i in feature_splits:
#                    entry = [str(suffix)+'_'+str(fid), feature_splits[i].mean(axis=0).values]
                   entry = [str(suffix)+'_'+str(fid), i.mean(axis=0)]
                   split_features.append(entry)
                   suffix+=1
                   
   return split_features
                   
       


# In[237]:


A_segment_features = segment_features(A_bkps, dynamic_features, filenames, 'A')


# In[256]:


# coerce segmented features into a dataframe

ft=[]
for i in A_segment_features:
    ft.append(i[1])

z=[]
for i in A_segment_features:
    z.append(i[0])

z = pd.Series(z, name='File_ID')


A_mu_dyn_features = pd.DataFrame(ft)


A_mu_dyn_features.insert(0, 'File_ID', z)


# In[448]:


# save segmented arousal features as csv

path = "/Users/jay/Documents/Jay's bits/Uni/Thesis/thesis-pipeline/data/processed/Features/"


# In[450]:


A_mu_dyn_features.to_csv(path+'Arousal_averaged_features.csv')


# #### Valence

# In[259]:


path = "/Users/jay/Documents/Jay's bits/Uni/Thesis/thesis-pipeline/data/interim/"

V_bkps = pd.read_pickle(path+'Valence_breaks')


# In[261]:


filenames = gen_ids(V_bkps)


# In[280]:


# segment audio features based on valence annotation changepoints

V_segment_features = segment_features(V_bkps, dynamic_features, filenames, 'V')


# In[282]:


ft=[]
for i in V_segment_features:
    ft.append(i[1])
    

z=[]
for i in V_segment_features:
    z.append(i[0])
    

z = pd.Series(z, name='File_ID')


V_mu_dyn_features = pd.DataFrame(ft)


V_mu_dyn_features.insert(0, 'File_ID', z)


# In[451]:


path = "/Users/jay/Documents/Jay's bits/Uni/Thesis/thesis-pipeline/data/processed/Features/"


# In[452]:


V_mu_dyn_features.to_csv(path+'Valence_averaged_features.csv')

