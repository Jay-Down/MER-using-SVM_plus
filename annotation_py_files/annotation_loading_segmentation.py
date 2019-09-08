#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import re, os, sys
import matplotlib.pyplot as plt


# # 1.0 Data Load

# In[2]:


# path = '/home/k1896084/Documents/Semester 2/Thesis/PMEmo/PMEmo/Annotations/Arousal/'
path = '/Volumes/Seagate Expansion Drive/PMEmo dataset/PMEmo/PMEmo/Annotations/Arousal/'

filenames = os.listdir(path)


# In[3]:


def load_annotations(filenames):
    """load indivudual annotation csv files into DataFrame"""
    annotations = []

    for fid in filenames:
            annotation = np.loadtxt(open(path+fid, 'r'), delimiter=',', skiprows=1)
            entry = [fid, annotation[:, 1], annotation[:, 2:]]
            annotations.append(entry)

    return pd.DataFrame(annotations, columns=['File_ID', 'static', 'dynamic'])


# In[4]:


# load Arousal annotations

A_annotations = load_annotations(filenames)


# In[8]:


# path = '/home/k1896084/Documents/Semester 2/Thesis/PMEmo/PMEmo/Annotations/Valence/'
path = '/Volumes/Seagate Expansion Drive/PMEmo dataset/PMEmo/PMEmo/Annotations/Valence/'

filenames = os.listdir(path)


# In[9]:


# load Valence annotations

V_annotations = load_annotations(filenames)


# ## 1.1 Exploratory Analysis

# In[5]:


def descriptive_stats(segments):
    """
    calculate descriptive statistics for annotation dataframes on row-wise and column-wise basis, 
    as in Fig. 7 of report
    """
    
    col_stats = []
    row_stats = []
    derived_stats = []

    for fid in segments['File_ID']:
        annotations = segments[segments['File_ID']==fid]['dynamic'].iloc[0]
        
        ### dynamic stats by column ### 

        # dynamic column means (mean emotion at time=t)
        col_mu = annotations.mean(axis=0)
        
        # dynamic column variances (level of disagreement in an annotation)
        col_vars = annotations.var(axis=0)
        
        col_entry = [fid, col_mu, col_vars]
        
        ### dynamic stats by row ### 

        # dynamic row means (overall mean emotion in a clip, per respondent. Ought to be comparable with static annotation values.)
        row_mu = annotations.mean(axis=1)
        
        # dynamic row variances (amount of variability in emotions portrayed throughout clip, per respondent)
        row_vars = annotations.var(axis=1)
        
        row_entry = [fid, row_mu, row_vars]
        
        ### derived stats of dynamic annotations ### 
        
        # mean of dynamic column means (average arousal/valence in a clip (at any one time?))
        mean_col_mu = col_mu.mean()
        
        # variance of dynamic column means (variability of A/V from the mean of col means at any point in time)
        var_col_mu = col_mu.var()
        
        # mean of dynamic column variances (i.e. average disagreement in a single time step annotation;
        # amount of noise around 'true' signal? average difficulty in annotation per time step?)
        mean_col_vars = col_vars.mean()
        
        # variance of dynamic column variances (variability in ambiguity of emotion in a track)
        var_col_vars = col_vars.var()
        
        # mean of dynamic row means (average overall emotion in a clip, comparable to mean of static annotations)
        mean_row_mu = row_mu.mean()
        
        # variance of row means (level of disagreement between respondents in overall emotion of clip)
        var_row_mu = row_mu.var()
        
        # mean of row variances (average level of emotion variation over course of a clip) 
        mean_row_vars = row_vars.mean()
        
        # variance of row variances (level of disagreement between respondents over variability of emotion in clip)
        var_row_vars = row_vars.var()
        
        derived_entry = [fid, mean_col_mu, var_col_mu, mean_col_vars, var_col_vars, mean_row_mu, var_row_mu, mean_row_vars, var_row_vars]
        
        col_stats.append(col_entry)
        
        row_stats.append(row_entry)
    
        derived_stats.append(derived_entry)        
    
    return pd.DataFrame(col_stats, columns=['File_ID', 'col_means', 'col_vars']), pd.DataFrame(row_stats, columns=['File_ID', 'row_means', 'row_vars']), pd.DataFrame(derived_stats, columns=['File_ID', 'mean_col_mu', 'var_col_mu', 'mean_col_vars', 'var_col_vars', 'mean_row_mu', 'var_row_mu', 'mean_row_vars', 'var_row_vars'])
    
        


# In[6]:


# arousal descriptive stats

A_column_stats, A_row_stats, A_derived_stats = descriptive_stats(A_annotations)


# In[10]:


# valence descriptive stats

V_column_stats, V_row_stats, V_derived_stats = descriptive_stats(V_annotations)


# Average length of clips

# In[13]:


# Average number of samples in dynamic annotations = 47.5

# Arousal
cusum = 0

for i in A_column_stats['col_means']:
    cusum += len(i)
    
cusum/len(A_column_stats)


# In[14]:


# Valence: 0.02 difference due to 0.5s (1 timestep value) length discrepancies in the following 20/767 files:
# [1, 2, 5, 16, 18, 19, 20, 23, 27, 32, 33, 34, 41, 42, 43, 55, 66, 107, 130, 157]

cusum = 0

for i in V_column_stats['col_means']:
    cusum += len(i)
    
cusum/len(V_column_stats)


# Distribution of annotations

# In[639]:


# left-skewed distribution => median and IQR rather than mean and s.d.

fig, ax = plt.subplots(1, 2, figsize=(15, 4))
ax[0].hist((V_derived_stats['mean_row_mu']))
ax[0].set_title('Mean Valence Annotation Distribution')
ax[1].hist((A_derived_stats['mean_row_mu']))
ax[1].set_title('Mean Arousal Annotation Distribution')
plt.show()


# Plotting music emotion variation over time

# In[245]:


def plot_signals(rand_ints):
    """plot average signals of four clips based on list of indices, as in Fig. 6 of report"""
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax[0,0].plot(A_column_stats['col_means'].iloc[rand_ints[0]])
    ax[0,0].set_title(A_column_stats.iloc[rand_ints[0]]['File_ID'])
    
    ax[0,1].plot(A_column_stats['col_means'].iloc[rand_ints[1]])
    ax[0,1].set_title(A_column_stats.iloc[rand_ints[1]]['File_ID'])
    
    ax[1,0].plot(A_column_stats['col_means'].iloc[rand_ints[2]])
    ax[1,0].set_title(A_column_stats.iloc[rand_ints[2]]['File_ID'])
    
    ax[1,1].plot(A_column_stats['col_means'].iloc[rand_ints[3]])
    ax[1,1].set_title(A_column_stats.iloc[rand_ints[3]]['File_ID'])
    
    fig.text(0.5, 0.04, 'time (s)', ha='center', va='center')
    fig.text(0.06, 0.5, 'mean dynamic arousal', ha='center', va='center', rotation='vertical')

    plt.show()


# In[256]:


# randomly generate four indices 

rand_ints = [np.random.randint(len(A_column_stats)) for i in range(4)]


# In[259]:


# files chosen for plotting in report: [753, 312, 413, 757]
plot_signals(rand_ints)


# Heatmap of A/V values to check if data biased

# In[295]:


x = A_derived_stats['mean_row_mu']
y = V_derived_stats['mean_row_mu']

heatmap, xedges, yedges = np.histogram2d(x, y, bins=15)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

plt.clf()
plt.imshow(heatmap.T, extent=extent, origin='lower')
plt.colorbar()
plt.xlabel('Arousal')
plt.ylabel('Valence')
plt.show()


# In[ ]:





# # 2.0 Segmentation

# ## 2.1 Synthetic signal and parameterisation

# In[298]:


import ruptures as rpt


# In[299]:


# using the mean of the variance per dynamic time step as a proxy for how much 'noise' is in the signal of a track's
# annotations in order to get a sigma value for changepoint detection

# average amount of variance in arousal annotation per dynamic time step = 0.09215330048169802
ave_dyn_var = A_derived_stats['mean_col_vars'].mean()

# => average s.d. = 0.303567620937573
ave_dyn_sd = np.sqrt(ave_dyn_var)


# In[301]:


# average signal parameters for synthetic signal
delta = np.median(A_derived_stats['mean_row_mu'])
diff = np.diff(np.quantile(A_derived_stats['mean_row_mu'], (0.25, 0.75)))


# In[402]:


# calculate synthetic signal based on ave_num_samples=48 and s.d. as noise

n = 48
sigma = ave_dyn_sd

signal, bkps = rpt.pw_constant(n, 1, 1, noise_std=sigma, delta=((delta-diff), (delta+diff)))
rpt.display(signal, bkps)
plt.show()


# In[404]:


# change point detection

# PELT algo

pen = np.log(n)*sigma**2
model = "l1"  # "l2", "rbf"
PELT = rpt.Pelt(model=model, min_size=10).fit(signal)
my_bkps = PELT.predict(pen=pen)

# show results
fig, (ax,) = rpt.display(signal, bkps, my_bkps)
plt.show()


# In[405]:


# actual vs. predicted changepoints

print('actual: %s \npredicted: %s'%(bkps, my_bkps))


# In[397]:


from ruptures.metrics import randindex

# evaluation of signal segmentation
n = 48
sigma = ave_dyn_sd
pen = np.log(n)*sigma**2
r_i = []

for i in range(1000):
    
    signal, bkps = rpt.pw_constant(n, 1, 1, noise_std=sigma, delta=((delta-diff), (delta+diff)))
    model = "l1"  # "l2", "rbf"
    PELT = rpt.Pelt(model=model, min_size=10).fit(signal)
    my_bkps = PELT.predict(pen=pen)
    r_i.append(randindex(bkps, my_bkps))

    
np.mean(r_i)


# ## 2.2 Segmentation

# In[30]:



def breakpoints(filenames, column_stats, derived_stats, model):
    """predict breakpoint locations in affective signals"""
    
    bkps = []
    problem_IDs = []
    sigma = np.sqrt(derived_stats['mean_col_vars'].mean())

    for fid in filenames:
        try:
            print('Segmenting '+fid)
            signal = column_stats[column_stats['File_ID']==fid]['col_means'].iloc[0]
            n = len(signal)
            pen = np.log(n)*sigma**2
            PELT = rpt.Pelt(model=model, min_size=10).fit(signal)
            breaks = PELT.predict(pen=pen)
            bkps.append([fid, breaks])
            print(fid+' Done')

        except ValueError:
            problem_IDs.append([fid, n])

    return pd.DataFrame(bkps, columns=['File_ID', 'bkps']), np.array(problem_IDs)
        #[('File_ID', 'U25'), ('bkps', 'O')]


# #### Arousal

# In[31]:


#path = '/home/k1896084/Documents/Semester 2/Thesis/PMEmo/PMEmo/Annotations/Arousal/'
path = '/Volumes/Seagate Expansion Drive/PMEmo dataset/PMEmo/PMEmo/Annotations/Arousal/'

filenames = os.listdir(path)


# In[32]:


# arousal changepoint detection

A_bkps, A_problem_IDs = breakpoints(filenames, A_column_stats, A_derived_stats, 'l1')


# In[43]:



def split_clips(bkps, annotations):
    """
    split annotations according to index position of breakpoints; if bkpt == len(clip) don't split, 
    else split and assign to new array
    """
    
    split_clips = []
    
    for fid in bkps['File_ID']:
        responses = annotations[annotations['File_ID']==fid]['dynamic'].iloc[0]
        clip_breaks = bkps[bkps['File_ID']==fid]['bkps'].iloc[0]
        
        if len(clip_breaks)==1:
            split_clips.append([fid, responses])
            
        else:
            temp_clips = []
            suffix=1
            
            for response in responses:
                segments=np.split(response, clip_breaks)
                temp_clips.append(segments)
                
            for i in range(len(clip_breaks)):
                tmp = []
                for segments in temp_clips:
                    tmp.append(segments[i])
                split_clips.append([str(suffix)+'_'+fid, tmp])
                suffix+=1
                    
    return pd.DataFrame(split_clips, columns=['File_ID', 'Annotations'])


# In[44]:


# arousal segmentation

A_splits = split_clips(A_bkps, A_annotations)


# In[47]:


# Arousal dataset size increased from 767 to 971
len(A_splits)


# In[415]:


# save down split segments as a dataframe for easy re-loading

path = "/Users/jay/Documents/Jay's bits/Uni/Thesis/thesis-pipeline/data/processed/Annotations/Arousal/"

A_splits.to_pickle(path+'Arousal_annotation_dataframe_segmented')


# In[56]:


#path = "/Users/jay/Documents/Jay's bits/Uni/Thesis/thesis-pipeline/data/processed/Annotations/Arousal"

#A_splits = pd.read_pickle(path+'Arousal_dataframe_segmented')


# #### Valence

# In[48]:


#path = '/home/k1896084/Documents/Semester 2/Thesis/PMEmo/PMEmo/Annotations/Valence/'
path = '/Volumes/Seagate Expansion Drive/PMEmo dataset/PMEmo/PMEmo/Annotations/Valence/'

filenames = os.listdir(path)


# In[49]:


# valence changepoint detection

V_bkps, V_problem_IDs = breakpoints(filenames, V_column_stats, V_derived_stats, 'l1')


# In[50]:


# valence segmentation

V_splits = split_clips(V_bkps, V_annotations)


# In[52]:


# valence dataset increased from 767 to 922
len(V_splits)


# ## 2.3 Segmentation appraisal

# In[53]:


def segmented_stats(segments):
    """descriptive statistics as in descriptive_stats(), calculated for segmented clips"""
    
    col_stats = []
    row_stats = []
    derived_stats = []

    for fid in segments['File_ID']:
        annotations = np.array(segments[segments['File_ID']==fid]['Annotations'].iloc[0])
        
        ### dynamic stats by column ### 

        # dynamic column means (mean emotion at time=t)
        col_mu = annotations.mean(axis=0)
        
        # dynamic column variances (level of disagreement in an annotation)
        col_vars = annotations.var(axis=0)
        
        col_entry = [fid, col_mu, col_vars]
        
        ### dynamic stats by row ### 

        # dynamic row means (overall mean emotion in a clip, per respondent. Ought to be comparable with static annotation values.)
        row_mu = annotations.mean(axis=1)
        
        # dynamic row variances (amount of variability in emotions portrayed throughout clip, per respondent)
        row_vars = annotations.var(axis=1)
        
        row_entry = [fid, row_mu, row_vars]
        
        ### derived stats of dynamic annotations ### 
        
        # mean of dynamic column means (average arousal/valence in a clip (at any one time?))
        mean_col_mu = col_mu.mean()
        
        # variance of dynamic column means (variability of A/V from the mean of col means at any point in time)
        var_col_mu = col_mu.var()
        
        # mean of dynamic column variances (i.e. average disagreement in a single time step annotation;
        # amount of noise around 'true' signal? average difficulty in annotation per time step?)
        mean_col_vars = col_vars.mean()
        
        # variance of dynamic column variances (variability in ambiguity of emotion in a track)
        var_col_vars = col_vars.var()
        
        # mean of dynamic row means (average overall emotion in a clip, comparable to mean of static annotations)
        mean_row_mu = row_mu.mean()
        
        # variance of row means (level of disagreement between respondents in overall emotion of clip)
        var_row_mu = row_mu.var()
        
        # mean of row variances (average level of emotion variation over course of a clip) 
        mean_row_vars = row_vars.mean()
        
        # variance of row variances (level of disagreement between respondents over variability of emotion in clip)
        var_row_vars = row_vars.var()
        
        derived_entry = [fid, mean_col_mu, var_col_mu, mean_col_vars, var_col_vars, mean_row_mu, var_row_mu, mean_row_vars, var_row_vars]
        
        col_stats.append(col_entry)
        
        row_stats.append(row_entry)
    
        derived_stats.append(derived_entry)        
    
    return pd.DataFrame(col_stats, columns=['File_ID', 'col_means', 'col_vars']), pd.DataFrame(row_stats, columns=['File_ID', 'row_means', 'row_vars']), pd.DataFrame(derived_stats, columns=['File_ID', 'mean_col_mu', 'var_col_mu', 'mean_col_vars', 'var_col_vars', 'mean_row_mu', 'var_row_mu', 'mean_row_vars', 'var_row_vars'])
    
        


# In[54]:


# segmented clips descriptive stats

A_seg_col_stats, A_seg_row_stats, A_seg_derived_stats = segmented_stats(A_splits)


# In[55]:


V_seg_col_stats, V_seg_row_stats, V_seg_derived_stats = segmented_stats(V_splits)


# #### Segmentation metrics for Arousal

# Average variability in row variance has decreased

# In[57]:


A_derived_stats['mean_row_vars'].mean()


# In[56]:


A_seg_derived_stats['mean_row_vars'].mean()


# Average variability in column variance has decreased

# In[58]:


A_derived_stats['var_col_vars'].mean()


# In[59]:


A_seg_derived_stats['var_col_vars'].mean()


# Mean of timestep variance not changed, indicating level of respondent disagreement not changed

# In[60]:


A_seg_derived_stats['mean_col_vars'].mean()


# In[61]:


A_derived_stats['mean_col_vars'].mean()


# Average variability in column mean has decreased

# In[62]:


A_derived_stats['var_col_mu'].mean()


# In[63]:


A_seg_derived_stats['var_col_mu'].mean()


# In[408]:


A_derived_stats['var_row_vars'].mean()


# In[409]:


A_seg_derived_stats['var_row_vars'].mean()


# In[64]:


def peak_to_peak(column_stats):
    """average range of arousal signal in clips"""
    
    jump = []

    for i in range(len(column_stats['col_means'])):
        jump.append(np.ptp(column_stats['col_means'][i]))

    return np.mean(jump)
    


# In[65]:


# average size of amplitude range

peak_to_peak(A_column_stats)


# In[66]:


peak_to_peak(A_seg_col_stats)


# #### Segmentation metrics for Valence

# Average variability in row variance has decreased

# In[67]:


V_derived_stats['mean_row_vars'].mean()


# In[68]:


V_seg_derived_stats['mean_row_vars'].mean()


# Average variability in column variance has decreased

# In[410]:


V_derived_stats['var_col_vars'].mean()


# In[411]:


V_seg_derived_stats['var_col_vars'].mean()


# Mean of timestep variance not changed, indicating level of respondent disagreement not changed

# In[71]:


V_seg_derived_stats['mean_col_vars'].mean()


# In[72]:


V_derived_stats['mean_col_vars'].mean()


# Average variability in column mean has decreased

# In[73]:


V_derived_stats['var_col_mu'].mean()


# In[74]:


V_seg_derived_stats['var_col_mu'].mean()


# In[75]:


peak_to_peak(V_column_stats)


# In[76]:


peak_to_peak(V_seg_col_stats)


# # 3.0 Privileged Info

# ## 3.1 Descriptive statistics

# In[79]:



from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, pairwise_distances

def cos_sim(splits, flag='s'):
    """
    cosine distance gives an indication of similarity of annotation vectors for a clip in terms of their angle;
    magnitude or absolute size variations are discounted by l2 normalisation
    """
    PI = []
    for fid in splits['File_ID']:
        tmp = []
        if flag=='s':
            annotations = splits[splits['File_ID']==fid]['Annotations'].iloc[0]
        elif flag=='a':
            annotations = splits[splits['File_ID']==fid]['dynamic'].iloc[0]
        t_steps = len(annotations[0])
        idxs = range(len(annotations))
        sidxs = set(idxs)
        si=set()
        for i in idxs:
            si.add(i)
            js = sidxs-si
            for j in js:
                c_s = cosine_similarity(annotations[i].reshape(1, -1), annotations[j].reshape(1, -1))
                tmp.append(c_s)
        PI.append([fid, np.mean(tmp)])
    
    return pd.DataFrame(PI, columns=['File_ID', 'cos_sim'])


# In[81]:


A_cos_sim = cos_sim(A_annotations, flag='a')


# In[82]:


A_seg_cos_sim = cos_sim(A_splits, flag='s')


# In[83]:


V_cos_sim = cos_sim(V_annotations, flag='a')


# In[84]:


V_seg_cos_sim = cos_sim(V_splits, flag='s')


# Cosine sim approx. 1 for almost all examples; shows that all individual annotator signals follow the same trajectory, not taking into account their differences in magnitude. Not very helpful for privileged info.

# In[246]:


# mean_col_vars seems most suitable contender for x* in LUPI

A_PI = A_seg_derived_stats['mean_col_vars']

V_PI = V_seg_derived_stats['mean_col_vars']


# In[661]:


# download whole descriptive stats csvs as PI for further use in Matlab

path = "/Users/jay/Documents/Jay's bits/Uni/Thesis/thesis-pipeline/data/processed/Annotations/Arousal/"

A_seg_derived_stats.to_csv(path+'derived_stats.csv')


# In[254]:


path = "/Users/jay/Documents/Jay's bits/Uni/Thesis/thesis-pipeline/data/processed/Annotations/Valence/"

V_seg_derived_stats.to_csv(path+'derived_stats.csv')

#V_pa.to_csv(path+'percent_agree.csv')


# ## 3.2 Percent agreement

# In[87]:


# threshold value for annotations determined from Matlab training partition

path = "/Users/jay/Documents/Jay's bits/Uni/Thesis/thesis-pipeline/data/interim/"

A_thresholds = pd.read_csv(path+'A_thresholds.csv', header=None, names=['run', 'thresh'])


# Not actually any need to calc for various thresholds of different train set splits; no variation in % agree results

# In[88]:


def percent_agreement(row_stats, thresh):
    ''' 
    compute pairwise agreement between each respondent then average
    '''
    PA=[]
    for fid in row_stats['File_ID']:
        binarised = row_stats[row_stats['File_ID']==fid]['row_means'].iloc[0]>thresh
        idxs = range(len(binarised))
        sidxs = set(idxs)
        si=set()
        bools = []
        for i in idxs:
            si.add(i)
            js = sidxs-si
            for j in js:
                bools.append(binarised[i]==binarised[j])
        p_a = sum(bools)/len(bools)
        PA.append([fid, p_a])
            
    return pd.DataFrame(PA, columns=['File_ID', 'percent_agreement'])
    


# In[89]:


# arousal percent agreement

A_pa = percent_agreement(A_seg_row_stats, A_thresholds.iloc[1,1])


# In[96]:


# valence percent agreement
# determined value for valence threshold = 0.6003

V_pa = percent_agreement(V_seg_row_stats, 0.6003)


# ## 3.3 Consensus level

# In[103]:


def consensus(row_stats, thresh):
    ''' calculate number of respondents whose responses constitute y=1
    '''
    PI=[]
    for fid in row_stats['File_ID']:
        y_counts = sum(row_stats[row_stats['File_ID']==fid]['row_means'].iloc[0]>thresh)        
        PI.append(y_counts)
    return pd.DataFrame(PI, columns=['y_count'])


# In[104]:


A_consensus = consensus(A_seg_row_stats, A_thresholds.iloc[1,1])


# In[105]:


V_consensus = consensus(V_seg_row_stats, 0.6003)


# # 4.0 Data Export

# #### Arousal

# In[107]:


path = "/Users/jay/Documents/Jay's bits/Uni/Thesis/thesis-pipeline/data/processed/Annotations/Arousal/"


# In[112]:


A_consensus.to_csv(path+'A_consensus.csv')


# In[ ]:


A_pa.to_csv(path+'percent_agree.csv')


# In[165]:


A_splits.to_pickle(path+'Raw_annotations_segmented')


# In[166]:


A_seg_col_stats.to_pickle(path+'column_stats')


# In[167]:


A_seg_row_stats.to_pickle(path+'row_stats')


# In[168]:


A_seg_derived_stats.to_pickle(path+'derived_stats')


# In[169]:


path = "/Users/jay/Documents/Jay's bits/Uni/Thesis/thesis-pipeline/data/interim/"

A_bkps.to_pickle(path+'Arousal_breaks')


# #### Valence

# In[109]:


path = "/Users/jay/Documents/Jay's bits/Uni/Thesis/thesis-pipeline/data/processed/Annotations/Valence/"


# In[111]:


V_consensus.to_csv(path+'V_consensus.csv')


# In[98]:


V_pa.to_csv(path+'percent_agree.csv')


# In[172]:


V_splits.to_pickle(path+'Raw_annotations_segmented')


# In[173]:


V_seg_col_stats.to_pickle(path+'column_stats')


# In[174]:


V_seg_row_stats.to_pickle(path+'row_stats')


# In[175]:


V_seg_derived_stats.to_pickle(path+'derived_stats')


# In[176]:


path = "/Users/jay/Documents/Jay's bits/Uni/Thesis/thesis-pipeline/data/interim/"

V_bkps.to_pickle(path+'Valence_breaks')

