# %%
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import decomposition
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn. ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score
# %%
#get data from original experiment
IFN0916_CellIntensity = pd.read_excel("../Data/09162022_CombineExtractFeature.xlsx", sheet_name = "IFN_celli")
IFN0916_CellIntensity = IFN0916_CellIntensity.dropna()
UT0916_CellIntensity = pd.read_excel("../Data/09162022_CombineExtractFeature.xlsx", sheet_name = "UT_celli")
UT0916_CellIntensity = UT0916_CellIntensity.dropna()
Intensity0916 = pd.concat([UT0916_CellIntensity, IFN0916_CellIntensity])

IFN0916_CellMorph = pd.read_excel("../Data/09162022_CombineExtractFeature.xlsx", sheet_name = "IFN_cellm")
IFN0916_CellMorph = IFN0916_CellMorph.dropna()
UT0916_CellMorph = pd.read_excel("../Data/09162022_CombineExtractFeature.xlsx", sheet_name = "UT_cellm")
UT0916_CellMorph = UT0916_CellMorph.dropna()
Morph0916 = pd.concat([UT0916_CellMorph, IFN0916_CellMorph])

df0916 = pd.concat([ Intensity0916, Morph0916.drop(['Treatment', 'ImageNumber', 'ObjectNumber', ],axis=1)],axis=1)
# %%# autoscale morph and intensity features of UT and IFN cells together
scaler=StandardScaler(copy=True, with_mean=True, with_std=True)

Data_0916 = df0916.drop(['Treatment', 'ImageNumber', 'ObjectNumber', 'Phase', 'Treatment_Phase',],axis=1)
Data_0916_s = scaler.fit_transform(Data_0916)
Data_0916_s = pd.DataFrame(Data_0916_s).reset_index().drop(['index'],axis=1)

df0916 = df0916.reset_index().drop(['index'],axis=1)
df0916_s = pd.concat([df0916.Treatment, df0916.ImageNumber, df0916.ObjectNumber, df0916.Phase, df0916.Treatment_Phase,
                      Data_0916_s],axis = 1,ignore_index = bool)
df0916_s.columns = [
    'Treatment', 'ImageNumber', 'ObjectNumber', 'Phase', 'Treatment_Phase',
    #intensity features
    'Cell_IntegratedIntensityEdge', 'Cell_IntegratedIntensity',
       'Cell_LowerQuartileIntensity', 'Cell_MADIntensity',
       'Cell_MassDisplacement', 'Cell_MaxIntensityEdge', 'Cell_MaxIntensity',
       'Cell_MeanIntensityEdge', 'Cell_MeanIntensity', 'Cell_MedianIntensity',
       'Cell_MinIntensityEdge', 'Cell_MinIntensity', 'Cell_StdIntensityEdge',
       'Cell_StdIntensity', 'Cell_UpperQuartileIntensity', 
    #morph features
    'Cell_Area', 'Cell_BoundingBoxArea', 'Cell_BoundingBoxMaximum_X',
    'Cell_BoundingBoxMaximum_Y', 'Cell_BoundingBoxMinimum_X',
    'Cell_BoundingBoxMinimum_Y', 'Cell_Center_X', 'Cell_Center_Y',
    'Cell_Compactness', 'Cell_ConvexArea', 'Cell_Eccentricity',
    'Cell_EquivalentDiameter', 'Cell_Extent', 'Cell_FormFactor',
    'Cell_MajorAxisLength', 'Cell_MaxFeretDiameter', 'Cell_MaximumRadius',
    'Cell_MeanRadius', 'Cell_MedianRadius', 'Cell_MinFeretDiameter',
    'Cell_MinorAxisLength', 'Cell_Orientation', 'Cell_Perimeter',
    'Cell_Solidity'
]
# %%
# keep 9 meaningful intensity features for UMAP spacing
Data9_Intensity0916_s = df0916_s.drop([
    'Treatment', 'ImageNumber', 'ObjectNumber', 'Phase', 'Treatment_Phase',
    #intensity features
    "Cell_IntegratedIntensityEdge","Cell_MaxIntensityEdge","Cell_MeanIntensityEdge","Cell_MinIntensityEdge",
    "Cell_MinIntensity","Cell_StdIntensityEdge",
    #morph features
    'Cell_Area', 'Cell_BoundingBoxArea', 'Cell_BoundingBoxMaximum_X',
    'Cell_BoundingBoxMaximum_Y', 'Cell_BoundingBoxMinimum_X',
    'Cell_BoundingBoxMinimum_Y', 'Cell_Center_X', 'Cell_Center_Y',
    'Cell_Compactness', 'Cell_ConvexArea', 'Cell_Eccentricity',
    'Cell_EquivalentDiameter', 'Cell_Extent', 'Cell_FormFactor',
    'Cell_MajorAxisLength', 'Cell_MaxFeretDiameter', 'Cell_MaximumRadius',
    'Cell_MeanRadius', 'Cell_MedianRadius', 'Cell_MinFeretDiameter',
    'Cell_MinorAxisLength', 'Cell_Orientation', 'Cell_Perimeter',
    'Cell_Solidity'
],axis=1)
# %%
#UMAP space for PE0916, including UT and IFN cells
import umap
#reducer1 for intensity UMAP space
reducer1 = umap.UMAP(n_neighbors=300,n_components = 2,min_dist=0, random_state=42,)
UMAP_intensity = reducer1.fit(Data9_Intensity0916_s)
embedding_intensity0916 = UMAP_intensity.transform(Data9_Intensity0916_s)
# %%
# create dataset by including UMAP embedding
UMAP0916 = pd.concat([pd.DataFrame(data=embedding_intensity0916, columns=['UMAP1_Intensity', 'UMAP2_Intensity']), df0916_s], axis=1)
# %%
#plot UMAP space of PE0916 based on 9 intensity features
sns.set(style='white', context='notebook',font_scale =2, rc={'figure.figsize':(6,6)})
colors = [ "#4668a3", #UT
          "#ed9b3e", #50IFN
         ]
g = sns.scatterplot(data=UMAP0916,
                   x='UMAP1_Intensity',
                   y='UMAP2_Intensity', 
                   hue="Treatment",
                   palette = sns.color_palette(colors),
                    s=20,
                    edgecolor = 'face',
                    alpha=0.8,
                       )
g.set_xlabel('UMAP1_Intensity', fontsize = 30, )
g.set_ylabel('UMAP2_Intensity', fontsize = 30,  )

legend = plt.legend(bbox_to_anchor=(0.4,0.25),borderaxespad=0)
for handle in legend.legendHandles:
    handle._sizes = [400] 
plt.savefig(f'../Plot/UTnIFN_UMAP.svg', format='svg', dpi=300, bbox_inches='tight')
# %%# K-means clustering of PE0916 to get C0(IDO high) and C1(IDO low) from IFN cells
from sklearn.cluster import KMeans
IFN0916 = UMAP0916.query('Treatment == "IFN"').reset_index(drop=True)
x1=IFN0916.UMAP1_Intensity
y1=IFN0916.UMAP2_Intensity
umap_IFN0916 = list(zip(x1,y1))
kmeans0916 = KMeans(n_clusters=2).fit(umap_IFN0916)
# %%
kclst_IFN0916=pd.concat([pd.DataFrame(data=kmeans0916.labels_, columns=['Cluster']), IFN0916],axis=1)

# %%
# replce clusters name
kclst_IFN0916 = kclst_IFN0916.replace({'Cluster': 0}, 'IDO_Low')
kclst_IFN0916 = kclst_IFN0916.replace({'Cluster': 1}, 'IDO_High')
# %%
# plot UMAP space colored in clusters
sns.set(style='white', context='notebook',font_scale =2, rc={'figure.figsize':(6,6)})
colors = [ 
   "#37a193", #cluster 1, high intensity
   "#942b2b",  #cluster 0, low intensity
]
g = sns.scatterplot(data=kclst_IFN0916,
                   x='UMAP1_Intensity', 
                   y='UMAP2_Intensity', 
                   hue="Cluster",
                   palette = sns.color_palette(colors),
                    s=20,
                    edgecolor = 'face',
                    alpha=0.8,
                
                       )

g.set_xlabel('UMAP1_Intensity', fontsize = 30, )
g.set_ylabel('UMAP2_Intensity',fontsize = 30, )

legend = plt.legend(bbox_to_anchor=(0.55,0.25),borderaxespad=0)
for handle in legend.legendHandles:
    handle._sizes = [400]  
plt.savefig(f'../Plot/Cluster_UMAP.svg', format='svg', dpi=300, bbox_inches='tight')


# %%
#################### transfer UMAP on PE1123####################
df1123 = pd.read_excel("../Data/11232022_autoscale_CombineFeatures.xlsx")
df1123 = df1123.dropna()
# %%#create datafile with 9 meaningful intensity features of PE1123 for transformed UMAP
Data9_Intensity1123 = df1123.drop([
    'Treatment', 'ImageNumber', 'ObjectNumber', 'Phase', 'Treatment_Phase',
       'Cell_Area', 'Cell_BoundingBoxArea', 'Cell_BoundingBoxMaximum_X',
       'Cell_BoundingBoxMaximum_Y', 'Cell_BoundingBoxMinimum_X',
       'Cell_BoundingBoxMinimum_Y', 'Cell_Center_X', 'Cell_Center_Y',
       'Cell_Compactness', 'Cell_ConvexArea', 'Cell_Eccentricity',
       'Cell_EquivalentDiameter', 'Cell_Extent', 'Cell_FormFactor',
       'Cell_MajorAxisLength', 'Cell_MaxFeretDiameter', 'Cell_MaximumRadius',
       'Cell_MeanRadius', 'Cell_MedianRadius', 'Cell_MinFeretDiameter',
       'Cell_MinorAxisLength', 'Cell_Orientation', 'Cell_Perimeter',
       'Cell_Solidity', 
       "Cell_IntegratedIntensityEdge","Cell_MaxIntensityEdge","Cell_MeanIntensityEdge","Cell_MinIntensityEdge",
    "Cell_MinIntensity","Cell_StdIntensityEdge",
],axis=1)
# %%Transform UMAP space
embedding_intensity1123 = UMAP_intensity.transform(Data9_Intensity1123)

# %%
UMAP1123 = pd.concat([pd.DataFrame(data=embedding_intensity1123, columns=['UMAP1_Intensity','UMAP2_Intensity']),df1123],axis=1)
UMAP1123 = UMAP1123.dropna()
# %%
#replace treatment name in UMAP1123
UMAP1123 = UMAP1123.replace({'Treatment': 'UT1123'}, 'UT')
UMAP1123 = UMAP1123.replace({'Treatment': 'IFN1123'}, 'IFN')
# %%
#plot UMAP space of PE0916 based on 9 intensity features
sns.set(style='white', context='notebook',font_scale =2, rc={'figure.figsize':(6,6)})
colors = [  "#6ea9cb",  #UT1123
    "#e67c83", #IFN1123
         ]
g = sns.scatterplot(data=UMAP1123,
                   x='UMAP1_Intensity',
                   y='UMAP2_Intensity', 
                   hue="Treatment",
                   palette = sns.color_palette(colors),
                    s=20,
                    edgecolor = 'face',
                    alpha=0.8,
                       )
g.set_xlabel('UMAP1 Intensity', fontsize = 30, )
g.set_ylabel('UMAP2 Intensity', fontsize = 30,  )
legend = plt.legend(bbox_to_anchor=(0.4,0.25),borderaxespad=0)
for handle in legend.legendHandles:
    handle._sizes = [400] 
plt.savefig(f'../Plot/IDO1123_UTnIFN_UMAP.svg', format='svg', dpi=300, bbox_inches='tight')
# %%
# Get PE0916 train-test split for model validation of prediction
C0_train, C0_test = train_test_split(kclst_IFN0916.query("Cluster == 'IDO_High'"), train_size = 1000)
C1_train, C1_test = train_test_split(kclst_IFN0916.query("Cluster == 'IDO_Low'"), train_size = 1000)
Train = pd.concat([C0_train, C1_train]).reset_index(drop=True)
Test = pd.concat([C0_test, C1_test]).reset_index(drop=True)
# %%Use overlapped cell morph for trainig
Xtrain = Train.drop(['Cluster', 'UMAP1_Intensity', 'UMAP2_Intensity', 'Treatment',
       'ImageNumber', 'ObjectNumber', 'Phase', 'Treatment_Phase',
       'Cell_IntegratedIntensityEdge', 'Cell_IntegratedIntensity',
       'Cell_LowerQuartileIntensity', 'Cell_MADIntensity',
       'Cell_MassDisplacement', 'Cell_MaxIntensityEdge', 'Cell_MaxIntensity',
       'Cell_MeanIntensityEdge', 'Cell_MeanIntensity', 'Cell_MedianIntensity',
       'Cell_MinIntensityEdge', 'Cell_MinIntensity', 'Cell_StdIntensityEdge',
       'Cell_StdIntensity', 'Cell_UpperQuartileIntensity', 
       #extra morph
       'Cell_BoundingBoxArea','Cell_BoundingBoxMaximum_X','Cell_BoundingBoxMaximum_Y',
    'Cell_BoundingBoxMinimum_X','Cell_BoundingBoxMinimum_Y', 'Cell_Center_X','Cell_Center_Y',
    'Cell_ConvexArea','Cell_EquivalentDiameter','Cell_Orientation'
       ],axis=1)
Y_train_U1 = Train.UMAP1_Intensity
Y_train_U2 = Train.UMAP2_Intensity
Y_train_c = Train.Cluster

XTest = Test.drop(['Cluster', 'UMAP1_Intensity', 'UMAP2_Intensity', 'Treatment',
       'ImageNumber', 'ObjectNumber', 'Phase', 'Treatment_Phase',
       'Cell_IntegratedIntensityEdge', 'Cell_IntegratedIntensity',
       'Cell_LowerQuartileIntensity', 'Cell_MADIntensity',
       'Cell_MassDisplacement', 'Cell_MaxIntensityEdge', 'Cell_MaxIntensity',
       'Cell_MeanIntensityEdge', 'Cell_MeanIntensity', 'Cell_MedianIntensity',
       'Cell_MinIntensityEdge', 'Cell_MinIntensity', 'Cell_StdIntensityEdge',
       'Cell_StdIntensity', 'Cell_UpperQuartileIntensity',
       #extra morph
       'Cell_BoundingBoxArea','Cell_BoundingBoxMaximum_X','Cell_BoundingBoxMaximum_Y',
    'Cell_BoundingBoxMinimum_X','Cell_BoundingBoxMinimum_Y', 'Cell_Center_X','Cell_Center_Y',
    'Cell_ConvexArea','Cell_EquivalentDiameter','Cell_Orientation'],axis=1)
Y_Test_U1 = Test.UMAP1_Intensity
Y_Test_U2 = Test.UMAP2_Intensity
Y_Test_c = Test.Cluster

# %%
# Use Kmeans clustering to cluster experiment IFN1123
IFN1123 = UMAP1123.query('Treatment == "IFN"').reset_index(drop=True)
x2=IFN1123.UMAP1_Intensity
y2=IFN1123.UMAP2_Intensity
umap_IFN1123 = list(zip(x2,y2))
kmeans1123 = KMeans(n_clusters=2).fit(umap_IFN1123)
label_IFN1123=pd.DataFrame(kmeans1123.labels_)

IFN1123 = pd.concat([pd.DataFrame(data=kmeans1123.labels_, columns=['Cluster']), UMAP1123.query('Treatment == "IFN"').reset_index(drop=True)],axis=1)

IFN1123 = IFN1123.replace({'Cluster': 1}, 'IDO_Low')
IFN1123 = IFN1123.replace({'Cluster': 0}, 'IDO_High')


# %%
# plot UMAP space colored in clusters
sns.set(style='white', context='notebook',font_scale =2, rc={'figure.figsize':(6,6)})
colors = [ 
      "#942b2b",  #cluster 0, low intensity
    "#37a193", #cluster 1, high intensity
]
g = sns.scatterplot(data=IFN1123,
                   x='UMAP1_Intensity', 
                   y='UMAP2_Intensity', 
                   hue="Cluster",
                   palette = sns.color_palette(colors),
                    s=20,
                    edgecolor = 'face',
                       )
g.set_xlabel('UMAP1 Intensity', fontsize = 30, )
g.set_ylabel('UMAP2 Intensity',fontsize = 30, )

legend = plt.legend(bbox_to_anchor=(0.55,0.25),borderaxespad=0)
for handle in legend.legendHandles:
    handle._sizes = [400] 
plt.savefig(f'../Plot/IDO1123_Cluster_UMAP.svg', format='svg', dpi=300, bbox_inches='tight')

# %%
##################################################
##################################################
############ MLPR predict IDO activity ############
##################################################
##################################################
# %%
# IFN donor with reported/measured IDO activity as target value
df = pd.read_excel('../Data/MultiDonor_Fluo_IFN.xlsx').drop(['Index'], axis=1)
df = df.dropna()
#Randomly seperate train and test dataset
Train_IFN, Test_IFN = train_test_split(df, train_size=900)

# %%
#smooth as bulk/group cell, 10%
#comments: prediction is not really good without smoothing,
# df_balance_RM = df_balance.groupby('Donor1').rolling(center=False,window=99,min_periods=1).mean().reset_index()
# df_balance_RM1 = df_balance_RM.drop(['Donor1', 'level_1', 'Donor2', 'RB_IDO', 'ImageNumber', 'ObjectNumber',],axis=1)

# %%
#keep 14 morph features as X for and reported/measured IDO activity as Y
Train_IFN = Train_IFN.reset_index(drop=True)
Xtrain_IFN = Train_IFN.drop(['Donor1','Donor2', 'RB_IDO', 'ImageNumber', 'ObjectNumber',],axis=1)
Ytrain_IFN = Train_IFN.RB_IDO
Test_IFN = Test_IFN.reset_index(drop=True)
Xtest_IFN = Test_IFN.drop(['Donor1','Donor2', 'RB_IDO', 'ImageNumber', 'ObjectNumber',],axis=1)
Ytest_IFN = Test_IFN.RB_IDO
# %%
from sklearn.neural_network import MLPRegressor
mlpr = MLPRegressor(activation='tanh', alpha= 201,hidden_layer_sizes= (50, 50, 50),learning_rate= 'constant',max_iter= 400,solver= 'sgd').fit(Xtrain_IFN, Ytrain_IFN)
# %%Apply multidonor IDO prediction regressor on IFN cells from IDOPE0916
# kclst_IFN0916_RM = kclst_IFN0916.rolling(center=False,window=265,min_periods=1).mean().reset_index(drop=True)
#Select overlaped 14 morphologies
Morph14_IFN0916 = kclst_IFN0916.drop([
      'Cluster', 'UMAP1_Intensity', 'UMAP2_Intensity', 'Treatment',
       'ImageNumber', 'ObjectNumber', 'Phase', 'Treatment_Phase',
       'Cell_IntegratedIntensityEdge', 'Cell_IntegratedIntensity',
       'Cell_LowerQuartileIntensity', 'Cell_MADIntensity',
       'Cell_MassDisplacement', 'Cell_MaxIntensityEdge', 'Cell_MaxIntensity',
       'Cell_MeanIntensityEdge', 'Cell_MeanIntensity', 'Cell_MedianIntensity',
       'Cell_MinIntensityEdge', 'Cell_MinIntensity', 'Cell_StdIntensityEdge',
       'Cell_StdIntensity', 'Cell_UpperQuartileIntensity',
       #extra morph
       'Cell_BoundingBoxArea','Cell_BoundingBoxMaximum_X','Cell_BoundingBoxMaximum_Y',
    'Cell_BoundingBoxMinimum_X','Cell_BoundingBoxMinimum_Y', 'Cell_Center_X','Cell_Center_Y',
    'Cell_ConvexArea','Cell_EquivalentDiameter','Cell_Orientation'
],axis=1)

# %%
Pred0916_IDO = pd.concat([pd.DataFrame(data=mlpr.predict(Morph14_IFN0916), columns=['MLPR_IDO']), kclst_IFN0916],axis=1)
# %%
#plot of transform UMAP space, hue in predicted IDO activity
sns.set(style='white', context='notebook',font_scale =2, rc={'figure.figsize':(7,6)})
g = sns.scatterplot(data=Pred0916_IDO,
                   x='UMAP1_Intensity',  
                   y='UMAP2_Intensity',
                    s=10, edgecolor='face',
                    hue="MLPR_IDO",        
                    palette='RdPu',        
                       )

norm = plt.Normalize(Pred0916_IDO['MLPR_IDO'].min(), Pred0916_IDO['MLPR_IDO'].max())
sm = plt.cm.ScalarMappable(cmap="RdPu", norm=norm)
sm.set_array([])
# Add the colorbar and set the label
cbar = g.figure.colorbar(sm, ax=g, orientation='vertical')
cbar.set_label('Predicted IDO Activity\n(pg KYN/cell/day)', rotation=90, labelpad=15)
# Remove the legend and add a colorbar
g.get_legend().remove()
plt.savefig(f'../Plot/PredictIDO0916.svg', format='svg', dpi=300, bbox_inches='tight')

# %%
########################################
######### color code on PE0916 #########
########################################
import os
from glob import glob
from scipy import ndimage
import tifffile as T
from numpy import *
from sklearn.preprocessing import normalize
import sys
np.set_printoptions(threshold=sys.maxsize)
com = ndimage.measurements.center_of_mass

# %% fn1=original image; fn2=16unit mask for labeling;fn3=output color coded image
fn1 = r'/Users/ruikejie/Desktop/OffLineWork/TransData/PE0916_ColorCode/D3_50IFN_Cy5.tif'
fn2 = r'/Users/ruikejie/Desktop/OffLineWork/TransData/PE0916_ColorCode/D3_50IFN_Cy5_ColorMatrix.tiff'
fno = r'/Users/ruikejie/Desktop/OffLineWork/TransData/PE0916_ColorCode/PE0916_D3_50IFN_CC_6layers.tif'

img = T.imread(fn1)
### Number of categories
num_cat = 7
arr = T.imread(fn2)
ny,nx = arr.shape
nz = int(arr.max())

MIN = Pred0916_IDO.pRFR_IDO.min()
MAX = Pred0916_IDO.pRFR_IDO.max()
rng = linspace(MIN,MAX,num_cat+1)
out = zeros((num_cat,ny,nx),dtype=int16)

Pred0916_IDO_array = Pred0916_IDO.to_numpy()
for i in range(nz):
#     print(i)
    a = Pred0916_IDO_array[i,0]
   
    msk = (arr==(i+1))

    aro = msk*img

    for j in range(num_cat):

        b = rng[j]

        c = rng[j+1]

        if ((a>b)and(a<c)):
            
#             np.add(num_cat-j-1, aro, out = num_cat-j-1, casting ="unsafe" )

            out[num_cat-j-1] += aro
T.imsave(fno,out)
# %%
