#%%
###Use all UT donor from Phase images with dilation for segmentation
##1Dimention reduction: PCA and UMAP##
#1.1Check Morph among donors
#1.2Compare Morph for all donors to fluorescent harvested morphology
##2

#%%
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import decomposition
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cross_decomposition import PLSRegression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn. ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# %%
#select2-1: delete image61-115 of RB71 due to bad segmentation;averaged texture features;
df = pd.read_excel("/Users/ruikejie/Desktop/OffLineWork/TransData/MultiDonor/MultiDonor_Phase_UTDilate_Combine.xlsx", sheet_name='select2-1')
df=df.dropna()
# %% Auto scaling all features
scaler=StandardScaler(copy=True, with_mean=True, with_std=True)
data = df.drop(['Exp', 'RB_Donor', 'Donor', 'Treatment', 'RB_IDO', 'ImageNumber','ObjectNumber',],axis=1)
data_s = pd.DataFrame(scaler.fit_transform(data))
df_s = pd.concat([
    df.Exp, df.RB_Donor, df.Donor, 
    df.Treatment, df.RB_IDO,
    df.ImageNumber, df.ObjectNumber, 
    data_s
],axis = 1,ignore_index = bool)
df_s.columns = ['Exp', 'RB_Donor', 'Donor', 'Treatment', 'RB_IDO', 'ImageNumber','ObjectNumber', 
    #    #14morph
    #    'Cell_Area', 'Cell_Compactness', 'Cell_Eccentricity',
    #    'Cell_Extent', 'Cell_FormFactor', 'Cell_MajorAxisLength',
    #    'Cell_MaxFeretDiameter', 'Cell_MaximumRadius', 'Cell_MeanRadius',
    #    'Cell_MedianRadius', 'Cell_MinFeretDiameter', 'Cell_MinorAxisLength',
    #    'Cell_Perimeter', 'Cell_Solidity'
       #morph
       'Cell_Area', 'Cell_BoundingBoxArea',
       'Cell_BoundingBoxMaximum_X', 'Cell_BoundingBoxMaximum_Y',
       'Cell_BoundingBoxMinimum_X', 'Cell_BoundingBoxMinimum_Y',
       'Cell_Center_X', 'Cell_Center_Y', 'Cell_Compactness', 'Cell_ConvexArea',
       'Cell_Eccentricity', 'Cell_EquivalentDiameter', 'Cell_Extent',
       'Cell_FormFactor', 'Cell_MajorAxisLength', 'Cell_MaxFeretDiameter',
       'Cell_MaximumRadius', 'Cell_MeanRadius', 'Cell_MedianRadius',
       'Cell_MinFeretDiameter', 'Cell_MinorAxisLength', 'Cell_Orientation',
       'Cell_Perimeter', 'Cell_Solidity',
       #intensity
       'DilateImage_IntegratedIntensityEdge',
       'DilateImage_IntegratedIntensity', 'DilateImage_LowerQuartileIntensity',
       'DilateImage_MADIntensity', 'DilateImage_MassDisplacement',
       'DilateImage_MaxIntensityEdge', 'DilateImage_MeanIntensityEdge',
       'DilateImage_MeanIntensity', 'DilateImage_MedianIntensity',
       'DilateImage_StdIntensityEdge', 'DilateImage_StdIntensity',
       'DilateImage_UpperQuartileIntensity', 
       #texture
       'Texture_AngularSecondMoment',
       'Texture_Contrast', 'Texture_Correlation', 'Texture_DifferenceEntropy',
       'Texture_DifferenceVariance', 'Texture_Entropy', 'Texture_InfoMeas1',
       'Texture_InfoMeas2', 'Texture_InverseDifferenceMoment',
       'Texture_SumAverage', 'Texture_SumEntropy', 'Texture_SumVariance',
       'Texture_Variance'
    ]

# %% get features for feature selections
feature = df_s.drop(['Exp', 'RB_Donor', 'Donor', 'Treatment', 'RB_IDO', 'ImageNumber','ObjectNumber',],axis=1)
# Calculate the correlation matrix
correlation_matrix = feature.corr().abs()
# Create a mask to ignore highly correlated features
mask = np.triu(np.ones(correlation_matrix.shape), k=0).astype(bool)
correlation_matrix_no_diag = correlation_matrix.mask(mask)
# Set a threshold for correlation
correlation_threshold = 0.9
plt.figure(figsize=(40, 32))
sns.heatmap(correlation_matrix_no_diag, cmap='coolwarm', linewidths=0.5,vmin=0, vmax=1)
plt.show()
# %%
# Find features that are highly correlated and remove them
highly_correlated_features = [column for column in correlation_matrix_no_diag.columns if any(correlation_matrix_no_diag[column] > correlation_threshold)]
feature_filtered = feature.drop(highly_correlated_features, axis=1)
df_filtered = df_s.drop(highly_correlated_features, axis=1)
# Calculate the correlation matrix
filter_matrix = feature_filtered.corr().abs()
# Create a mask to ignore highly correlated features
filter_mask = np.triu(np.ones(filter_matrix.shape), k=0).astype(bool)
filter_matrix_no_diag = filter_matrix.mask(filter_mask)
plt.figure(figsize=(40, 32))
sns.heatmap(filter_matrix_no_diag, cmap='coolwarm', linewidths=0.5,vmin=0, vmax=1)
plt.show()

# %%
RB48 = df_filtered.query('RB_Donor == "RB48_UT"').sample(n=323)
RB71 = df_filtered.query('RB_Donor == "RB71_UT"').sample(n=323)
RB183 = df_filtered.query('RB_Donor == "RB183_UT"').sample(n=323)
RB179 = df_filtered.query('RB_Donor == "RB179_UT"').sample(n=323)
RB177 = df_filtered.query('RB_Donor == "RB177_UT"').sample(n=323)
RB175 = df_filtered.query('RB_Donor == "RB175_UT"').sample(n=323)
# %%
df_RM = pd.concat([RB48,RB71,RB183,RB179,RB177,RB175]).groupby('Donor').rolling(center=False,window=32,min_periods=1).mean().reset_index()
df_RM1 = df_RM.drop(['Donor', 'level_1', 'RB_IDO', 'ImageNumber', 'ObjectNumber',],axis=1)

# %%
def PCA_Python(X, y, scaled=False):
    pca = decomposition.PCA(n_components=3)
    
    if scaled == True:
        X_centered = (X - X.mean(axis=0))/np.sqrt(X.std(axis=0)) # pareto scaling.
        # X_centered =(X - X.mean(axis=0))/X.std(axis=0) #autoscaling
    elif scaled == False:
        X_centered = X
    
    pca.fit(X_centered)
    X_pca = pca.transform(X_centered)
    Var = pca.explained_variance_ratio_ # returns variance ratio of the selected component.
    Ld = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2', 'PC3'], index=X.columns)
  
    a01 = plt.plot(X_pca[y=="RB48_UT_Phase"][:,0], X_pca[y=="RB48_UT_Phase"][:,1], marker = 'o',color = '#8caff5', linestyle = 'None',label='RB48_UT_Phase');
    a02 = plt.plot(X_pca[y=="RB175_UT_Phase"][:,0], X_pca[y=="RB175_UT_Phase"][:,1], marker = 'o',color = '#98999c', linestyle = 'None',label='RB175_UT_Phase');
    a03 = plt.plot(X_pca[y=="RB179_UT_Phase"][:,0], X_pca[y=="RB179_UT_Phase"][:,1], marker = 'o',color = '#cb8deb', linestyle = 'None',label='RB179_UT_Phase');
    a04 = plt.plot(X_pca[y=="RB71_UT_Phase"][:,0], X_pca[y=="RB71_UT_Phase"][:,1], marker = 'o',color = '#636999', linestyle = 'None',label='RB71_UT_Phase');
    a05 = plt.plot(X_pca[y=="RB183_UT_Phase"][:,0], X_pca[y=="RB183_UT_Phase"][:,1], marker = 'o',color = '#e38a05', linestyle = 'None',label='RB183_UT_Phase');
    a06 = plt.plot(X_pca[y=="RB177_UT_Phase"][:,0], X_pca[y=="RB177_UT_Phase"][:,1], marker = 'o',color = '#ed63ff', linestyle = 'None',label='RB177_UT_Phase');
   
    # a11 = plt.plot(X_pca[y=="RB48_IFN"][:,0], X_pca[y=="RB48_IFN"][:,1], marker = 'x',color = '#8caff5', linestyle = 'None',label='RB48_IFN');
    # a12 = plt.plot(X_pca[y=="RB175_IFN"][:,0], X_pca[y=="RB175_IFN"][:,1], marker = 'x',color = '#98999c', linestyle = 'None',label='RB175_IFN');
    # a13 = plt.plot(X_pca[y=="RB179_IFN"][:,0], X_pca[y=="RB179_IFN"][:,1], marker = 'x',color = '#cb8deb', linestyle = 'None',label='RB179_IFN');
    # a14 = plt.plot(X_pca[y=="RB71_IFN"][:,0], X_pca[y=="RB71_IFN"][:,1], marker = 'x',color = '#636999', linestyle = 'None',label='RB71_IFN');
    # a15 = plt.plot(X_pca[y=="RB183_IFN"][:,0], X_pca[y=="RB183_IFN"][:,1], marker = 'x',color = '#e38a05', linestyle = 'None',label='RB183_IFN');
    # a16 = plt.plot(X_pca[y=="RB177_IFN"][:,0], X_pca[y=="RB177_IFN"][:,1], marker = 'x',color = '#ed63ff', linestyle = 'None',label='RB177_IFN');
   
    # a01 = plt.plot(X_pca[y=="RB48_UT_Phase"][:,0], X_pca[y=="RB48_UT_Phase"][:,1], marker = 'o',color = '#4668a3', linestyle = 'None',label='RB48_UT_Phase');
    # a02 = plt.plot(X_pca[y=="RB175_UT_Phase"][:,0], X_pca[y=="RB175_UT_Phase"][:,1], marker = 'o',color = '#4668a3', linestyle = 'None',label='RB175_UT_Phase');
    # a03 = plt.plot(X_pca[y=="RB179_UT_Phase"][:,0], X_pca[y=="RB179_UT_Phase"][:,1], marker = 'o',color = '#4668a3', linestyle = 'None',label='RB179_UT_Phase');
    # a04 = plt.plot(X_pca[y=="RB71_UT_Phase"][:,0], X_pca[y=="RB71_UT_Phase"][:,1], marker = 'o',color = '#4668a3', linestyle = 'None',label='RB71_UT_Phase');
    # a05 = plt.plot(X_pca[y=="RB183_UT_Phase"][:,0], X_pca[y=="RB183_UT_Phase"][:,1], marker = 'o',color = '#4668a3', linestyle = 'None',label='RB183_UT_Phase');
    # a06 = plt.plot(X_pca[y=="RB177_UT_Phase"][:,0], X_pca[y=="RB177_UT_Phase"][:,1], marker = 'o',color = '#4668a3', linestyle = 'None',label='RB177_UT_Phase');
   
    # a11 = plt.plot(X_pca[y=="RB48_IFN"][:,0], X_pca[y=="RB48_IFN"][:,1], marker = 'x',color = '#ed9b3e', linestyle = 'None',label='RB48_IFN');
    # a12 = plt.plot(X_pca[y=="RB175_IFN"][:,0], X_pca[y=="RB175_IFN"][:,1], marker = 'x',color = '#ed9b3e', linestyle = 'None',label='RB175_IFN');
    # a13 = plt.plot(X_pca[y=="RB179_IFN"][:,0], X_pca[y=="RB179_IFN"][:,1], marker = 'x',color = '#ed9b3e', linestyle = 'None',label='RB179_IFN');
    # a14 = plt.plot(X_pca[y=="RB71_IFN"][:,0], X_pca[y=="RB71_IFN"][:,1], marker = 'x',color = '#ed9b3e', linestyle = 'None',label='RB71_IFN');
    # a15 = plt.plot(X_pca[y=="RB183_IFN"][:,0], X_pca[y=="RB183_IFN"][:,1], marker = 'x',color = '#ed9b3e', linestyle = 'None',label='RB183_IFN');
    # a16 = plt.plot(X_pca[y=="RB177_IFN"][:,0], X_pca[y=="RB177_IFN"][:,1], marker = 'x',color = '#ed9b3e', linestyle = 'None',label='RB177_IFN');
   
    
    c = plt.xlabel("PC1: " + str(round(Var[0]*100, 2)) + "%", fontsize = 30);
    d = plt.ylabel("PC2: " + str(round(Var[1]*100, 2))+ "%",fontsize = 30);
    e = plt.legend(loc='best', fontsize = 15);
    f = plt.tick_params(labelsize = 27)
    return (X_pca,Var,Ld,X_centered)

(pcaResult,pcaVar,pcaLoadings,X_scaled)=PCA_Python(df_RM1, df_RM.Donor,scaled=False)
# fig  = PCA_Python(dfRM1PC, dfRM.Label,scaled=True)
sns.set(context='notebook', style='white', font='sans-serif', font_scale=2, color_codes=True, rc={'figure.figsize':(8,6)})
plt.legend(bbox_to_anchor=(1.08, 0.8),fontsize = 18, borderaxespad=0)

# plt.xlim(-3,3.5)
# plt.ylim(-3,4.5)
# %%
PCLoadingArray = np.array(pcaLoadings)
PCLoading = pd.DataFrame(data=PCLoadingArray, index=df_RM1.columns, columns=["PC"+str(i+1) for i in range(PCLoadingArray.shape[1])])
PCLoading['MorphFeatures'] = PCLoading.index.astype(str)
PC1Loading = PCLoading.sort_values(['PC1']).reset_index(drop=True)
plt.figure(figsize=(6,16))

ax = sns.barplot(x='PC1',y='MorphFeatures',data=PC1Loading, orient = 'h',palette = "Wistia")

ax.set_xlabel('PC1 Loading', fontsize = 30)
ax.set_ylabel(" ", fontsize = 30)
ax.tick_params(labelsize = 10)
# plt.savefig('/Users/ruikejie/Desktop/Ph.D Program/Python/vscodeproject/MultiDonorPhase/MultiDonor_Phase_PCA_PC1.svg',dpi=300,bbox_inches='tight')

# %%

PC2Loading = PCLoading.sort_values(['PC2']).reset_index(drop=True)
plt.figure(figsize=(6,16))

ax = sns.barplot(x='PC2',y='MorphFeatures',data=PC2Loading, orient = 'h',palette = "Wistia")

ax.set_xlabel('PC2 Loading', fontsize = 30)
ax.set_ylabel(" ", fontsize = 30)
ax.tick_params(labelsize = 10)

# %%
import umap
reducer = umap.UMAP(n_neighbors=50,n_components = 2,min_dist=0.5, random_state=42,)

UMAP_all = reducer.fit(df_RM1)
embedding_all = UMAP_all.transform(df_RM1)

# %%
Uloading_all = pd.concat([pd.DataFrame(embedding_all),df_RM],axis=1)
Uloading_all.columns=['UMAP1','UMAP2', 
                    'Donor', 'level_1', 'RB_IDO', 'ImageNumber', 'ObjectNumber',
       'Cell_BoundingBoxArea', 'Cell_Center_X', 'Cell_Center_Y',
       'Cell_Compactness', 'Cell_Eccentricity', 'Cell_EquivalentDiameter',
       'Cell_Extent', 'Cell_FormFactor', 'Cell_MaxFeretDiameter',
       'Cell_MedianRadius', 'Cell_MinorAxisLength', 'Cell_Orientation',
       'Cell_Perimeter', 'Cell_Solidity',
       'DilateImage_IntegratedIntensityEdge',
       'DilateImage_IntegratedIntensity', 'DilateImage_MADIntensity',
       'DilateImage_MassDisplacement', 'DilateImage_MeanIntensityEdge',
       'DilateImage_StdIntensityEdge', 'Texture_Contrast',
       'Texture_Correlation', 'Texture_DifferenceEntropy', 'Texture_InfoMeas1',
       'Texture_InfoMeas2', 'Texture_SumAverage', 'Texture_SumEntropy',
       'Texture_Variance'
       ]
# %%

plt.scatter(
    Uloading_all.query('Donor == "RB179_UT_Phase"').UMAP1,
    Uloading_all.query('Donor == "RB179_UT_Phase"').UMAP2, 
    c='#cb8deb',marker='^',label='RB179_UT_Phase',s=20)
plt.scatter(
    Uloading_all.query('Donor == "RB48_UT_Phase"').UMAP1,
    Uloading_all.query('Donor == "RB48_UT_Phase"').UMAP2, 
    c='#8caff5',marker='^',label='RB48_UT_Phase', s=20)
plt.scatter(
    Uloading_all.query('Donor == "RB177_UT_Phase"').UMAP1,
    Uloading_all.query('Donor == "RB177_UT_Phase"').UMAP2, 
    c='#ed63ff',marker='^',label='RB177_UT_Phase',s=20)
plt.scatter(
    Uloading_all.query('Donor == "RB71_UT_Phase"').UMAP1,
    Uloading_all.query('Donor == "RB71_UT_Phase"').UMAP2, 
    c='#636999',marker='^',label='RB71_UT_Phase',s=20)
plt.scatter(
    Uloading_all.query('Donor == "RB183_UT_Phase"').UMAP1,
    Uloading_all.query('Donor == "RB183_UT_Phase"').UMAP2, 
    c='#e38a05',marker='^',label='RB183_UT_Phase',s=20)
plt.scatter(
    Uloading_all.query('Donor == "RB175_UT_Phase"').UMAP1,
    Uloading_all.query('Donor == "RB175_UT_Phase"').UMAP2, 
    c='#98999c',marker='^',label='RB175_UT_Phase',s=20)

# plt.xlim(-6,15)
# plt.ylim(-6,15)
plt.xlabel("UMAP1", fontsize = 30);
plt.ylabel("UMAP2",fontsize = 30);
plt.legend(loc='best', fontsize = 15);
plt.tick_params(labelsize = 27) 
plt.legend(bbox_to_anchor=(1.51, 0.8),fontsize = 18, borderaxespad=0)
# %%
#train test split
Train_RB179_Dilate, Test_RB179_Dilate = train_test_split(RB179, train_size=300)
Train_RB48_Dilate, Test_RB48_Dilate = train_test_split(RB48, train_size=300)
Train_RB177_Dilate, Test_RB177_Dilate = train_test_split(RB177, train_size=300)
Train_RB71_Dilate, Test_RB71_Dilate = train_test_split(RB71, train_size=300)
Train_RB183_Dilate, Test_RB183_Dilate = train_test_split(RB183, train_size=300)
Train_RB175_Dilate, Test_RB175_Dilate = train_test_split(RB175, train_size=300)

Train_all = pd.concat([Train_RB179_Dilate,Train_RB48_Dilate,Train_RB177_Dilate,Train_RB71_Dilate,Train_RB183_Dilate,Train_RB175_Dilate
                    ]).reset_index(drop=True)
# Train_all_RM = Train_all.groupby('Donor').rolling(center=False,window=30,min_periods=1).mean().reset_index()

Test_all = pd.concat([Test_RB179_Dilate,Test_RB48_Dilate,Test_RB177_Dilate,Test_RB71_Dilate,Test_RB183_Dilate,Test_RB175_Dilate
                    ]).reset_index(drop=True)
# Test_all_RM = Test_all.groupby('Donor').rolling(center=False,window=2,min_periods=1).mean().reset_index()


Xtrain_all = Train_all.drop([ 'Exp', 'RB_Donor', 'Donor','RB_IDO', 'Treatment', 'ImageNumber', 'ObjectNumber',],axis=1)
# Xtrain_all_RM = Train_all_RM.drop([ 'Donor','level_1',  'RB_IDO', 'ImageNumber', 'ObjectNumber',],axis=1)
Ytrain_all = Train_all.RB_IDO

Xtest_all = Test_all.drop([ 'Exp', 'RB_Donor', 'Donor', 'RB_IDO','Treatment', 'ImageNumber', 'ObjectNumber',],axis=1)
# Xtest_all_RM = Test_all_RM.drop([ 'Donor','level_1',  'RB_IDO', 'ImageNumber', 'ObjectNumber',],axis=1)
Ytest_all = Test_all.RB_IDO

# %% RFR
rfr = RandomForestRegressor(n_estimators = 500,oob_score=True )
rfr.fit(Xtrain_all, Ytrain_all)
# %%
score_rfr = rfr.score(Xtrain_all, Ytrain_all)
score_rfr
# %%
Importance_rfr = pd. concat([pd.DataFrame(Xtrain_all.columns),
                         pd.DataFrame(rfr.feature_importances_)], 
                         axis = 1)
Importance_rfr.columns = ["Features", "Importance" ]
Importance_rfr = Importance_rfr.sort_values(['Importance']).reset_index(drop=True)

plt.figure(figsize=(6,15))
ax = sns.barplot(x='Importance',y='Features',data=Importance_rfr, palette = "Wistia", orient = 'h')
ax.set_xlabel('Features Importance', fontsize = 30)
ax.set_ylabel("Morph Features", fontsize = 0)
ax.tick_params(labelsize = 10)
# %%
Ypred_rfr = pd.DataFrame(rfr.predict(Xtest_all))
# %%
from sklearn.metrics import mean_squared_error
test_error = mean_squared_error(Ytest_all, Ypred_rfr)
test_error

# %%
rfr.oob_score_
#%% #Pred_all
Pred_all = pd. concat([Ypred_rfr,
                       Test_all,
                       ],
                        axis = 1,
                        ignore_index = bool
                        )
Pred_all.columns = ["Predict_IDOA",
                    'Exp', 'RB_Donor', 'Donor', 'Treatment', 'RB_IDO', 'ImageNumber',
       'ObjectNumber', 'Cell_BoundingBoxArea', 'Cell_Center_X',
       'Cell_Center_Y', 'Cell_Compactness', 'Cell_Eccentricity',
       'Cell_EquivalentDiameter', 'Cell_Extent', 'Cell_FormFactor',
       'Cell_MaxFeretDiameter', 'Cell_MedianRadius', 'Cell_MinorAxisLength',
       'Cell_Orientation', 'Cell_Perimeter', 'Cell_Solidity',
       'DilateImage_IntegratedIntensityEdge',
       'DilateImage_IntegratedIntensity', 'DilateImage_MADIntensity',
       'DilateImage_MassDisplacement', 'DilateImage_MeanIntensityEdge',
       'DilateImage_StdIntensityEdge', 'Texture_Contrast',
       'Texture_Correlation', 'Texture_DifferenceEntropy', 'Texture_InfoMeas1',
       'Texture_InfoMeas2', 'Texture_SumAverage', 'Texture_SumEntropy',
       'Texture_Variance'
                   ]
#%% #rfr single prediction
#colored as expriment batch
plt.figure(figsize=(10,7))
plt.scatter(
    Pred_all.query('Donor == "RB179_UT_Phase"').RB_IDO,
    Pred_all.query('Donor == "RB179_UT_Phase"').Predict_IDOA, 
    c='#cb8deb',marker='^',label='RB179_DilatePhase',s=50)
plt.scatter(
    Pred_all.query('Donor == "RB48_UT_Phase"').RB_IDO,
    Pred_all.query('Donor == "RB48_UT_Phase"').Predict_IDOA, 
    c='#8caff5',marker='^',label='RB48_DilatePhase', s=50)
plt.scatter(
    Pred_all.query('Donor == "RB177_UT_Phase"').RB_IDO,
    Pred_all.query('Donor == "RB177_UT_Phase"').Predict_IDOA, 
    c='#ed63ff',marker='^',label='RB177_DilatePhase',s=50)
plt.scatter(
    Pred_all.query('Donor == "RB71_UT_Phase"').RB_IDO,
    Pred_all.query('Donor == "RB71_UT_Phase"').Predict_IDOA, 
    c='#636999',marker='^',label='RB71_DilatePhase',s=50)
plt.scatter(
    Pred_all.query('Donor == "RB183_UT_Phase"').RB_IDO,
    Pred_all.query('Donor == "RB183_UT_Phase"').Predict_IDOA, 
    c='#e38a05',marker='^',label='RB183_DilatePhase',s=50)
plt.scatter(
    Pred_all.query('Donor == "RB175_UT_Phase"').RB_IDO,
    Pred_all.query('Donor == "RB175_UT_Phase"').Predict_IDOA, 
    c='#98999c',marker='^',label='RB175_DilatePhase',s=50)

plt.axline((10,10),slope=1,c='black',linestyle='--')
plt.xlim(0,90)
plt.ylim(0,90)
plt.xlabel("Measured IDO Activity\n(pg KYN/cell/day)", fontsize = 30);
plt.ylabel("Predict IDO Activity\n(pg KYN/cell/day)",fontsize = 30);
plt.legend(loc='best', fontsize = 15);
plt.tick_params(labelsize = 27) 
plt.legend(bbox_to_anchor=(1.3, 1),fontsize = 18, borderaxespad=0)
# plt.savefig('/Users/ruikejie/Desktop/Ph.D Program/Python/rfr_Prediction_IFN_Smooth.svg',dpi=300,bbox_inches='tight')


# %% rfr bulk prediction
import statistics
Pred_all_M = Pred_all.groupby('Donor').median()
Pred_all_M_list = Pred_all_M.values.tolist()
Pred_all_M_array = Pred_all_M.to_numpy()

plt.figure(figsize=(10,7))

plt.scatter(
    Pred_all_M.query('Donor == "RB179_UT_Phase"').RB_IDO,
    Pred_all_M.query('Donor == "RB179_UT_Phase"').Predict_IDOA, 
    c='#cb8deb',marker='^',label='RB179_Phase',s=200)
plt.scatter(
    Pred_all_M.query('Donor == "RB48_UT_Phase"').RB_IDO,
    Pred_all_M.query('Donor == "RB48_UT_Phase"').Predict_IDOA, 
    c='#8caff5',marker='^',label='RB48_Phase', s=200)
plt.scatter(
    Pred_all_M.query('Donor == "RB177_UT_Phase"').RB_IDO,
    Pred_all_M.query('Donor == "RB177_UT_Phase"').Predict_IDOA, 
    c='#ed63ff',marker='^',label='RB177_Phase',s=200)
plt.scatter(
    Pred_all_M.query('Donor == "RB71_UT_Phase"').RB_IDO,
    Pred_all_M.query('Donor == "RB71_UT_Phase"').Predict_IDOA, 
    c='#636999',marker='^',label='RB71_Phase',s=200)
plt.scatter(
    Pred_all_M.query('Donor == "RB183_UT_Phase"').RB_IDO,
    Pred_all_M.query('Donor == "RB183_UT_Phase"').Predict_IDOA, 
    c='#e38a05',marker='^',label='RB183_Phase',s=200)
plt.scatter(
    Pred_all_M.query('Donor == "RB175_UT_Phase"').RB_IDO,
    Pred_all_M.query('Donor == "RB175_UT_Phase"').Predict_IDOA, 
    c='#98999c',marker='^',label='RB175_Phase',s=200)

X = Pred_all_M_array[:,1]
Y = Pred_all_M_array[:,0]
z = np.polyfit(X,Y, 1)
p = np.poly1d(z)

yhat = p(X)
ybar = np.sum(Y)/len(Y)
ssreg = np.sum((yhat-ybar)**2)
sstot = np.sum((Y-ybar)**2)
plt.plot(X,p(X),"r--")

plt.text( 8, 80, "y=%.2fx+%.2f"%(z[0],z[1]), fontsize=25,fontweight = 'bold')
plt.text( 8, 73, 'R^2=' "%.2f"%(ssreg/sstot), fontsize=25,fontweight = 'bold')

plt.xlabel("Measured IDO Activity\n(pg KYN/cell/day)", fontsize = 30);
plt.ylabel("Predict IDO Activity\n(pg KYN/cell/day)",fontsize = 30);
plt.legend(loc='best', fontsize = 15);
plt.tick_params(labelsize = 27) 
plt.xlim(5,90)
plt.ylim(5,90)
plt.legend(bbox_to_anchor=(1.35, 1),fontsize = 18, borderaxespad=0)
# %% alpha value--test error
# Define a list of alpha from 0 to 1
from sklearn.linear_model import Lasso
# %%
# range = np.arange(0,1,0.05)
# # Initialize lists to store test error and train size
# test_errors = []
# test_stdevs = []
# # Calculate the model scores for each train size
# cv_scores = []
# # Loop over train sizes and fit the model
# for i in range:

#     # Instantiate the random forest regressor with desired hyperparameters
#     Lasso_dp = Lasso(alpha=i)
#     Lasso_dp.fit(Xtrain_all, Ytrain_all)

#     # Append cross calidation and std to list for plot
#     cv_scores.append(Lasso_dp.score(Xtrain_all, Ytrain_all))
#     # Make predictions on the test data
#     Y_pred = Lasso_dp.predict(Xtest_all)
#     # Append the test error and train size to the lists
#     test_errors.append(mean_squared_error(Ytest_all, Y_pred))
#     test_stdevs.append(np.std(Y_pred))

# # %% Test Error w/std in shadow
# plt.figure(figsize=(8,6))
# # Create x-axis values
# x = np.arange(len(test_errors))

# # Plot the test error line
# plt.plot(range, test_errors, marker='o', label='Test Error')

# # Plot the shaded area for standard deviation
# plt.fill_between(range, np.array(test_errors) - np.array(test_stdevs),
#                  np.array(test_errors) + np.array(test_stdevs),
#                  alpha=0.3, label='Standard Deviation')
# plt.xlabel('Alpha',fontsize = 30)
# plt.ylabel('Test error (MSE)',fontsize = 30)
# plt.title('LASSO Test Error over hyperparameter n_estimator\n',fontsize=15)

# %%
Lasso_phase = Lasso (alpha=1)
Lasso_phase.fit(Xtrain_all, Ytrain_all)
Y_LASSO = pd.DataFrame(Lasso_phase.predict(Xtest_all))
# %%
Pred_LASSO = pd. concat([Y_LASSO,
                       Test_all,
                       ],
                        axis = 1,
                        ignore_index = bool
                        )
Pred_LASSO.columns = ["Predict_IDOA",
                    # 'Donor', 'level_1', 'RB_IDO', 'ImageNumber', 'ObjectNumber',
                    'Exp', 'RB_Donor', 'Donor', 'Treatment', 'RB_IDO', 'ImageNumber',
       'ObjectNumber', 'Cell_BoundingBoxArea', 'Cell_Center_X',
       'Cell_Center_Y', 'Cell_Compactness', 'Cell_Eccentricity',
       'Cell_EquivalentDiameter', 'Cell_Extent', 'Cell_FormFactor',
       'Cell_MaxFeretDiameter', 'Cell_MedianRadius', 'Cell_MinorAxisLength',
       'Cell_Orientation', 'Cell_Perimeter', 'Cell_Solidity',
       'DilateImage_IntegratedIntensityEdge',
       'DilateImage_IntegratedIntensity', 'DilateImage_MADIntensity',
       'DilateImage_MassDisplacement', 'DilateImage_MeanIntensityEdge',
       'DilateImage_StdIntensityEdge', 'Texture_Contrast',
       'Texture_Correlation', 'Texture_DifferenceEntropy', 'Texture_InfoMeas1',
       'Texture_InfoMeas2', 'Texture_SumAverage', 'Texture_SumEntropy',
       'Texture_Variance'
       ]
#%% #rfr single prediction
#colored as expriment batch
plt.figure(figsize=(10,7))
plt.scatter(
    Pred_LASSO.query('Donor == "RB179_UT_Phase"').RB_IDO,
    Pred_LASSO.query('Donor == "RB179_UT_Phase"').Predict_IDOA, 
    c='#cb8deb',marker='^',label='RB179_DilatePhase',s=50)
plt.scatter(
    Pred_LASSO.query('Donor == "RB48_UT_Phase"').RB_IDO,
    Pred_LASSO.query('Donor == "RB48_UT_Phase"').Predict_IDOA, 
    c='#8caff5',marker='^',label='RB48_DilatePhase', s=50)
plt.scatter(
    Pred_LASSO.query('Donor == "RB177_UT_Phase"').RB_IDO,
    Pred_LASSO.query('Donor == "RB177_UT_Phase"').Predict_IDOA, 
    c='#ed63ff',marker='^',label='RB177_DilatePhase',s=50)
plt.scatter(
    Pred_LASSO.query('Donor == "RB71_UT_Phase"').RB_IDO,
    Pred_LASSO.query('Donor == "RB71_UT_Phase"').Predict_IDOA, 
    c='#636999',marker='^',label='RB71_DilatePhase',s=50)
plt.scatter(
    Pred_LASSO.query('Donor == "RB183_UT_Phase"').RB_IDO,
    Pred_LASSO.query('Donor == "RB183_UT_Phase"').Predict_IDOA, 
    c='#e38a05',marker='^',label='RB183_DilatePhase',s=50)
plt.scatter(
    Pred_LASSO.query('Donor == "RB175_UT_Phase"').RB_IDO,
    Pred_LASSO.query('Donor == "RB175_UT_Phase"').Predict_IDOA, 
    c='#98999c',marker='^',label='RB175_DilatePhase',s=50)

plt.axline((10,10),slope=1,c='black',linestyle='--')
plt.xlabel("Measured IDO Activity\n(pg KYN/cell/day)", fontsize = 30);
plt.ylabel("Predict IDO Activity\n(pg KYN/cell/day)",fontsize = 30);
plt.legend(loc='best', fontsize = 15);
plt.tick_params(labelsize = 27) 
plt.legend(bbox_to_anchor=(1.3, 1),fontsize = 18, borderaxespad=0)
# plt.savefig('/Users/ruikejie/Desktop/Ph.D Program/Python/rfr_Prediction_IFN_Smooth.svg',dpi=300,bbox_inches='tight')


# %% rfr bulk prediction
import statistics
Pred_LASSO_M = Pred_LASSO.groupby('Donor').median()
Pred_LASSO_M_list = Pred_LASSO_M.values.tolist()
Pred_LASSO_M_array = Pred_LASSO_M.to_numpy()

plt.figure(figsize=(10,7))

plt.scatter(
    Pred_LASSO_M.query('Donor == "RB179_UT_Phase"').RB_IDO,
    Pred_LASSO_M.query('Donor == "RB179_UT_Phase"').Predict_IDOA, 
    c='#cb8deb',marker='^',label='RB179_Phase',s=200)
plt.scatter(
    Pred_LASSO_M.query('Donor == "RB48_UT_Phase"').RB_IDO,
    Pred_LASSO_M.query('Donor == "RB48_UT_Phase"').Predict_IDOA, 
    c='#8caff5',marker='^',label='RB48_Phase', s=200)
plt.scatter(
    Pred_LASSO_M.query('Donor == "RB177_UT_Phase"').RB_IDO,
    Pred_LASSO_M.query('Donor == "RB177_UT_Phase"').Predict_IDOA, 
    c='#ed63ff',marker='^',label='RB177_Phase',s=200)
plt.scatter(
    Pred_LASSO_M.query('Donor == "RB71_UT_Phase"').RB_IDO,
    Pred_LASSO_M.query('Donor == "RB71_UT_Phase"').Predict_IDOA, 
    c='#636999',marker='^',label='RB71_Phase',s=200)
plt.scatter(
    Pred_LASSO_M.query('Donor == "RB183_UT_Phase"').RB_IDO,
    Pred_LASSO_M.query('Donor == "RB183_UT_Phase"').Predict_IDOA, 
    c='#e38a05',marker='^',label='RB183_Phase',s=200)
plt.scatter(
    Pred_LASSO_M.query('Donor == "RB175_UT_Phase"').RB_IDO,
    Pred_LASSO_M.query('Donor == "RB175_UT_Phase"').Predict_IDOA, 
    c='#98999c',marker='^',label='RB175_Phase',s=200)

X = Pred_LASSO_M_array[:,1]
Y = Pred_LASSO_M_array[:,0]
z = np.polyfit(X,Y, 1)
p = np.poly1d(z)

yhat = p(X)
ybar = np.sum(Y)/len(Y)
ssreg = np.sum((yhat-ybar)**2)
sstot = np.sum((Y-ybar)**2)
plt.plot(X,p(X),"r--")


plt.text( 8, 80, "y=%.2fx+%.2f"%(z[0],z[1]), fontsize=25,fontweight = 'bold')
plt.text( 8, 73, 'R^2=' "%.2f"%(ssreg/sstot), fontsize=25,fontweight = 'bold')
plt.xlim(5,90)
plt.ylim(5,90)
plt.xlabel("Measured IDO Activity\n(pg KYN/cell/day)", fontsize = 30);
plt.ylabel("Predict IDO Activity\n(pg KYN/cell/day)",fontsize = 30);
plt.legend(loc='best', fontsize = 15);
plt.tick_params(labelsize = 27) 
plt.legend(bbox_to_anchor=(1.35, 1),fontsize = 18, borderaxespad=0)

# %% grid search for best SVR model
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
# param_grid = {'C': [0.1,5,10,20],
#               'epsilon': [0.1, 0.5, 1.0],
#               'kernel': ['linear', 'rbf', 'poly']}
# svr = SVR()
# grid_search = GridSearchCV(svr, param_grid, cv=5)
# grid_search.fit(Xtrain_all, Ytrain_all)
# best_svr_model = grid_search.best_estimator_
# best_svr_params = grid_search.best_params_
# %%
svr_phase = SVR(kernel='rbf', C=20, epsilon=1)
svr_phase.fit(Xtrain_all, Ytrain_all)
Y_SVR = pd.DataFrame(svr_phase.predict(Xtest_all))
# %%
Pred_SVR = pd. concat([Y_SVR,
                       Test_all,
                       ],
                        axis = 1,
                        ignore_index = bool
                        )
Pred_SVR.columns = ["Predict_IDOA",
                    # 'Donor', 'level_1', 'RB_IDO', 'ImageNumber', 'ObjectNumber',
                    'Exp', 'RB_Donor', 'Donor', 'Treatment', 'RB_IDO', 'ImageNumber',
       'ObjectNumber', 'Cell_BoundingBoxArea', 'Cell_Center_X',
       'Cell_Center_Y', 'Cell_Compactness', 'Cell_Eccentricity',
       'Cell_EquivalentDiameter', 'Cell_Extent', 'Cell_FormFactor',
       'Cell_MaxFeretDiameter', 'Cell_MedianRadius', 'Cell_MinorAxisLength',
       'Cell_Orientation', 'Cell_Perimeter', 'Cell_Solidity',
       'DilateImage_IntegratedIntensityEdge',
       'DilateImage_IntegratedIntensity', 'DilateImage_MADIntensity',
       'DilateImage_MassDisplacement', 'DilateImage_MeanIntensityEdge',
       'DilateImage_StdIntensityEdge', 'Texture_Contrast',
       'Texture_Correlation', 'Texture_DifferenceEntropy', 'Texture_InfoMeas1',
       'Texture_InfoMeas2', 'Texture_SumAverage', 'Texture_SumEntropy',
       'Texture_Variance' 
    ]
#%% #rfr single prediction
#colored as expriment batch
plt.figure(figsize=(10,7))
plt.scatter(
    Pred_SVR.query('Donor == "RB179_UT_Phase"').RB_IDO,
    Pred_SVR.query('Donor == "RB179_UT_Phase"').Predict_IDOA, 
    c='#cb8deb',marker='^',label='RB179_DilatePhase',s=50)
plt.scatter(
    Pred_SVR.query('Donor == "RB48_UT_Phase"').RB_IDO,
    Pred_SVR.query('Donor == "RB48_UT_Phase"').Predict_IDOA, 
    c='#8caff5',marker='^',label='RB48_DilatePhase', s=50)
plt.scatter(
    Pred_SVR.query('Donor == "RB177_UT_Phase"').RB_IDO,
    Pred_SVR.query('Donor == "RB177_UT_Phase"').Predict_IDOA, 
    c='#ed63ff',marker='^',label='RB177_DilatePhase',s=50)
plt.scatter(
    Pred_SVR.query('Donor == "RB71_UT_Phase"').RB_IDO,
    Pred_SVR.query('Donor == "RB71_UT_Phase"').Predict_IDOA, 
    c='#636999',marker='^',label='RB71_DilatePhase',s=50)
plt.scatter(
    Pred_SVR.query('Donor == "RB183_UT_Phase"').RB_IDO,
    Pred_SVR.query('Donor == "RB183_UT_Phase"').Predict_IDOA, 
    c='#e38a05',marker='^',label='RB183_DilatePhase',s=50)
plt.scatter(
    Pred_SVR.query('Donor == "RB175_UT_Phase"').RB_IDO,
    Pred_SVR.query('Donor == "RB175_UT_Phase"').Predict_IDOA, 
    c='#98999c',marker='^',label='RB175_DilatePhase',s=50)

plt.axline((10,10),slope=1,c='black',linestyle='--')
plt.xlabel("Measured IDO Activity\n(pg KYN/cell/day)", fontsize = 30);
plt.ylabel("Predict IDO Activity\n(pg KYN/cell/day)",fontsize = 30);
plt.legend(loc='best', fontsize = 15);
plt.tick_params(labelsize = 27) 
plt.legend(bbox_to_anchor=(1.3, 1),fontsize = 18, borderaxespad=0)
# plt.savefig('/Users/ruikejie/Desktop/Ph.D Program/Python/rfr_Prediction_IFN_Smooth.svg',dpi=300,bbox_inches='tight')


# %% rfr bulk prediction
import statistics
Pred_SVR_M = Pred_SVR.groupby('Donor').median()
Pred_SVR_M_list = Pred_SVR_M.values.tolist()
Pred_SVR_M_array = Pred_SVR_M.to_numpy()

plt.figure(figsize=(10,7))

plt.scatter(
    Pred_SVR_M.query('Donor == "RB179_UT_Phase"').RB_IDO,
    Pred_SVR_M.query('Donor == "RB179_UT_Phase"').Predict_IDOA, 
    c='#cb8deb',marker='^',label='RB179_Phase',s=200)
plt.scatter(
    Pred_SVR_M.query('Donor == "RB48_UT_Phase"').RB_IDO,
    Pred_SVR_M.query('Donor == "RB48_UT_Phase"').Predict_IDOA, 
    c='#8caff5',marker='^',label='RB48_Phase', s=200)
plt.scatter(
    Pred_SVR_M.query('Donor == "RB177_UT_Phase"').RB_IDO,
    Pred_SVR_M.query('Donor == "RB177_UT_Phase"').Predict_IDOA, 
    c='#ed63ff',marker='^',label='RB177_Phase',s=200)
plt.scatter(
    Pred_SVR_M.query('Donor == "RB71_UT_Phase"').RB_IDO,
    Pred_SVR_M.query('Donor == "RB71_UT_Phase"').Predict_IDOA, 
    c='#636999',marker='^',label='RB71_Phase',s=200)
plt.scatter(
    Pred_SVR_M.query('Donor == "RB183_UT_Phase"').RB_IDO,
    Pred_SVR_M.query('Donor == "RB183_UT_Phase"').Predict_IDOA, 
    c='#e38a05',marker='^',label='RB183_Phase',s=200)
plt.scatter(
    Pred_SVR_M.query('Donor == "RB175_UT_Phase"').RB_IDO,
    Pred_SVR_M.query('Donor == "RB175_UT_Phase"').Predict_IDOA, 
    c='#98999c',marker='^',label='RB175_Phase',s=200)

X = Pred_SVR_M_array[:,1]
Y = Pred_SVR_M_array[:,0]
z = np.polyfit(X,Y, 1)
p = np.poly1d(z)

yhat = p(X)
ybar = np.sum(Y)/len(Y)
ssreg = np.sum((yhat-ybar)**2)
sstot = np.sum((Y-ybar)**2)
plt.plot(X,p(X),"r--")


plt.text( 8, 80, "y=%.2fx+%.2f"%(z[0],z[1]), fontsize=25,fontweight = 'bold')
plt.text( 8, 73, 'R^2=' "%.2f"%(ssreg/sstot), fontsize=25,fontweight = 'bold')
plt.xlim(5,90)
plt.ylim(5,90)
plt.xlabel("Measured IDO Activity\n(pg KYN/cell/day)", fontsize = 30);
plt.ylabel("Predict IDO Activity\n(pg KYN/cell/day)",fontsize = 30);
plt.legend(loc='best', fontsize = 15);
plt.tick_params(labelsize = 27) 
plt.legend(bbox_to_anchor=(1.35, 1),fontsize = 18, borderaxespad=0)

# %%
svrL_phase = SVR(kernel='linear', C=100, epsilon=1)
svrL_phase.fit(Xtrain_all, Ytrain_all)
Y_SVRL = pd.DataFrame(svrL_phase.predict(Xtest_all))
# %%
Pred_SVRL = pd. concat([Y_SVRL,
                       Test_all,
                       ],
                        axis = 1,
                        ignore_index = bool
                        )
Pred_SVRL.columns = ["Predict_IDOA",
   'Exp', 'RB_Donor', 'Donor', 'Treatment', 'RB_IDO', 'ImageNumber',
       'ObjectNumber', 'Cell_BoundingBoxArea', 'Cell_Center_X',
       'Cell_Center_Y', 'Cell_Compactness', 'Cell_Eccentricity',
       'Cell_EquivalentDiameter', 'Cell_Extent', 'Cell_FormFactor',
       'Cell_MaxFeretDiameter', 'Cell_MedianRadius', 'Cell_MinorAxisLength',
       'Cell_Orientation', 'Cell_Perimeter', 'Cell_Solidity',
       'DilateImage_IntegratedIntensityEdge',
       'DilateImage_IntegratedIntensity', 'DilateImage_MADIntensity',
       'DilateImage_MassDisplacement', 'DilateImage_MeanIntensityEdge',
       'DilateImage_StdIntensityEdge', 'Texture_Contrast',
       'Texture_Correlation', 'Texture_DifferenceEntropy', 'Texture_InfoMeas1',
       'Texture_InfoMeas2', 'Texture_SumAverage', 'Texture_SumEntropy',
       'Texture_Variance'
     ]
#%% #rfr single prediction
#colored as expriment batch
plt.figure(figsize=(10,7))
plt.scatter(
    Pred_SVRL.query('Donor == "RB179_UT_Phase"').RB_IDO,
    Pred_SVRL.query('Donor == "RB179_UT_Phase"').Predict_IDOA, 
    c='#cb8deb',marker='^',label='RB179_DilatePhase',s=50)
plt.scatter(
    Pred_SVRL.query('Donor == "RB48_UT_Phase"').RB_IDO,
    Pred_SVRL.query('Donor == "RB48_UT_Phase"').Predict_IDOA, 
    c='#8caff5',marker='^',label='RB48_DilatePhase', s=50)
plt.scatter(
    Pred_SVRL.query('Donor == "RB177_UT_Phase"').RB_IDO,
    Pred_SVRL.query('Donor == "RB177_UT_Phase"').Predict_IDOA, 
    c='#ed63ff',marker='^',label='RB177_DilatePhase',s=50)
plt.scatter(
    Pred_SVRL.query('Donor == "RB71_UT_Phase"').RB_IDO,
    Pred_SVRL.query('Donor == "RB71_UT_Phase"').Predict_IDOA, 
    c='#636999',marker='^',label='RB71_DilatePhase',s=50)
plt.scatter(
    Pred_SVRL.query('Donor == "RB183_UT_Phase"').RB_IDO,
    Pred_SVRL.query('Donor == "RB183_UT_Phase"').Predict_IDOA, 
    c='#e38a05',marker='^',label='RB183_DilatePhase',s=50)
plt.scatter(
    Pred_SVRL.query('Donor == "RB175_UT_Phase"').RB_IDO,
    Pred_SVRL.query('Donor == "RB175_UT_Phase"').Predict_IDOA, 
    c='#98999c',marker='^',label='RB175_DilatePhase',s=50)

plt.axline((10,10),slope=1,c='black',linestyle='--')
plt.xlabel("Measured IDO Activity\n(pg KYN/cell/day)", fontsize = 30);
plt.ylabel("Predict IDO Activity\n(pg KYN/cell/day)",fontsize = 30);
plt.legend(loc='best', fontsize = 15);
plt.tick_params(labelsize = 27) 
plt.legend(bbox_to_anchor=(1.3, 1),fontsize = 18, borderaxespad=0)
# plt.savefig('/Users/ruikejie/Desktop/Ph.D Program/Python/rfr_Prediction_IFN_Smooth.svg',dpi=300,bbox_inches='tight')


# %% rfr bulk prediction
import statistics
Pred_SVRL_M = Pred_SVRL.groupby('Donor').median()
Pred_SVRL_M_list = Pred_SVRL_M.values.tolist()
Pred_SVRL_M_array = Pred_SVRL_M.to_numpy()

plt.figure(figsize=(10,7))

plt.scatter(
    Pred_SVRL_M.query('Donor == "RB179_UT_Phase"').RB_IDO,
    Pred_SVRL_M.query('Donor == "RB179_UT_Phase"').Predict_IDOA, 
    c='#cb8deb',marker='^',label='RB179_Phase',s=200)
plt.scatter(
    Pred_SVRL_M.query('Donor == "RB48_UT_Phase"').RB_IDO,
    Pred_SVRL_M.query('Donor == "RB48_UT_Phase"').Predict_IDOA, 
    c='#8caff5',marker='^',label='RB48_Phase', s=200)
plt.scatter(
    Pred_SVRL_M.query('Donor == "RB177_UT_Phase"').RB_IDO,
    Pred_SVRL_M.query('Donor == "RB177_UT_Phase"').Predict_IDOA, 
    c='#ed63ff',marker='^',label='RB177_Phase',s=200)
plt.scatter(
    Pred_SVRL_M.query('Donor == "RB71_UT_Phase"').RB_IDO,
    Pred_SVRL_M.query('Donor == "RB71_UT_Phase"').Predict_IDOA, 
    c='#636999',marker='^',label='RB71_Phase',s=200)
plt.scatter(
    Pred_SVRL_M.query('Donor == "RB183_UT_Phase"').RB_IDO,
    Pred_SVRL_M.query('Donor == "RB183_UT_Phase"').Predict_IDOA, 
    c='#e38a05',marker='^',label='RB183_Phase',s=200)
plt.scatter(
    Pred_SVRL_M.query('Donor == "RB175_UT_Phase"').RB_IDO,
    Pred_SVRL_M.query('Donor == "RB175_UT_Phase"').Predict_IDOA, 
    c='#98999c',marker='^',label='RB175_Phase',s=200)

X = Pred_SVRL_M_array[:,1]
Y = Pred_SVRL_M_array[:,0]
z = np.polyfit(X,Y, 1)
p = np.poly1d(z)

yhat = p(X)
ybar = np.sum(Y)/len(Y)
ssreg = np.sum((yhat-ybar)**2)
sstot = np.sum((Y-ybar)**2)
plt.plot(X,p(X),"r--")


plt.text( 8, 80, "y=%.2fx+%.2f"%(z[0],z[1]), fontsize=25,fontweight = 'bold')
plt.text( 8, 73, 'R^2=' "%.2f"%(ssreg/sstot), fontsize=25,fontweight = 'bold')
plt.xlim(5,90)
plt.ylim(5,90)
plt.xlabel("Measured IDO Activity\n(pg KYN/cell/day)", fontsize = 30);
plt.ylabel("Predict IDO Activity\n(pg KYN/cell/day)",fontsize = 30);
plt.legend(loc='best', fontsize = 15);
plt.tick_params(labelsize = 27) 
plt.legend(bbox_to_anchor=(1.35, 1),fontsize = 18, borderaxespad=0)

# %%
from sklearn.ensemble import GradientBoostingRegressor
gb_regressor = GradientBoostingRegressor(n_estimators=300, learning_rate=0.1)
gb_regressor.fit(Xtrain_all, Ytrain_all)
Ypred_gb = pd.DataFrame(gb_regressor.predict(Xtest_all))

# %%
Pred_GB = pd. concat([Ypred_gb,
                       Test_all,
                       ],
                        axis = 1,
                        ignore_index = bool
                        )
Pred_GB.columns = ["Predict_IDOA",
                   'Exp', 'RB_Donor', 'Donor', 'Treatment', 'RB_IDO', 'ImageNumber',
       'ObjectNumber', 'Cell_BoundingBoxArea', 'Cell_Center_X',
       'Cell_Center_Y', 'Cell_Compactness', 'Cell_Eccentricity',
       'Cell_EquivalentDiameter', 'Cell_Extent', 'Cell_FormFactor',
       'Cell_MaxFeretDiameter', 'Cell_MedianRadius', 'Cell_MinorAxisLength',
       'Cell_Orientation', 'Cell_Perimeter', 'Cell_Solidity',
       'DilateImage_IntegratedIntensityEdge',
       'DilateImage_IntegratedIntensity', 'DilateImage_MADIntensity',
       'DilateImage_MassDisplacement', 'DilateImage_MeanIntensityEdge',
       'DilateImage_StdIntensityEdge', 'Texture_Contrast',
       'Texture_Correlation', 'Texture_DifferenceEntropy', 'Texture_InfoMeas1',
       'Texture_InfoMeas2', 'Texture_SumAverage', 'Texture_SumEntropy',
       'Texture_Variance'
     ]
#%% #rfr single prediction
#colored as expriment batch
plt.figure(figsize=(10,7))
plt.scatter(
    Pred_GB.query('Donor == "RB179_UT_Phase"').RB_IDO,
    Pred_GB.query('Donor == "RB179_UT_Phase"').Predict_IDOA, 
    c='#cb8deb',marker='^',label='RB179_DilatePhase',s=50)
plt.scatter(
    Pred_GB.query('Donor == "RB48_UT_Phase"').RB_IDO,
    Pred_GB.query('Donor == "RB48_UT_Phase"').Predict_IDOA, 
    c='#8caff5',marker='^',label='RB48_DilatePhase', s=50)
plt.scatter(
    Pred_GB.query('Donor == "RB177_UT_Phase"').RB_IDO,
    Pred_GB.query('Donor == "RB177_UT_Phase"').Predict_IDOA, 
    c='#ed63ff',marker='^',label='RB177_DilatePhase',s=50)
plt.scatter(
    Pred_GB.query('Donor == "RB71_UT_Phase"').RB_IDO,
    Pred_GB.query('Donor == "RB71_UT_Phase"').Predict_IDOA, 
    c='#636999',marker='^',label='RB71_DilatePhase',s=50)
plt.scatter(
    Pred_GB.query('Donor == "RB183_UT_Phase"').RB_IDO,
    Pred_GB.query('Donor == "RB183_UT_Phase"').Predict_IDOA, 
    c='#e38a05',marker='^',label='RB183_DilatePhase',s=50)
plt.scatter(
    Pred_GB.query('Donor == "RB175_UT_Phase"').RB_IDO,
    Pred_GB.query('Donor == "RB175_UT_Phase"').Predict_IDOA, 
    c='#98999c',marker='^',label='RB175_DilatePhase',s=50)

plt.axline((10,10),slope=1,c='black',linestyle='--')
plt.xlabel("Measured IDO Activity\n(pg KYN/cell/day)", fontsize = 30);
plt.ylabel("Predict IDO Activity\n(pg KYN/cell/day)",fontsize = 30);
plt.legend(loc='best', fontsize = 15);
plt.tick_params(labelsize = 27) 
plt.legend(bbox_to_anchor=(1.3, 1),fontsize = 18, borderaxespad=0)
# plt.savefig('/Users/ruikejie/Desktop/Ph.D Program/Python/rfr_Prediction_IFN_Smooth.svg',dpi=300,bbox_inches='tight')


# %% rfr bulk prediction
import statistics
Pred_GB_M = Pred_GB.groupby('Donor').median()
Pred_GB_M_list = Pred_GB_M.values.tolist()
Pred_GB_M_array = Pred_GB_M.to_numpy()

plt.figure(figsize=(10,7))

plt.scatter(
    Pred_GB_M.query('Donor == "RB179_UT_Phase"').RB_IDO,
    Pred_GB_M.query('Donor == "RB179_UT_Phase"').Predict_IDOA, 
    c='#cb8deb',marker='^',label='RB179_Phase',s=200)
plt.scatter(
    Pred_GB_M.query('Donor == "RB48_UT_Phase"').RB_IDO,
    Pred_GB_M.query('Donor == "RB48_UT_Phase"').Predict_IDOA, 
    c='#8caff5',marker='^',label='RB48_Phase', s=200)
plt.scatter(
    Pred_GB_M.query('Donor == "RB177_UT_Phase"').RB_IDO,
    Pred_GB_M.query('Donor == "RB177_UT_Phase"').Predict_IDOA, 
    c='#ed63ff',marker='^',label='RB177_Phase',s=200)
plt.scatter(
    Pred_GB_M.query('Donor == "RB71_UT_Phase"').RB_IDO,
    Pred_GB_M.query('Donor == "RB71_UT_Phase"').Predict_IDOA, 
    c='#636999',marker='^',label='RB71_Phase',s=200)
plt.scatter(
    Pred_GB_M.query('Donor == "RB183_UT_Phase"').RB_IDO,
    Pred_GB_M.query('Donor == "RB183_UT_Phase"').Predict_IDOA, 
    c='#e38a05',marker='^',label='RB183_Phase',s=200)
plt.scatter(
    Pred_GB_M.query('Donor == "RB175_UT_Phase"').RB_IDO,
    Pred_GB_M.query('Donor == "RB175_UT_Phase"').Predict_IDOA, 
    c='#98999c',marker='^',label='RB175_Phase',s=200)

X = Pred_GB_M_array[:,1]
Y = Pred_GB_M_array[:,0]
z = np.polyfit(X,Y, 1)
p = np.poly1d(z)

yhat = p(X)
ybar = np.sum(Y)/len(Y)
ssreg = np.sum((yhat-ybar)**2)
sstot = np.sum((Y-ybar)**2)
plt.plot(X,p(X),"r--")


plt.text( 8, 80, "y=%.2fx+%.2f"%(z[0],z[1]), fontsize=25,fontweight = 'bold')
plt.text( 8, 73, 'R^2=' "%.2f"%(ssreg/sstot), fontsize=25,fontweight = 'bold')
plt.xlim(5,90)
plt.ylim(5,90)
plt.xlabel("Measured IDO Activity\n(pg KYN/cell/day)", fontsize = 30);
plt.ylabel("Predict IDO Activity\n(pg KYN/cell/day)",fontsize = 30);
plt.legend(loc='best', fontsize = 15);
plt.tick_params(labelsize = 27) 
plt.legend(bbox_to_anchor=(1.35, 1),fontsize = 18, borderaxespad=0)

# %%
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
# Define the parameter grid
param_grid = {
    'hidden_layer_sizes': [(100,), (100, 50), (50,)],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.001, 0.01],
}

# Create the Neural Network Regressor
nn_reg = MLPRegressor()

# Perform grid search
grid_search = GridSearchCV(nn_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(Xtrain_all, Ytrain_all)

# Get the best hyperparameters and model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Use the best model to make predictions
Ypred_NN = best_model.predict(Xtest_all)

# Calculate mean squared error and R2 score
mse = mean_squared_error(Ytest_all, Ypred_NN)
r2 = r2_score(Ytest_all, Ypred_NN)

# Print the best hyperparameters and performance metrics
print("Best Hyperparameters:", best_params)
print("Mean Squared Error:", mse)
print("R2 Score:", r2)
# %%
Pred_NN = pd. concat([pd.DataFrame(Ypred_NN),
                       Test_all,
                       ],
                        axis = 1,
                        ignore_index = bool
                        )
Pred_NN.columns = ["Predict_IDOA",
    'Exp', 'RB_Donor', 'Donor', 'Treatment', 'RB_IDO', 'ImageNumber',
       'ObjectNumber', 'Cell_BoundingBoxArea', 'Cell_Center_X',
       'Cell_Center_Y', 'Cell_Compactness', 'Cell_Eccentricity',
       'Cell_EquivalentDiameter', 'Cell_Extent', 'Cell_FormFactor',
       'Cell_MaxFeretDiameter', 'Cell_MedianRadius', 'Cell_MinorAxisLength',
       'Cell_Orientation', 'Cell_Perimeter', 'Cell_Solidity',
       'DilateImage_IntegratedIntensityEdge',
       'DilateImage_IntegratedIntensity', 'DilateImage_MADIntensity',
       'DilateImage_MassDisplacement', 'DilateImage_MeanIntensityEdge',
       'DilateImage_StdIntensityEdge', 'Texture_Contrast',
       'Texture_Correlation', 'Texture_DifferenceEntropy', 'Texture_InfoMeas1',
       'Texture_InfoMeas2', 'Texture_SumAverage', 'Texture_SumEntropy',
       'Texture_Variance'
     ]

#%% #rfr single prediction
#colored as expriment batch
plt.figure(figsize=(10,7))
plt.scatter(
    Pred_NN.query('Donor == "RB179_UT_Phase"').RB_IDO,
    Pred_NN.query('Donor == "RB179_UT_Phase"').Predict_IDOA, 
    c='#cb8deb',marker='^',label='RB179_DilatePhase',s=50)
plt.scatter(
    Pred_NN.query('Donor == "RB48_UT_Phase"').RB_IDO,
    Pred_NN.query('Donor == "RB48_UT_Phase"').Predict_IDOA, 
    c='#8caff5',marker='^',label='RB48_DilatePhase', s=50)
plt.scatter(
    Pred_NN.query('Donor == "RB177_UT_Phase"').RB_IDO,
    Pred_NN.query('Donor == "RB177_UT_Phase"').Predict_IDOA, 
    c='#ed63ff',marker='^',label='RB177_DilatePhase',s=50)
plt.scatter(
    Pred_NN.query('Donor == "RB71_UT_Phase"').RB_IDO,
    Pred_NN.query('Donor == "RB71_UT_Phase"').Predict_IDOA, 
    c='#636999',marker='^',label='RB71_DilatePhase',s=50)
plt.scatter(
    Pred_NN.query('Donor == "RB183_UT_Phase"').RB_IDO,
    Pred_NN.query('Donor == "RB183_UT_Phase"').Predict_IDOA, 
    c='#e38a05',marker='^',label='RB183_DilatePhase',s=50)
plt.scatter(
    Pred_NN.query('Donor == "RB175_UT_Phase"').RB_IDO,
    Pred_NN.query('Donor == "RB175_UT_Phase"').Predict_IDOA, 
    c='#98999c',marker='^',label='RB175_DilatePhase',s=50)

plt.axline((10,10),slope=1,c='black',linestyle='--')
plt.xlabel("Measured IDO Activity\n(pg KYN/cell/day)", fontsize = 30);
plt.ylabel("Predict IDO Activity\n(pg KYN/cell/day)",fontsize = 30);
plt.legend(loc='best', fontsize = 15);
plt.tick_params(labelsize = 27) 
plt.legend(bbox_to_anchor=(1.3, 1),fontsize = 18, borderaxespad=0)
# plt.savefig('/Users/ruikejie/Desktop/Ph.D Program/Python/rfr_Prediction_IFN_Smooth.svg',dpi=300,bbox_inches='tight')


# %% rfr bulk prediction
import statistics
Pred_NN_M = Pred_NN.groupby('Donor').median()
Pred_NN_M_list = Pred_NN_M.values.tolist()
Pred_NN_M_array = Pred_NN_M.to_numpy()

plt.figure(figsize=(10,7))

plt.scatter(
    Pred_NN_M.query('Donor == "RB179_UT_Phase"').RB_IDO,
    Pred_NN_M.query('Donor == "RB179_UT_Phase"').Predict_IDOA, 
    c='#cb8deb',marker='^',label='RB179_Phase',s=200)
plt.scatter(
    Pred_NN_M.query('Donor == "RB48_UT_Phase"').RB_IDO,
    Pred_NN_M.query('Donor == "RB48_UT_Phase"').Predict_IDOA, 
    c='#8caff5',marker='^',label='RB48_Phase', s=200)
plt.scatter(
    Pred_NN_M.query('Donor == "RB177_UT_Phase"').RB_IDO,
    Pred_NN_M.query('Donor == "RB177_UT_Phase"').Predict_IDOA, 
    c='#ed63ff',marker='^',label='RB177_Phase',s=200)
plt.scatter(
    Pred_NN_M.query('Donor == "RB71_UT_Phase"').RB_IDO,
    Pred_NN_M.query('Donor == "RB71_UT_Phase"').Predict_IDOA, 
    c='#636999',marker='^',label='RB71_Phase',s=200)
plt.scatter(
    Pred_NN_M.query('Donor == "RB183_UT_Phase"').RB_IDO,
    Pred_NN_M.query('Donor == "RB183_UT_Phase"').Predict_IDOA, 
    c='#e38a05',marker='^',label='RB183_Phase',s=200)
plt.scatter(
    Pred_NN_M.query('Donor == "RB175_UT_Phase"').RB_IDO,
    Pred_NN_M.query('Donor == "RB175_UT_Phase"').Predict_IDOA, 
    c='#98999c',marker='^',label='RB175_Phase',s=200)

X = Pred_NN_M_array[:,1]
Y = Pred_NN_M_array[:,0]
z = np.polyfit(X,Y, 1)
p = np.poly1d(z)

yhat = p(X)
ybar = np.sum(Y)/len(Y)
ssreg = np.sum((yhat-ybar)**2)
sstot = np.sum((Y-ybar)**2)
plt.plot(X,p(X),"r--")


plt.text( 8, 80, "y=%.2fx+%.2f"%(z[0],z[1]), fontsize=25,fontweight = 'bold')
plt.text( 8, 73, 'R^2=' "%.2f"%(ssreg/sstot), fontsize=25,fontweight = 'bold')
plt.xlim(5,90)
plt.ylim(5,90)
plt.xlabel("Measured IDO Activity\n(pg KYN/cell/day)", fontsize = 30);
plt.ylabel("Predict IDO Activity\n(pg KYN/cell/day)",fontsize = 30);
plt.legend(loc='best', fontsize = 15);
plt.tick_params(labelsize = 27) 
plt.legend(bbox_to_anchor=(1.35, 1),fontsize = 18, borderaxespad=0)

# %%
RFR_error = mean_squared_error(Ytest_all, Ypred_rfr)
GB_error = mean_squared_error(Ytest_all, Ypred_gb)
SVR_error = mean_squared_error(Ytest_all, Y_SVR)
NN_error = mean_squared_error(Ytest_all, Ypred_NN)
LASSO_error = mean_squared_error(Ytest_all, Y_LASSO)
SVRL_error = mean_squared_error(Ytest_all, Y_SVRL)

# %%
from scipy import stats
# %%
error_175 = mean_squared_error(Pred_all.query('RB_Donor == "RB175_UT"').RB_IDO, Pred_all.query('RB_Donor == "RB175_UT"').Predict_IDOA)
error_177  = mean_squared_error(Pred_all.query('RB_Donor == "RB177_UT"').RB_IDO, Pred_all.query('RB_Donor == "RB177_UT"').Predict_IDOA)
error_179 = mean_squared_error(Pred_all.query('RB_Donor == "RB179_UT"').RB_IDO, Pred_all.query('RB_Donor == "RB179_UT"').Predict_IDOA)
error_183 = mean_squared_error(Pred_all.query('RB_Donor == "RB183_UT"').RB_IDO, Pred_all.query('RB_Donor == "RB183_UT"').Predict_IDOA)
error_48 = mean_squared_error(Pred_all.query('RB_Donor == "RB48_UT"').RB_IDO, Pred_all.query('RB_Donor == "RB48_UT"').Predict_IDOA)
error_71 = mean_squared_error(Pred_all.query('RB_Donor == "RB71_UT"').RB_IDO, Pred_all.query('RB_Donor == "RB71_UT"').Predict_IDOA)
error_rfr = np.mean([error_175,error_177,error_179,error_183,error_48,error_71])
# mse_rfr = stats.sem([error_175,error_177,error_179,error_183,error_48,error_71])
std_rfr =np.std([error_175,error_177,error_179,error_183,error_48,error_71])
print(error_rfr)
print(std_rfr)
# %%
error_175 = mean_squared_error(Pred_GB.query('RB_Donor == "RB175_UT"').RB_IDO, Pred_GB.query('RB_Donor == "RB175_UT"').Predict_IDOA)
error_177  = mean_squared_error(Pred_GB.query('RB_Donor == "RB177_UT"').RB_IDO, Pred_GB.query('RB_Donor == "RB177_UT"').Predict_IDOA)
error_179 = mean_squared_error(Pred_GB.query('RB_Donor == "RB179_UT"').RB_IDO, Pred_GB.query('RB_Donor == "RB179_UT"').Predict_IDOA)
error_183 = mean_squared_error(Pred_GB.query('RB_Donor == "RB183_UT"').RB_IDO, Pred_GB.query('RB_Donor == "RB183_UT"').Predict_IDOA)
error_48 = mean_squared_error(Pred_GB.query('RB_Donor == "RB48_UT"').RB_IDO, Pred_GB.query('RB_Donor == "RB48_UT"').Predict_IDOA)
error_71 = mean_squared_error(Pred_GB.query('RB_Donor == "RB71_UT"').RB_IDO, Pred_GB.query('RB_Donor == "RB71_UT"').Predict_IDOA)
error_gb = np.mean([error_175,error_177,error_179,error_183,error_48,error_71])
std_gb =np.std([error_175,error_177,error_179,error_183,error_48,error_71])
print(error_gb)
print(std_gb)
# %%
error_175 = mean_squared_error(Pred_LASSO.query('RB_Donor == "RB175_UT"').RB_IDO, Pred_LASSO.query('RB_Donor == "RB175_UT"').Predict_IDOA)
error_177  = mean_squared_error(Pred_LASSO.query('RB_Donor == "RB177_UT"').RB_IDO, Pred_LASSO.query('RB_Donor == "RB177_UT"').Predict_IDOA)
error_179 = mean_squared_error(Pred_LASSO.query('RB_Donor == "RB179_UT"').RB_IDO, Pred_LASSO.query('RB_Donor == "RB179_UT"').Predict_IDOA)
error_183 = mean_squared_error(Pred_LASSO.query('RB_Donor == "RB183_UT"').RB_IDO, Pred_LASSO.query('RB_Donor == "RB183_UT"').Predict_IDOA)
error_48 = mean_squared_error(Pred_LASSO.query('RB_Donor == "RB48_UT"').RB_IDO, Pred_LASSO.query('RB_Donor == "RB48_UT"').Predict_IDOA)
error_71 = mean_squared_error(Pred_LASSO.query('RB_Donor == "RB71_UT"').RB_IDO, Pred_LASSO.query('RB_Donor == "RB71_UT"').Predict_IDOA)
error_lasso = np.mean([error_175,error_177,error_179,error_183,error_48,error_71])
std_lasso =np.std([error_175,error_177,error_179,error_183,error_48,error_71])
print(error_lasso)
print(std_lasso)
# %%
error_175 = mean_squared_error(Pred_NN.query('RB_Donor == "RB175_UT"').RB_IDO, Pred_NN.query('RB_Donor == "RB175_UT"').Predict_IDOA)
error_177  = mean_squared_error(Pred_NN.query('RB_Donor == "RB177_UT"').RB_IDO, Pred_NN.query('RB_Donor == "RB177_UT"').Predict_IDOA)
error_179 = mean_squared_error(Pred_NN.query('RB_Donor == "RB179_UT"').RB_IDO, Pred_NN.query('RB_Donor == "RB179_UT"').Predict_IDOA)
error_183 = mean_squared_error(Pred_NN.query('RB_Donor == "RB183_UT"').RB_IDO, Pred_NN.query('RB_Donor == "RB183_UT"').Predict_IDOA)
error_48 = mean_squared_error(Pred_NN.query('RB_Donor == "RB48_UT"').RB_IDO, Pred_NN.query('RB_Donor == "RB48_UT"').Predict_IDOA)
error_71 = mean_squared_error(Pred_NN.query('RB_Donor == "RB71_UT"').RB_IDO, Pred_NN.query('RB_Donor == "RB71_UT"').Predict_IDOA)
error_nn = np.mean([error_175,error_177,error_179,error_183,error_48,error_71])
std_nn =np.std([error_175,error_177,error_179,error_183,error_48,error_71])
print(error_nn)
print(std_nn)
# %%
error_175 = mean_squared_error(Pred_SVR.query('RB_Donor == "RB175_UT"').RB_IDO, Pred_SVR.query('RB_Donor == "RB175_UT"').Predict_IDOA)
error_177  = mean_squared_error(Pred_SVR.query('RB_Donor == "RB177_UT"').RB_IDO, Pred_SVR.query('RB_Donor == "RB177_UT"').Predict_IDOA)
error_179 = mean_squared_error(Pred_SVR.query('RB_Donor == "RB179_UT"').RB_IDO, Pred_SVR.query('RB_Donor == "RB179_UT"').Predict_IDOA)
error_183 = mean_squared_error(Pred_SVR.query('RB_Donor == "RB183_UT"').RB_IDO, Pred_SVR.query('RB_Donor == "RB183_UT"').Predict_IDOA)
error_48 = mean_squared_error(Pred_SVR.query('RB_Donor == "RB48_UT"').RB_IDO, Pred_SVR.query('RB_Donor == "RB48_UT"').Predict_IDOA)
error_71 = mean_squared_error(Pred_SVR.query('RB_Donor == "RB71_UT"').RB_IDO, Pred_SVR.query('RB_Donor == "RB71_UT"').Predict_IDOA)
error_svr = np.mean([error_175,error_177,error_179,error_183,error_48,error_71])
std_svr =np.std([error_175,error_177,error_179,error_183,error_48,error_71])
print(error_svr)
print(std_svr)
# %%
error_175 = mean_squared_error(Pred_SVRL.query('RB_Donor == "RB175_UT"').RB_IDO, Pred_SVRL.query('RB_Donor == "RB175_UT"').Predict_IDOA)
error_177  = mean_squared_error(Pred_SVRL.query('RB_Donor == "RB177_UT"').RB_IDO, Pred_SVRL.query('RB_Donor == "RB177_UT"').Predict_IDOA)
error_179 = mean_squared_error(Pred_SVRL.query('RB_Donor == "RB179_UT"').RB_IDO, Pred_SVRL.query('RB_Donor == "RB179_UT"').Predict_IDOA)
error_183 = mean_squared_error(Pred_SVRL.query('RB_Donor == "RB183_UT"').RB_IDO, Pred_SVRL.query('RB_Donor == "RB183_UT"').Predict_IDOA)
error_48 = mean_squared_error(Pred_SVRL.query('RB_Donor == "RB48_UT"').RB_IDO, Pred_SVRL.query('RB_Donor == "RB48_UT"').Predict_IDOA)
error_71 = mean_squared_error(Pred_SVRL.query('RB_Donor == "RB71_UT"').RB_IDO, Pred_SVRL.query('RB_Donor == "RB71_UT"').Predict_IDOA)
error_svrl = np.mean([error_175,error_177,error_179,error_183,error_48,error_71])
std_svrl =np.std([error_175,error_177,error_179,error_183,error_48,error_71])
print(error_svrl)
print(std_svrl)
# %%
import matplotlib.pyplot as plt
import math
# Values for the x-axis (categories)
categories = ['RFR', 'GBR', 'MLPR', 'SVR','SVRL', 'LASSO',]

# Values for the y-axis (values calculated in your code)
# values = [RFR_error, GB_error, NN_error, SVR_error, SVRL_error, LASSO_error, ]
values = [error_rfr, error_gb, error_nn, error_svr, error_svrl, error_lasso]
# std = [std_rfr/(math.sqrt(6)), std_gb/(math.sqrt(6)), std_nn/(math.sqrt(6)), std_svr/(math.sqrt(6)), std_svrl/(math.sqrt(6)), std_lasso/(math.sqrt(6))]
std = [std_rfr, std_gb, std_nn, std_svr, std_svrl, std_lasso]

# Plot the error bars first with lower zorder value
plt.figure(figsize=(10,7))
plt.errorbar(categories, values, yerr = std,fmt='none', capsize=5,color='black',ecolor='grey',zorder=0)
# Plot the bars on top of the error bars with higher zorder value
plt.bar(categories, values,color='black', zorder=2)

# Adding labels and title
plt.tick_params(labelsize = 27) 
plt.ylim(-1, 700)
plt.ylabel('MSE', fontsize=30)
plt.title('Mean-squared Test Errors -- Phase Images\n', fontsize=30)
# Displaying the chart
plt.show()

# %%
TE_Phase = {'Model': categories, 'TestError': values, 'Std': std}
TE_Phase = pd.DataFrame(TE_Phase)
TE_Phase

# %%
TE_Phase.to_excel('/Users/ruikejie/Desktop/OffLineWork/TE_phase.xlsx', index=False)
# %%
