# %% Import packages
import numpy as np
import pandas as pd
import umap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import seaborn as sns
# %% Read data, Media comparison
df = pd.read_excel('../Data/RevisionSup_Cell.xlsx', sheet_name="Confluency")
# %%Auto Scale to standardzie the features
Data = df.drop(['Condition', 'ImageNumber', 'ObjectNumber', ],axis=1)
Data_s = StandardScaler().fit_transform(Data)
df_s = pd.concat([df.Condition, df.ImageNumber, df.ObjectNumber, pd.DataFrame(Data_s)], axis=1)
df_s.columns=['Condition', 'ImageNumber', 'ObjectNumber', 'AreaShape_Area',
       'AreaShape_BoundingBoxArea', 'AreaShape_BoundingBoxMaximum_X',
       'AreaShape_BoundingBoxMaximum_Y', 'AreaShape_BoundingBoxMinimum_X',
       'AreaShape_BoundingBoxMinimum_Y', 'AreaShape_Center_X',
       'AreaShape_Center_Y', 'AreaShape_Compactness', 'AreaShape_ConvexArea',
       'AreaShape_Eccentricity', 'AreaShape_EquivalentDiameter',
       'AreaShape_Extent', 'AreaShape_FormFactor', 'AreaShape_MajorAxisLength',
       'AreaShape_MaxFeretDiameter', 'AreaShape_MaximumRadius',
       'AreaShape_MeanRadius', 'AreaShape_MedianRadius',
       'AreaShape_MinFeretDiameter', 'AreaShape_MinorAxisLength',
       'AreaShape_Orientation', 'AreaShape_Perimeter', 'AreaShape_Solidity',
       'Granularicy', 'Intensity_IntegratedIntensityEdge_RescaleIntensityCell',
       'Intensity_IntegratedIntensity_RescaleIntensityCell',
       'Intensity_LowerQuartileIntensity_RescaleIntensityCell',
       'Intensity_MADIntensity_RescaleIntensityCell',
       'Intensity_MassDisplacement_RescaleIntensityCell',
       'Intensity_MaxIntensityEdge_RescaleIntensityCell',
       'Intensity_MaxIntensity_RescaleIntensityCell',
       'Intensity_MeanIntensityEdge_RescaleIntensityCell',
       'Intensity_MeanIntensity_RescaleIntensityCell',
       'Intensity_MedianIntensity_RescaleIntensityCell',
       'Intensity_MinIntensityEdge_RescaleIntensityCell',
       'Intensity_MinIntensity_RescaleIntensityCell',
       'Intensity_StdIntensityEdge_RescaleIntensityCell',
       'Intensity_StdIntensity_RescaleIntensityCell',
       'Intensity_UpperQuartileIntensity_RescaleIntensityCell',
       'Texture_AngularSecondMoment_RescaleIntensityCell',
       'Texture_Contrast_RescaleIntensityCell',
       'Texture_Correlation_RescaleIntensityCell',
       'Texture_DifferenceEntropy_RescaleIntensityCell',
       'Texture_DifferenceVariance_RescaleIntensityCell',
       'Texture_Entropy_RescaleIntensityCell',
       'Texture_InfoMeas1_RescaleIntensityCell',
       'Texture_InfoMeas2_RescaleIntensityCell',
       'Texture_InverseDifferenceMoment_RescaleIntensityCell',
       'Texture_SumAverage_RescaleIntensityCell',
       'Texture_SumEntropy_RescaleIntensityCell',
       'Texture_SumVariance_RescaleIntensityCell',
       'Texture_Variance_RescaleIntensityCell']
# %% Create balacned data
df_balance = df_s.groupby('Condition', group_keys=False).apply(lambda x: x.sample(n=260, random_state=42)).reset_index(drop=True)
# %%Smooth data
df_RM = df_balance.groupby('Condition').rolling(center=False,window=1,min_periods=1).mean().reset_index()
df_RM1 = df_RM.drop(['Condition', 'level_1', 'ImageNumber', 'ObjectNumber'],axis=1)
# %%#Perform PCA for reducing dimentionality
pca = PCA(n_components=2)
principal_components = pca.fit_transform(df_RM1)
explained_var = pca.explained_variance_ratio_ * 100  # in %

pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df['Condition'] = df_RM.Condition

plt.figure(figsize=(7,7))
colors = ["#5e316e", "#c48cc9"]
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Condition', s=60, palette = sns.color_palette(colors))

plt.xlabel(f'PC1 ({explained_var[0]:.1f}% Variance)', fontsize=30)
plt.ylabel(f'PC2 ({explained_var[1]:.1f}% Variance)', fontsize=30)
plt.tick_params(labelsize = 25) 
plt.title('PCA of DPC Features: Confluency\n', fontsize=20)
legend=plt.legend(fontsize=25,)
# Resize legend marker points
for handle in legend.legendHandles:
    handle._sizes = [250]  
plt.tight_layout()
plt.savefig('../Plot/RevisionSup/RB183_ConluencyComparison_DPCFeatures.svg', format="svg",dpi=300,bbox_inches='tight') 
# %% PC1 Feature Importance
PCLoading = pd.DataFrame(data=pca.components_.T, columns=['PC1', 'PC2'],index=df_RM1.columns)
PCLoading['Features'] = PCLoading.index.astype(str)
# %%
PC1Loading = PCLoading.sort_values(['PC1']).reset_index(drop=True)
plt.figure(figsize=(6,16))

ax = sns.barplot(x='PC1',y='Features',data=PC1Loading, orient = 'h',palette = "Wistia")

ax.set_xlabel('PC1 Loading', fontsize = 30)
ax.set_ylabel(" ", fontsize = 30)
ax.tick_params(labelsize = 10)
plt.savefig('../Plot/RevisionSup/RB183_ConfluencyComparison_DPCFeatures_PC1Importance.svg', format="svg",dpi=300,bbox_inches='tight') 
# %%
PC2Loading = PCLoading.sort_values(['PC2']).reset_index(drop=True)
plt.figure(figsize=(6,16))

ax = sns.barplot(x='PC2',y='Features',data=PC2Loading, orient = 'h',palette = "Wistia")

ax.set_xlabel('PC2 Loading', fontsize = 30)
ax.set_ylabel(" ", fontsize = 30)
ax.tick_params(labelsize = 10)
plt.savefig('../Plot/RevisionSup/RB183_ConfluencyComparison_DPCFeatures_PC2Importance.svg', format="svg",dpi=300,bbox_inches='tight') 
# %%Screen out the cell shape
Morph = [col for col in df_RM.columns if col.startswith('AreaShape_')]

principal_components = pca.fit_transform(df_RM[Morph])
explained_var = pca.explained_variance_ratio_ * 100  # in %

pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df['Condition'] = df_RM['Condition']

plt.figure(figsize=(7,7))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Condition', s=60, palette = sns.color_palette(colors))

plt.xlabel(f'PC1 ({explained_var[0]:.1f}% Variance)', fontsize=30)
plt.ylabel(f'PC2 ({explained_var[1]:.1f}% Variance)', fontsize=30)
plt.tick_params(labelsize = 25) 
plt.title('PCA of Morph Features: Confluency\n', fontsize=20)
legend=plt.legend(fontsize=25,)
# Resize legend marker points
for handle in legend.legendHandles:
    handle._sizes = [250]  
plt.tight_layout()
plt.savefig('../Plot/RevisionSUp/RB183_ConfluencyComparison_MorphFeatures.svg', format="svg",dpi=300,bbox_inches='tight') 
# %%
PCLoading = pd.DataFrame(data=pca.components_.T, columns=['PC1', 'PC2'],index=df_RM[Morph].columns)
PCLoading['Features'] = PCLoading.index.astype(str)

PC1Loading = PCLoading.sort_values(['PC1']).reset_index(drop=True)
plt.figure(figsize=(6,16))

ax = sns.barplot(x='PC1',y='Features',data=PC1Loading, orient = 'h',palette = "Wistia")

ax.set_xlabel('PC1 Loading', fontsize = 30)
ax.set_ylabel(" ", fontsize = 30)
ax.tick_params(labelsize = 10)
plt.savefig('../Plot/RevisionSup/RB183_ConfluencyComparison_MorphFeatures_PC1Importance.svg', format="svg",dpi=300,bbox_inches='tight') 
# %%
PC2Loading = PCLoading.sort_values(['PC2']).reset_index(drop=True)
plt.figure(figsize=(6,16))

ax = sns.barplot(x='PC2',y='Features',data=PC2Loading, orient = 'h',palette = "Wistia")

ax.set_xlabel('PC2 Loading', fontsize = 30)
ax.set_ylabel(" ", fontsize = 30)
ax.tick_params(labelsize = 10)
plt.savefig('../Plot/RevisionSup/RB183_ConfluencyComparison_MorphFeatures_PC2Importance.svg', format="svg",dpi=300,bbox_inches='tight') 


# %%
