# %%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
import umap
from sklearn.ensemble import IsolationForest
from sklearn.utils import shuffle
from scipy import stats
import os
# %%
df = pd.read_excel("../Data/IDOPE0916_UMAP.xlsx")
df = df.dropna()
# %%select 14 morphology used for prediction
Data14_Morph0916_s = df.drop([
'UMAP1_Intensity', 'UMAP2_Intensity', 'Treatment', 'ImageNumber',
       'ObjectNumber', 'Phase', 'Treatment_Phase',
    #Intensity
       'Cell_IntegratedIntensityEdge', 'Cell_IntegratedIntensity',
       'Cell_LowerQuartileIntensity', 'Cell_MADIntensity',
       'Cell_MassDisplacement', 'Cell_MaxIntensityEdge', 'Cell_MaxIntensity',
       'Cell_MeanIntensityEdge', 'Cell_MeanIntensity', 'Cell_MedianIntensity',
       'Cell_MinIntensityEdge', 'Cell_MinIntensity', 'Cell_StdIntensityEdge',
       'Cell_StdIntensity', 'Cell_UpperQuartileIntensity', 
       # Extra Morph
       'Cell_BoundingBoxArea', 'Cell_BoundingBoxMaximum_X',
       'Cell_BoundingBoxMaximum_Y', 'Cell_BoundingBoxMinimum_X',
       'Cell_BoundingBoxMinimum_Y', 'Cell_Center_X', 'Cell_Center_Y',
       'Cell_ConvexArea', 'Cell_EquivalentDiameter', 'Cell_Orientation', 
], axis=1)
# %%UMAP
reducer1 = umap.UMAP(n_neighbors=30,n_components = 2,min_dist=0, random_state=42,)
embedding1 = reducer1.fit(Data14_Morph0916_s).transform(Data14_Morph0916_s)
# %%combine UMAP embedding with data
UMAP_df1 = pd.concat([pd.DataFrame(embedding1, columns=['UMAP1', 'UMAP2']),df],axis=1)
#10% smooth for each group seperately
UMAP0916_UT_RM = UMAP_df1.query('Treatment == "UT"').groupby('Treatment').rolling(center=False,window=13,min_periods=1).mean().reset_index(drop=True)
UMAP0916_IFN_RM = UMAP_df1.query('Treatment == "IFN"').groupby('Treatment').rolling(center=False,window=26,min_periods=1).mean().reset_index(drop=True)
UMAP0916_UT_RM['Treatment'] = 'UT'
UMAP0916_IFN_RM['Treatment'] = 'IFN'
UMAP0916_RM = pd.concat([UMAP0916_IFN_RM, UMAP0916_UT_RM ]).reset_index(drop=True)
# %%UMAP plot
sns.set(style='white', context='notebook',font_scale =2, rc={'figure.figsize':(6,6)})
colors = [ "#4668a3", #UT
          "#ed9b3e", #50IFN
         ]
g = sns.scatterplot(data=UMAP0916_RM,x='UMAP1',y='UMAP2', hue="Treatment",
                   palette = sns.color_palette(colors),s=20,edgecolor = 'face',alpha=0.8,)
g.set_xlabel('UMAP1 Intensity', fontsize = 30, )
g.set_ylabel('UMAP2 Intensity', fontsize = 30,  )

plt.legend(bbox_to_anchor=(1.1,0.3),borderaxespad=0)
legend=plt.legend(bbox_to_anchor=(1.05, 0.95),fontsize = 25, borderaxespad=0)
# Resize legend marker points
for handle in legend.legendHandles:
    handle._sizes = [400]  
plt.savefig("../Plot/MorphUMAP_UTnIFN.svg", format="svg", bbox_inches='tight', dpi=300)
# %% approxiamte contribution of Morph features on UMAP
from sklearn.linear_model import LinearRegression
#Fit Linear Model to Approximate UMAP Coordinates
linear_model = LinearRegression()
feature = Data14_Morph0916_s
linear_model.fit(feature, embedding1)
#Extract Feature Importances
feature_importances = linear_model.coef_
# plot
Importance=pd.DataFrame(feature_importances, columns=feature.columns).T
Importance.insert(0,'Morphology',feature.columns)
Importance.columns=["Morphology", "UMAP1_Importance", "UMAP2_Importance"]
Importance = Importance.sort_values(['UMAP1_Importance']).reset_index(drop=True)
Importance10 = pd.concat([Importance.head(10), Importance.tail(10)])
plt.figure(figsize=(8,6))
ax = sns.barplot(x='UMAP1_Importance',y='Morphology',data=Importance, palette = "Spectral", orient = 'h')
ax.set_xlabel('Features Importance', fontsize = 30)
ax.set_ylabel("Cell Morphology", fontsize = 30)
ax.tick_params(labelsize = 15)
plt.savefig("../Plot/MorphUMAP_UTnIFN_FeatureImportance.svg", format="svg", bbox_inches='tight', dpi=300)
# %%
# Define the directory to save plots
save_dir = "../Plot/UTnIFN_box"
os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist

# Loop through all columns for hue
# Select all columns starting with 'Cell_'
cell_columns = [col for col in UMAP0916_RM.columns if col.startswith("Cell_")]

for col in cell_columns:
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(4,7))

    # box plot with current column
    sns.boxplot(data=UMAP0916_RM, x='Treatment', y=col, ax=ax)
    # Axis labels
    text_label = col.replace("Cell_", "")
    plt.xlabel(" ")
    plt.ylabel(f"{text_label} (a.u.)", fontsize=25)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=15)
   # Create a valid filename (remove spaces or special characters if necessary)
    safe_filename = "".join(c if c.isalnum() or c in "_-" else "_" for c in col)

    # Save the plot as an SVG file with 300 dpi
    save_path = os.path.join(save_dir, f"{safe_filename}.svg", )
    plt.savefig(save_path, format="svg", dpi=300, bbox_inches='tight')

    plt.close(fig)

print("Plots saved successfully in:", save_dir)

# %%
kclst_IFN0916 = pd.read_excel("../Data/IFN0916_UMAP_kclst.xlsx")
# %%select 14 morphology used for prediction
Data14_ClusterMorph = kclst_IFN0916.drop([
       'Cluster', 'UMAP1_Intensity', 'UMAP2_Intensity', 'Treatment',
       'ImageNumber', 'ObjectNumber', 'Phase', 'Treatment_Phase',
       # intensity
       'Cell_IntegratedIntensityEdge', 'Cell_IntegratedIntensity',
       'Cell_LowerQuartileIntensity', 'Cell_MADIntensity',
       'Cell_MassDisplacement', 'Cell_MaxIntensityEdge', 'Cell_MaxIntensity',
       'Cell_MeanIntensityEdge', 'Cell_MeanIntensity', 'Cell_MedianIntensity',
       'Cell_MinIntensityEdge', 'Cell_MinIntensity', 'Cell_StdIntensityEdge',
       'Cell_StdIntensity', 'Cell_UpperQuartileIntensity', 
       # Extra Morph
       'Cell_BoundingBoxArea', 'Cell_BoundingBoxMaximum_X',
       'Cell_BoundingBoxMaximum_Y', 'Cell_BoundingBoxMinimum_X',
       'Cell_BoundingBoxMinimum_Y', 'Cell_Center_X', 'Cell_Center_Y',
       'Cell_ConvexArea', 'Cell_EquivalentDiameter', 'Cell_Orientation', 
],axis=1)
reducer2 = umap.UMAP(n_neighbors=30,n_components = 2,min_dist=0, random_state=42,)
UMAP_ClusterMorph = reducer2.fit(Data14_ClusterMorph)
embedding_ClusterMorph = UMAP_ClusterMorph.transform(Data14_ClusterMorph)
# %%
UMAP_clsIFN0916 = pd.concat([pd.DataFrame(data=embedding_ClusterMorph, columns=['UMAP1', 'UMAP2']), kclst_IFN0916],axis=1)
#10% smooth for each group seperately
IDOhigh = UMAP_clsIFN0916.query('Cluster == "IDO_High"').rolling(center=False,window=14,min_periods=1).mean().reset_index(drop=True)
IDOlow = UMAP_clsIFN0916.query('Cluster == "IDO_Low"').rolling(center=False,window=12,min_periods=1).mean().reset_index(drop=True)
IDOhigh['Cluster'] = 'IDO_High'
IDOlow['Cluster'] = 'IDO_Low'
UMAP_clsIFN0916_RM = pd.concat([ IDOlow, IDOhigh,]).reset_index(drop=True)
#%%
sns.set(style='white', context='notebook',font_scale =2, rc={'figure.figsize':(6,6)})
colors = [ 
   "#37a193", #cluster 1, high intensity
   "#942b2b",  #cluster 0, low intensity
]
g = sns.scatterplot(data=UMAP_clsIFN0916_RM,x='UMAP1',y='UMAP2', hue="Cluster",
                   palette = sns.color_palette(colors),s=20,edgecolor = 'face',alpha=0.8,)
g.set_xlabel('UMAP1 Intensity', fontsize = 30, )
g.set_ylabel('UMAP2 Intensity', fontsize = 30,  )

legend=plt.legend(bbox_to_anchor=(1.05, 0.95),fontsize = 25, borderaxespad=0)
# Resize legend marker points
for handle in legend.legendHandles:
    handle._sizes = [400]  
plt.savefig("../Plot/MorphUMAP_Cluster.svg", format="svg", bbox_inches='tight', dpi=300)
# %% approxiamte contribution of FITC features on UMAP(Morph)
from sklearn.linear_model import LinearRegression
#Fit Linear Model to Approximate UMAP Coordinates
linear_model = LinearRegression()
feature = Data14_ClusterMorph
linear_model.fit(feature, embedding_ClusterMorph) 
#Extract Feature Importances
feature_importances = linear_model.coef_
# plot
Importance=pd.DataFrame(feature_importances, columns=feature.columns).T
Importance.insert(0,'Morphology',feature.columns)
Importance.columns=["Morphology", "UMAP1_Importance", "UMAP2_Importance"]
Importance = Importance.sort_values(['UMAP1_Importance']).reset_index(drop=True)
Importance10 = pd.concat([Importance.head(10), Importance.tail(10)])
plt.figure(figsize=(8,6))
ax = sns.barplot(x='UMAP1_Importance',y='Morphology',data=Importance, palette = "Spectral", orient = 'h')
ax.set_xlabel('Features Importance', fontsize = 30)
ax.set_ylabel("Cell Morphology", fontsize = 30)
ax.tick_params(labelsize = 15)
plt.savefig("../Plot/MorphUMAP_Cluster_FeatureImporteance.svg", format="svg", bbox_inches='tight', dpi=300)
# %%
# Define the directory to save plots
save_dir = "../Plot/IFNCluster_box"
os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist

# Loop through all columns for hue
# Select all columns starting with 'Cell_'
cell_columns = [col for col in UMAP0916_RM.columns if col.startswith("Cell_")]

for col in cell_columns:
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(4,7))

    # box plot with current column
    sns.boxplot(data=UMAP0916_RM, x='Treatment', y=col, ax=ax)
    # Axis labels
    text_label = col.replace("Cell_", "")
    plt.xlabel(" ")
    plt.ylabel(f"{text_label} (a.u.)", fontsize=25)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=15)
   # Create a valid filename (remove spaces or special characters if necessary)
    safe_filename = "".join(c if c.isalnum() or c in "_-" else "_" for c in col)

    # Save the plot as an SVG file with 300 dpi
    save_path = os.path.join(save_dir, f"{safe_filename}.svg", )
    plt.savefig(save_path, format="svg", dpi=300, bbox_inches='tight')

    plt.close(fig) 

print("Plots saved successfully in:", save_dir)

# %%
