# %% Import packages
import numpy as np
import pandas as pd
import umap
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
import seaborn as sns
# %% Read data
df = pd.read_excel('../Data/MultiDonor_Fluo_UTnIFN_Balance770.xlsx')
# %% transfer Donor/treatment into numerical way as target and insert into dataset
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
target = label_encoder.fit_transform(df.Donor)
df.insert(0, 'Target', target) 
# %%Randomize/Smooth the dataset
balance_RM = df.groupby('Donor').rolling(center=False,window=37,min_periods=1).mean().reset_index()
balance_RM1 = balance_RM.drop(['Target', 'Donor','level_1','RB_IDO', 'ImageNumber', 'ObjectNumber',], axis=1)

# %%Perform supervised UMAP
reducer = umap.UMAP(n_neighbors=20,n_components = 2,min_dist=1, random_state=42,)
embedding0 = reducer.fit_transform(balance_RM1)
embedding = reducer.fit_transform(balance_RM1, y=balance_RM.Target)
Loadings = pd.concat([pd.DataFrame(data=embedding, columns=['UMAP1', 'UMAP2']), balance_RM], axis=1)

# %% Figure 1E
sns.set(context='notebook', style='white', font='sans-serif', font_scale=3, color_codes=True, rc={'figure.figsize':(7,7)})
plt.scatter(
    Loadings[Loadings['Donor'].str.contains("UT")].UMAP1,
    Loadings[Loadings['Donor'].str.contains("UT")].UMAP2, 
    c='#4668a3',marker='o',label='UT',s=40)
plt.scatter(
    Loadings[Loadings['Donor'].str.contains("IFN")].UMAP1,
    Loadings[Loadings['Donor'].str.contains("IFN")].UMAP2, 
    c='#ed9b3e',marker='o',label='IFN', s=40)
plt.xlabel("UMAP1", fontsize = 50);
plt.ylabel("UMAP2",fontsize = 50);
plt.tick_params(labelsize = 25) 
legend=plt.legend(bbox_to_anchor=(0.34, 0.21),fontsize = 25, borderaxespad=0)
# Resize legend marker points
for handle in legend.legendHandles:
    handle._sizes = [250]  
# plt.savefig('../Plot/UMAP_UTnIFN.svg', format="svg",dpi=300,bbox_inches='tight')  
# %%
sns.set(context='notebook', style='white', font='sans-serif', font_scale=3, color_codes=True, rc={'figure.figsize':(7,7)})
# grey background of all scatters
plt.scatter(Loadings.UMAP1, Loadings.UMAP2, 
    c='whitesmoke',marker='o',label=' ',s=40) 
# colors according to donors
plt.scatter(Loadings.query('Donor == "RB179_UT"').UMAP1,Loadings.query('Donor == "RB179_UT"').UMAP2, 
    c='#cb8deb',marker='o',label='RB179_UT',s=40)
plt.scatter(Loadings.query('Donor == "RB48_UT"').UMAP1,Loadings.query('Donor == "RB48_UT"').UMAP2, 
    c='#8caff5',marker='o',label='RB48_UT', s=40)
plt.scatter(Loadings.query('Donor == "RB177_UT"').UMAP1,Loadings.query('Donor == "RB177_UT"').UMAP2, 
    c='#ed63ff',marker='o',label='RB177_UT',s=40)
plt.scatter(Loadings.query('Donor == "RB71_UT"').UMAP1,Loadings.query('Donor == "RB71_UT"').UMAP2, 
    c='#636999',marker='o',label='RB71_UT',s=40)
plt.scatter(Loadings.query('Donor == "RB183_UT"').UMAP1,Loadings.query('Donor == "RB183_UT"').UMAP2, 
    c='#e38a05',marker='o',label='RB183_UT',s=40)
plt.scatter(Loadings.query('Donor == "RB175_UT"').UMAP1,Loadings.query('Donor == "RB175_UT"').UMAP2, 
    c='#98999c',marker='o',label='RB175_UT',s=40)

plt.xlabel("UMAP1", fontsize = 50);
plt.ylabel("UMAP2",fontsize = 50);
plt.tick_params(labelsize = 25) 

plt.savefig('../Plot/UMAP_UT.svg', format="svg",dpi=300,bbox_inches='tight')  
# %%
sns.set(context='notebook', style='white', font='sans-serif', font_scale=3, color_codes=True, rc={'figure.figsize':(7,7)})
# grey background of all scatters
plt.scatter(Loadings.UMAP1, Loadings.UMAP2, 
    c='whitesmoke',marker='o',label=' ',s=40) 
# colors according to donors
plt.scatter(Loadings.query('Donor == "RB179_IFN"').UMAP1,Loadings.query('Donor == "RB179_IFN"').UMAP2, 
    c='#cb8deb',marker='x',label='RB179_IFN',s=40)
plt.scatter(Loadings.query('Donor == "RB48_IFN"').UMAP1,Loadings.query('Donor == "RB48_IFN"').UMAP2, 
    c='#8caff5',marker='x',label='RB48_IFN', s=40)
plt.scatter(Loadings.query('Donor == "RB177_IFN"').UMAP1,Loadings.query('Donor == "RB177_IFN"').UMAP2, 
    c='#ed63ff',marker='x',label='RB177_IFN',s=40)
plt.scatter(Loadings.query('Donor == "RB71_IFN"').UMAP1,Loadings.query('Donor == "RB71_IFN"').UMAP2, 
    c='#636999',marker='x',label='RB71_IFN',s=40)
plt.scatter(Loadings.query('Donor == "RB183_IFN"').UMAP1,Loadings.query('Donor == "RB183_IFN"').UMAP2, 
    c='#e38a05',marker='x',label='RB183_IFN',s=40)
plt.scatter(Loadings.query('Donor == "RB175_IFN"').UMAP1,Loadings.query('Donor == "RB175_IFN"').UMAP2, 
    c='#98999c',marker='x',label='RB175_IFN',s=40)

plt.xlabel("UMAP1", fontsize = 50);
plt.ylabel("UMAP2",fontsize = 50);
plt.tick_params(labelsize = 25)

plt.savefig('../Plot/UMAP_IFN.svg', format="svg",dpi=300,bbox_inches='tight')  
# %%Generate a separate legend
from matplotlib.patches import Patch
# Define colors and labels for each donor group
legend_items = [
    {'label': 'RB179', 'color': '#cb8deb'},
    {'label': 'RB48', 'color': '#8caff5'},
    {'label': 'RB177', 'color': '#ed63ff'},
    {'label': 'RB71', 'color': '#636999'},
    {'label': 'RB183', 'color': '#e38a05'},
    {'label': 'RB175', 'color': '#98999c'},
]
# Create square (box) marker patches
custom_legend = [
    Patch(facecolor=item['color'], edgecolor='white', label=item['label'])
    for item in legend_items
]
# Create the separate legend figure
fig_leg = plt.figure(figsize=(3, 4)) 
ax_leg = fig_leg.add_subplot(111)
ax_leg.axis('off')  # Hide axes
# Plot the legend
ax_leg.legend(handles=custom_legend, loc='center', fontsize=15, borderpad=1, handlelength=1.5,handleheight=1.5)

fig_leg.savefig("../Plot/Legend.svg", format="svg", bbox_inches='tight', dpi=300)
# %%
# Set the plotting style
sns.set(context='notebook', style='white', font='sans-serif', font_scale=3,color_codes=True, rc={'figure.figsize': (7,7)})

# Select all columns starting with 'Cell_' to use as hue
cell_columns = [col for col in Loadings.columns if col.startswith("Cell_")]

# Loop through the selected columns
for col in cell_columns:
    plt.figure()
    # Create scatter plot
    g = sns.scatterplot(data=Loadings, x='UMAP1', y='UMAP2',hue=col,s=40,edgecolor='face',palette='RdPu')

    # Add customized text at top-left
    text_label = col.replace("Cell_", "")
    plt.text(0.03, 0.98, text_label, fontsize=35,
             transform=plt.gca().transAxes,  # Use axes coords (0-1)
             verticalalignment='top', horizontalalignment='left')

    # Customize plot
    plt.xlabel("UMAP1", fontsize=50)
    plt.ylabel("UMAP2", fontsize=50)
    plt.tick_params(labelsize=25)
    g.get_legend().remove()

    # Save the figure
    save_name = f"UMAP_{text_label}.svg"
    plt.savefig(f'../Plot/{save_name}', format='svg', dpi=300, bbox_inches='tight')
    plt.close()
# %% approxiamte contribution of features on UMAP
from sklearn.linear_model import LinearRegression
#Fit Linear Model to Approximate UMAP Coordinates
linear_model = LinearRegression()
feature = Loadings.drop(['UMAP1', 'UMAP2', 'Donor', 'level_1', 'Target', 'RB_IDO', 'ImageNumber', 'ObjectNumber', ],axis=1)
linear_model.fit(feature, embedding) # embedding2 as supervised UMAP
#Extract Feature Importances
feature_importances = linear_model.coef_
# plot
Importance=pd.DataFrame(feature_importances, columns=feature.columns).T
Importance.insert(0,'Morphology',feature.columns)
Importance.columns=["Morphology", "UMAP1_Importance", "UMAP2_Importance"]
Importance = Importance.sort_values(['UMAP1_Importance']).reset_index(drop=True)
plt.figure(figsize=(7,7))
ax = sns.barplot(x='UMAP1_Importance',y='Morphology',data=Importance, palette = "Spectral", orient = 'h')
ax.set_xlabel('Features Importance', fontsize = 40)
ax.set_ylabel("Cell Morphology", fontsize = 40)
ax.tick_params(labelsize = 20)
plt.savefig('../Plot/FeatureImportance.svg', format="svg",dpi=300,bbox_inches='tight')  
# %%
