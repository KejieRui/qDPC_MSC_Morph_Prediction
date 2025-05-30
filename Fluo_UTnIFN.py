# %% Import packages
import numpy as np
import pandas as pd
import umap
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
import seaborn as sns
# %%
df_IFN = pd.read_excel('../Data/MultiDonor_Fluo_IFN.xlsx').drop(['Index'], axis=1
# %%
# Set the plotting style
sns.set(context='notebook', style='white', font='sans-serif', font_scale=3,color_codes=True)
donor_styles = {
    "RB182_LBL": {'color': '#e38a05', 'marker': 'x', 'label': 'RB182_A'},
    "RB37_LBL": {'color': '#3ddde3', 'marker': 'x', 'label': 'RB37_A'},
    "RB114_LBL": {'color': '#147a00', 'marker': 'x', 'label': 'RB114_A'},
    "RB179_MD": {'color': '#cb8deb', 'marker': '^', 'label': 'RB179_B'},
    "RB48_MD": {'color': '#8caff5', 'marker': '^', 'label': 'RB48_B'},
    "RB177_MD": {'color': '#ed63ff', 'marker': '^', 'label': 'RB177_B'},
    "RB71_MD": {'color': '#636999', 'marker': '^', 'label': 'RB71_B'},
    "RB183_MD": {'color': '#e38a05', 'marker': '^', 'label': 'RB183_B'},
    "RB175_MD": {'color': '#98999c', 'marker': '^', 'label': 'RB175_B'}
}

# Select all columns starting with 'Cell_'
cell_columns = [col for col in df_IFN.columns if col.startswith("Cell_")]

# Take donor means
df_IFN_M = df_IFN.groupby('Donor1').mean().reset_index()

# Loop through the selected features
for i, col in enumerate(cell_columns):
    plt.figure(figsize=(10,7))

    # Plot each donor point
    for donor, style in donor_styles.items():
        donor_data = df_IFN_M[df_IFN_M['Donor1'] == donor]
        if not donor_data.empty:
            plt.scatter(
                donor_data['RB_IDO'],              # x-axis: Measured IDO Activity
                donor_data[col],                   # y-axis: Feature
                c=style['color'],
                marker=style['marker'],
                s=200  # bigger marker
            )

    # Linear regression
    X = df_IFN_M['RB_IDO'].values
    Y = df_IFN_M[col].values
    z = np.polyfit(X, Y, 1)
    p = np.poly1d(z)
    plt.plot(X, p(X), "r--", linewidth=3)

    # Axis labels
    text_label = col.replace("Cell_", "")
    plt.xlabel("Measured IDO Activity\n(pg KYN/cell/day)", fontsize=40)
    plt.ylabel(f"{text_label} (a.u.)", fontsize=40)

    plt.tick_params(labelsize=25)

    # Remove legend
    plt.legend([],[], frameon=False)

    # Save
    save_name = f"{text_label}.svg"
    plt.savefig(f'../Plot/Fluo_UTnIFN/{save_name}', format='svg', dpi=300, bbox_inches='tight')
    plt.close()
# %%
#Correlation matrix for cell morphology to IDO activity, compare between UT and IFN cells used for prediction
# %%
df_UT = pd.read_excel('../Data/MultiDonor_Fluo_UT.xlsx').drop(['Index'], axis=1)
# %%
df_UT_M = df_UT.groupby('Donor1').mean().reset_index()

# %% create dataset including 14morph for correlation matrix
Data_IFN = df_IFN_M.drop(['Donor1', 'Donor2', 'ImageNumber', 'ObjectNumber', ],axis=1)
Data_UT = df_UT_M.drop(['Donor1', 'Donor2', 'ImageNumber', 'ObjectNumber',  ],axis=1)

# %%
# Compute correlation matrix
Matrix_IFN = Data_IFN.corr()
# Extract correlations of 'IDO' with other columns
Correlation_IFN = Matrix_IFN['RB_IDO'].drop('RB_IDO')  # Drop 'IDO' itself from the correlations
# Plot the correlation values
plt.figure(figsize=(6, 12))
# Create a custom color palette: blue for negative, red for positive correlations
colors = ['#616fc9' if val < 0 else '#c95655' for val in Correlation_IFN.values]
sns.barplot(x=Correlation_IFN.values, y=Correlation_IFN.index, palette=colors)
plt.title('Correlation with IDO (IFN)\n', fontsize=30)
# plt.ylabel('Morphology', fontsize=0)
plt.xlabel('Correlation Coefficient', fontsize=30)
plt.xlim(-0.8,0.9)
plt.xticks(rotation=0, ha='right', fontsize=13)
plt.yticks(rotation=0, ha='right', fontsize=30)
plt.savefig('../Plot/Fluo_UTnIFN/IFN.svg', format="svg",dpi=300,bbox_inches='tight')  

# %%
# %%
# Compute correlation matrix
Matrix_UT = Data_UT.corr()
# Extract correlations of 'IDO' with other columns
Correlation_UT = Matrix_UT['RB_IDO'].drop('RB_IDO')  # Drop 'IDO' itself from the correlations
# Plot the correlation values
plt.figure(figsize=(6, 12))
# Create a custom color palette: blue for negative, red for positive correlations
colors = ['#616fc9' if val < 0 else '#c95655' for val in Correlation_UT.values]
sns.barplot(x=Correlation_UT.values, y=Correlation_UT.index, palette=colors)
plt.title('Correlation with IDO (UT)\n', fontsize=30)
plt.xlabel('Correlation Coefficient', fontsize=30)
plt.xlim(-0.8,0.9)
plt.xticks(rotation=0, ha='right', fontsize=13)
plt.yticks(rotation=0, ha='right', fontsize=30)
plt.tick_params(labelleft = False,) 
plt.savefig('../Plot/Fluo_UTnIFN/UT.svg', format="svg",dpi=300,bbox_inches='tight')  
# %%
