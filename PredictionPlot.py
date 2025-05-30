# %%
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
import statistics
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
# %%
df = pd.read_excel('F:/MortensenLab_Jessica/mLab_Work/Data/MultipleDonor/MultiDonorIDOPrediction/BestModel_NMR/LASSO_NMR_Prediction_5foldcv.xlsx')
# %%
df_M = df.groupby('Donor').mean()
df_M_array = df_M.to_numpy()
# %%
donor_styles = {
    "RB179_UT": {'color': '#cb8deb', 'marker': 'o', 'size': 150, 'label': 'RB179_Phase_A'},
    # "RB48_UT": {'color': '#8caff5', 'marker': 'o', 'size': 150, 'label': 'RB48_Phase_A'},
    "RB177_UT": {'color': '#ed63ff', 'marker': 'o','size': 150, 'label': 'RB177_Phase_A'},
    "RB71_UT": {'color': '#636999', 'marker': 'o', 'size': 150,'label': 'RB71_Phase_A'},
    "RB183_UT": {'color': '#e38a05', 'marker': 'o','size': 150, 'label': 'RB183_Phase_A'},
    "RB175_UT": {'color': '#98999c', 'marker': 'o', 'size': 150,'label': 'RB175_Phase_A'},

    # "RB277_UT": {'color': '#32a852', 'marker': '*', 'size': 600, 'label': 'RB277_Phase_B'}
}

# %%
# Create the scatter plot
plt.figure(figsize=(9,6))

# Loop through each donor and plot their data points
for donor, style in donor_styles.items():
    donor_data = df.query(f'Donor == "{donor}"')
    plt.scatter(
        donor_data['Label'],           # x-axis data (Labels)
        donor_data['Prediction'],      # y-axis data (Predictions)
        c=style['color'],              # color from dictionary
        marker=style['marker'],        # marker from dictionary
        label=style['label'],          # label from dictionary
        s=style['size'],            # size from dictionary
    )
# Add labels and title
plt.xlabel('Label')
plt.ylabel('Prediction')
plt.title('Scatter Plot of Predictions by Donor')
#plot linear regression
X = df_M_array[:,0]
Y = df_M_array[:,1]
z = np.polyfit(X,Y, 1)
p = np.poly1d(z)

yhat = p(X)
ybar = np.sum(Y)/len(Y)
ssreg = np.sum((yhat-ybar)**2)
sstot = np.sum((Y-ybar)**2)
plt.plot(X,p(X),"r--")


plt.text(0.05, 0.98, "y=%.2fx+%.2f"%(z[0],z[1]), fontsize=25,fontweight = 'bold',
          transform=plt.gca().transAxes,  # Use axes coords (0-1)
             verticalalignment='top', horizontalalignment='left')
plt.text(0.05, 0.9, 'R^2=' "%.2f"%(ssreg/sstot), fontsize=25,fontweight = 'bold',
          transform=plt.gca().transAxes,  # Use axes coords (0-1)
             verticalalignment='top', horizontalalignment='left')

#customize plot
plt.title("LASSO\n", fontsize=30,fontweight = 'bold')
plt.xlabel("Measured IDO Activity\n(pg KYN/cell/day)", fontsize = 30);
plt.ylabel("Predict IDO Activity\n(pg KYN/cell/day)",fontsize = 30);
plt.tick_params(labelsize = 27) 
legend = plt.legend(bbox_to_anchor=(1.5, 1),fontsize = 18, borderaxespad=0)
for handle in legend.legendHandles:
    handle._sizes = [250]  # Adjust to your preferred size
plt.savefig('../Plot/Phase_Pred/Phase_NMR/Phase_IDOPrediction_LASSO.svg', format="svg",dpi=300,bbox_inches='tight')  

# %%
