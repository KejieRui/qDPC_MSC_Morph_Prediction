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

# %% #Train Test Set
df = pd.read_excel('../Data/MultiDonor_Phase_MorphOnly.xlsx').drop(['Index'], axis=1)
label_group = list(set(np.around(df.RB_IDO.to_numpy(),2)))
# %%Initialize models - grid search
rfr = RandomForestRegressor(n_estimators=117, max_depth=7, random_state=42)
gbr = GradientBoostingRegressor(n_estimators=29, learning_rate=0.2, max_depth=3,random_state=42)
svr = SVR(C= 5, epsilon=0.5, kernel='rbf')
mlpr = MLPRegressor(activation='relu', alpha= 1,hidden_layer_sizes= (100,),learning_rate= 'constant',max_iter= 200,solver= 'adam')
svrl = SVR(C= 0.1, epsilon=1, kernel='linear')
lasso =  Lasso(alpha=0.8172727272727273)

# %%
# Define the scoring methods
scoring_methods = {'R2': r2_score}
# List of models
models = {
    'RFR': rfr,
    'SVR': svr,
    'GBR': gbr,
    'MLPR': mlpr,
    'SVR_L': svrl,
    'LASSO': lasso
}
#5-fold cross validation
N_cv = 5
n_each = int(df.shape[0]/N_cv)
model_scores = {}
model_pred = {}
for i in range(N_cv):
    print(f'Current val round {i}')

    Train = pd.concat([df.iloc[0:i*n_each], df.iloc[(i+1)*n_each:]], axis=0)
    X_train = Train.drop(['RB_Donor', 'level_1','Index', 'RB_IDO', 'ImageNumber', 'ObjectNumber',],axis=1)
    y_train = np.around(Train.RB_IDO.to_numpy(),2)

    X_test = df.iloc[i*n_each:(i+1)*n_each].drop(['RB_Donor', 'level_1', 'Index', 'RB_IDO', 'ImageNumber', 'ObjectNumber',],axis=1)
    y_test = np.around(df.iloc[i*n_each:(i+1)*n_each].RB_IDO.to_numpy(),2)

    
    for model_name, model in models.items():
        try: 
            model_scores[model_name]
            model_pred[model_name]
        except: 
            model_scores[model_name] = {}
            model_pred[model_name] = []

        model.fit(X_train, y_train)
        y_predict = model.predict(X_test)

        group_y = []
        group_predict = []
        for each_ in label_group:
            cur_index = np.where(y_test==each_)
            group_predict.append(np.mean(y_predict[cur_index]))
            group_y.append(each_)

        model_pred[model_name].append(group_predict)
        
        for scoring_name, scoring_method in scoring_methods.items():
            score_ = scoring_method(group_y, group_predict)

            if i == 0:
                model_scores[model_name][scoring_name] = [score_]
            else:
                model_scores[model_name][scoring_name].append(score_)
# %%export model performance R2 score for GraphPad Prism
df_score = pd.DataFrame(model_scores)
df_score.to_excel(r'../Data/ModelCVScore_Phase_MorphOnly.xlsx')
# %%
avg_result = {}
for scoring_name, scoring_method in scoring_methods.items():
    try:
        avg_result[scoring_name]
    except:
        avg_result[scoring_name] = {}
    for model_name, model in models.items():
        avg_result[scoring_name][model_name] = [np.mean(model_scores[model_name][scoring_name]),np.std(model_scores[model_name][scoring_name])]

# %% Generate prediction results of each model
result_dict_df = {}
for model_name, model in models.items():
    cur_data = np.asarray(model_pred[model_name])

    cur_data_w_label = np.concatenate((cur_data, np.asarray(group_y)[None,:]), axis=0)
    index = np.argsort(cur_data_w_label[-1,:])

    cur_data_w_label_sort = cur_data_w_label[:,index]

    label_chunk = np.array(cur_data_w_label_sort[-1].tolist()*5).reshape(N_cv,-1)

    cur_model_data = []

    for j in range(cur_data_w_label_sort.shape[1]):
        for i in range(N_cv):
            donor1 = list(set(df.loc[df['RB_IDO']==label_chunk[i,j]]['Donor1'].to_numpy()))[0]
            cur_model_data.append([donor1, label_chunk[i,j], cur_data_w_label_sort[i,j]])

    result_df = pd.DataFrame(np.asarray(cur_model_data),columns=['Donor','Label','Prediction'])
    result_dict_df[model_name] = result_df
# %%
# Define a dictionary to assign specific colors and markers to each donor
donor_styles = {
    "RB179_UT": {'color': '#cb8deb', 'marker': 'o', 'size': 150, 'label': 'RB179_Phase_A'},
    "RB48_UT": {'color': '#8caff5', 'marker': 'o', 'size': 150, 'label': 'RB48_Phase_A'},
    "RB177_UT": {'color': '#ed63ff', 'marker': 'o','size': 150, 'label': 'RB177_Phase_A'},
    "RB71_UT": {'color': '#636999', 'marker': 'o', 'size': 150,'label': 'RB71_Phase_A'},
    "RB183_UT": {'color': '#e38a05', 'marker': 'o','size': 150, 'label': 'RB183_Phase_A'},
    "RB175_UT": {'color': '#98999c', 'marker': 'o', 'size': 150,'label': 'RB175_Phase_A'},
}
# %%Separate out prediction results from each model, and save as excel sheet
# %%RFR
Pred_RFR = pd.DataFrame(result_dict_df['RFR'])
Pred_RFR.to_excel(r'../Data/PredictionResult/Fluo_UT_RFR_Prediction_5foldcv.xlsx', index=False)
Pred_RFR = pd.read_excel('../Data/PredictionResult/Fluo_UT_RFR_Prediction_5foldcv.xlsx')
# %%
Pred_RFR_M = Pred_RFR.groupby('Donor').mean()
Pred_RFR_M_array = Pred_RFR_M.to_numpy()
# %% RFR
# Create the scatter plot
plt.figure(figsize=(9,6))
# Loop through each donor and plot their data points
for donor, style in donor_styles.items():
    donor_data = Pred_RFR.query(f'Donor == "{donor}"')
    plt.scatter(
        donor_data['Label'],           # x-axis data (Labels)
        donor_data['Prediction'],      # y-axis data (Predictions)
        c=style['color'],              # color from dictionary
        marker=style['marker'],        # marker from dictionary
        label=style['label'],          # label from dictionary
        s=100                          # size of the markers
    )
# Add labels and title
plt.xlabel('Label')
plt.ylabel('Prediction')
plt.title('Scatter Plot of Predictions by Donor')
#plot linear regression
X = Pred_RFR_M_array[:,0]
Y = Pred_RFR_M_array[:,1]
z = np.polyfit(X,Y, 1)
p = np.poly1d(z)

yhat = p(X)
ybar = np.sum(Y)/len(Y)
ssreg = np.sum((yhat-ybar)**2)
sstot = np.sum((Y-ybar)**2)
plt.plot(X,p(X),"r--")
plt.text(0.03, 0.98, "y=%.2fx+%.2f"%(z[0],z[1]), fontsize=25,fontweight = 'bold',
          transform=plt.gca().transAxes,  # Use axes coords (0-1)
             verticalalignment='top', horizontalalignment='left')
plt.text(0.03, 0.9, 'R^2=' "%.2f"%(ssreg/sstot), fontsize=25,fontweight = 'bold',
          transform=plt.gca().transAxes,  # Use axes coords (0-1)
             verticalalignment='top', horizontalalignment='left')
#customize plot
plt.title("RFR\n", fontsize=30,fontweight = 'bold')
plt.xlabel("Measured IDO Activity\n(pg KYN/cell/day)", fontsize = 30);
plt.ylabel("Predict IDO Activity\n(pg KYN/cell/day)",fontsize = 30);
plt.tick_params(labelsize = 27) 
plt.legend(bbox_to_anchor=(1.3, 1),fontsize = 18, borderaxespad=0)
# plt.savefig('../Plot/IFN_Fluo_Pred/Fluo_IFN_RFR.svg', format="svg",dpi=300,bbox_inches='tight')  

# %%
Pred_SVR = pd.DataFrame(result_dict_df['SVR'])
Pred_SVR.to_excel(r'F:/MortensenLab_Jessica/mLab_Work/Data/MultipleDonor/MultiDonorIDOPrediction/Fluo_UT_SVR_Prediction_5foldcv.xlsx', index=False)
Pred_SVR = pd.read_excel('F:/MortensenLab_Jessica/mLab_Work/Data/MultipleDonor/MultiDonorIDOPrediction/Fluo_UT_SVR_Prediction_5foldcv.xlsx')

# %%
Pred_SVR_M = Pred_SVR.groupby('Donor').mean()
Pred_SVR_M_array = Pred_SVR_M.to_numpy()
# %%SVR
# Create the scatter plot
plt.figure(figsize=(10,7))

# Loop through each donor and plot their data points
for donor, style in donor_styles.items():
    donor_data = Pred_SVR.query(f'Donor == "{donor}"')
    plt.scatter(
        donor_data['Label'],           # x-axis data (Labels)
        donor_data['Prediction'],      # y-axis data (Predictions)
        c=style['color'],              # color from dictionary
        marker=style['marker'],        # marker from dictionary
        label=style['label'],          # label from dictionary
        s=100                          # size of the markers
    )

# Add labels and title
plt.xlabel('Label')
plt.ylabel('Prediction')
plt.title('Scatter Plot of Predictions by Donor')
#plot linear regression
X = Pred_SVR_M_array[:,0]
Y = Pred_SVR_M_array[:,1]
z = np.polyfit(X,Y, 1)
p = np.poly1d(z)

yhat = p(X)
ybar = np.sum(Y)/len(Y)
ssreg = np.sum((yhat-ybar)**2)
sstot = np.sum((Y-ybar)**2)
plt.plot(X,p(X),"r--")

plt.text(0.03, 0.98, "y=%.2fx+%.2f"%(z[0],z[1]), fontsize=25,fontweight = 'bold',
          transform=plt.gca().transAxes,  # Use axes coords (0-1)
             verticalalignment='top', horizontalalignment='left')
plt.text(0.03, 0.9, 'R^2=' "%.2f"%(ssreg/sstot), fontsize=25,fontweight = 'bold',
          transform=plt.gca().transAxes,  # Use axes coords (0-1)
             verticalalignment='top', horizontalalignment='left')
#customize plot
plt.title("SVR\n", fontsize=30,fontweight = 'bold')
plt.xlabel("Measured IDO Activity\n(pg KYN/cell/day)", fontsize = 30);
plt.ylabel("Predict IDO Activity\n(pg KYN/cell/day)",fontsize = 30);
plt.tick_params(labelsize = 27) 
plt.legend(bbox_to_anchor=(1.3, 1),fontsize = 18, borderaxespad=0)
plt.savefig('../Plot/UT_Fluo_Pred/Fluo_UT_SVR.svg', format="svg",dpi=300,bbox_inches='tight')  
# %%
Pred_GBR = pd.DataFrame(result_dict_df['GBR'])
Pred_GBR.to_excel(r'F:/MortensenLab_Jessica/mLab_Work/Data/MultipleDonor/MultiDonorIDOPrediction/Fluo_UT_GBR_Prediction_5foldcv.xlsx', index=False)
Pred_GBR = pd.read_excel('F:/MortensenLab_Jessica/mLab_Work/Data/MultipleDonor/MultiDonorIDOPrediction/Fluo_UT_GBR_Prediction_5foldcv.xlsx')

# %%
Pred_GBR_M = Pred_GBR.groupby('Donor').mean()
Pred_GBR_M_array = Pred_GBR_M.to_numpy()
# %%GBR
# Create the scatter plot
plt.figure(figsize=(10,7))

# Loop through each donor and plot their data points
for donor, style in donor_styles.items():
    donor_data = Pred_GBR.query(f'Donor == "{donor}"')
    plt.scatter(
        donor_data['Label'],           # x-axis data (Labels)
        donor_data['Prediction'],      # y-axis data (Predictions)
        c=style['color'],              # color from dictionary
        marker=style['marker'],        # marker from dictionary
        label=style['label'],          # label from dictionary
        s=100                          # size of the markers
    )

# Add labels and title
plt.xlabel('Label')
plt.ylabel('Prediction')
plt.title('Scatter Plot of Predictions by Donor')
#plot linear regression
X = Pred_GBR_M_array[:,0]
Y = Pred_GBR_M_array[:,1]
z = np.polyfit(X,Y, 1)
p = np.poly1d(z)

yhat = p(X)
ybar = np.sum(Y)/len(Y)
ssreg = np.sum((yhat-ybar)**2)
sstot = np.sum((Y-ybar)**2)
plt.plot(X,p(X),"r--")
plt.text(0.03, 0.98, "y=%.2fx+%.2f"%(z[0],z[1]), fontsize=25,fontweight = 'bold',
          transform=plt.gca().transAxes,  # Use axes coords (0-1)
             verticalalignment='top', horizontalalignment='left')
plt.text(0.03, 0.9, 'R^2=' "%.2f"%(ssreg/sstot), fontsize=25,fontweight = 'bold',
          transform=plt.gca().transAxes,  # Use axes coords (0-1)
             verticalalignment='top', horizontalalignment='left')
#customize plot
plt.title("GBR\n", fontsize=30,fontweight = 'bold')
plt.xlabel("Measured IDO Activity\n(pg KYN/cell/day)", fontsize = 30);
plt.ylabel("Predict IDO Activity\n(pg KYN/cell/day)",fontsize = 30);
plt.tick_params(labelsize = 27) 
plt.legend(bbox_to_anchor=(1.3, 1),fontsize = 18, borderaxespad=0)
plt.savefig('../Plot/UT_Fluo_Pred/Fluo_UT_GBR.svg', format="svg",dpi=300,bbox_inches='tight')  
# %%
Pred_MLPR = pd.DataFrame(result_dict_df['MLPR'])
Pred_MLPR.to_excel(r'F:/MortensenLab_Jessica/mLab_Work/Data/MultipleDonor/MultiDonorIDOPrediction/Fluo_UT_MLPR_Prediction_5foldcv.xlsx', index=False)
Pred_MLPR = pd.read_excel('F:/MortensenLab_Jessica/mLab_Work/Data/MultipleDonor/MultiDonorIDOPrediction/Fluo_UT_MLPR_Prediction_5foldcv.xlsx')

# %%
Pred_MLPR_M = Pred_MLPR.groupby('Donor').mean()
Pred_MLPR_M_array = Pred_MLPR_M.to_numpy()
# %%MLPR
# Create the scatter plot
plt.figure(figsize=(10,7))

# Loop through each donor and plot their data points
for donor, style in donor_styles.items():
    donor_data = Pred_MLPR.query(f'Donor == "{donor}"')
    plt.scatter(
        donor_data['Label'],           # x-axis data (Labels)
        donor_data['Prediction'],      # y-axis data (Predictions)
        c=style['color'],              # color from dictionary
        marker=style['marker'],        # marker from dictionary
        label=style['label'],          # label from dictionary
        s=100                          # size of the markers
    )

# Add labels and title
plt.xlabel('Label')
plt.ylabel('Prediction')
plt.title('Scatter Plot of Predictions by Donor')
#plot linear regression
X = Pred_MLPR_M_array[:,0]
Y = Pred_MLPR_M_array[:,1]
z = np.polyfit(X,Y, 1)
p = np.poly1d(z)

yhat = p(X)
ybar = np.sum(Y)/len(Y)
ssreg = np.sum((yhat-ybar)**2)
sstot = np.sum((Y-ybar)**2)
plt.plot(X,p(X),"r--")
plt.text(0.03, 0.98, "y=%.2fx+%.2f"%(z[0],z[1]), fontsize=25,fontweight = 'bold',
          transform=plt.gca().transAxes,  # Use axes coords (0-1)
             verticalalignment='top', horizontalalignment='left')
plt.text(0.03, 0.9, 'R^2=' "%.2f"%(ssreg/sstot), fontsize=25,fontweight = 'bold',
          transform=plt.gca().transAxes,  # Use axes coords (0-1)
             verticalalignment='top', horizontalalignment='left')
#customize plot
plt.title("GBR\n", fontsize=30,fontweight = 'bold')
plt.xlabel("Measured IDO Activity\n(pg KYN/cell/day)", fontsize = 30);
plt.ylabel("Predict IDO Activity\n(pg KYN/cell/day)",fontsize = 30);
plt.tick_params(labelsize = 27) 
plt.legend(bbox_to_anchor=(1.3, 1),fontsize = 18, borderaxespad=0)
plt.savefig('../Plot/UT_Fluo_Pred/Fluo_UT_GBR.svg', format="svg",dpi=300,bbox_inches='tight')  
# %%
# %%
Pred_SVR_L = pd.DataFrame(result_dict_df['SVR_L'])
Pred_SVR_L.to_excel(r'F:/MortensenLab_Jessica/mLab_Work/Data/MultipleDonor/MultiDonorIDOPrediction/Fluo_UT_SVR_L_Prediction_5foldcv.xlsx', index=False)
Pred_SVR_L = pd.read_excel('F:/MortensenLab_Jessica/mLab_Work/Data/MultipleDonor/MultiDonorIDOPrediction/Fluo_UT_SVR_L_Prediction_5foldcv.xlsx')

# %%
Pred_SVR_L_M = Pred_SVR_L.groupby('Donor').mean()
Pred_SVR_L_M_array = Pred_SVR_L_M.to_numpy()
# %%SVR_L
# Create the scatter plot
plt.figure(figsize=(10,7))

# Loop through each donor and plot their data points
for donor, style in donor_styles.items():
    donor_data = Pred_SVR_L.query(f'Donor == "{donor}"')
    plt.scatter(
        donor_data['Label'],           # x-axis data (Labels)
        donor_data['Prediction'],      # y-axis data (Predictions)
        c=style['color'],              # color from dictionary
        marker=style['marker'],        # marker from dictionary
        label=style['label'],          # label from dictionary
        s=100                          # size of the markers
    )

# Add labels and title
plt.xlabel('Label')
plt.ylabel('Prediction')
plt.title('Scatter Plot of Predictions by Donor')
#plot linear regression
X = Pred_SVR_L_M_array[:,0]
Y = Pred_SVR_L_M_array[:,1]
z = np.polyfit(X,Y, 1)
p = np.poly1d(z)

yhat = p(X)
ybar = np.sum(Y)/len(Y)
ssreg = np.sum((yhat-ybar)**2)
sstot = np.sum((Y-ybar)**2)
plt.plot(X,p(X),"r--")
plt.text(0.03, 0.89, "y=%.2fx+%.2f"%(z[0],z[1]), fontsize=25,fontweight = 'bold',
          transform=plt.gca().transAxes,  # Use axes coords (0-1)
             verticalalignment='top', horizontalalignment='left')
plt.text(0.03, 0.81, 'R^2=' "%.2f"%(ssreg/sstot), fontsize=25,fontweight = 'bold',
          transform=plt.gca().transAxes,  # Use axes coords (0-1)
             verticalalignment='top', horizontalalignment='left')
#customize plot
plt.title("SVR_L\n", fontsize=30,fontweight = 'bold')
plt.xlabel("Measured IDO Activity\n(pg KYN/cell/day)", fontsize = 30);
plt.ylabel("Predict IDO Activity\n(pg KYN/cell/day)",fontsize = 30);
plt.tick_params(labelsize = 27) 
plt.legend(bbox_to_anchor=(1.3, 1),fontsize = 18, borderaxespad=0)
plt.savefig('../Plot/UT_Fluo_Pred/Fluo_UT_SVR_L.svg', format="svg",dpi=300,bbox_inches='tight')  
# %%
# %%
Pred_LASSO = pd.DataFrame(result_dict_df['LASSO'])
Pred_LASSO.to_excel(r'F:/MortensenLab_Jessica/mLab_Work/Data/MultipleDonor/MultiDonorIDOPrediction/Fluo_UT_LASSO_Prediction_5foldcv.xlsx', index=False)
Pred_LASSO = pd.read_excel('F:/MortensenLab_Jessica/mLab_Work/Data/MultipleDonor/MultiDonorIDOPrediction/Fluo_UT_LASSO_Prediction_5foldcv.xlsx')

# %%
Pred_LASSO_M = Pred_LASSO.groupby('Donor').mean()
Pred_LASSO_M_array = Pred_LASSO_M.to_numpy()
# %%LASSO
# Create the scatter plot
plt.figure(figsize=(10,7))

# Loop through each donor and plot their data points
for donor, style in donor_styles.items():
    donor_data = Pred_LASSO.query(f'Donor == "{donor}"')
    plt.scatter(
        donor_data['Label'],           # x-axis data (Labels)
        donor_data['Prediction'],      # y-axis data (Predictions)
        c=style['color'],              # color from dictionary
        marker=style['marker'],        # marker from dictionary
        label=style['label'],          # label from dictionary
        s=100                          # size of the markers
    )

# Add labels and title
plt.xlabel('Label')
plt.ylabel('Prediction')
plt.title('Scatter Plot of Predictions by Donor')
#plot linear regression
X = Pred_LASSO_M_array[:,0]
Y = Pred_LASSO_M_array[:,1]
z = np.polyfit(X,Y, 1)
p = np.poly1d(z)

yhat = p(X)
ybar = np.sum(Y)/len(Y)
ssreg = np.sum((yhat-ybar)**2)
sstot = np.sum((Y-ybar)**2)
plt.plot(X,p(X),"r--")
plt.text(0.03, 0.98, "y=%.2fx+%.2f"%(z[0],z[1]), fontsize=25,fontweight = 'bold',
          transform=plt.gca().transAxes,  # Use axes coords (0-1)
             verticalalignment='top', horizontalalignment='left')
plt.text(0.03, 0.9, 'R^2=' "%.2f"%(ssreg/sstot), fontsize=25,fontweight = 'bold',
          transform=plt.gca().transAxes,  # Use axes coords (0-1)
             verticalalignment='top', horizontalalignment='left')
#customize plot
plt.title("LASSO\n", fontsize=30,fontweight = 'bold')
plt.xlabel("Measured IDO Activity\n(pg KYN/cell/day)", fontsize = 30);
plt.ylabel("Predict IDO Activity\n(pg KYN/cell/day)",fontsize = 30);
plt.tick_params(labelsize = 27) 
plt.legend(bbox_to_anchor=(1.3, 1),fontsize = 18, borderaxespad=0)
plt.savefig('../Plot/UT_Fluo_Pred/Fluo_UT_LASSO.svg', format="svg",dpi=300,bbox_inches='tight')  