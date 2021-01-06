import numpy as np
import pandas as pd
from IPython.display import clear_output,display, Markdown, Latex
import scikit_posthocs as sp
from scipy import stats
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.style as style; style.use('seaborn-muted')
from matplotlib.patches import Ellipse

import itertools
from statsmodels.stats.outliers_influence import variance_inflation_factor 
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor, BaggingRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score, learning_curve






def data_describe(data,ret=False):
    '''
    Function to enhance Pandas Describe function with additional information
    '''
    df_description = data.describe().T
    df_description['Median'] = [round(data[vals].median(),4) for vals in df_description.index]
    df_description['Range'] = [round(data[vals].max() - data[vals].min(),4) for vals in df_description.index]
    df_description['IQR'] = [round(data[vals].quantile(0.75) - data[vals].quantile(0.25),4) for vals in df_description.index]
    df_description['Kurtosis'] = [round(data[vals].kurt(),4) for vals in df_description.index]
    df_description['Skewness'] = [round(data[vals].skew(),4) for vals in df_description.index]
    df_description['MeanAbsoluteDeviation'] = [round(data[vals].mad(),4) for vals in df_description.index]
    df_description['Num_Outliers'] = [ (len(data) - len(sp.outliers_iqr(data[vals]))) for vals in df_description.index]
    if (ret==False) :
        display(df_description)
    else:
        return(df_description)

def print_message(message):
    print(message)

def find_na(data,ret=False):
    columns_to_show = data.columns
    na_df = pd.DataFrame(pd.DataFrame(data={'NANs':list(data[columns_to_show].isna().sum()),
                        'Zeros':[data[data[i] == 0 ][i].count() for i in columns_to_show],
                        'Negatives':[data[data[j] < 0 ][j].count() for j in columns_to_show]},
                        index=columns_to_show))
    if ret == False:
        display(na_df)
    else:
        return na_df

## Function to create the list of continuous and categorical variables.
def create_variable_list(df,min_cats,stats=False):
    '''
    Function for categoring the continuous and categorical columns 
    Usage: 
        create_variable_list(df,min_cats,stats):
        (1) df : Pandas data frame.
        (2) min_cats : Threshold for a variables to be deemed as a Categorical Variable
        (3) stats :( dafult=False) Return a table of variable name, variable type, number of unique values and list of unique values.
                    
        Return : Returns two lists each with categorical and continuous variables respectively.
                 Also if stats=True, then return a table of data types
        
    '''
    categorical =[]
    continuous = []
    objects = []
    var_df = pd.DataFrame(columns=['Variable',
                                       'Type',
                                       'Categorical_Class',
                                       'Uniques',
                                       'N-Uniques'])
    
    for col in df.columns:
        if (df[col].dtype.name == 'int64' or df[col].dtype.name == 'float64'):
            if df[col].nunique() > min_cats :
                continuous.append(col)
            else:
                categorical.append(col)
        elif (df[col].dtype.name == 'category'):
            categorical.append(col)
        else:
            objects.append(col)
            
    if stats == True : 
        
        for cats in categorical:
            if df[cats].nunique() == 2 :
                cat_class = 'Binary'
            else:
                cat_class = 'Multi'
            var_df = var_df.append({'Variable' : cats,
                                    'Type' :'Categorical',
                                    'Categorical_Class':cat_class,
                                    'Uniques': df[cats].unique(),
                                    'N-Uniques': len(df[cats].unique())},
                                   ignore_index=True)
            
        for conts in continuous:
            var_df = var_df.append({'Variable' : conts,
                                    'Type' :'Numeric',
                                    #'Uniques': df[conts].unique(),
                                    'N-Uniques': len(df[conts].unique())},
                                   ignore_index=True)
        for obs in objects:
            var_df = var_df.append({'Variable' : obs,
                                    'Type' :'Objects',
                                    #'Uniques': df[conts].unique(),
                                    'N-Uniques': len(df[obs].unique())},
                                   ignore_index=True)
        return categorical,continuous,var_df
    else:
        return categorical,continuous



def correlation_significance(df,target_var,alpha=0.05):
    p_col,coef_col,feature=[],[],[]
    for cols in df.drop(target_var,axis=1).columns:
        pearson_coef, p_value=stats.pearsonr(df[cols], df[target_var])
        if p_value < alpha :
            feature.append(cols)
            p_col.append(p_value)
            coef_col.append(pearson_coef)

    pearson_corr_df = pd.DataFrame({'Features':feature,
                    'Pearson_Coefficients':coef_col,
                    'P_Value':p_col}).sort_values(by='Pearson_Coefficients',ascending=False)

    pearson_corr_df.set_index('Features',inplace=True)
    return pearson_corr_df

def get_correlation_matrix(df,alpha=0.05):
    p_col,coef_col,feature=[],[],[]
    for cols in itertools.combinations(df.columns,2):
        pearson_coef, p_value=stats.pearsonr(df[cols[0]], df[cols[1]])
        if p_value < alpha :
            feature.append(cols)
            p_col.append(p_value)
            coef_col.append(pearson_coef)

    pearson_corr_df = pd.DataFrame({'Features':feature,
                    'Pearson_Coefficients':coef_col,
                    'P_Value':p_col}).sort_values(by='Pearson_Coefficients',ascending=False)

    pearson_corr_df.set_index('Features',inplace=True)
    return pearson_corr_df


def get_VIF_Table(df):
    '''
    Function to get the Variance Inflation Factor from the data frame:
    Usage: get_VIF_Table(pd.DataFrame)
    (1) df : Dataframe
    
    Return:
        Returns the Table containing the list of column names or features and VIF for each of them.
    '''
    X = df
    vif_data = pd.DataFrame() 
    vif_data["feature"] = df.columns 
    vif_data["VIF"] =[variance_inflation_factor(df.values, i) for i in range(len(df.columns))]
#     vif_data['VIF'] = vif_data['VIF'].map(lambda a: '%2.2f' % a) 
    return vif_data

def remove_multicollinearity(df,target_var,threshold=5.0,depth=None,ret_vif_table=False):
    if target_var in df.columns:
        X_vif = df.drop(target_var,axis=1)
    else:
        X_vif = df

    vif_df = get_VIF_Table(X_vif)
    high_vif = list(vif_df[vif_df['VIF'] > threshold].sort_values(by='VIF',ascending=False)['feature'])
    if depth == None:
        depth = len(high_vif)
    else:
        pass
    n_features = 1
    break_loop = 0
    while ((break_loop == 0) & (n_features <= depth )) :
        # print(n_features, ((break_loop == 0) | (n_features <= depth )))
        for hvs in itertools.combinations(high_vif,n_features):
            # print(hvs)
            # clear_output(wait=False)
            vif_df = get_VIF_Table(X_vif.drop(list(hvs),axis=1))
            if  vif_df['VIF'].max() < threshold:
                break_loop = 1
                break
            else:
                continue
        n_features += 1
    if ret_vif_table == False:
        return(df[vif_df['feature']])      
    else:
        return(vif_df)


    


def get_Normality_Check(df):
    '''
    This function uses Shpiro-Wilk method to test Normality at an alpha of 0.05.
    Ho: The sample  data is Normally distributed.
    H1: The sample data is not Normally distributed.
    Usage:
        1) df : Data Frame with all features who's normality needs to be tested.
    Returns:
        1) DataFrame with Features, Normality (yes/No) and corresponding P_value as columns.
    '''
    from scipy.stats import shapiro

    alpha = 0.05
    normal,p_value,cols=[],[],[]
    for i in df.columns:
        if (df[i].dtype.name == 'int64' or df[i].dtype.name == 'float64'):
            data = df[i]
            if shapiro(data)[1] > alpha :
                normal.append('Yes')
            else:
                normal.append('No')
            p_value.append(shapiro(data)[1],)
            cols.append(i)
        else:
            pass
    normality_df = pd.DataFrame({'Features':cols,'Normality':normal,'P-Value':p_value})
    normality_df.set_index('Features',inplace=True)
    return(normality_df)

def show_multivar_corr_plot(df):
    columns_to_show = df.columns
    plots = [_ for _ in itertools.combinations(columns_to_show,2)]
    num_cols=2
    for i in range(0,len(plots),2):
        fig, ax = plt.subplots(1,num_cols,figsize=(20,4)) 
        
        sns.regplot(plots[i][0], 
                plots[i][1], 
                data=df, 
                fit_reg=1,
                line_kws={'color': 'red'},
                ax=ax[0]);
        ax[0].set_title(f"\n({i+1}) {plots[i][0]} / {plots[i][1]} \n ρ = {round(df[plots[i][0]].corr(df[plots[i][1]]),2)}\n",fontdict=dict(fontsize=20));
        
        if (len(plots)-i) >=2 :
            sns.regplot(plots[i+1][0], 
                    plots[i+1][1], 
                    data=df, 
                    fit_reg=1,
                    line_kws={'color': 'red'},
                    ax=ax[1]);
            ax[1].set_title(f"\n({i+2}) {plots[i+1][0]} / {plots[i+1][1]} \n ρ = {round(df[plots[i+1][0]].corr(df[plots[i+1][1]]),2)}\n",fontdict=dict(fontsize=20));
            plt.show()

    # clear_output(wait=True)         
    # fig.show();
    return

def show_variance_threshold(df,target_var):
    from sklearn.feature_selection import VarianceThreshold

    columns_to_show = df.drop(target_var,axis=1).columns
    vthreshold = VarianceThreshold(threshold=0.0)
    vthreshold.fit(df[columns_to_show].values)
    vt_df = pd.DataFrame({'Feature':columns_to_show,
                        'Variance Threshold':vthreshold.variances_}
                        ).sort_values(by='Variance Threshold',ascending=False)
    sns.barplot(y=vt_df['Feature'],x=vt_df['Variance Threshold'],orient='h');
    display(Markdown("### <center> <u> Variance Threshold </u> </center>"));
    return

def show_corr_target(df,target_var):
    columns_to_show = df.drop(target_var,axis=1).columns

    if (df[target_var].dtype.name == 'object' or df[target_var].dtype.name == 'category'):
        y = df[target_var].astype('category').cat.codes
    else:
        y=df[target_var]
        
    correlations = df[columns_to_show].corrwith(y,method='pearson').to_frame()
    sorted_correlations = correlations.sort_values(0,ascending=False)
    fig, ax = plt.subplots(figsize=(5,10))
    sns.heatmap(sorted_correlations, cmap='coolwarm', annot=True, vmin=-1, vmax=1, ax=ax);
    plt.title(f"\nCorrelation of '{target_var.upper()}' with other predictors\n",fontsize=20);
    return  


def basemodel_score(X,y,score_type='r2'):
    random_state = 123
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = random_state)
    basemodel = ExtraTreesRegressor(random_state = random_state)
    kf = KFold(n_splits = 10, random_state = random_state,shuffle=True)
    cv_results = cross_val_score(basemodel, X_train, y_train, cv = kf, scoring = 'r2')
    base_score_mean = round(cv_results.mean(),4)
    base_score_std = round(cv_results.std(),4)
    return(base_score_mean,base_score_std)

def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    
    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))

def plot_gmm(gmm, X, label=True, ax=None):
    cols = X.columns
    X = X.values
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    ax.axis('equal')
    ax.set_xlabel(cols[0])
    ax.set_ylabel(cols[1])
    
    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor,ax=ax)