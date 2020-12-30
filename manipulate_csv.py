import os
import csv
import math
import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from skgof import cvm_test
from scipy import stats
from pylab import savefig
from matplotlib import cm
from collections import Counter
from sklearn import preprocessing
from dython.nominal import associations
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.stattools import medcouple 
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif, f_classif, mutual_info_regression, chi2, f_regression
from info_gain import info_gain
import scipy.stats as ss


def make_dataset(dataset,text):
    start_point = os.getcwd()
    os.chdir(start_point)
    os.chdir("static/samples")
    aux = dataset.head(n = 20)
    aux.to_csv(text+".csv", index = False)
    dataset.to_csv(text+"dataset.csv", index = False)
    os.chdir(start_point)

def shapiro_test(data):
    D,p = st.shapiro(data)
    alpha = 0.05
    
    if p > alpha:
        return 'Sample looks Gaussian (fail to reject H0)'
    else:
        return 'Sample does not look Gaussian (reject H0)'    

def von_misses(data):
    p = cvm_test(data, "norm").pvalue
    alpha = 0.05
    if p > alpha:
        return 'Sample looks Gaussian (fail to reject H0)'
    else:
        return 'Sample does not look Gaussian (reject H0)'

def lillierfos_test(data):
    D,p = sm.stats.diagnostic.lilliefors(data, dist="norm")
    alpha = 0.05
    
    if p > alpha:
        return 'Sample looks Gaussian (fail to reject H0)'
    else:
        return 'Sample does not look Gaussian (reject H0)'
    
    
def normal_test(data):
    _,p = st.normaltest(data)
    alpha = 0.05
    if p > alpha:
        binario = 0
    else:
        binario = 1
        
    return binario

def kolmogorov_test(data, dist_names):
    dist_results = []
    params = {}
    for dist_name in dist_names:
        dist = getattr(st, dist_name)
        param = dist.fit(data)
        params[dist_name] = param
        # Applying the Kolmogorov-Smirnov test
        D, p = st.kstest(data, dist_name, args=param)
        dist_results.append((dist_name, p))

    # select the best fitted distribution
    best_dist, best_p = (max(dist_results, key=lambda item: item[1]))
        
    return best_dist

def all_normal_tests(dataset):
    column1  = []
    column2  = []
    column3  = []
    for col in dataset.columns:
        column1.append(shapiro_test(dataset[col]))
        column2.append(von_misses(dataset[col]))
        column3.append(lillierfos_test(dataset[col]))
    return column1,column2,column3

def best_fit(dist_names,dataset):
    column  = []
    for col in dataset.columns:
        column.append(kolmogorov_test(dataset[col],dist_names))
    return column

def get_vector_of_normality(dataset):
    column  = []
    for col in dataset.columns:
        column.append(normal_test(dataset[col]))
    return column

# 0 columns for string
# 1 columns for int
# 2 columns for float
def get_columns(dataframe):
    dataset = dataframe
    cols1 = []
    cols2 = []
    cols3 = []
    for col in dataset.columns:
        if dataset[col].dtypes == 'object':
            cols1.append(col)
    for col in dataset.columns:
        if dataset[col].dtypes == 'int64':
            cols2.append(col)
    for col in dataset.columns:
        if dataset[col].dtypes == 'float64':
            cols3.append(col)
    return cols1, cols2, cols3


# function to make plots from categorical variables in the dataset
def categorical_plots(df):
    start_point = os.getcwd()
    os.chdir("static/plots")
    for col in df.columns:
        if df[col].dtype != "object":
            df = df.drop(columns = col)

    array = df.to_numpy()
    array = array.transpose()
    
    for i in range(len(array)):
        for j in range(len(array[i])):
            array[i][j] = str(array[i][j])
    array = array.transpose()
    col_string = []
    
    for col in df.columns:
        if df[col].dtype == "object":
            col_string.append(col)
    
    df = pd.DataFrame(array, columns = col_string)
    vetor = len(col_string)*[[]]
    
    for i in range(len(col_string)):
        vetor[i] = list(pd.unique(df[col_string[i]]))
    for i in range(len(vetor)):
        for j in range(len(vetor[i])):
            vetor[i][j] = str(vetor[i][j])
    array = len(col_string)*[[]]
    for i in range(len(col_string)):
        array[i] = len(vetor[i])*[[]]
    for i in range(len(col_string)):
        for j in range(len(vetor[i])):
            array[i][j] = df[ df[ col_string[i] ] == vetor[i][j] ].shape[0]
    
    my_colors = 'rgbkymc'
    
    for i in range(len(array)):
        if len(vetor[i]) > 3:
            pass
        else:    
            fig1, ax1 = plt.subplots(figsize=(8, 8))
            ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            ax1.pie(array[i], labels=vetor[i], autopct='%1.1f%%', shadow = False , startangle=90)
            plt.savefig("pie_"+col_string[i]+"_.jpg", dpi = 500)
            plt.clf()

    for i in range(len(array)):    
        fig1, ax1 = plt.subplots(figsize=(8, 8))
        ax1.bar(vetor[i], height=array[i], color = my_colors )
        plt.savefig("bar_"+col_string[i]+"_.jpg", dpi = 500)
        plt.clf()    
    os.chdir(start_point)
    return vetor

# function to make plots from discrete variables in the dataset
def discrete_plots(df):
    start_point = os.getcwd()
    os.chdir("static/plot_int")
    integers = df
    for col in df.columns:
        if df[col].dtype != "int64":
            integers = integers.drop(columns = col)
    for col in integers.columns:
        plt.xlabel('Values')
        plt.ylabel('Quantities')
        plt.title(r'Histogram of {}'.format(col))
        plt.hist(integers[col])
        plt.savefig("hist_"+col+"_.jpg", dpi = 400)
        plt.clf()
    os.chdir(start_point)

def generate_plots(dist_list,df):
    start_point = os.getcwd()
    os.chdir("static/images")
    for col,dist_ in zip(df.columns,dist_list):
        plt.figure()
        plt.hist(df[col], bins=25, density=True, alpha=0.6, color='g')
        # Plot norm PDF.
        mu, std = st.norm.fit(df[col])
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, len(df))
        p = st.norm.pdf(x, mu, std)
        plt.plot(x, p,"k:", linewidth=2, label = "norm")
        # Plot second PDF.
        dist = getattr(st, dist_)
        param = dist.fit(df[col])
        mu1 = param[len(param)-2] 
        std1 = param[len(param)-1]
        ymin, ymax = plt.xlim()
        y = np.linspace(ymin, ymax, len(df))
        q = dist.pdf(y,*param[:-2] ,loc = mu1, scale = std1)
        plt.plot(y, q, 'r', linewidth = 1, label = dist_)
        plt.legend()
        plt.savefig(col+".jpg")
        plt.clf()
    os.chdir(start_point)

def corrdot_pearson(*args, **kwargs):
    corr_r = args[0].corr(args[1], 'pearson')
    corr_text = round(corr_r, 2)
    ax = plt.gca()
    font_size = abs(corr_r) * 60 + 5
    ax.annotate(corr_text, [.5, .5,],  xycoords="axes fraction", ha='center', va='center', fontsize=font_size)

def corrfunc_pearson(x, y, **kws):
    r, p = stats.pearsonr(x, y)
    p_stars = ''
    if p <= 0.05:
        p_stars = '*'
    if p <= 0.01:
        p_stars = '**'
    if p <= 0.001:
        p_stars = '***'
    ax = plt.gca()
    ax.annotate(p_stars, xy=(0.65, 0.6), xycoords=ax.transAxes, color='red', fontsize= 55)

def corrdot_spearman(*args, **kwargs):
    corr_r = args[0].corr(args[1], 'spearman')
    corr_text = round(corr_r, 2)
    ax = plt.gca()
    font_size = abs(corr_r) * 50 + 15
    ax.annotate(corr_text, [.5, .5,],  xycoords="axes fraction", ha='center', va='center', fontsize=font_size)

def corrfunc_spearman(x, y, **kws):
    r, p = stats.spearmanr(x, y)
    p_stars = ''
    if p <= 0.05:
        p_stars = '*'
    if p <= 0.01:
        p_stars = '**'
    if p <= 0.001:
        p_stars = '***'
    ax = plt.gca()
    ax.annotate(p_stars, xy=(0.65, 0.6), xycoords=ax.transAxes, color='red', fontsize = 55)

def generate_correlations_pearson(dataframe,text):
    start_point = os.getcwd()
    os.chdir("static/samples")
    df = dataframe
    for col in df.columns:
        if df[col].dtypes == 'float64' or df[col].dtypes == 'int64' :
           pass
        else:
            df = df.drop( columns = [col])
    sns.set(style='white', font_scale=1.6)
    g = sns.PairGrid(df, aspect=1.5, diag_sharey=False, despine=False)
    g.map_lower(sns.regplot, lowess=True, ci=False,
                line_kws={'color': 'red', 'lw': 1},
                scatter_kws={'color': 'black', 's': 20})
    g.map_diag(sns.distplot, color='black',
            kde_kws={'color': 'red', 'cut': 0.7, 'lw': 1,'bw': 0.3},
            hist_kws={'histtype': 'bar', 'lw': 2,
                        'edgecolor': 'k', 'facecolor':'grey'})
    g.map_diag(sns.rugplot, color='black')
    g.map_upper(corrdot_pearson)
    g.map_upper(corrfunc_pearson)
    g.fig.subplots_adjust(wspace=0, hspace=0)
    # Remove axis labels
    for ax in g.axes.flatten():
        ax.set_ylabel('')
        ax.set_xlabel('')
    # Add titles to the diagonal axes/subplots
    for ax, col in zip(np.diag(g.axes), df.columns):
        ax.set_title(col, y=0.82, fontsize=26)
    # Save plot    
    g.savefig("../correlations/"+text+"pearson.jpg", dpi = 200)
    os.chdir(start_point)

def generate_correlations_spearman(dataframe, text):
    start_point = os.getcwd()
    os.chdir("static/samples")
    df = dataframe
    for col in df.columns:
        if df[col].dtypes == 'float64' or df[col].dtypes == 'int64' :
            pass
        else:
            df = df.drop( columns = [col])
    sns.set(style='white', font_scale=1.6)
    g = sns.PairGrid(df, aspect=1.5, diag_sharey=False, despine=False)
    g.map_lower(sns.regplot, lowess=True, ci=False,
                line_kws={'color': 'red', 'lw': 1},
                scatter_kws={'color': 'black', 's': 20})
    g.map_diag(sns.distplot, color='black',
            kde_kws={'color': 'red', 'cut': 0.7, 'lw': 1, 'bw': 0.3},
            hist_kws={'histtype': 'bar', 'lw': 2,
                        'edgecolor': 'k', 'facecolor':'grey'})
    g.map_diag(sns.rugplot, color='black')
    g.map_upper(corrdot_spearman)
    g.map_upper(corrfunc_spearman)
    g.fig.subplots_adjust(wspace=0, hspace=0)
    # Remove axis labels
    for ax in g.axes.flatten():
        ax.set_ylabel('')
        ax.set_xlabel('')
    # Add titles to the diagonal axes/subplots
    for ax, col in zip(np.diag(g.axes), df.columns):
        ax.set_title(col, y=0.82, fontsize=26)
    # Save plot    
    g.savefig("../correlations/"+text+"spearman.jpg", dpi = 200)
    os.chdir(start_point)

def corr_cramer_v(data,text):
    start_point = os.getcwd()
    os.chdir("static/samples")
    
    for col in data.columns:
        if data[col].dtypes == 'float64' or data[col].dtypes == 'int64':
            data = data.drop( columns = [col])
    corr = associations(data)
    
    fig, ax = plt.subplots(figsize=(15,10))
    map3 = sns.heatmap(corr['corr'], annot= True,  ax = ax, linewidth=0.5, cmap='YlGnBu')
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    figure3 = map3.get_figure()
    figure3.savefig("../correlations/"+text+"cramer.jpg", dpi = 400)
    os.chdir(start_point)

def sample_csv(dataframe,text):
    start_point = os.getcwd()
    os.chdir("static/samples")
    df = dataframe
    # creation of datatypes.csv
    df_datatypes = pd.DataFrame(df.dtypes)
    df_datatypes.to_csv("aux1.csv")
    to_datatype = pd.read_csv("aux1.csv", names = ["Columns", "Type"])
    to_datatype = to_datatype[1:]
    to_datatype = to_datatype.set_index('Columns')
    to_datatype = to_datatype.swapaxes("index", "columns")
    to_datatype = to_datatype.reset_index()
    to_datatype.to_csv("aux2.csv", index = False)
    to_datatype = pd.read_csv("aux2.csv")
    to_datatype.rename(columns={'index':'Columns'}, inplace=True) 
    to_datatype.to_csv(text+"datatypes.csv", index=False)
    
    # creation descriptive statistics and add cv row
    dataset = df
    string_set = df

    for col in dataset.columns:
        if dataset[col].dtypes == 'object':
            dataset = dataset.drop( columns = [col])

    for col in string_set.columns:
        if string_set[col].dtypes == 'int64' or string_set[col].dtypes == 'float64' :
            string_set = string_set.drop( columns = [col])

    if len(list(string_set.columns)) > 0:
        cat_describe = string_set.describe()
        cat_describe.to_csv(text+"catdescribe.csv")
    
    if len(list(dataset.columns)) > 0:
        describe = dataset.describe()
        std = describe.iloc[[2]]
        mean = describe.iloc[[1]]
        std = std.values
        mean = mean.values
        # print("std:", std)
        # print("mean:", mean)
        cv = []
        for i in range(len(std)):
            cv.append(std[i]/mean[i])
        describe.loc["cv"] = cv[0]
        describe = describe.reindex(["count","mean","std","cv","min","25%","50%","75%","max"])
        describe.to_csv(text+"numericdescribe.csv")

    os.chdir(start_point)


def miss_value(code, special, df, text):
    start_point = os.getcwd()
    os.chdir("static/samples")
    # print("Inside  miss_value function:",code)

    if "nan" in code:
        print("True NAN PASS")
        sum_of_nulls = df.isnull().sum()
        sum_of_nulls.to_csv("temp1.csv")
        temp1 = pd.read_csv("temp1.csv", names = ["Columns","Count of miss values"])
        percent = df.isnull().sum() / len(df) * 100
        percent.to_csv("temp2.csv")
        temp2 = pd.read_csv("temp2.csv", names = ["Columns", "Percent of miss values"])
        temp1["Percent of miss values"] = temp2["Percent of miss values"]
        temp1 = temp1.iloc[1:]
        temp1 = temp1.set_index('Columns')
        temp1 = temp1.swapaxes("index","columns")
        temp1 = temp1.reset_index()
        temp1.to_csv("temp3.csv", index = False)
        temp3 = pd.read_csv("temp3.csv")
        temp3.rename(columns={'index':'Columns'}, inplace=True)
        temp3.to_csv(text+"nanreport.csv", index = False)

    if "empty" in code:
        code = str("")
        miss = [[],[]]
        for col in df.columns:
            miss[0].append(df[df[col] == code].shape[0])
            miss[1].append((df[df[col] == code].shape[0]/df.shape[0])*100)
        miss_shaped = np.reshape(miss, (2, len(df.columns)))
        miss_values = pd.DataFrame(miss_shaped, columns = list(df.columns))
        miss_values.to_csv("temp1x.csv")
        miss_values = pd.read_csv("temp1x.csv")
        miss_values = miss_values.rename(columns = {'Unnamed: 0': 'Columns'}, inplace = False)
        miss_values = miss_values.set_index('Columns')
        miss_values = miss_values.rename(index = { 0:"Count of miss values", 1:"Percent of miss values"})
        miss_values.to_csv(text+"emptyreport.csv")

    

    if special != None:
        code = str(special)
        miss = [[],[]]
        for col in df.columns:
            miss[0].append(df[df[col] == code].shape[0])
            miss[1].append((df[df[col] == code].shape[0]/df.shape[0])*100)
        miss_shaped = np.reshape(miss, (2, len(df.columns)))
        miss_values = pd.DataFrame(miss_shaped, columns = list(df.columns))
        miss_values.to_csv("temp2x.csv")
        miss_values = pd.read_csv("temp2x.csv")
        miss_values = miss_values.rename(columns = {'Unnamed: 0': 'Columns'}, inplace = False)
        miss_values = miss_values.set_index('Columns')
        miss_values = miss_values.rename(index = { 0:"Count of miss values", 1:"Percent of miss values"})
        miss_values.to_csv(text+"specialreport.csv")
    os.chdir(start_point)
    


### List graphics on the page correctly
def get_name_graphics():
    start_point = os.getcwd()
    os.chdir("static/images")
    names = os.listdir()
    os.chdir(start_point)
    return names
    
### divisão dos conjuntos de treino, validação e test / normalização do conjunto de treinamento
def split_and_norm(choiced,df, text, test_percent):
    start_point = os.getcwd()
    le = preprocessing.LabelEncoder()
    for col in df.columns:
        if df[col].dtypes == 'object' :
            df[col] = le.fit_transform(df[col])
    dataset = df.drop(columns= [choiced])
    label = df[[choiced]]
    os.chdir("static/samples")
    X_train, X_test, y_train, y_test = train_test_split(dataset, label, test_size = test_percent)
    X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.5)
    ### Train set
    train_set = X_train.copy()
    train_set[y_train.columns[0]] = y_train
    train_set.to_csv(text+"train_data.csv", index = False)
    ### Validation set
    validation_set = X_validation.copy()
    validation_set[y_validation.columns[0]] = y_validation
    validation_set.to_csv(text+"validation_data.csv", index = False)
    ### Test set
    test_set = X_train.copy()
    test_set[y_test.columns[0]] = y_test
    test_set.to_csv(text+"test_data.csv", index = False)
    z_score = st.zscore(train_set)
    dataset_norm = pd.DataFrame(z_score,columns = train_set.columns)
    aux = dataset_norm.head(n=20)
    aux.to_csv(text+"train_norm_data20.csv", index = False)
    dataset_norm.to_csv(text+"train_norm_data.csv", index = False)
    os.chdir(start_point)
    return X_train,y_train

### drop a column get from filter_categorical.html
def drop_col(not_drop, dataframe, text):
    start_point = os.getcwd()
    df = dataframe
    os.chdir("static/samples")
    if len(not_drop)>0:
        df = df.drop(columns = not_drop)
        df.to_csv(text+"dataset.csv", index = False)
    os.chdir(start_point)
    return df


### IQR ajustado para detecção de outliers
def outliers_value(ys):
    aux = []
    q1, q3 = np.percentile(ys, [25, 75])
    iqr = q3 - q1
    mc = medcouple(ys)
    if mc >= 0:
        lower_bound = q1 - (1.5 * math.exp(-4.0 * mc) * iqr) 
        upper_bound = q3 + (1.5 * math.exp(3 * mc) * iqr)
    else:
        lower_bound = q1 - (1.5 * math.exp(-3.0 * mc) * iqr) 
        upper_bound = q3 + (1.5 * math.exp(4 * mc) * iqr)
    for y in ys:
        if (y > upper_bound) or (y < lower_bound):
            aux.append(y)
    return aux

def outliers_position(ys):
    q1, q3 = np.percentile(ys, [25, 75])
    iqr = q3 - q1
    mc = medcouple(ys)
    if mc >= 0:
        lower_bound = q1 - (1.5 * math.exp(-4.0 * mc) * iqr) 
        upper_bound = q3 + (1.5 * math.exp(3 * mc) * iqr)
    else:
        lower_bound = q1 - (1.5 * math.exp(-3.0 * mc) * iqr) 
        upper_bound = q3 + (1.5 * math.exp(4 * mc) * iqr)
    return np.where((ys > upper_bound) | (ys < lower_bound))

def adjust_iqr(df, text):
    for col in df.columns:
        if df[col].dtypes == 'float64' or df[col].dtypes == 'int64' :
            pass
        else:
            df = df.drop( columns = [col])

    out_posi = len(df.columns)*[[]]
    for col,i in zip(df.columns,range(len(df.columns))):
        out_posi[i] = outliers_position(df[col].values)[0]

    outlier_values = len(df.columns)*[[]]
    for col,i in zip(df.columns,range(len(out_posi))):
        outlier_values[i] = outliers_value(df[col].values)

    my_locations = list(df.columns)
    outlier_value = list(df.columns)

    for i in range(len(outlier_value)):
        outlier_value[i] = outlier_value[i] + " outlier candidate"

    for i in range(len(my_locations)):
        my_locations[i] = my_locations[i] + " line"

    new_df = pd.DataFrame(out_posi)
    new_df = new_df.T
    new_df.columns = my_locations

    new_df2 = pd.DataFrame(outlier_values)
    new_df2 = new_df2.T
    new_df2.columns = outlier_value

    for col in new_df.columns:
        new_df2[col] = new_df[col]

    col = list(new_df2.columns.values)
    col.sort()
    new_df2 = new_df2[col]
    new_df2 = new_df2.replace(np.nan,"")
    start_point = os.getcwd()
    os.chdir("static/samples")
    new_df2.to_csv(text+"outliers.csv", index = False)
    os.chdir(start_point)



# Criacao de boxplots
def create_boxplots(df):
    start_point = os.getcwd()
    os.chdir("static/boxplots")
    for col in df.columns:
        if df[col].dtypes == 'float64' or df[col].dtypes == 'int64' :
            pass
        else:
            df = df.drop( columns = [col])
    for col in df.columns:
        fig1, ax1 = plt.subplots()
        ax1.boxplot(df[col])
        ax1.set_xticklabels([col])
        plt.savefig(col+".jpg")
        plt.clf()
    os.chdir(start_point)

def get_name_boxplots():
    start_point = os.getcwd()
    os.chdir("static/boxplots")
    names = os.listdir()
    os.chdir(start_point)
    return names

# Resample Techniques

def before_reasample(target,text):
    counter = Counter(target.values[:,0])
    min_value = min(counter.items(), key=lambda x: x[1])[1]
    max_value = max(counter.items(), key=lambda x: x[1])[1]
    start_point = os.getcwd()
    os.chdir("static/samples")
    with open(text+'_before.csv', mode='w') as csv_file:
        fieldnames = ['Class', 'Count', 'Percentage']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for k,v in counter.items():
            per = round(v / len(target) * 100, 2)
            writer.writerow({'Class': k, 'Count': v, 'Percentage': per})
    os.chdir(start_point)
    return min_value, max_value

from imblearn.under_sampling import RandomUnderSampler

def after_undersampling(dataset,labels,text):
    rus = RandomUnderSampler(random_state=0)
    data = dataset.values
    target = labels.values
    x_res, y_res = rus.fit_sample(data, target)
    counter = Counter(y_res)
    start_point = os.getcwd()
    os.chdir("static/samples")
    with open(text+'_after_under.csv', mode='w') as csv_file:
        fieldnames = ['Class', 'Count', 'Percentage']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for k,v in counter.items():
            per = round(v / len(target) * 100, 2)
            writer.writerow({'Class': k, 'Count': v, 'Percentage': per})
    os.chdir(start_point)

from imblearn.over_sampling import RandomOverSampler

def after_oversampling(dataset,labels,text):    
    ros = RandomOverSampler(random_state=0)
    data = dataset.values
    target = labels.values
    x_res, y_res = ros.fit_sample(data, target)
    counter = Counter(y_res)
    start_point = os.getcwd()
    os.chdir("static/samples")
    with open(text+'_after_over.csv', mode='w') as csv_file:
        fieldnames = ['Class', 'Count', 'Percentage']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for k,v in counter.items():
            per = round(v / len(target) * 100, 2)
            writer.writerow({'Class': k, 'Count': v, 'Percentage': per})
    os.chdir(start_point)

# Feature Selection
def create_table_feature_selection(dataset, labels, type_problem, text):
    ch,_ =  chi2(dataset,labels)
    if type_problem == "Classification":
        mi = mutual_info_classif(dataset, labels)
        f,_ =  f_classif(dataset, labels)
    else:
        mi = mutual_info_regression(dataset, labels)
        f,_ =  f_regression(dataset, labels)
    ig = []
    igr = []
    for col in dataset.columns:
        ig.append(info_gain.info_gain(dataset[col],labels))
        igr.append(info_gain.info_gain_ratio(dataset[col],labels))
    feature_ranking = pd.DataFrame(ss.rankdata(ch), index = [dataset.columns], columns = ['Ch'])
    feature_ranking["Ig"] = ss.rankdata(ig)
    feature_ranking["Gr"] = ss.rankdata(igr)
    feature_ranking["Mi"] = ss.rankdata(mi)
    feature_ranking["F"] = ss.rankdata(f)
    feature_ranking["Sum"] = feature_ranking.sum(axis=1)
    feature_ranking = feature_ranking.sort_values(by=['Sum'], ascending = False)
    start_point = os.getcwd()
    os.chdir("static/samples")
    feature_ranking.to_csv(text+"_fs.csv")
    os.chdir(start_point)
    count = 0
    for i in feature_ranking['Sum']:
        if i > (feature_ranking['Sum'][0]/2):
            count = count + 1
    columns = feature_ranking['Sum'].index[0:count]
    len(columns)
    list_col = []
    for i in columns:
        list_col.append(i[0])
    return list_col

def filter_on_feature_selection(col_to_remove, text):
    start_point = os.getcwd()
    os.chdir("static/samples")
    dataframe1 = pd.read_csv(text+"train_data.csv")
    dataframe2 = pd.read_csv(text+"validation_data.csv")
    dataframe3 = pd.read_csv(text+"test_data.csv")
    
    dataframe1 = dataframe1.drop(columns = col_to_remove)
    dataframe1.to_csv(text+"train_data.csv", index = False)

    dataframe2 = dataframe2.drop(columns = col_to_remove)
    dataframe2.to_csv(text+"validation_data.csv", index = False)

    dataframe3 = dataframe3.drop(columns = col_to_remove)
    dataframe3.to_csv(text+"test_data.csv", index = False)

    os.chdir(start_point)
