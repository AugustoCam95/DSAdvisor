import os
import pandas as pd
import delete
# import statistics
import warnings
import manipulate_csv
from flask import Flask, render_template, request, flash, send_file, url_for, redirect
from werkzeug.utils import secure_filename


warnings.filterwarnings('ignore') 

delete.delete_trash()

UPLOAD_FOLDER = 'static/uploads/dataset'

app = Flask(__name__)
app.jinja_env.filters['zip'] = zip
app.config["SECRET_KEY"] = 'secret'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = 'csv'


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    delete.delete_trash()
    return render_template("index.html")

log_user_execution = {}
file_name = None
dataframe = None

@app.route('/upload_csv', methods = [ "GET", "POST"])
def upload_csv():
    delete.delete_trash()
    global log_user_execution
    if request.method == 'POST':
        log_user_execution["Dataset details"] = request.form["text_details"]
        if 'file' not in request.files:
            flash('Nenhum arquivo encontrado')
            return redirect(url_for('upload_csv'))
        
        file = request.files['file']
        
        if file.filename == "":
            flash('Nenhum arquivo selecionado')
            return redirect(url_for('upload_csv'))
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            global file_name
            global dataframe
            file_name = filename.split(".")[0]
            log_user_execution["file_name"] = file_name
            # statistics.statistics_Creator(file.filename)
            # jupyter.jupyter_Creator()
            # zip.create_zip()
            dataframe = pd.read_csv("static/uploads/dataset/"+filename, index_col=False, keep_default_na=False)
            manipulate_csv.make_dataset(dataframe,file_name)
            columns = dataframe.columns
            if len(columns) == 0:
                signal = True
            else:
                signal = False
            return render_template("upload_csv.html", message = "Success to upload", filename = file_name, signal = signal, path ="static/samples/"+file.filename )

        
    return render_template("upload_csv.html", message = "Waiting for upload")

@app.route('/check_variables_type', methods = ["GET", "POST"])
def check_variables_type():
    if request.method == "POST":
        global dataframe
        global log_user_execution
        drop_values = request.form.getlist("checkbox")
        log_user_execution["removed variables"] = drop_values
        dataframe = manipulate_csv.drop_col(drop_values, dataframe, file_name)
        return render_template("check_variables_type.html" , message = "Success to choice" , list_x = drop_values)

    return render_template("check_variables_type.html", message = "Waiting for choice" , string = manipulate_csv.get_columns(dataframe)[0], float = manipulate_csv.get_columns(dataframe)[2], int = manipulate_csv.get_columns(dataframe)[1] )

choices_miss = None
special_code = None

@app.route('/filter_miss_values', methods = [ "GET", "POST"])
def filter_miss_values():
    if request.method == "POST":
        global choices_miss
        global special_code
        global log_user_execution
        choices_miss = request.form.getlist("checkbox")
        log_user_execution["missing values codes"] = choices_miss
        
        if "other_code" in choices_miss:
            choices_miss.remove("other_code")
            special_code = request.form["text123"]
            log_user_execution["special missing value code"] = special_code
            manipulate_csv.miss_value(choices_miss,special_code,dataframe, file_name)
            manipulate_csv.sample_csv(dataframe, file_name)
            choices_miss.append("other_code")
        else:
            manipulate_csv.miss_value(choices_miss,None,dataframe, file_name)
            manipulate_csv.sample_csv(dataframe, file_name)
            

        return render_template("filter_miss_values.html" , message = "Success to choice", miss_code = choices_miss )
    return render_template("filter_miss_values.html", message = "Waiting for choice")


@app.route('/descriptive_statistics', methods = [ "GET"])
def descriptive_statistics():
    dataset = dataframe
    string_set = dataframe

    for col in dataset.columns:
        if dataset[col].dtypes == 'object':
            dataset = dataset.drop( columns = [col])

    for col in string_set.columns:
        if string_set[col].dtypes == 'int64' or string_set[col].dtypes == 'float64' :
            string_set = string_set.drop( columns = [col])
    
    signal_1 = 0
    signal_2 = 0
    signal_3 = 0

    if len(list(dataset.columns)) > 0:
        signal_1 = 1

    if len(list(string_set.columns)) > 0:
        signal_2 = 1

    if len(choices_miss) > 0:
        signal_3 = 1

    return render_template("descriptive_statistics.html", signal_1 = signal_1, signal_2 = signal_2, signal_3 = signal_3 ,filename = file_name, choices_miss = choices_miss, special_code = special_code)

@app.route('/plot_variables', methods = [ "GET"])
def plot_variables():
    lst = []
    temp1 = []
    temp2 = []
    list_images = []
    for col in dataframe.columns:
        if dataframe[col].dtype == "int64":
            temp1.append(col)
        if dataframe[col].dtype == "object":
            temp2.append(col)
    if len(temp1)>0:
        list_images = manipulate_csv.discrete_plots(dataframe)
    if len(temp2)>0:
        cat_col_num, bar_cat_images, pie_images = manipulate_csv.categorical_plots(dataframe)
        for i in range(len(cat_col_num)):
            d = {}
            d['temp2'] = temp2[i]
            d['bar'] = bar_cat_images[i]
            d['pie'] = pie_images[i]
            d['cat_col'] = cat_col_num[i]
            lst.append(d)
    if (len(temp1)>0 and len(temp2)>0):
        return render_template("plot_variables.html", temp1 = temp1, temp2 = temp2 , elements = lst , list_images= list_images)
    if (len(temp1)==0 and len(temp2)>0):
        return render_template("plot_variables.html", temp1 = temp1, temp2 = temp2 , elements = lst )
    if (len(temp1)>0):
        return render_template("plot_variables.html", temp1 = temp1, list_images= list_images)
    if (len(temp1) == 0 and len(temp2) == 0):
        return render_template("plot_variables.html", temp1 = temp1, temp2 = temp2 )


lazy_dist = ['crystalball', 'johnsonsb', 'burr', 'fisk', 'exponweib', 'powerlognorm', 'johnsonsu',
                'kappa4', 'vonmises_line', 'vonmises', 'ncx2', 'gausshyper', 'argus', 'genexpon',
                'ncf', 'genextreme', 'gengamma', 'kappa3', 'ksone', 'skewnorm', 'powernorm', 'trapz',
                'burr12', 'kstwobign', 'exponpow', 'halfgennorm', 'gompertz', 'triang', 'genhalflogistic', 
                'mielke', 'rice']

not_lazy = ['alpha', 'anglit', 'arcsine', 'beta', 'betaprime', 'bradford', 'cauchy', 'chi', 'chi2',
                'cosine','dgamma',  'dweibull', 'erlang', 'expon', 'exponnorm', 'f', 'fatiguelife', 'foldcauchy',
                'foldnorm', 'gamma', 'genlogistic', 'gennorm', 'genpareto', 'gilbrat',
                'gumbel_l', 'gumbel_r', 'halfcauchy', 'halflogistic', 'halfnorm', 'hypsecant', 'invgamma',
                'invweibull', 'laplace', 'levy', 'levy_l', 'loggamma', 'logistic', 'loglaplace', 'lognorm',
                'loguniform', 'lomax', 'maxwell', 'moyal', 'nakagami', 'norm', 'pareto', 'pearson3',
                'powerlaw', 'rayleigh', 'rdist', 'semicircular', 't', 'truncexpon', 'uniform', 'wald',
                'weibull_max', 'weibull_min','wrapcauchy']

dist_names = None

@app.route('/distribution_analysis_part_1', methods = [ "GET", "POST"])
def distribution_analysis_part_1():
    if request.method == "POST":
        dist_list = request.form.getlist("checkbox")
        global dist_names
        global log_user_execution
        log_user_execution["selected_ditribution_to_best_fit_method"] = dist_list
        dist_names = dist_list
        return render_template("distribution_analysis_part_1.html" , message = "Success to choice" , list_x = dist_list)

    global lazy_dist
    global not_lazy
    return render_template("distribution_analysis_part_1.html", message = "Waiting for choice"  , lazy_dist =  lazy_dist, not_lazy = not_lazy )


user_choice_dist = None
@app.route('/distribution_analysis_part_2',  methods = [ "GET", "POST"])
def distribution_analysis_part_2():
    if request.method == "POST":
        global user_choice_dist 
        global log_user_execution
        user_choice_dist = []
        df = dataframe
        for col in dataframe.columns:
            if dataframe[col].dtype != "float64":
                df = df.drop(columns = col)
        for i in range(len(df.columns)):
            user_choice_dist.append(request.form["radio"+str(i)])
        log_user_execution["users_distribution_selected"] = user_choice_dist
        return render_template("distribution_analysis_part_2.html" , message = "Success to choice", user_answer= user_choice_dist)

    df = dataframe
    for col in dataframe.columns:
        if dataframe[col].dtype != "float64":
            df = df.drop(columns = col)

    temp = []
    for col in dataframe.columns:
        if dataframe[col].dtype == "float64":
            temp.append(col)

    aux = temp
    aux2 = manipulate_csv.get_vector_of_normality(df)
    aux3 = manipulate_csv.best_fit(dist_names,df)
    aux4 = manipulate_csv.all_normal_tests(df)[0]
    aux5 = manipulate_csv.all_normal_tests(df)[1]
    list_plots = manipulate_csv.generate_plots(aux3,df)
    lst = []
    for i in range(len(list_plots)):
        d = {}
        d['name'] = aux[i]
        d['test'] = aux2[i]
        d['shap'] = aux4[i]
        d['lillie'] = aux5[i] 
        d['best_fit'] = aux3[i].upper()
        d['base64'] = list_plots[i]
        lst.append(d)
        
    return render_template("distribution_analysis_part_2.html", message = "Waiting for choice" , elements = lst, num = len(dataframe.columns))


@app.route('/correlations', methods = [ "GET"])
def correlations():
    df = dataframe
    string = dataframe
    for col in dataframe.columns:
        if dataframe[col].dtype != "float64":
            df = df.drop(columns = col)
        if dataframe[col].dtype != "object":
            string = string.drop(columns = col)
    
    signal_2 = 0
    cramer_64 = None 
    pearson_64 = None
    spearman_64 = None
    if len(string.columns) >1:
        cramer_64 = manipulate_csv.corr_cramer_v(string,file_name)
        signal_2 = 1
    
    j = 0
    if "norm" in user_choice_dist:
        for i in range(len(user_choice_dist)):
            if user_choice_dist[i] == "norm":
                j = j+1

    signal_1 = 0
    if j > 1:
        signal_1 = 1
        df_norm = df
        for choice,col in zip(user_choice_dist,df_norm):
            if choice != "norm":
                df_norm = df_norm.drop(columns = col)                
        pearson_64 = manipulate_csv.generate_correlations_pearson(df_norm,file_name)
    
    spearman_64 = manipulate_csv.generate_correlations_spearman(df,file_name)
    
    return render_template("correlations.html",  filename = file_name, signal_1 = signal_1, signal_2 = signal_2, cramer_64 = cramer_64, pearson_64 = pearson_64, spearman_64 = spearman_64)


type_problem = None
X_train =  None
y_train = None   
X = None 
y = None
@app.route('/problem_setup_part_1', methods = [ "GET", "POST"])
def problem_setup_part_1():
    if request.method == "POST":
        global X_train, y_train, type_problem, log_user_execution, X, y 
        dependent_variable = request.form["radiobutton"]
        test_percent = float(request.form["radiobutton2"])
        type_problem = request.form["radiobutton3"]
        log_user_execution["dependent_variable"] = dependent_variable
        log_user_execution["test_size_percent"] = test_percent
        log_user_execution["problem_type"] = type_problem
        X_train, y_train, X, y = manipulate_csv.split_and_norm(dependent_variable, dataframe, file_name, test_percent)
        return render_template("problem_setup_part_1.html" , message = "Success to choice" ,user_answer= dependent_variable, train_percent = test_percent, user_answer2 = type_problem)

    return render_template("problem_setup_part_1.html", message = "Waiting for choice" ,columns = dataframe.columns)


@app.route('/outlier_report', methods = [ "GET"])
def outlier_report():
    boxplot_list = []
    boxplot_list = manipulate_csv.create_boxplots(X_train)
    list_names = X_train.columns
    manipulate_csv.adjust_iqr(X_train, file_name)
    out_posi = len(X_train.columns)*[[]]
    for col,i in zip(X_train.columns, range(len(X_train.columns))):
        out_posi[i] = manipulate_csv.outliers_position(X_train[col].values)[0]
    lst = []
    for col,i in zip(X_train.columns,range(len(X_train.columns))):
        d = {}
        d["name"] = list_names[i]
        d["img"] = boxplot_list[i]
        d["elements"] = len(X_train[col])
        d["sum_outliers"] = len(out_posi[i])
        d["percent_outliers"] = round((len(out_posi[i])*100)/len(X_train[col]),2)
        lst.append(d)
    
    return render_template("outlier_report.html", elements = lst , num = len(X_train.columns))


@app.route('/table_outlier', methods = [ "GET"])
def table_outlier():
    return render_template("table_outlier.html",  path1 ="static/samples/"+file_name+"outliers.csv")


list_col = None
@app.route('/normalization', methods = [ "GET", "POST"])
def normalization():
    if request.method == "POST":
        global list_col, log_user_execution
        anwser = request.form["radiobutton"]
        log_user_execution["normalization"] = anwser
        list_col = manipulate_csv.create_table_feature_selection(X_train, y_train, type_problem, file_name)
        return render_template("normalization.html", message = "Sucess", anwser = anwser )
    return render_template("normalization.html", message = "Waiting for choice", path1 ="static/samples/"+file_name+".csv", path2 ="static/samples/"+file_name+"train_norm_data20.csv"  )


@app.route('/feature_selection', methods = [ "GET", "POST"] )
def feature_selection():
    if request.method == "POST":
        global log_user_execution
        selected_variables = request.form.getlist("checkbox")
        log_user_execution["feature_selection_variables"] = selected_variables
        drop_variables = list(set(X_train.columns) - set(selected_variables))
        manipulate_csv.filter_on_feature_selection(drop_variables, file_name)
        return render_template("feature_selection.html", message = "Sucess", type_problem = type_problem, selected_variables = selected_variables)
    return render_template("feature_selection.html", message = "Waiting for choice", filename = file_name, variables = list(set(X_train.columns) - set(list_col)), list_col = list_col)


@app.route('/resemple_techniques', methods = [ "GET", "POST"])
def resemple_techniques():
    global log_user_execution, X_train, y_train
    manipulate_csv.before_reasample(y_train,file_name)
    if request.method == "POST":
        resampling_choice = request.form["radiobutton"]
        if resampling_choice == "oversampling":
            log_user_execution["resample_technique_choiced"] = "SMOTE(random_state=42)"
            manipulate_csv.after_oversampling(X_train, y_train,file_name)
            return render_template("resemple_techniques.html" , message = "Success to choice" , resampling_choice = resampling_choice, path2 = "static/samples/"+file_name+"_after_over.csv")
        if resampling_choice == "undersampling":
            log_user_execution["resample_technique_choiced"] = "RandomUnderSampler(random_state=42)"
            manipulate_csv.after_undersampling(X_train, y_train,file_name)
            return render_template("resemple_techniques.html" , message = "Success to choice" , resampling_choice = resampling_choice, path2 = "static/samples/"+file_name+"_after_under.csv")
        if resampling_choice == "without":
            return render_template("resemple_techniques.html" , message = "Success to choice" , resampling_choice = resampling_choice, path2 = "static/samples/"+file_name+"_before.csv")
    return render_template("resemple_techniques.html", message = "Waiting for choice", path1 = "static/samples/"+file_name+"_before.csv")


dict_exec_models = None
@app.route('/generate_models', methods = [ "GET", "POST"])
def generate_models():
    global type_problem, log_user_execution, X, y, dict_exec_models 
    if request.method == "POST":
        
        predictive_alg_list = request.form.getlist("checkbox")
        metrics_list = request.form.getlist("checkbox2")
        
        log_user_execution["predictive_alg_list"] = predictive_alg_list
        log_user_execution["metrics_list"] = metrics_list
        
        dict_exec_models = manipulate_csv.generate_models(X, y, log_user_execution)
        return render_template("generate_models.html", message = "Success to choice", user_answer1 = predictive_alg_list,  user_answer2 = metrics_list)

    return render_template("generate_models.html", message = "Waiting for choice", type_problem = type_problem)


@app.route('/metrics', methods = [ "GET"])
def metrics():
    global log_user_execution, dict_exec_models
    manipulate_csv.convertdict(log_user_execution)
    return render_template("metrics.html", dict_exec_models = dict_exec_models,  metrics = log_user_execution["metrics_list"])

@app.route('/reproducibility', methods = [ "GET"])
def reproducibility():
    return render_template("reproducibility.html")

@app.route('/return_files_choices/', methods = [ "GET"])
def return_files_choices():
	return send_file('static/samples/AllConfigurations.txt', attachment_filename= file_name+'AllConfigurations.txt')

@app.route('/return_files_train/', methods = [ "GET"])
def return_files_train():
	return send_file('static/samples/'+file_name+'train_data.csv', attachment_filename= file_name+'train_data.csv')

@app.route('/return_files_test/', methods = [ "GET"])
def return_files_test():
	return send_file('static/samples/'+file_name+'test_data.csv', attachment_filename= file_name+'test_data.csv')


if __name__ == "__main__":    
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug= True)
    
