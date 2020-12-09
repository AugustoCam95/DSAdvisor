import sklearn.datasets
import sklearn.metrics
import os
import autosklearn.regression

X, y = sklearn.datasets.load_boston(return_X_y=True)

X_train, X_test, y_train, y_test = \
    sklearn.model_selection.train_test_split(X, y, random_state=1)

automl = autosklearn.regression.AutoSklearnRegressor(
    time_left_for_this_task=120,
    per_run_time_limit=30,
    tmp_folder='\tmp',
    output_folder= '\final',
)
automl.fit(X_train, y_train, dataset_name='boston')


print(automl.show_models())

predictions = automl.predict(X_test)
print("R2 score:", sklearn.metrics.r2_score(y_test, predictions))