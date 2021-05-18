# import os
# import pandas as pd
# from pandas_profiling import ProfileReport

# def statistics_Creator(file_name):
#     os.chdir("static/uploads/dataset")
#     df = pd.read_csv(file_name)
#     profile = ProfileReport(df, title=file_name + ' report', html={'style': {'full_width': True}})
#     os.chdir("../../..")
#     profile.to_file(output_file="templates/dataset.html")