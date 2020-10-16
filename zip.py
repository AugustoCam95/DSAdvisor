import os
import shutil

def create_zip():
    os.chdir("../../..")
    output_filename = "output"
    shutil.make_archive(output_filename, 'zip', "static/uploads")
