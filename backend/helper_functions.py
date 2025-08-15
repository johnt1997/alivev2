import os
import shutil


def clear_dir(directory_path: str):
    """Clears a directory with all its subdirectories and files"""
    if os.path.exists(directory_path):
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)

            if os.path.isfile(file_path) and filename != ".gitignore":
                os.remove(file_path)

            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)


def string_to_bool(string: str):
    return string == "True"


def get_document_name(document_path: str):
    return document_path.split("/")[-1]


def delete_dir(directory_path: str):
    """Delete a directory with all its subdirectories and files"""
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)
