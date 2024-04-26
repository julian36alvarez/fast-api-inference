# -*- coding: utf-8 -*-
"""
This module provides utility functions for working with file paths.

Functions:
- make_path(*args): returns an absolute path given a list of path components
- make_item_list(*args): returns a list of absolute file paths in a specified folder
- make_directories_list(*args): returns a list of absolute directory paths in a specified folder
"""

import os


def make_path(*args):
    """function to make an absolute path

    Returns:
        data_dir(str):a path with the input arguments
    """ """"""
    current_dir = os.getcwd()
    new_dir = os.path.join(current_dir, os.pardir, *args)
    return new_dir


def make_item_list(*args, file_ext=None):
    """return the items in a specified path in this

    Args:
        file_ext (str, optional): The file extension to filter by. If None, all files are returned.

    Returns:
        item(list): a list with all absolute file path in the specified folder
    """
    data_dir = make_path(*args)
    if file_ext:
        items = [
            os.path.join(data_dir, item)
            for item in os.listdir(data_dir)
            if os.path.isfile(os.path.join(data_dir, item)) and item.endswith(file_ext)
        ]
    else:
        items = [
            os.path.join(data_dir, item)
            for item in os.listdir(data_dir)
            if os.path.isfile(os.path.join(data_dir, item))
        ]
    return items


def make_directories_list(*args):
    """return the directories in a specified path

    Returns:
        item(list): a list with all absolute directories paths in the specified folder
    """
    data_dir = make_path(*args)
    directories = [
        os.path.join(data_dir, item)
        for item in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, item))
    ]
    return directories


def get_directories(*args):
    """wrapper function to get directories in a specified path

    Returns:
        directories(list): a list with all absolute directories paths in the specified folder
    """
    directories = list(make_directories_list(*args))
    return directories
