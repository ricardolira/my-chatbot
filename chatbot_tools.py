"""Chatbot tutorial."""
import numpy as np
import tensorflow as tf
import re
import time


# Importing the dataset
def import_dataset(data_path, encoding='utf-8', errors='ignore'):
    """Import data from file path nad return parsed data."""
    with open(
        data_path, encoding=encoding, errors=errors
    ) as data:
        return_data = data.read().split('\n')
    data.close()
    return return_data

def map_lines(all_lines):
    """Create a dict contaning all dialogs mapped to its id."""
    lines_splitter = re.compile(r'\+{3}\$\+{3}')
    lines_dict = dict()
    for line in all_lines:
        key = re.split(lines_splitter, line)[0].strip()
        value = re.split(lines_splitter, line)[-1].strip()
        lines_dict.update({key: value})
    return lines_dict
