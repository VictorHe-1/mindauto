import json
import yaml
import pickle
from pathlib import Path


def list_from_file(filename,
                   prefix='',
                   offset=0,
                   max_num=0,
                   encoding='utf-8'):
    """Load a text file and parse the content as a list of strings.

    Args:
        filename (str): Filename.
        prefix (str): The prefix to be inserted to the beginning of each item.
        offset (int): The offset of lines.
        max_num (int): The maximum number of lines to be read,
            zeros and negatives mean no limitation.
        encoding (str): Encoding used to open the file. Default utf-8.

    Examples:
        >>> list_from_file('/path/of/your/file')
        ['hello', 'world']

    Returns:
        list[str]: A list of strings.
    """
    cnt = 0
    item_list = []
    with open(filename, 'r', encoding=encoding) as f:
        for _ in range(offset):
            f.readline()
        for line in f:
            if 0 < max_num <= cnt:
                break
            item_list.append(prefix + line.rstrip('\n\r'))
            cnt += 1
    return item_list

def load_from_serialized(file, file_format=None, **kwargs):
    """Load data from json/yaml/pickle files.

    This method provides a unified api for loading data from serialized files.

    Args:
        file (str or :obj:`Path` or file-like object): Filename or a file-like
            object.
        file_format (str, optional): If not specified, the file format will be
            inferred from the file extension, otherwise use the specified one.
            Currently supported formats include "json", "yaml/yml" and
            "pickle/pkl".

    Returns:
        The content from the file.
    """
    if isinstance(file, Path):
        file = str(file)
    if file_format is None and isinstance(file, str):
        file_format = file.split('.')[-1]
    if file_format not in ['json', 'yaml', 'yml', 'pickle', 'pkl']:
        raise TypeError(f'Unsupported format: {file_format}')

    if isinstance(file, str):
        with open(file, 'rb' if file_format in ['pickle', 'pkl'] else 'r') as f:
            if file_format in ['json']:
                obj = json.load(f, **kwargs)
            elif file_format in ['yaml', 'yml']:
                obj = yaml.safe_load(f, **kwargs)
            elif file_format in ['pickle', 'pkl']:
                obj = pickle.load(f, **kwargs)
    elif hasattr(file, 'read'):
        if file_format in ['json']:
            obj = json.load(file, **kwargs)
        elif file_format in ['yaml', 'yml']:
            obj = yaml.safe_load(file, **kwargs)
        elif file_format in ['pickle', 'pkl']:
            obj = pickle.load(file, **kwargs)
    else:
        raise TypeError('"file" must be a filepath str or a file-object')
    return obj


def dump(obj, file=None, file_format=None, **kwargs):
    """Dump data to json/yaml/pickle strings or files.

    This method provides a unified api for dumping data as strings or to files,
    and also supports custom arguments for each file format.

    Args:
        obj (any): The python object to be dumped.
        file (str or :obj:`Path` or file-like object, optional): If not
            specified, then the object is dumped to a str, otherwise to a file
            specified by the filename or file-like object.
        file_format (str, optional): Same as :func:`load`.

    Returns:
        bool: True for success, False otherwise.
    """
    if isinstance(file, Path):
        file = str(file)
    if file_format is None:
        if isinstance(file, str):
            file_format = file.split('.')[-1]
        elif file is None:
            raise ValueError('file_format must be specified since file is None')
    if file_format not in ['json', 'yaml', 'yml', 'pickle', 'pkl']:
        raise TypeError(f'Unsupported format: {file_format}')

    if file is None:
        if file_format in ['json']:
            return json.dumps(obj, **kwargs)
        elif file_format in ['yaml', 'yml']:
            return yaml.dump(obj, **kwargs)
        elif file_format in ['pickle', 'pkl']:
            return pickle.dumps(obj, **kwargs)
    elif isinstance(file, str):
        if file_format in ['json']:
            with open(file, 'w') as f:
                json.dump(obj, f, **kwargs)
        elif file_format in ['yaml', 'yml']:
            with open(file, 'w') as f:
                yaml.dump(obj, f, **kwargs)
        elif file_format in ['pickle', 'pkl']:
            with open(file, 'wb') as f:
                pickle.dump(obj, f, **kwargs)
    elif hasattr(file, 'write'):
        if file_format in ['json']:
            json.dump(obj, file, **kwargs)
        elif file_format in ['yaml', 'yml']:
            yaml.dump(obj, file, **kwargs)
        elif file_format in ['pickle', 'pkl']:
            pickle.dump(obj, file, **kwargs)
    else:
        raise TypeError('"file" must be a filename str or a file-object')

    return True
