import codecs
from collections.abc import Iterable
import pickle
import json


def save_array(arr, path, print_len=False, sort_by=None, descending=False, sep=u'\t', encoding='utf-8'):
    if sort_by:
        arr = sorted(arr, key=lambda x: x[sort_by], reverse=descending)

    with codecs.open(path, 'w', encoding) as f:
        if print_len:
            f.write('{}\n'.format(len(arr)))
        for item in arr:
            if sep and isinstance(item, Iterable):
                f.write('{}\n'.format(sep.join([str(i) for i in item])))
            else:
                f.write('{}\n'.format(item))


def make_file(path):
    save_array([], path)


def save_map(mp, path, reverse_kv=False, sort_by_key=False, **kwargs):
    arr = [(v, k) if reverse_kv else (k, v) for k, v in mp.items()]
    if sort_by_key:
        kwargs['sort_by'] = 0
    save_array(arr, path, **kwargs)


def saveobj(obj, fname):
    with open(fname, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def readobj(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


def to_json(obj):
    return json.dumps(obj, default=lambda o: o.__dict__, sort_keys=False, indent=4)
