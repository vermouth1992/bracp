import base64
import os.path as osp
import pickle
import sys
import zlib

sys.path.append(osp.join(osp.abspath(osp.dirname(__file__)), '../'))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('encoded_thunk')
    args = parser.parse_args()
    thunk = pickle.loads(zlib.decompress(base64.b64decode(args.encoded_thunk)))
    thunk()
