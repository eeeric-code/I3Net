from opt import *
import torch.nn as nn
import json
from importlib import import_module

def args_add_additinoal_attr(args,json_path):
    dic = json.load(open(json_path,'r',))
    for key,value in dic.items():
        if key == '//':
            continue
        setattr(args,key,value)

def select_model(args):
    opt_path = f'opt/{args.model}.json'
    args_add_additinoal_attr(args, opt_path)
    module = import_module(f'model_zoo.{args.model.lower()}.basic_model')
    model = module.make_model(args)
    return model


