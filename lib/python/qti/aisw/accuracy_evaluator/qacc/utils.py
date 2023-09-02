##############################################################################
#
# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# All data and information contained in or disclosed by this document are
# confidential and proprietary information of Qualcomm Technologies, Inc., and
# all rights therein are expressly reserved. By accepting this material, the
# recipient agrees that this material and the information contained therein
# are held in confidence and in trust and will not be used, copied, reproduced
# in whole or in part, nor its contents revealed in any manner to others
# without the express written permission of Qualcomm Technologies, Inc.
#
##############################################################################
import os
import sys
import yaml
import json

import qti.aisw.accuracy_evaluator.common.defaults as df
import qti.aisw.accuracy_evaluator.common.exceptions as ce
from qti.aisw.accuracy_evaluator.common.utilities import Helper, ModelType

defaults = df.Defaults.getInstance()

def check_model_dir(config_file):
    """
    Checks if model path or dir is given
    """
    with open(config_file, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise ce.ConfigurationException('incorrect configuration file', exc)

    is_model_dir = False
    try:
        model_path = config['model']['inference-engine']['model_path']
        if os.path.isdir(model_path):
            is_model_dir = True
    except Exception as e:
        raise ce.ConfigurationException(f'Failed to read the config file', exc)

    return is_model_dir, model_path

def process_model_dir(model_dir_path, parent_work_dir):
    models = []
    work_dirs = []
    preproc_files = []
    for root, dirs, files in os.walk(model_dir_path):
        for f in files:
            if (f.endswith(".onnx") or f.endswith(".pb") or
               f.endswith(".tflite") or f.endswith(".pt")):
                models.append(os.path.join(root , f))
                model_dir_name = root.split("/")[-2]
                model_work_dir = os.path.join(parent_work_dir, model_dir_name)
                work_dirs.append(model_work_dir)

    for model_path in models:
        curr_dir = os.path.dirname(os.path.dirname(model_path)) # Going to the parent directory
        data_list = os.path.join(curr_dir, "inputs", "input_list.txt")
        if os.path.isfile(data_list):
            preproc_files.append(data_list)
        else:
            raise ce.ConfigurationException(f'input_list.txt not present in the {curr_dir} -', exc)

    return models, preproc_files, work_dirs

def create_default_config(work_dir, model_path, backend, target_arch, comparator, tol_thresh,
                          input_info=None, act_bw=None, bias_bw=None, box_input=None):
    """
    Create temp config file from the command line parameters
    """
    temp_dir = os.path.join(work_dir, "temp")
    os.makedirs(temp_dir)
    temp_config_file = os.path.join(temp_dir, "temp_config.yaml")
    temp_config = {"model": {"info": {"desc": "Default Config"}, "inference-engine": None}}
    temp_config["model"]["inference-engine"] = {"model_path": None, "comparator": None, "platforms": None}

    if input_info is not None:
        temp_config["model"]["inference-engine"]["inputs_info"] = []
        for inp in input_info:
            inp_name = inp[0]
            inp_shape = [int(i) for i in inp[1].split(',')]
            info_dict = {inp_name: {"shape": inp_shape, "type": "float32"}}
            temp_config["model"]["inference-engine"]["inputs_info"].append(info_dict)

    if backend == "htp" or backend == "aic":
        temp_config["model"]["inference-engine"]["platforms"] = []
        temp_config["model"]["inference-engine"]["platforms"].append(
            defaults.get_value("qacc.default_platforms.cpu"))

        if target_arch == "x86_64-linux-clang":
            if backend == "htp":
                temp_config["model"]["inference-engine"]["platforms"].append(
                defaults.get_value("qacc.default_platforms.htp_x86"))
            elif backend == "aic":
                temp_config["model"]["inference-engine"]["platforms"].append(
                defaults.get_value("qacc.default_platforms.aic_x86"))
        elif target_arch == "aarch64-android":
            if backend == "htp":
                temp_config["model"]["inference-engine"]["platforms"].append(
                defaults.get_value("qacc.default_platforms.htp_aarch64"))
            elif backend == "aic":
                raise ce.ConfigurationException(f'Target architecture {target_arch} not supported for backend {backend}')

        temp_config["model"]["inference-engine"]["model_path"] = model_path
        #Update the comparator config
        temp_config["model"]["inference-engine"]["comparator"] = defaults.get_value("qacc.comparator")
        temp_config["model"]["inference-engine"]["comparator"]["type"] = comparator
        temp_config["model"]["inference-engine"]["comparator"]["tol"] = tol_thresh
        if comparator == "box":
            temp_config["model"]["inference-engine"]["comparator"]["box_input_json"] = box_input

    with open(temp_config_file, "w") as stream:
        yaml.dump(temp_config, stream, default_flow_style=False)

    return temp_config_file

def convert_npi_to_json(npi_yaml_file, output_json):
    """
    Converts npi yaml file to output json in qnn
    quantization_overrides format and saves it at output_json location
    """

    with open(npi_yaml_file) as F:
        data = yaml.safe_load(F)

    list_of_tensors_in_fp16 = []
    list_of_tensors_in_fp32 = []

    for key in data:
        if key == 'FP16NodeInstanceNames':
            list_of_tensors_in_fp16.extend(data[key])
        elif key == 'FP32NodeInstanceNames':
            list_of_tensors_in_fp32.extend(data[key])
        else:
            print('Incorrect entry in YAML file: ',key)
            exit(1)

    overrides_dict = { "activation_encodings":{}, "param_encodings":{}}

    activation_encodings_dict  = {}
    for tensor in list_of_tensors_in_fp32:
        value = [
            {
                "bitwidth": 32,
                "dtype": "float"
            }
        ]
        activation_encodings_dict[tensor] = value

    for tensor in list_of_tensors_in_fp16:
        value = [
            {
                "bitwidth": 32,
                "dtype": "float"
            }
        ]
        activation_encodings_dict[tensor] = value

    overrides_dict["activation_encodings"] = activation_encodings_dict

    with open(output_json, 'w') as fp:
        json.dump(overrides_dict, fp)
