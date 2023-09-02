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
import enum
import logging
import numpy as np
import onnx
import os
import tensorflow as tf
import re
import sys
import shutil

import qti.aisw.accuracy_evaluator.common.exceptions as ce
from qti.aisw.accuracy_evaluator.common.defaults import qaic_logger
import qti.aisw.accuracy_evaluator.common.defaults as df

# to avoid printing logs on console
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

defaults = df.Defaults.getInstance()

class ModelType(enum.Enum):
    ONNX = 0
    TORCHSCRIPT = 1
    TENSORFLOW = 2
    TFLITE = 3
    FOLDER = 4

class Constants:
    OUTPUT_FILE_EXTENSION = '-activation-0-inf-0.bin'


class AIC_ERROR_MSGS:
    RESHAPE_CONST_ERROR = 'Reshape: Non-constant shape tensors are unsupported'
    SEG_FAULT = 'Segmentation Fault'


class Helper:
    """
    Utility class contains common utility methods
    To use:
    >>>Helper.get_np_dtype(type)
    >>>Helper.get_model_type(path)
    """

    @classmethod
    def get_np_dtype(cls, dtype):
        """
        This method gives the appropriate numpy datatype for given data type
        Args:
            dtype  : onnx data type

        Returns:
            corresponding numpy datatype
        """
        # returns dtype if it is already a numpy dtype
        # else get the corresponding numpy datatype
        try:
            if dtype.__module__ == np.__name__:
                return dtype
        except AttributeError as e:
            if dtype.__class__ == np.dtype:
                dtype = dtype.name

        if (dtype == 'tensor(float)' or dtype == 'float' or dtype == 'float32' or
            dtype == tf.float32):
            return np.float32
        elif (dtype == 'tensor(int)' or dtype == 'int'):
            return np.int
        elif (dtype == 'tensor(float64)' or dtype == 'float64' or dtype == tf.float64):
            return np.float64
        elif (dtype == 'tensor(int64)' or dtype == 'int64' or dtype == tf.int64):
            return np.int64
        elif (dtype == 'tensor(int32)' or dtype == 'int32' or dtype == tf.int32):
            return np.int32
        elif dtype == 'tensor(bool)' or dtype == 'bool' or dtype == tf.bool:
            return np.bool
        else:
            assert False, "Unsupported OP type " + str(dtype)

    @classmethod
    def get_model_type(cls, path):
        if os.path.isdir(path):
            return ModelType.FOLDER
        else:
            extn = os.path.splitext(path)[1]
        if extn == '.onnx':
            return ModelType.ONNX
        elif extn == '.pt':
            return ModelType.TORCHSCRIPT
        elif extn == '.pb':
            return ModelType.TENSORFLOW
        elif extn == ".tflite":
            return ModelType.TFLITE
        else:
            # TODO : support other model types.
            raise ce.UnsupportedException('model type not supported :' + path)

    @classmethod
    def onnx_type_to_numpy(cls, type):
        """
        This method gives the corresponding numpy datatype for given onnx tensor element type
        Args:
            type : onnx tensor element type
        Returns:
            corresponding numpy datatype and size
        """
        if type == '1':
            return (np.float32, 4)
        elif type == '2':
            return (np.uint8, 1)
        elif type == '3':
            return (np.int8, 1)
        elif type == '4':
            return (np.uint16, 2)
        elif type == '5':
            return (np.int16, 2)
        elif type == '6':
            return (np.int32, 4)
        elif type == '7':
            return (np.int64, 8)
        elif type == '9':
            return (np.bool8, 1)
        else:
            raise ce.UnsupportedException('Unsupported type : {}'.format(str(type)))

    @classmethod
    def tf_type_to_numpy(cls, type):
        """
        This method gives the corresponding numpy datatype for given tensorflow tensor element type
        Args:
            type : tensorflow tensor element type
        Returns:
            corresponding tensorflow datatype
        """
        # TODO: Add QINT dtypes
        tf_to_numpy = {
            1: np.float32,
            2: np.float64,
            3: np.int32,
            4: np.uint8,
            5: np.int16,
            6: np.int8,
            9: np.int64,
            10: np.bool8
        }
        if type in tf_to_numpy:
            return tf_to_numpy[type]
        else:
            raise ce.UnsupportedException('Unsupported type : {}'.format(str(type)))

    @classmethod
    def ort_to_tensorProto(cls, type):
        """
        This method gives the appropriate numpy datatype for given onnx data type
        Args:
            type  : onnx data type

        Returns:
            corresponding numpy datatype
        """
        if (type == 'tensor(float)' or type == 'float'):
            return onnx.TensorProto.FLOAT
        elif (type == 'tensor(int)' or type == 'int'):
            return onnx.TensorProto.INT8
        elif (type == 'tensor(float64)' or type == 'float64'):
            return onnx.TensorProto.DOUBLE
        elif (type == 'tensor(int64)' or type == 'int64'):
            return onnx.TensorProto.INT64
        elif (type == 'tensor(int32)' or type == 'int32'):
            return onnx.TensorProto.INT32
        else:
            assert ("TODO: fix unsupported OP type " + str(type))

    @classmethod
    def parse_compile_log(cls, log_path, err_msg):
        """
        searches the log file for a specific err_msg AIC_ERROR_MSGS
        If found, returns the line, or specific section from the log based on the err_msg
        """
        found_err = False
        msg = None

        with open(log_path, 'r') as log_file:
            lines = log_file.readlines()
            for line in lines:
                if line.__contains__(err_msg):
                    found_err = True
                    if err_msg == AIC_ERROR_MSGS.RESHAPE_CONST_ERROR:
                        msg = re.findall(r".*\[Operator-'(.*)'\]", line)

        return found_err, msg

    @classmethod
    def dump_error_into_log(self, is_compilation_fail, dir, dump_all=False):
        """
        This method dumps the aic compile and/or run logs into log file
        """
        aic_compile_log = os.path.join(dir, 'aiclog.txt')
        aic_run_log = os.path.join(dir, 'runlog.txt')
        if (is_compilation_fail or dump_all) and os.path.exists(aic_compile_log):
            with open(aic_compile_log, 'r') as f:
                qaic_logger.error('Last AIC compilation log::\n')
                for line in f:
                    qaic_logger.error('\t' + line.strip())
                qaic_logger.error('\n')

        if (not is_compilation_fail or dump_all) and os.path.exists(aic_run_log):
            qaic_logger.error('Last AIC execution log::\n')
            with open(aic_run_log, 'r') as f:
                for line in f:
                    qaic_logger.error('\t' + line.strip())
                qaic_logger.error('\n')

    @classmethod
    def get_average_match_percentage(cls, outputs_match_percentage, output_comp_map):
        """
        Return the average match for all the outputs for a given comparator
        """
        all_op_match = []
        for op, match in outputs_match_percentage.items():
            comparator = output_comp_map[op]
            comp_name = comparator.display_name()
            all_op_match.append(match[comp_name])

        return sum(all_op_match) / len(all_op_match)

    @classmethod
    def show_progress(cls, total_count, cur_count, info='', key='='):
        """
        Displays the progress bar
        """
        completed = int(round(80 * cur_count / float(total_count)))
        percent = round(100.0 * cur_count / float(total_count), 1)
        bar = key * completed + '-' * (80 - completed)

        sys.stdout.write('[%s] %s%s (%s)\r' % (bar, percent, '%', info))
        sys.stdout.flush()


    @classmethod
    def map_outputs_to_files(cls, out_node_names, output_files, file_ext):
        """
        This method maps the output node name to aic output file.
        Args:
            out_node_names : list of output node names
            output_files   : list of aic output file paths
            file_ext       : extention of file
        Returns:
            output_file_map : dictionary mapping output node names to file names
        """
        output_file_map = {}
        out_node_names = sorted(out_node_names, key = len, reverse = True)
        mark_b = [False] * len(output_files)
        for op in out_node_names:
            idx = -1
            for i, file in enumerate(output_files):
                # fetch the node name from the output file path
                node_name = file.split('/')[-1].split('-activation-')[0]
                if (not mark_b[i] and op in node_name and file_ext in file):
                    idx = i
                    break
            if (idx != -1):
                output_file_map[op] = output_files[idx]
                mark_b[idx] = True
        return output_file_map

    @classmethod
    def get_aic_output_path(cls, model_type, work_dir, layer_output_name, out_files,iter=None):
        """
        This method returns aic_output_path
        Args:
            model_type        : framework of model
            work_dir          : path of working directory
            layer_output_name : output name of layer
        Returns:
            aic_path : path of aic output file
        """
        tag = str(iter) if iter else '1'
        extention =  '-activation-0-inf-' + str(int(tag) - 1) + '.bin'
        out_path = work_dir if iter else work_dir + '/aic/'
        aic_output_map = Helper.map_outputs_to_files([layer_output_name],out_files,extention)
        return aic_output_map[layer_output_name]

    @classmethod
    def get_model_reader(cls, model):
        """
        This method returns the appropriate model reader for the model.
        """
        from qti.aisw.accuracy_evaluator.qdbg.reader import OnnxModelReader, TensorflowModelReader, \
            TorchModelReader
        model_type = Helper.get_model_type(model)

        if model_type == ModelType.ONNX:
            return OnnxModelReader(model)
        elif model_type == ModelType.TENSORFLOW:
            return TensorflowModelReader(model)
        elif model_type == ModelType.TORCHSCRIPT:
            return TorchModelReader(model)
        else:
            raise ce.UnsupportedException('Invalid Model Type ' + str(model_type))

    @classmethod
    def validate_aic_device_id(self, device_ids):
        '''
        device_ids: list containing the device ids
        '''
        # TODO: Need to validate device count on remote machine
        if Helper.useRemoteDevice():
            return True
        try:
            valid_devices = [d.strip() for d in
                             os.popen('/opt/qti-aic/tools/qaic-util -q |grep "QID"').readlines()]
            device_count = len(valid_devices)
        except:
            raise ce.ConfigurationException(
                'Failed to get Device Count. Check Devices are connected and Platform SDK '
                'Installation')
        for dev_id in device_ids:
            if f'QID {dev_id}' not in valid_devices:
                raise ce.ConfigurationException(
                    f'Invalid Device Id(s) Passed. Device used must be one of '
                    f'{", ".join(valid_devices)}')
        return True

    @classmethod
    def get_aic_device_count(self):
        device_count = 0
        try:
            device_count = len(
                os.popen('/opt/qti-aic/tools/qaic-util -q |grep "Status"').readlines())
        except:
            raise ce.ConfigurationException(
                'Failed to get Device Count. Check Devices are connected and Platform SDK '
                'Installation')
        return device_count

    @classmethod
    def useRemoteDevice(cls):
        """
        This method returns true if remote exec is enabled
        """
        return defaults.get_value('common.remote_exec.enabled')

    @classmethod
    def prepare_work_dir(self, work_dir):
        temp_dir = os.path.join(work_dir)
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        # create empty temp dir
        os.makedirs(temp_dir)
        defaults = df.Defaults.getInstance()
        defaults.set_log(work_dir + '/qacc.log')

    @classmethod
    def dump_stage_error_log(self, logfile):
        with open(logfile) as f:
            log = f.read()
        qaic_logger.error(log)
