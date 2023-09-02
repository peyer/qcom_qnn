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
import logging
import numpy as np
import os
import shutil
import sys
import time
from collections import OrderedDict

import qti.aisw.accuracy_evaluator.common.exceptions as ce
from qti.aisw.accuracy_evaluator.common.defaults import qaic_logger
from qti.aisw.accuracy_evaluator.common.infer_engines.infer_engine import InferenceEngine
from qti.aisw.accuracy_evaluator.common.utilities import Helper
from qti.aisw.accuracy_evaluator.common.transforms import ModelHelper

try:
    os.environ['GLOG_minloglevel'] = '3'
    import caffe
except:
    qaic_logger.info(
        'Caffe engine is unsupported in this environment.')

class CaffeInferenceEngine(InferenceEngine):
    """
    CaffeInferenceEngine class takes required inputs supplied by user from commandline
    options and
    calls validate and execute methods.
    TODO: Add an extra dictionary parameter to class which enables to uses extra caffe
    options
    To use:
    >>> engine = CaffeInferenceEngine(model, inputlistfile, output_path, multithread,
    input_info,
                output_info, gen_out_file, extra_params)
    >>> engine.validate()
    >>> engine.execute()
    """

    def __init__(self, model, inputlistfile, output_path, multithread, input_info, output_info=None,
                 gen_out_file=None, extra_params=None):
        super().__init__(model, inputlistfile, output_path, multithread, input_info,
                         output_info, gen_out_file, extra_params)
        self.validate()

    def execute(self):
        """
        This method runs the given model on caffe
        Returns:
            status: execution status
            res   : dictionary containing ort_session
        """
        qaic_logger.debug("CaffeInferenceEngine start execution")
        # capture inference time
        inf_time = 0
        res = {}
        inp_shapes_map = OrderedDict()
        do_profile = False
        save_intermediate_outputs = False
        profile_data = None
        if self.extra_params:
            profile_data = {}
            if '-profile' in self.extra_params:
                do_profile = self.extra_params['-profile']
            if '-save-intermediate-outputs' in self.extra_params:
                save_intermediate_outputs = self.extra_params['-save-intermediate-outputs']

        # Load the caffe model
        prototxt_path, weights_file_path = ModelHelper.get_caffe_paths(self.model_path)
        test_mode = caffe.TEST
        loaded = ModelHelper.load_caffe(prototxt_path, test_mode, weights_file_path)

        # create a dictionary mapping input node names to corresponding shapes
        for i_node_name in loaded.inputs:
            inp_shapes_map[i_node_name] = (
                loaded.blobs[i_node_name].data.shape, loaded.blobs[i_node_name].data.dtype)

            # Create the output file if requested.
        out_list_file = None
        if self.gen_out_file:
            out_list_file = open(self.gen_out_file, 'w')

        start_time = time.time()

        # Create input dictionary and run the inference for each input.
        with open(self.input_path) as f:
            for iter, line in enumerate(f):
                input_map = {}
                inps = line.strip().split(',')
                inps = [inp.strip() for inp in inps if inp.strip()]

                for idx, inp in enumerate(inps):
                    if self.input_info is None:
                        # When input shapes and dtypes are not passed by user,inputs dictionary
                        # is formed using loaded caffe model
                        try:
                            input_np = np.fromfile(inp,
                                                   dtype=inp_shapes_map[loaded.inputs[idx]][
                                                       1]).reshape(
                                inp_shapes_map[loaded.inputs[idx]][0])
                            input_map[loaded.inputs[idx]] = input_np
                        except Exception as e:
                            logging.error(
                                'Unable to extract input info from model.Please try '
                                'passing input-info ')
                            qaic_logger.exception(e)
                            raise ce.InferenceEngineException(
                                "Unable to extract input info from model.Please try "
                                "passing input-info",
                                e)
                    else:
                        if loaded.inputs[idx] not in self.input_info:
                            qaic_logger.error(
                                'Input info name not valid for this model. expected: {} '.format(
                                    inp_nodes[idx].name))
                            raise ce.ConfigurationException("Invalid Configuration")
                        input_np = np.fromfile(inp, dtype=(
                            Helper.get_np_dtype(self.input_info[loaded.inputs[idx]] \
                                                    [0]))).reshape(
                            self.input_info[loaded.inputs[idx]][1])
                        input_map[loaded.inputs[idx]] = input_np

                # Run inference
                try:
                    result = loaded.forward(**input_map)
                    outputs = []
                    interm_out_names = []
                    if '-save-single-layer-output-name' in self.extra_params:
                        outputs.append(loaded.blobs[self.extra_params['-save-single-layer-output-name']].data)
                    elif '-save-input' in self.extra_params:
                        for input_layer in self.extra_params['-save-input']:
                            outputs.append(loaded.blobs[input_layer].data)
                    elif save_intermediate_outputs:
                        for layer, outs in loaded.top_names.items():
                            if outs[0] not in loaded.outputs:
                                interm_out_names.append(outs[0])
                        for item in interm_out_names:
                            if item not in loaded.outputs:
                                outputs.append(loaded.blobs[item].data)

                    if ('-save-single-layer-output-name' not in self.extra_params) and ('-save-input' not in self.extra_params):
                        for elem in loaded.outputs:
                            outputs.append(result[elem])

                except Exception as e:
                    qaic_logger.error('inference.run failed')
                    qaic_logger.exception(e)
                    raise ce.InferenceEngineException("inference failed", e)

                caffe_outputs = interm_out_names + loaded.outputs
                # Store the names of the outputs. Use the same for raw file name generation.
                if self.output_info:
                    output_names = list(self.output_info.keys())
                    # reorder caffe outputs if needed as per output info names
                    _temp = []
                    _temp_outs = []
                    for name in output_names:
                        _name_found = False
                        for out, node_name in zip(outputs, caffe_outputs):
                            if str(node_name) == str(name):
                                _temp.append(node_name)
                                _temp_outs.append(out)
                                _name_found = True
                                break
                        if not _name_found:
                            logging.error('Output name {} in config is incorrect.'.format(name))
                            raise ce.InferenceEngineException(
                                "Invalid model config. Please fix output_info {}".format(name))
                    caffe_outputs = _temp
                    outputs = _temp_outs
                else:
                    if '-save-single-layer-output-name' in self.extra_params:
                        output_names = [self.extra_params['-save-single-layer-output-name']]
                    elif '-save-input' in self.extra_params:
                        output_names = self.extra_params['-save-input']
                    else:
                        # set same as caffe session
                        output_names = interm_out_names + loaded.outputs
                if self.output_path:
                    path = self.output_path + '/'
                    if not os.path.exists(path):
                        os.makedirs(path)

                    # Check if the output info is configured correctly.
                    if len(output_names) != len(caffe_outputs):
                        raise ce.ConfigurationException('The number of outputs in config file'
                                                        '({}) does not match with caffe model'
                                                        ' outputs ({})'
                                                        .format(len(output_names),
                                                                len(caffe_outputs)))

                    # Write output files and get profile data.
                    profile_data, _paths = self.save_outputs_and_profile(output_names, outputs,
                                                                         iter,
                                                                         True,
                                                                         do_profile)

                    # generate output text file for each of the inputs.
                    if self.gen_out_file:
                        out_list_file.write(','.join(_paths) + '\n')

        if self.gen_out_file:
            out_list_file.close()

        res['caffe_session'] = loaded
        res['profile'] = profile_data

        # For generating histogram profile
        output_dtypes = [op.dtype for op in outputs]
        output_array_map = list(zip(output_names, output_dtypes, outputs))

        if do_profile:
            logging.info('Captured caffe profile')

        inf_time = time.time() - start_time
        qaic_logger.debug("CaffeInferenceEngine execution success")
        return True, res, inf_time, output_array_map

    def get_profile(self):
        return self.profile_data

    def validate(self):
        """
        This method checks whether the given model_path ,model,input_path and output_path are
        valid or not
        Returns:
            status: validation status
        """
        qaic_logger.debug("CaffeInferenceEngine validation")
        # check the existence of model path and its autenticity
        if not os.path.exists(self.model_path):
            qaic_logger.error('Model path : {} does not exist '.format(self.model_path))
            raise ce.InferenceEngineException('Model path : {} does not exist '.format(
                self.model_path))

        # check whether the output path exists and create the path otherwise
        if self.output_path and not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        # check the existence of input path
        if not os.path.exists(self.input_path):
            logging.error('Input path : {} does not exist '.format(self.input_path))
            raise ce.InferenceEngineException(
                'Input path : {} does not exist '.format(self.input_path))
        qaic_logger.debug("CaffeInferenceEngine validation success")
        return True
