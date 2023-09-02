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
import copy
import logging
import numpy as np
import os
from itertools import chain


import qti.aisw.accuracy_evaluator.common.exceptions as ce
import qti.aisw.accuracy_evaluator.qacc.configuration as qc
import qti.aisw.accuracy_evaluator.qacc.dataset as ds
from qti.aisw.accuracy_evaluator.qacc import *
from qti.aisw.accuracy_evaluator.qacc import qaic_logger
from qti.aisw.accuracy_evaluator.common.utilities import *
from qti.aisw.accuracy_evaluator.qacc.constants import Constants as qcc
from qti.aisw.accuracy_evaluator.common.infer_engines.QnnInferenceEngine import QnnInferenceEngine


class InferenceManager:

    def __init__(self, platform_config, infer_config, binary_path):
        self.plat_config = platform_config
        self.binary_path = binary_path
        self.infer_config = infer_config

        # capture execution time
        # (quantization time, compilation time, inference time)
        self.execution_time = [0, 0, 0]

    def execute(self, model_path, output_dir, input_file, output_file, calibration, device_id,
                precompiled_path, console_tag, compile_only, enable_perf=False,
                perf_iter_count=0, qnn_sdk_dir=""):
        if self.plat_config._name == qcc.INFER_ENGINE_QNN:
            return self.execute_qnn(model_path, output_dir, input_file, output_file, device_id,
                                    precompiled_path, console_tag,
                                    calibration=calibration, compile_only=compile_only,
                                    qnn_sdk_dir=qnn_sdk_dir)
        elif self.plat_config._name == qcc.INFER_ENGINE_ONNXRT:
            return self.execute_onnxrt(model_path, output_dir, input_file, output_file)
        elif self.plat_config._name == qcc.INFER_ENGINE_TFRT:
            return self.execute_tfrt(model_path, output_dir, input_file, output_file)
        elif self.plat_config._name == qcc.INFER_ENGINE_TORCHSCRIPTRT:
            return self.execute_torchscriptrt(model_path, output_dir, input_file, output_file)
        elif self.plat_config._name == qcc.INFER_ENGINE_TFRT_SESSION:
            return self.execute_tfrt_session(model_path, output_dir, input_file, output_file)

        assert ('Invalid Inference Platform ' + self.plat_config._name)

    def execute_qnn(self, model_path, output_dir, input_file, output_file, device_id,
                    precompiled_path, console_tag, calibration=None,
                    compile_only=False, qnn_sdk_dir=""):

        backend = self.plat_config._backend
        target_arch = self.plat_config._target_arch

        compiler_args = self._parse_platform_params(self.plat_config._compiler_params)
        runtime_args = self._parse_platform_params(self.plat_config._runtime_params)
        converter_args = self._parse_platform_params(self.plat_config._converter_params)

        calibration_file = None
        if calibration and self.plat_config._precision in [qcc.PRECISION_QUANT, qcc.PRECISION_INT8]:
            calibration_file = self.parse_generate_calibration(calibration,
                                                               input_file,
                                                               os.path.dirname(input_file))

        engine = QnnInferenceEngine(model=model_path,
                                    inputlistfile=input_file,
                                    calibration_file=calibration_file,
                                    output_path=output_dir,
                                    input_info=self.plat_config._input_info,
                                    output_info=self.plat_config._output_info,
                                    gen_out_file=output_file,
                                    compiler_params=compiler_args,
                                    runtime_params=runtime_args,
                                    converter_params=converter_args,
                                    backend=backend,
                                    target_arch=target_arch,
                                    qnn_sdk_dir=qnn_sdk_dir,
                                    device_id=device_id)

        try:
            engine.execute()
            ret_status = True
            qaic_logger.info('Inference success on QNN in execution stage.')
        except Exception as e:
            logging.info(e)
            qaic_logger.error('Inference failed on QNN in execution stage.')
            ret_status = False
        finally:
            infer_stages_status = engine.stage_status

        infer_fail_stage = self._get_first_fail_stage(infer_stages_status)
        return not ret_status, infer_fail_stage, [0,0,0]

    def execute_onnxrt(self, model_path, output_dir, input_file, output_file):
        from qti.aisw.accuracy_evaluator.common.infer_engines.OnnxRTEngine import OnnxInferenceEngine
        engine = OnnxInferenceEngine(model=model_path,
                                     inputlistfile=input_file,
                                     multithread=self.plat_config._multithreaded,
                                     output_path=output_dir,
                                     input_info=self.plat_config._input_info,
                                     output_info=self.plat_config._output_info,
                                     gen_out_file=output_file,
                                     convert_nchw=self.plat_config._convert_nchw)

        try:
            status, _, self.execution_time[2], _ = engine.execute()
            infer_fail_stage = None
        except Exception as e:
            logging.error('(onnxrt) Inference failed. See qacc.log for more details.')
            qaic_logger.error('Exception - {}'.format(e))
            status = 0
            infer_fail_stage = 'onnx-inference'

        return not status, infer_fail_stage, self.execution_time

    def execute_tfrt(self, model_path, output_dir, input_file, output_file):
        from qti.aisw.accuracy_evaluator.common.infer_engines.TensorflowRTEngine import TensorflowInferenceEngine
        engine = TensorflowInferenceEngine(model=model_path,
                                           inputlistfile=input_file,
                                           multithread=self.plat_config._multithreaded,
                                           output_path=output_dir,
                                           input_info=self.plat_config._input_info,
                                           output_info=self.plat_config._output_info,
                                           gen_out_file=output_file)
        try:
            status, _, self.execution_time[2], _ = engine.execute()
            infer_fail_stage = None
        except Exception as e:
            logging.error('tensorflow runtime inference failed. See qacc.log for more details.')
            qaic_logger.error('Exception - {}'.format(e))
            status = 0
            infer_fail_stage = 'tensorflow-inference'

        return not status, infer_fail_stage, self.execution_time

    def execute_torchscriptrt(self, model_path, output_dir, input_file, output_file):
        from qti.aisw.accuracy_evaluator.common.infer_engines.TorchScriptRTEngine import TorchScriptInferenceEngine
        engine = TorchScriptInferenceEngine(model=model_path,
                                      inputlistfile=input_file,
                                      multithread=self.plat_config._multithreaded,
                                      output_path=output_dir,
                                      input_info=self.plat_config._input_info,
                                      output_info=self.plat_config._output_info,
                                      gen_out_file=output_file)
        try:
            status, _, self.execution_time[2], _ = engine.execute()
            infer_fail_stage = None
        except Exception as e:
            logging.error('torchscript runtime inference failed. See qacc.log for more details.')
            qaic_logger.error('Exception - {}'.format(e))
            status = 0
            infer_fail_stage = 'torchscript-inference'

        return not status, infer_fail_stage, self.execution_time

    def execute_tfrt_session(self, model_path, output_dir, input_file, output_file):
        from qti.aisw.accuracy_evaluator.common.infer_engines.TensorflowSessionRTEngine import TensorflowSessionInferenceEngine
        engine = TensorflowSessionInferenceEngine(model=model_path,
                                           inputlistfile=input_file,
                                           multithread=self.plat_config._multithreaded,
                                           output_path=output_dir,
                                           input_info=self.plat_config._input_info,
                                           output_info=self.plat_config._output_info,
                                           gen_out_file=output_file)
        try:
            status, _, self.execution_time[2], _ = engine.execute()
            infer_fail_stage = None
        except Exception as e:
            logging.error('tensorflow runtime inference failed. See qacc.log for more details.')
            qaic_logger.error('Exception - {}'.format(e))
            status = 0
            infer_fail_stage = 'tensorflow-session-inference'

        return not status, infer_fail_stage, self.execution_time

    def _parse_range(self, index_str):
        if len(index_str) == 0:
            return []
        nums = index_str.split("-")
        assert len(nums) <= 2, 'Invalid range in calibration file '
        start = int(nums[0])
        end = int(nums[-1]) + 1
        return range(start, end)

    def parse_generate_calibration(self, calibration, input_file, output_dir):
        if calibration is None or input_file is None:
            return None
        (calib_type, calib_file) = calibration

        if calib_type == qcc.CALIBRATION_TYPE_RAW:
            return calib_file
        elif calib_type == qcc.CALIBRATION_TYPE_INDEX:
            cf = open(calib_file, 'r')
            indexes_str = cf.read().replace('\n', ',').strip()
            indexes = sorted(set(chain.from_iterable(map(self._parse_range,
                                                         indexes_str.split(",")))))
            cf.close()
            _path = os.path.join(output_dir, 'calibration.txt')
            qaic_logger.info('Generating calibration file')
            with open(input_file) as f, open(_path, 'w') as f2:
                for index, line in enumerate(f):
                    if index in indexes:
                        f2.write(line)
            return _path
        else:
            raise RuntimeError('Invalid calibration type {}'.format(calib_type))

    def _parse_platform_params(self, params):
        """
        Cleans up the unnecessary params
        """
        #TODO: Verfiy if there are any params which needs the False flag.
        param_args = {}
        for k, v in params.items():
            if v is False:
                pass
            else:
                param_args[k] = v

        return param_args

    def _get_first_fail_stage(self, stage_status):
        for stage in stage_status:
            if not stage_status[stage]:
                return stage
        return ""



class PlatformManager:

    def __init__(self, platforms, config):
        self.platforms = platforms
        self.aic_device_ids = config._inference_config._aic_device_ids
        self.schedule = None

    def scan_and_add_platform_permutations(self):
        """
        Scans the platform section and finds all the possible
        AIC permutations. Once the scan is complete, these possible
        platform permutations are added to the existing platform list

        example:
        Given a platform
            platform:
                name: aic
                precision: <value>
                params:
                    param1: input1 | input2 =>
                    param2: input3 | input4 =>
                use_precompiled:

        will create following platforms
            platform:
                name: aic
                precision: <value>
                params:
                    param1: input1
                    param2: input3
                use_precompiled:platform:
                name: aic

            platform:
                name: aic
                precision: <value>
                params:
                    param1: input1
                    param2: input4
                use_precompiled:platform:

            platform:
                name: aic
                precision: <value>
                params:
                    param1: input2
                    param2: input3
                use_precompiled:platform:
                name: aic

            platform:
                name: aic
                precision: <value>
                params:
                    param1: input2
                    param2: input4
                use_precompiled:platform:

        """

        def calibration_check(plat, keys):
            '''
            Returns True if calibration is required else return False
            '''
            if plat._precision == qcc.PRECISION_QUANT \
                    and plat._use_precompiled is None:
                    #and set(keys).isdisjoint(QnnInferenceEngine.get_calibration_skip_params()):
                return True
            else:
                return False

        # updated platforms consisting of original plus newly
        # generated platforms
        updated_platforms = []

        # used to tag all the generated platforms from
        # one original platform with same group id
        group_id = -1

        # use same group_id across same pqq_group
        # key: pqq_group tag and val: group_id
        pgq_group_dict = {}

        # used to perform calibration if int8 platform available
        is_calib_req = False

        for plat in self.platforms:

            if (qcc.INFER_ENGINE_QNN != plat._name) and (qcc.INFER_ENGINE_AIC != plat._name)\
                 and (qcc.INFER_ENGINE_AIC_SIM != plat._name):
                qaic_logger.debug('scan_and_add: Non QNN platform {} added'.format(plat._name))
                updated_platforms.append(plat)
                continue

            # get nested list of values
            param_values = []
            param_keys = []

            for key, val in plat._converter_params.items():
                if isinstance(val, list):
                    # skip keys which have list of values
                    param_values.append(val)
                    param_keys.append(key)
                else:
                    val = str(val)
                    # store list of values
                    vals = [v.strip() for v in val.split(qcc.SEARCH_SPACE_DELIMITER)]
                    val2remove = []  # Values to Remove
                    for v_idx, v in enumerate(vals):
                        if v.startswith(qcc.RANGE_BASED_SWEEP_PREFIX) and v.endswith(')'):
                            try:
                                start, end, step = v[len(qcc.RANGE_BASED_SWEEP_PREFIX):-1].split(
                                    qcc.RANGE_BASED_DELIMITER)
                                start, end, step = start.strip(), end.strip(), step.strip()
                                val_precision = max([len(start.split('.')[-1]), len(end.split('.')[-1]),
                                                     len(step.split('.')[-1])])
                            except:
                                raise ce.ConfigurationException(
                                    f"Check range based parameter syntax in platform params in config "
                                    f"file")
                            _, start = self.get_param_dtype(start, return_val=True)
                            _, end = self.get_param_dtype(end, return_val=True)
                            _, step = self.get_param_dtype(step, return_val=True)
                            range_values = [f'{range_val:0.{val_precision}f}' for range_val in
                                            np.arange(start, end, step)]
                            val2remove.append(v)
                            vals.extend(range_values)
                    for val in val2remove:
                        vals.remove(val)  # Remove the Range based param post expansion of range params

                    param_values.append(list(set(vals)))  # Remove Duplicates if any
                    qaic_logger.debug('Plat-{} Added {}:{} values for search space scan'
                                      .format(plat._name, key, vals))

                    # store keys
                    param_keys.append(key)

            qaic_logger.debug('scan_and_add: Options for keys-{} values-{} added'
                              .format(param_keys, param_values))

            # check whether for current platform calibration is needed.
            # The key is needed in estimating disk space and performing
            # preprocessing for calibration inputs.
            if not is_calib_req:
                # check only if is_calib_req is False
                # if even platform needs calibration then this field will be True
                is_calib_req = calibration_check(plat, param_keys)

            if 0 != len(param_values):
                # check if group_id already present
                if plat._pgq_group in pgq_group_dict:
                    group_id = pgq_group_dict[plat._pgq_group]
                else:
                    group_id += 1
                    if plat._pgq_group:
                        pgq_group_dict[plat._pgq_group] = group_id
                self.scan_over_params(param_keys, param_values, plat, updated_platforms, group_id,
                                      True)
                qaic_logger.debug(updated_platforms)
            else:
                # add aic platform with empty params
                updated_platforms.append(plat)

        for up_plat in updated_platforms:
            qaic_logger.info('Platform: {} - params: {}'.format(up_plat._name, up_plat._converter_params))

        qaic_logger.debug('pgq_groups: {}'
                          .format(pgq_group_dict.items()))

        # updating platform list
        self.platforms = updated_platforms

        return updated_platforms, is_calib_req

    def scan_over_params(self, param_keys, param_values, plat, updated_platforms, group_id,
                         is_parent, row=0, new_param_values=None):
        """
        Scan and add platforms

        example format for param_values:
        [[][] ... []]
        [
        [param_val_0 ... param_val_N] => from param_key_0
        [param_val_0 ... param_val_N] => from param_key_1
        .
        .
        .
        [param_val_0 ... param_val_N] => from param_key_N
        ]

        Based on nested param values the function sweeps across all possible combinations
        and adds it as a new AIC platform with modified params.
        """

        # new param values
        if new_param_values is None:
            new_param_values = []

        # terminating case
        if row == len(param_values):
            # reached the end so add the platform to updated platforms
            new_plat = copy.deepcopy(plat)

            # create new param dict
            new_param_dict = dict(zip(param_keys, new_param_values))
            qaic_logger.debug(f"New param dict: {new_param_dict}")
            #TODO: Remove all the invalid combinations for the QNN quant params
            # Remove invalid combinations: percentile-calibration-value required only for
            # quantization-calibration == Percentile
            # if 'quantization-calibration' in temp_dict and temp_dict[
            #     'quantization-calibration'] != 'Percentile':
            #     if 'percentile-calibration-value' in temp_dict:
            #         del temp_dict['percentile-calibration-value']
            new_plat._converter_params = new_param_dict

            # mark platform with unique group id.
            # This filed is used while reusing pgq profile.
            new_plat._group_id = group_id

            # add new platform
            if new_plat not in updated_platforms:
                if is_parent:
                    # add parent platform at index 1 (second platform)
                    # so that it scheduled before its child platforms.
                    # This is done to reuse pgq profile generated by
                    # the parent platform. The index 0 is reserved for
                    # reference platform eg onnx.
                    updated_platforms.insert(1, copy.deepcopy(new_plat))

                    # add child platforms at the end of the list
                    is_parent = False
                else:
                    updated_platforms.append(copy.deepcopy(new_plat))
                qaic_logger.debug('Platform added: {} - new params: {}'
                                  .format(new_plat._name, new_plat._converter_params))

            return is_parent

        for idx, val in enumerate(param_values[row]):

            # check for first element
            if 0 != idx:
                # remove last inserted element
                new_param_values.pop()

            # adding next element
            new_param_values.append(val)

            # call for next params
            is_parent = self.scan_over_params(param_keys, param_values, plat, updated_platforms,
                                              group_id,
                                              is_parent, row + 1, new_param_values)

        # remove last inserted element
        new_param_values.pop()

        # informing previous caller to change it to false
        return is_parent

    def create_schedule(self, strategy='distributed'):
        """
        Creates a schedule based on selected strategy.
        Default is distributed inference strategy.

        A schedule has following format:
            [parallel_chuck_1, parallel_chuck_2, ... , parallel_chuck_n]

        Each parallel chunk has following format:
            [(platform_idx, device_id), ... , (platform_idx, device_id)]

        Note: device_id for platforms other than aic is -1

        example:
            case1:
                aic_device_ids = [0,1]
                platforms = [onnx, aic, aic, aic, aic]
                schedule = [[(0,-1), (1,0), (2,1)], [(3,0), (4,1)]]
        """
        #logging.info('Creating Schedule using available AIC device ids: {}'.format(self.aic_device_ids))

        if strategy == 'distributed':
            self.schedule = []
            aic_slots = len(self.aic_device_ids)
            distributed_plats = []
            used_slots = 0

            for idx, plat in enumerate(self.platforms):
                if plat._name == qcc.INFER_ENGINE_AIC:

                    # if all slots filled
                    if used_slots == aic_slots:
                        self.schedule.append(copy.deepcopy(distributed_plats))
                        distributed_plats = []
                        used_slots = 0

                    distributed_plats.append((idx, int(self.aic_device_ids[used_slots])))

                    # inc used slots
                    used_slots += 1


                else:
                    # device id for non aic platform is -1
                    distributed_plats.append((idx, self.aic_device_ids[0]))

            # copy the last chuck
            self.schedule.append(copy.deepcopy(distributed_plats))
            qaic_logger.info('Distributed schedule: {}'.format(self.schedule))
        else:
            # will be handled while invoking inference platforms
            pass

    def get_schedule(self):
        return self.schedule

    def get_param_dtype(self, param_str, return_val=False):
        '''Determine given String is int,float or string
        Used to in
        '''
        try:
            val = int(param_str)
            if return_val: return int, val
            return int
        except:
            pass
        try:
            val = float(param_str)
            if return_val: return float, val
            return float
        except:
            pass
        if return_val: return str, val
        return str
