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
import builtins
import copy
import csv
import datetime
import logging
import math
import os
import shutil
import sys
import time
from itertools import chain
from joblib import Parallel, delayed
from tabulate import tabulate

import qti.aisw.accuracy_evaluator.common.exceptions as ce
import qti.aisw.accuracy_evaluator.qacc.configuration as qc
import qti.aisw.accuracy_evaluator.qacc.dataset as ds
import qti.aisw.accuracy_evaluator.qacc.plugin as pl
import qti.aisw.accuracy_evaluator.qacc.inference as infer
from qti.aisw.accuracy_evaluator.common.transforms import ModelHelper
from qti.aisw.accuracy_evaluator.qacc import qaic_logger
from qti.aisw.accuracy_evaluator.common.comparators import *
from qti.aisw.accuracy_evaluator.qacc.constants import Constants as qcc
from qti.aisw.accuracy_evaluator.qacc import *
from qti.aisw.accuracy_evaluator.common.utilities import Helper, ModelType

console = builtins.print

# stage completion status
STAGE_PREPROC_PASS = False
STAGE_INFER_PASS = False

pipeline_cache = qc.PipelineCache.getInstance()
DATASETS_YAML_PATH = os.path.join(os.path.dirname(__file__), '..', 'configs', 'datasets.yaml')

class QACCManager:

    def __init__(self, config_path, work_dir, set_global=None, batchsize=None,
                 dataset_config_yaml=DATASETS_YAML_PATH, model_path=None):

        # Stores the runtime info for each platform.
        self.platform_run_status = {}
        # available plugins must be loaded before loading configuration.
        pl.PluginManager.findAvailablePlugins()

        #Loading the model config from the yaml file.
        config = qc.Configuration.getInstance()
        logging.info('Loading model config')
        try:
            config.load_config_from_yaml(config_path=config_path, work_dir=work_dir,
                                         set_global=set_global, batchsize=batchsize,
                                         dataset_config_yaml=dataset_config_yaml,
                                         model_path=model_path)
        except Exception as e:
            logging.error('qacc failed to load config file. check log for more details.')
            qaic_logger.exception(e)
            sys.exit(1)

    def process_dataset(self, config, dataset_config):
        logging.info('Executing dataset plugins')
        out_dir = self.get_output_path(config._work_dir, qcc.DATASET_DIR)
        plugin_manager = pl.PluginManager()
        return plugin_manager.execute_dataset_transformations(dataset_config, out_dir)

    def preprocess(self, config, dataset, is_calibration=False):
        # Execute preprocessing.
        if is_calibration:
            logging.info('Executing Preprocessors for calibration inputs')
            out_dir = self.get_output_path(config._work_dir, qcc.STAGE_PREPROC_CALIB)
            pipeline_cache.set_val(qcc.PIPELINE_CALIB_DIR, out_dir)
            pipeline_cache.set_val(qcc.PIPELINE_PREPROC_IS_CALIB, is_calibration)
        else:
            logging.info('Executing Preprocessors')
            out_dir = self.get_output_path(config._work_dir, qcc.STAGE_PREPROC)
            pipeline_cache.set_val(qcc.PIPELINE_PREPROC_DIR, out_dir)
            pipeline_cache.set_val(qcc.PIPELINE_PREPROC_IS_CALIB, is_calibration)
        transformations_config = config._preprocessing_config._transformations
        transformations = pl.Transformations(
            plugin_config_list=transformations_config._plugin_config_list,
            max_input_len=dataset.get_total_entries())
        plugin_manager = pl.PluginManager(dataset)

        ret_status = plugin_manager.execute_transformations(
                         transformations=transformations,
                         output_dir=out_dir,
                         batch_offset=0,
                         input_names=config._inference_config._input_names)
        return ret_status, self.get_output_path(out_dir, qcc.QNN_PROCESSED_OUTFILE)

    def infer(self, model_path, config, processed_input_file, platform, dataset, device_id,
              platform_name, compile_only=False, load_binary_from_dir=False,
              enable_perf=False, perf_iter_count=200):

        # Execute Inference.
        logging.info('({}) Starting inference engine'.format(platform_name))
        dir_name = self.get_output_path(dir=config._work_dir, type=qcc.STAGE_INFER,
                                        plat_name=platform_name)
        infer_ds_path = self.get_output_path(dir=dir_name, type=qcc.INFER_OUTFILE)
        pipeline_cache.set_val(qcc.PIPELINE_INFER_DIR, dir_name, platform_name)
        pipeline_cache.set_val(qcc.PIPELINE_INFER_FILE, infer_ds_path, platform_name)
        binary_path = self.get_output_path(dir=config._work_dir, type=qcc.BINARY_PATH,
                                           plat_name=platform_name)

        # set network binary directory
        network_bin_dir = platform._precompiled_path \
            if platform._precompiled_path is not None else binary_path

        # store values in pipeline cache
        pipeline_cache.set_val(qcc.PIPELINE_NETWORK_BIN_DIR, network_bin_dir, platform_name)
        pipeline_cache.set_val(qcc.PIPELINE_NETWORK_DESC,
                               os.path.join(network_bin_dir, qcc.NETWORK_DESC_FILE), platform_name)
        pipeline_cache.set_val(qcc.PIPELINE_PROGRAM_QPC,
                               os.path.join(network_bin_dir, qcc.PROGRAM_QPC_FILE), platform_name)

        # update exec batchsize in platform params
        bs_key = qcc.INTERNAL_EXEC_BATCH_SIZE
        if pipeline_cache.get_val(bs_key) is not None:
            platform._params[qcc.MODEL_INFO_BATCH_SIZE] = pipeline_cache.get_val(bs_key)

        # Set Precompiled path to the corresponding binary path
        qnn_sdk_dir = pipeline_cache.get_val(qcc.QNN_SDK_DIR)
        precompiled_path = binary_path if load_binary_from_dir else None
        infer_mgr = infer.InferenceManager(platform, config._inference_config, binary_path)
        if config._dataset_config is not None:
            calibration_file = dataset.get_dataset_calibration()
        else:
            #TODO: Handle cli calibration file.
            calibration_file = (qcc.CALIBRATION_TYPE_RAW, processed_input_file)
        err_status, infer_fail_stage, execution_time = infer_mgr.execute(model_path=model_path,
                                                       output_dir=dir_name,
                                                       input_file=processed_input_file,
                                                       output_file=infer_ds_path,
                                                       calibration=calibration_file,
                                                       device_id=device_id,
                                                       precompiled_path=precompiled_path,
                                                       console_tag=platform_name,
                                                       compile_only=compile_only,
                                                       enable_perf=enable_perf,
                                                       perf_iter_count=perf_iter_count,
                                                       qnn_sdk_dir=qnn_sdk_dir)

        return err_status, infer_fail_stage, infer_ds_path, execution_time

    def postprocess(self, idx, config, dataset, infer_ds_path, platform_name):
        # Execute post processing for this inference results.
        # Get inference output dataset
        if config._postprocessing_config:
            logging.info('({}) Executing Postprocessors'.format(platform_name))
            infer_dataset = ds.DataSet(input_list_file=infer_ds_path)
            squash_results = config._postprocessing_config._squash_results
            transformations_config = config._postprocessing_config._transformations
            transformations = pl.Transformations(
                plugin_config_list=transformations_config._plugin_config_list,
                max_input_len=dataset.get_total_entries())
            plugin_manager = pl.PluginManager(infer_dataset, orig_dataset=dataset)
            dir_name = self.get_output_path(dir=config._work_dir, type=qcc.STAGE_POSTPROC,
                                            plat_name=platform_name)
            pipeline_cache.set_val(qcc.PIPELINE_POSTPROC_DIR, dir_name, platform_name)
            err_status = plugin_manager.execute_transformations(transformations=transformations,
                                                                output_dir=dir_name,
                                                                batch_offset=0,
                                                                squash_results=squash_results)
            if err_status:
                return 1, None

            metrics_input_file = self.get_output_path(dir=dir_name, type=qcc.PROCESSED_OUTFILE)
        else:
            metrics_input_file = infer_ds_path
        pipeline_cache.set_val(qcc.PIPELINE_POSTPROC_FILE, metrics_input_file, platform_name)

        return 0, metrics_input_file

    def evaluate_metrics(self, idx, config, dataset, postproc_file, platform):
        """Evaluate the given metrics on the inferred data"""
        platform_name = platform.get_platform_name()
        if config._evaluator_config:
            logging.info('({}) Evaluating metrics'.format(platform_name))
            processed_dataset = ds.DataSet(input_list_file=postproc_file)
            plugin_manager = pl.PluginManager(processed_dataset, orig_dataset=dataset)
            metrics_pl_cfg = config._evaluator_config._metrics_plugin_list
            metrics_results = []
            metrics_results_dict = {}
            dir_name = self.get_output_path(config._work_dir, qcc.STAGE_METRIC, platform_name)
            err_status = plugin_manager.execute_metrics(metrics_plugin_config=metrics_pl_cfg,
                                                        output_dir=dir_name,
                                                        results_str_list=metrics_results,
                                                        results_dict=metrics_results_dict)
            if err_status:
                self.platform_run_status[platform_name]['metrics'] = {}
                self.platform_run_status[platform_name]['status'] = qcc.PLAT_METRIC_FAIL
                return 1
            metrics_info = ''
            for res in metrics_results:
                logging.info('({}) metric: {}'.format(platform_name, res.replace('\n',' ')))
                if len(metrics_info) > 0:
                    metrics_info += '\n' + res
                else:
                    metrics_info = res
            self.platform_run_status[platform_name]['metrics'] = metrics_results_dict
        else:
            self.platform_run_status[platform_name]['metrics'] = {}
            return 0

    def compare_infer_results(self, config, preproc_file):
        """
        Compare inference outputs with configured comparator.
        Comparison can be done if there are more than 1 platforms.
        User can configure a reference platform by is_ref=True in platform section in yaml.
        In absense of is_ref, The first defined platform is considered as reference, the other platforms results are
        compared against the reference.
        """

        def getComparator(config, out_info=None, ref_out_file=None):
            """
            Return the configured comparators and datatypes for each output
            The order is same as defined in config file.
            """
            output_comparators = []
            output_comparator_names = []
            output_comparator_dtypes = []
            output_names = []
            if out_info:
                for outname, val in out_info.items():
                    if len(val) > 2:
                        # output specific comparator
                        cmp = val[2]['type']
                        tol_thresh = val[2]['tol'] if 'tol' in val[2] else 0.001
                        qaic_logger.info('Using output specific comparator : ' + cmp)
                    else:
                        cmp = config._inference_config._comparator['type']
                        tol_thresh = float(config._inference_config._comparator['tol'])

                    if cmp == 'abs':
                        _comparator = TolComparator(tol_thresh)
                    elif cmp == 'avg':
                        _comparator = AvgComparator(tol_thresh)
                    elif cmp == 'rme':
                        _comparator = RMEComparator(tol_thresh)
                    elif cmp == 'l1norm':
                        _comparator = NormComparator(order=1, tol=tol_thresh)
                    elif cmp == 'l2norm':
                        _comparator = NormComparator(order=2, tol=tol_thresh)
                    elif cmp == 'cos':
                        _comparator = CosComparator(tol_thresh)
                    elif cmp == 'std':
                        _comparator = StdComparator(tol_thresh)
                    elif cmp == 'maxerror':
                        _comparator = MaxErrorComparator(tol_thresh)
                    elif cmp == "snr":
                        _comparator = SnrComparator(tol_thresh)
                    elif cmp == "topk":
                        _comparator = TopKComparator(tol_thresh)
                    elif cmp == "pixelbypixel":
                        _comparator = PixelByPixelComparator(tol_thresh)
                    elif cmp == "box":
                        _comparator = BoxComparator(config._inference_config._comparator['box_input_json'], tol_thresh)
                    else:
                        logging.error(
                            'Unknown comparator {}. Using default {} instead:'.format(cmp, 'avg'))
                        cmp = 'avg'
                        _comparator = AvgComparator(0.001)

                    output_comparators.append(_comparator)
                    output_comparator_names.append(cmp)
                    output_comparator_dtypes.append(val[0])
                    output_names.append(outname)
            else:
                out_names = self.get_out_names(ref_out_file)
                for outname in out_names:
                    cmp = config._inference_config._comparator['type']
                    tol_thresh = float(config._inference_config._comparator['tol'])
                    if cmp == 'abs':
                        _comparator = TolComparator(tol_thresh)
                    elif cmp == 'avg':
                        _comparator = AvgComparator(tol_thresh)
                    elif cmp == 'rme':
                        _comparator = RMEComparator(tol_thresh)
                    elif cmp == 'l1norm':
                        _comparator = NormComparator(order=1, tol=tol_thresh)
                    elif cmp == 'l2norm':
                        _comparator = NormComparator(order=2, tol=tol_thresh)
                    elif cmp == 'cos':
                        _comparator = CosComparator(tol_thresh)
                    elif cmp == 'std':
                        _comparator = StdComparator(tol_thresh)
                    elif cmp == 'maxerror':
                        _comparator = MaxErrorComparator(tol_thresh)
                    elif cmp == "snr":
                        _comparator = SnrComparator(tol_thresh)
                    elif cmp == "topk":
                        _comparator = TopKComparator(tol_thresh)
                    elif cmp == "pixelbypixel":
                        _comparator = PixelByPixelComparator(tol_thresh)
                    elif cmp == "box":
                        _comparator = BoxComparator(config._inference_config._comparator['box_input_json'], tol_thresh)
                    else:
                        logging.error(
                            'Unknown comparator {}. Using default {} instead:'.format(cmp, 'avg'))
                        cmp = 'avg'
                        _comparator = AvgComparator(0.001)

                    output_comparators.append(_comparator)
                    output_comparator_names.append(cmp)
                    #Default dtype to float32
                    output_comparator_dtypes.append("float32")
                    output_names.append(outname)

            return output_names, output_comparators, output_comparator_dtypes, \
                   output_comparator_names

        platforms = config._inference_config._platforms
        if platforms and len(platforms) < 2:
            logging.info('Not enough platforms to compare inference outputs')
            return 0

        ref_platform = config.get_ref_platform()

        # ref_out_dir = self.get_output_path(config._work_dir, qcc.STAGE_INFER,
        #                                    self.get_platform_name(ref_platform._idx, ref_platform._name))
        ref_out_dir = self.get_output_path(config._work_dir, qcc.STAGE_INFER,
                                           ref_platform.get_platform_name())
        ref_out_file = self.get_output_path(ref_out_dir, qcc.INFER_OUTFILE)
        if not os.path.exists(ref_out_file):
            qaic_logger.error('reference inference out file {} does not exist'.format(ref_out_file))
            return 1

        outputs_ref = []
        with open(ref_out_file) as ref_file:
            for line in ref_file:
                outputs_ref.append(line.split(','))

        fcomp = FileComparator()
        out_names, comp, comp_dtypes, comp_names = getComparator(config, ref_platform._output_info,
                                                                 ref_out_file=ref_out_file)
        qaic_logger.info('comparators: {}'.format(comp))
        qaic_logger.info('comparator dtypes: {}'.format(comp_dtypes))

        # compare outputs for all platforms with reference.
        top = int(config._inference_config._comparator['fetch_top'])

        qaic_logger.info('Comparing inference output files. This may take some time..')
        qaic_logger.info('================ Inference output comparisons ====================')
        qaic_logger.info('Comparing all files ...')
        for idx, platform in enumerate(platforms):
            if idx == ref_platform._idx:
                continue

            # platform_name = self.get_platform_name(idx, platform._name)
            platform_name = platform.get_platform_name()

            try:
                out_file = self.get_output_path(
                    self.get_output_path(config._work_dir, qcc.STAGE_INFER, platform_name),
                    qcc.INFER_OUTFILE)

                if not os.path.exists(out_file):
                    qaic_logger.error('Platform infer out file does not exist {}'.format(out_file))
                    continue

                outputs_plat = []
                with open(out_file) as plat_file:
                    for line in plat_file:
                        outputs_plat.append(line.split(','))

                if len(outputs_ref) != len(outputs_plat):
                    qaic_logger.error('Infer output files count for {}:{} does not match for ' +
                                      '{}: {}'.format(ref_platform._name + str(ref_platform._idx), len(outputs_ref),
                                                      platform._name + str(idx), len(outputs_plat)))
                    return 1

                # compare each output of each platform and reference.
                output_results = {}
                output_results_per_output = {}
                for i, ref_inps in enumerate(outputs_ref):
                    plat_inps = outputs_plat[i]
                    if len(ref_inps) != len(plat_inps):
                        qaic_logger.error('Record {} :Reference number of inputs {} must match '
                                          'with plat {} inputs {}'.format(i, len(ref_inps),
                                                                          platform._name + str(idx),
                                                                          len(plat_inps)))
                        return 1

                    if comp[0].name() == "box":
                        match, percent_match, _ = fcomp.compare([a_path.strip() for a_path in plat_inps],
                                                                [r_path.strip() for r_path in ref_inps],
                                                                comp[0], comp_dtypes[0])

                        output_results[i] = round(percent_match, 3)
                    else:
                        out_i_per_match = []
                        for out_i, (a_path, r_path) in enumerate(zip(plat_inps, ref_inps)):
                            if comp[out_i].name() == "pixelbypixel":
                                save_dir = os.path.dirname(a_path)
                                match, percent_match, _ = fcomp.compare(a_path.strip(), r_path.strip(),
                                                                    comp[out_i], comp_dtypes[out_i], save_dir=save_dir)
                            else:
                                match, percent_match, _ = fcomp.compare(a_path.strip(), r_path.strip(),
                                                                        comp[out_i], comp_dtypes[out_i])
                            out_i_per_match.append(percent_match)
                            if out_i in output_results_per_output:
                                output_results_per_output[out_i].append(percent_match)
                            else:
                                output_results_per_output[out_i] = [percent_match]

                        output_results[i] = round(sum(out_i_per_match) / len(out_i_per_match), 3)

                # sorting by values
                output_results = dict(sorted(output_results.items(), key=lambda item: item[1]))

                mean = round(sum(output_results.values()) / len(output_results), 3)
                qaic_logger.info('Avg Match (all outputs) : {} vs {} = {} %'.format(
                    'plat'+ str(ref_platform._idx) + '_' + ref_platform._name,
                    platform_name,
                    mean))
                self.platform_run_status[platform_name]['comparator'] = {f'Avg Match (all outputs)': mean}
                for out_i, oname in enumerate(out_names):
                    _mean = round(sum(output_results_per_output[out_i]) / len(
                        output_results_per_output[out_i]), 3)
                    self.platform_run_status[platform_name]['comparator'].update(
                        {f'({comp_names[out_i]}) {oname}': _mean})
                    qaic_logger.info('\t({}) {} => {} %'.format(comp_names[out_i], oname, _mean))

                matches = sum(float(x) == 100.0 for x in output_results.values())
                qaic_logger.info(
                    'Complete Matches {} %'.format(round(matches * 100 / len(output_results)), 3))

                qaic_logger.info('Top mismatched inputs:')
                qaic_logger.info('------------------------------')

                # create 'top' mismatched files reading from preproc file.
                top_indexes = []
                for x in list(output_results)[0:top]:
                    qaic_logger.info('Index {} matched {} %  '.format(x, output_results[x]))
                    top_indexes.append(x)

                cache_lines = {}
                with open(preproc_file, 'r') as f1, open(
                        os.path.join(config._work_dir, platform_name + '_mm.txt'), 'w') as f2:
                    for pos, line in enumerate(f1):
                        if pos in top_indexes:
                            cache_lines[pos] = line
                    # write the file inputs in order of top indexes
                    for i in top_indexes:
                        f2.write(cache_lines[i])

                qaic_logger.info('')
                qaic_logger.info('Top matched inputs:')
                qaic_logger.info('-------------------------------')
                for x in list(reversed(list(output_results)))[0:top]:
                    qaic_logger.info('Index {} matched {} %  '.format(x, output_results[x]))

            except Exception as e:
                qaic_logger.error(e)
                return 1

        return 0

    def _set_test_params(self, max_calib=5):
        config = qc.Configuration.getInstance()
        config._inference_config._max_calib = max_calib

    def run_pipeline(self, work_dir='qacc_temp', platform_name=None,
                     platform_tag=None, cleanup='', onnx_symbol=None,
                     device_id=None, platform_tag_params=None,
                     inference_strategy='distributed',
                     cli_preproc_file=None, cli_infer_file=None, enable_perf_flag=False,
                     perf_iter_count=100, qnn_sdk_dir="", silent=False, backend=None):
        """
        Executes the E2E pipeline based on the args and model configuration

        Args:
            Arguments passed from cmd line are supplied to respective variables
        work_dir: path to directory to store the evaluation results and associated artifacts
        platform_name: run only on this platform type Allowed values ['qnn','aic','onnxrt',
        'tensorflow','torchscript']
        platform_tag: run only this platform tag
        cleanup:'cleanup preprocessing, inference and postprocessing output files.
            cleanup = 'end': deletes the files after all stages are completed.
            cleanup = 'intermediate' : deletes the intermediate inference and postprocessing
            output files. Selecting intermediate option saves space but disables comparator option'
        onnx_symbol: Replace onnx symbols in input/output shapes. Can be passed as list of
        multiple items. Default replaced by 1. e.g __unk_200:1
        device_id: Target Device to be used for accuracy evaluation
        preproc_file: preprocessed output file (if starting at infer stage)
        infer_file: Inference output file (if starting at postproc stage)
        enable_perf_flag: Flag to Capture Throughput and Latency metrics
        perf_iter_count: Number of iterations used for capturing performance metrics

        Returns:
            status: 0 if success otherwise 1
        """
        ret_status = 0
        config = qc.Configuration.getInstance()
        pipeline_stages, pipeline_start, pipeline_end = self.get_pipeline_stages_from_config(config)
        if len(pipeline_stages):
            qaic_logger.info('Configured stages: {}'.format(pipeline_stages))
        else:
            logging.error('Invalid pipeline start and end stages')
            return 1

        if pipeline_start == 'compiled':
            # reuse the existing files in work-dir
            dataset_dir = self.get_output_path(config._work_dir, qcc.DATASET_DIR)
            input_list_path = self.get_output_path(dataset_dir, qcc.INPUT_LIST_FILE)
            calib_list_path = self.get_output_path(dataset_dir, qcc.CALIB_FILE)
            max_inputs = sum(1 for line in open(input_list_path))
            max_calib = sum(1 for line in open(calib_list_path))
            pipeline_cache.set_val(qcc.PIPELINE_MAX_INPUTS, max_inputs)
            pipeline_cache.set_val(qcc.PIPELINE_MAX_CALIB, max_calib)

            config._dataset_config._inputlist_file = input_list_path
            config._dataset_config._calibration_file = calib_list_path
            config._dataset_config._max_inputs = max_inputs

            dataset = ds.DataSet(dataset_config=config._dataset_config)
        else:
            # execute dataset plugins
            if config._dataset_config:
                config._dataset_config = self.process_dataset(config, config._dataset_config)
                # handle max_inpts and max_calib for backward compatibility
                # override max_calib from dataset plugin to dataset
                if pipeline_cache.get_val(qcc.PIPELINE_MAX_INPUTS):
                    config._dataset_config._update_max_inputs()  # Update max_inputs post
                    # process_dataset()
                # Set max_inputs in pipeline cache with updated values
                pipeline_cache.set_val(qcc.PIPELINE_MAX_INPUTS, config._dataset_config._max_inputs)

                # override max_calib from dataset plugin to inference section for backward compatibility
                if pipeline_cache.get_val(qcc.PIPELINE_MAX_CALIB):
                    config._inference_config._max_calib = pipeline_cache.get_val(qcc.PIPELINE_MAX_CALIB)
                else:
                    pipeline_cache.set_val(qcc.PIPELINE_MAX_CALIB, config._inference_config._max_calib)
                # create dataset with modified dataset config
                dataset = ds.DataSet(dataset_config=config._dataset_config, caching=True)

        preproc_file = None
        infer_file = None
        plat_manager = None
        model_path = None
        platforms = []

        if qcc.STAGE_INFER in pipeline_stages:
            # platform config must be present
            platforms = config._inference_config._platforms  # list of platforms from config
            # Filter platforms using supplied cli platform and platform-tag
            platforms = self.filter_platforms(platforms, platform_name=platform_name,
                                              platform_tag=platform_tag)
            # Update the platform params via supplied cli args
            if platform_tag_params:
                platform_tag_params = self.parse_platform_tag_params(platform_tag_params)
                platforms = self.update_platform_params(platform_tag_params, platforms)

            if device_id:
                if backend is not None and backend=="htp":
                    device_ids = [device_id]
                elif isinstance(device_id,int):
                    # device_id=0 format
                    device_ids = [device_id]
                elif isinstance(device_id,str):
                    # Assumes '0,1' like string Format. Handle Sting parsing to list conversion
                    device_ids = [int(device) for device in device_id.strip().split(',')]
                else:
                    # Assumes [0,1] like list Format.
                    device_ids = [int(device) for device in device_id]
                # Validate in right range
                if backend is not None and backend == "htp":
                    status = True
                else:
                    status = Helper.validate_aic_device_id(device_ids)
                if status:
                    config._inference_config._aic_device_ids = device_ids

            # once platform(s) is selected perform further actions
            # create platform manager
            plat_manager = infer.PlatformManager(platforms, config)

            # search space scan and adding platform combination
            platforms, is_calib_req = plat_manager.scan_and_add_platform_permutations()

            # update the config object with all platform permutation
            config._inference_config._platforms = platforms
            config._inference_config._is_calib_req = is_calib_req

            # create schedule for different platforms
            plat_manager.create_schedule(inference_strategy)

        # get the pipeline_inputs
        if config._dataset_config:
            total_inputs = dataset.get_total_entries()

        # clean model if configured.
        if qcc.STAGE_INFER in pipeline_stages:
            # confirm for platforms and estimate space
            if config._dataset_config:
                self.confirmation_prompt(platforms, config, total_inputs, dataset,
                                         plat_manager, cleanup, silent)

            # clean only if the model is not a tf session or pytorch module
            # config._inference_config._model_object is True for tf session and pytorch module. False otherwise
            if not config._inference_config._model_object:
                if config._inference_config and config._inference_config._clean_model:
                    logging.info('Cleaning up model..')
                    symbols = {}
                    if config._inference_config._onnx_define_symbol:
                        sym_from_config = config._inference_config._onnx_define_symbol.split(' ')
                        for sym in sym_from_config:
                            elems = sym.split('=')
                            symbols[elems[0]] = int(elems[1])
                    if onnx_symbol:
                        for sym in onnx_symbol:
                            elems = sym[0].split(':')
                            symbols[elems[0]] = int(elems[1])
                    model_path = ModelHelper.clean_model_for_aic(config._inference_config._model_path,
                                                                out_dir=config._work_dir,
                                                                symbols=symbols,
                                                                check_model=config._inference_config._check_model)
                else:
                    model_path = config._inference_config._model_path
                # check batchsize to be passed to inference engine
                if config._inference_config._platforms[0]._input_info:
                    inp_dims = config._inference_config._platforms[0]._input_info
                    key_list = list(inp_dims.keys())
                    if len(key_list) == 1:
                        in_node = key_list[0]
                        bs = ModelHelper.get_model_batch_size(model_path, in_node)
                        qaic_logger.info(f'Batchsize from Model graph: {bs}')
                        if bs != config._info_config._batchsize:
                            # When Model bs != input_bs (supplied) override  input_bs for execution
                            pipeline_cache.set_val(qcc.INTERNAL_EXEC_BATCH_SIZE,
                                                config._info_config._batchsize)
                    else:
                        qaic_logger.warning(
                            'Setting batchsize for multiple inputs is currently unsupported')
            else:
                model_path = config._inference_config._model_path

        # set values in pipeline pipeline_cache
        pipeline_cache.set_val(qcc.PIPELINE_BATCH_SIZE, config._info_config._batchsize)
        pipeline_cache.set_val(qcc.PIPELINE_WORK_DIR, config._work_dir)
        pipeline_cache.set_val(qcc.PIPELINE_INFER_INPUT_INFO,
                               config._inference_config._platforms[0]._input_info)
        pipeline_cache.set_val(qcc.PIPELINE_INFER_OUTPUT_INFO,
                               config._inference_config._platforms[0]._output_info)
        pipeline_cache.set_val(qcc.QNN_SDK_DIR, qnn_sdk_dir)

        ret_status, preproc_file = self.execute_pipeline_stages(config, pipeline_stages,
                                                                plat_manager, model_path,
                                                                cli_work_dir=work_dir,
                                                                pipeline_start=pipeline_start,
                                                                pipeline_end=pipeline_end,
                                                                cli_preproc_file=cli_preproc_file,
                                                                cli_infer_file=cli_infer_file,
                                                                cleanup=cleanup,
                                                                enable_perf_flag=enable_perf_flag,
                                                                perf_iter_count=perf_iter_count)

        if ret_status:
            qaic_logger.info('Pipeline execution interrupted')

        # do comaparison of infer outputs across platforms if configured.
        if qcc.STAGE_INFER in pipeline_stages and config._inference_config._comparator['enabled'] \
                and len(platforms) > 1 and preproc_file is not None and (STAGE_INFER_PASS):
            ret_status = self.compare_infer_results(config, preproc_file)
            if ret_status:
                logging.error('qacc comparator failed.')

        # Constant for all platforms
        if config._dataset_config:
            ds_name = config._dataset_config._name
            batch_size = config._info_config._batchsize
            max_inputs = pipeline_cache.get_val(qcc.PIPELINE_MAX_INPUTS)
            qaic_logger.info(f"Using Dataset: {ds_name} Batch Size: {batch_size} Max Inputs : {max_inputs}")
        if pipeline_end !=qcc.STAGE_PREPROC:# Print Summary
            summary = []
            self.results = [] # Used for Api
            # print the reults in the same order as config
            for plat_idx, platform in enumerate(platforms):
                entry = []
                #platform_name = self.get_platform_name(plat_idx, platform._name)
                platform_name = platform.get_platform_name()
                status_code = self.platform_run_status[platform_name]['status']
                entry.append(platform_name)
                if self.platform_run_status[platform_name]['infer_stage_status']:
                    plat_status_str = f"{qcc.get_plat_status(status_code)} \nin {self.platform_run_status[platform_name]['infer_stage_status']}"
                else:
                    plat_status_str = qcc.get_plat_status(status_code)
                entry.append(plat_status_str)
                entry.append(platform._precision)
                backend_str = ''
                params_str = ''
                converter_params_str = ''
                if qcc.get_plat_status(status_code) != 'Success':
                    ret_status = 1
                if platform._backend:
                    backend_str = f"{platform._backend}\n{platform._target_arch}"
                entry.append(backend_str)
                if platform._compiler_params:
                    for k, v in platform._compiler_params.items():
                        params_str += '{}:{} \n'.format(k, v)
                if platform._runtime_params:
                    for k, v in platform._runtime_params.items():
                        params_str += '{}:{} \n'.format(k, v)
                entry.append(params_str)
                if platform._converter_params:
                    for k, v in platform._converter_params.items():
                        converter_params_str += '{}:{} \n'.format(k, v)
                entry.append(converter_params_str)
                if 'metrics' in self.platform_run_status[platform_name] and \
                        self.platform_run_status[platform_name]['metrics']:
                    metric_str =''
                    mertics_dict = self.platform_run_status[platform_name]['metrics']
                    for k, v in mertics_dict.items():
                        metric_str += '{}: {} \n'.format(k, v)
                    entry.append(metric_str)
                else:
                    mertics_dict = {}
                    entry.append('-')
                if 'comparator' in self.platform_run_status[platform_name]:
                    comparator_dict = self.platform_run_status[platform_name]['comparator']
                    comparator_str = ''
                    compare_value = float(list(comparator_dict.values())[0])
                    for k, v in comparator_dict.items():
                        comparator_str += '{}: {} \n'.format(k, v)
                    entry.append(comparator_str)
                    entry.append(compare_value)
                else:
                    comparator_dict = {}
                    entry.append('-')
                    entry.append(float("-inf"))
                summary.append(entry)
                self.results.append(
                    [platform._idx, platform._tag, platform_name, qcc.get_plat_status(status_code),
                     platform._precision,
                     platform._converter_params, mertics_dict, comparator_dict])  # appending metric results
            summary.sort(reverse=True, key=lambda x: x[-1])
            summary = [i[:-1] for i in summary]
            logging.info('Execution Summary:')
            headers = ['Platform', 'Status', 'Precision', 'Backend', 'Params', 'Converter Params' ,'Metrics', 'Comparator']
            console(tabulate(summary, headers=headers))
            result_csv_path = self.get_output_path(config._work_dir, qcc.RESULTS_TABLE_CSV)
            self.write2csv(result_csv_path, summary, header=headers)
            logging.info(f"\n{tabulate(summary, headers=headers)}")

        # delete output files of all stages.
        if qcc.CLEANUP_AT_END == cleanup:
            self.cleanup_files(config=config, stage='all')
        qaic_logger.debug(pipeline_cache._pipeline_cache)

        return ret_status

    def execute_pipeline_stages(self, config, pipeline_stages, plat_manager, model_path,cli_work_dir,
                                pipeline_start, pipeline_end, cli_preproc_file,cli_infer_file=None,cleanup=None,
                                enable_perf_flag=False, perf_iter_count=100 ):
        """
        Execute pipeline stages
        """
        # using global stage variables
        global STAGE_PREPROC_PASS
        global STAGE_INFER_PASS

        compile_only = pipeline_end == qcc.STAGE_COMPILE
        load_compiled_binary_from_dir = pipeline_start == qcc.STAGE_COMPILE

        # This is used during inference stage to support reuse_pgq option.
        # The dictionary stores
        # key: group_id and val: path to pgq profile
        # initially has no values and updated once profiles are generated
        group_pgq_dict = {}

        # capturing start time of calibration
        start_time = time.time()

        # perform preprocessing for calibration inputs
        # this adds support to supply calibration file with inputs files.
        # These inputs can be filenames other than what is mentioned in inputlist.
        # To use this add calibration section in the dataset.yaml as below:
        # calibration:
        #             type: filename
        #             file: calibration_file.txt
        if config._dataset_config is not None and config._dataset_config._calibration_file \
                and config._dataset_config._calibration_type == qcc.CALIBRATION_TYPE_DATASET \
                and (pipeline_end == qcc.STAGE_PREPROC
                     or config._inference_config._is_calib_req) and (qcc.STAGE_PREPROC in pipeline_stages):

            # modify the inputlist file to calibration file
            # this is done to execute all the preprocessing plugins
            # using files in calibration file
            calib_dataset_config = copy.deepcopy(config._dataset_config)
            calib_dataset_config._inputlist_file = config._dataset_config._calibration_file
            calib_dataset_config._max_inputs = config._inference_config._max_calib

            # create dataset object with inputlist as calibration file
            calib_dataset = ds.DataSet(dataset_config=calib_dataset_config, caching=True)

            # using batch index 0
            err_status, calib_file = self.preprocess(config, calib_dataset, True)
            if err_status:
                qaic_logger.info('Calibration preprocessing failed')
                return 1
            else:
                # Setting it to RAW as these inputs are already preprocessed
                config._dataset_config._calibration_type = qcc.CALIBRATION_TYPE_RAW
                config._dataset_config._calibration_file = calib_file

                # updating the max calib
                # This is added as in certain scenarios the number of processed outputs
                # could increase or decrease based on processing technique used like
                # in the case of BERT model.
                config._inference_config._max_calib = len(open(calib_file).readlines())
                pipeline_cache.set_val(qcc.PIPELINE_CALIB_FILE, calib_file)
                qaic_logger.info('Calibration preprocessing complete. calibration file: {}'
                                 .format(calib_file))
        else:
            # Setting calibration file to None in case of INT8 is not given as calibration is not required
            if not config._inference_config._is_calib_req and config._dataset_config:
                config._dataset_config._calibration_file = None

        # set calibration time
        self.capture_time(qcc.INTERNAL_CALIB_TIME, start_time)
        start_time = None  # reset start time

        # run the pipeline
        # create new dataset object
        dataset = ds.DataSet(dataset_config=config._dataset_config, caching=True)

        # Preprocessing
        # capturing start time of preprocessing
        start_time = time.time()
        if (qcc.STAGE_PREPROC in pipeline_stages) and (cli_preproc_file is None):
            err_status, preproc_file = self.preprocess(config, dataset)
            if err_status:
                STAGE_PREPROC_PASS = False
                return 1
            else:
                # calibration_file = dataset.get_dataset_calibration()
                STAGE_PREPROC_PASS = self.validate_pipeline_stage(qcc.STAGE_PREPROC, config)
                if not STAGE_PREPROC_PASS:
                    qaic_logger.info('{} stage validation failed'.format(qcc.STAGE_PREPROC))
                else:
                    pipeline_cache.set_val(qcc.PIPELINE_PREPROC_FILE, preproc_file)
        else:
            # required for idx based dataset access while
            # post processing
            if config._dataset_config:
                qaic_logger.info('Loading dataset')
                dataset.load_dataset()

            # if cli preproc not supplied and preprocessing stage is skipped then treat
            # input list as preproc. This is used for supporting scenarios where only
            # preprocessed data is available.
            logging.info('Loading preprocessed data')
            if cli_preproc_file:
                #Creates new file with absolute paths
                preproc_file = self.update_relative_paths(cli_preproc_file, cli_work_dir)
            elif pipeline_start == qcc.STAGE_COMPILE:
                # When loading from existing compiled output
                dir = self.get_output_path(cli_work_dir, qcc.STAGE_PREPROC)
                preproc_file = self.get_output_path(dir, qcc.QNN_PROCESSED_OUTFILE)
            else:
                # To support AUTO team where generally only preprocessed data is available
                preproc_file = dataset.get_input_list_file()
            STAGE_PREPROC_PASS = True
            pipeline_cache.set_val(qcc.PIPELINE_PREPROC_FILE, preproc_file)


        # set preprocessing time
        self.capture_time(qcc.INTERNAL_PREPROC_TIME, start_time)
        start_time = None  # reset start time
        if qcc.STAGE_INFER in pipeline_stages:

            if config._inference_config is None:
                logging.error('No inference section found in model config.'
                              'Use -pipeline-start and -pipeline-end flag to skip inference')
                return 1

            # get all the platforms
            platforms = config._inference_config._platforms

            # run a schedule in distributed manner
            for schedule in plat_manager.get_schedule():

                # get the scheduled platforms
                # schedule format: [(platform_idx, device_id), ... , (platform_idx, device_id)]
                # example: [[(0,-1), (1,0), (2,1)], [(3,0), (4,1)]]
                schd_plats = []

                for schd_plat in schedule:
                    plat_idx = schd_plat[0]
                    platform = platforms[plat_idx]
                    device_id = schd_plat[1]

                    # update load-profile with pgq profile path if available
                    if (platform._reuse_pgq) and (hasattr(platform, '_group_id')) \
                            and (platform._group_id in group_pgq_dict):
                        #platform_name = self.get_platform_name(plat_idx, platform._name)
                        platform_name = platform.get_platform_name()
                        platform._params['load-profile'] = group_pgq_dict[platform._group_id]
                        qaic_logger.debug('({}) loaded pgq profile {} for groupid {}'
                                          .format(platform_name, pgq_dir, platform._group_id))

                    # store in schd_plats
                    plat_tuple = (plat_idx, platform, device_id)
                    schd_plats.append(plat_tuple)

                # run inference sequentially for QNN
                #TODO: Parallelize the inference for QNN backends.
                for plat_idx, platform, device_id in schd_plats:
                    self.run_schedule_in_parallel(config, preproc_file,
                                                model_path, dataset, plat_idx, platform,
                                                device_id,
                                                pipeline_stages,
                                                compile_only_flag=compile_only,
                                                load_compiled_binary_from_dir_flag=load_compiled_binary_from_dir,
                                                enable_perf_flag=enable_perf_flag,
                                                perf_iter_count=perf_iter_count,cleanup=cleanup,
                                                cli_infer_file=cli_infer_file)

                # Parallel(n_jobs=-1, verbose=0, prefer="threads") \
                #     (delayed(self.run_schedule_in_parallel)(config, preproc_file,
                #                                             model_path, dataset, plat_idx, platform,
                #                                             device_id,
                #                                             pipeline_stages,
                #                                             compile_only_flag=compile_only,
                #                                             load_compiled_binary_from_dir_flag=load_compiled_binary_from_dir,
                #                                             enable_perf_flag=enable_perf_flag,
                #                                             perf_iter_count=perf_iter_count,cleanup=cleanup,
                #                                             cli_infer_file=cli_infer_file)
                #      for plat_idx, platform, device_id in schd_plats)

                # save pgq profile path
                for plat_idx, platform, device_id in schd_plats:
                    # update group_pgq_dict with pgq profile path if available
                    if (platform._reuse_pgq) and (hasattr(platform, '_group_id')) \
                            and (platform._group_id not in group_pgq_dict):
                        #platform_name = self.get_platform_name(plat_idx, platform._name)
                        platform_name = platform.get_platform_name()
                        # path where profile is stored
                        # pgq_dir = self.get_output_path(config._work_dir, qcc.STAGE_INFER,
                        #                                self.get_platform_name(plat_idx,
                        #                                                       platform._name))
                        pgq_dir = self.get_output_path(config._work_dir, qcc.STAGE_INFER,
                                                       platform.get_platform_name())
                        pgq_path = os.path.join(pgq_dir, qcc.PROFILE_YAML)
                        if os.path.exists(pgq_path):
                            group_pgq_dict[platform._group_id] = pgq_path
                            qaic_logger.debug('({}) updated groupid {} with path {}'
                                              .format(platform_name, platform._group_id, pgq_path))

            # marking infer stage passed
            STAGE_INFER_PASS = self.validate_pipeline_stage(qcc.STAGE_INFER, config)
            if not STAGE_INFER_PASS:
                qaic_logger.info('{} stage validation failed'.format(qcc.STAGE_INFER))

        # delete preprocessed outputs
        if STAGE_PREPROC_PASS and (qcc.CLEANUP_INTERMEDIATE == cleanup):
            self.cleanup_files(config, qcc.STAGE_PREPROC)

        # terminate pipeline if only preprocessing is configured
        if STAGE_PREPROC_PASS and pipeline_end == qcc.STAGE_PREPROC:
            # squash preproc files
            preproc_dir = self.get_output_path(config._work_dir, qcc.STAGE_PREPROC)
            if os.path.exists(preproc_dir):
                preproc_file = self.get_output_path(preproc_dir, qcc.QNN_PROCESSED_OUTFILE)
                pipeline_cache.set_val(qcc.PIPELINE_PREPROC_DIR, preproc_dir)
                pipeline_cache.set_val(qcc.PIPELINE_PREPROC_FILE, preproc_file)

            if not STAGE_INFER_PASS:
                return 0, preproc_file

        # setting paths and starting metric evaluation
        for plat_idx, platform in enumerate(platforms):
            #platform_name = self.get_platform_name(plat_idx, platform._name)
            platform_name = platform.get_platform_name()
            if self.platform_run_status[platform_name]['status'] in [qcc.PLAT_POSTPROC_FAIL,
                                                                     qcc.PLAT_INFER_FAIL]:
                continue

            # setting postprocessing file
            if qcc.STAGE_POSTPROC in pipeline_stages and config._postprocessing_config \
                    and self.platform_run_status[platform_name][
                'status'] == qcc.PLAT_POSTPROC_SUCCESS:
                # validate postproc stage
                if not self.validate_pipeline_stage(qcc.STAGE_POSTPROC, config):
                    qaic_logger.info('{} stage validation failed'.format(qcc.STAGE_POSTPROC))
                else:
                    postproc_file = pipeline_cache.get_val(qcc.PIPELINE_POSTPROC_FILE,
                                                           platform_name)
            else:
                postproc_file = pipeline_cache.get_val(qcc.PIPELINE_INFER_FILE, platform_name)

        return 0, preproc_file

    def confirmation_prompt(self, platforms, config, pipeline_batch_inputs, dataset,plat_manager,
                            cleanup, silent):
        """
        Prompts the user with
            - number of platforms
            - total space required in Distributed and Sequential Strategies
            - disabling of comparator
        """

        # disable comparator for intermediate delete
        # calculate based on delete option
        def log_disk_usage(size, msg):
            if size >= 1024:
                logging.info(msg + '  - {} GB'.format(round(size / 1024, 2)))
            else:
                logging.info(msg + '  - {} MB'.format(round(size, 2)))

        cleanup_inter = False
        if qcc.CLEANUP_INTERMEDIATE == cleanup:
            logging.info('Disabling comparator as -cleanup intermediate is selected')
            config._inference_config._comparator['enabled'] = False
            cleanup_inter = True

        num_platforms = len(platforms)
        total_req_sizes = self.get_estimated_req_size(num_platforms, config, dataset, cleanup_inter)
        logging.info('Total platform configurations: {}'.format(num_platforms))
        logging.info('Total inputs for execution: {} and calibration: {}'.format(
            config._dataset_config._max_inputs, config._inference_config._max_calib))
        preproc_size, calib_size, infer_size = total_req_sizes[0], total_req_sizes[1], \
                                               total_req_sizes[2]
        log_disk_usage(preproc_size + calib_size + infer_size, 'Approximate disk usage')
        if not cleanup_inter and plat_manager.get_schedule() is not None:
            plats = len(plat_manager.get_schedule()[0])  # get len of first schedule
            size = ((infer_size / num_platforms) * plats) + calib_size + preproc_size
            log_disk_usage(size, 'Approximate disk usage if -cleanup intermediate option is used')

        user_input = input('Do you want to continue execution? (yes/no) :').lower() \
            if not silent else 'y'
        if user_input not in ['yes', 'y']:
            logging.info('User terminated execution')
            sys.exit(1)

    def get_estimated_req_size(self, num_platforms, config, dataset, cleanup_inter=False):
        """
        Estimate the required size for Distributed strategy.

        Returns:
            total_req_sizes: [preproc, calib, infer]
        """

        def _parse_range(index_str):
            if len(index_str) == 0:
                return []
            nums = index_str.split("-")
            assert len(nums) <= 2, 'Invalid range in calibration file '
            start = int(nums[0])
            end = int(nums[-1]) + 1
            return range(start, end)

        if not hasattr(config, '_inference_config'):
            return [0, 0]  # inference section not availabe for calculation

        inputs = dataset.get_total_entries()

        size_dict = {'bool': 1, 'float': 4, 'float32': 4, 'float16': 2, 'float64': 8,
                     'int8': 1, 'int16': 2, 'int32': 4, 'int64': 8}

        # calculate preproc output size
        platforms = config._inference_config._platforms
        input_dims = platforms[0]._input_info
        qaic_logger.debug('input_dims type{} value {}'.format(type(input_dims), input_dims))
        preproc_size = 0
        batch_size = 1
        for in_node, val in input_dims.items():
            qaic_logger.debug('val {} for node {}'.format(val, in_node))
            if val[0] not in size_dict: # datatype
                qaic_logger.error('input type {} not supported in input_info '
                                  'in config'.format(val[0]))
            preproc_size_per_out = 1
            for idx, v in enumerate(val[1]): # tensor shape
                if 0 == idx:
                    batch_size = v
                preproc_size_per_out *= v
            preproc_size += preproc_size_per_out * size_dict[val[0]]
        total_preproc_size = preproc_size * (
                inputs / batch_size)  # (inputs/batch_size) --> num of preproc files
        qaic_logger.info('preproc size: {} MB'.format(round(total_preproc_size / (1024 * 1024)), 3))

        # calculate calibration output size
        calib_size = 0
        if config._dataset_config._calibration_file and config._inference_config._is_calib_req:
            calib_file = config._dataset_config._calibration_file
            if config._dataset_config._calibration_type == qcc.CALIBRATION_TYPE_DATASET \
                    or config._dataset_config._calibration_type == qcc.CALIBRATION_TYPE_RAW:
                calib_inputs = sum(1 for input in open(calib_file))
            else:
                cf = open(calib_file, 'r')
                indexes_str = cf.read().replace('\n', ',').strip()
                indexes = sorted(set(chain.from_iterable(map(_parse_range,
                                                             indexes_str.split(",")))))
                cf.close()
                calib_inputs = len(indexes)

            if -1 != config._inference_config._max_calib:
                calib_inputs = min(calib_inputs, config._inference_config._max_calib)
            else:
                config._inference_config._max_calib = calib_inputs
            calib_size = (calib_inputs / batch_size) * preproc_size
            qaic_logger.info('calib_inputs {} preproc_size {} batch_size {}'
                             .format(calib_inputs, preproc_size, batch_size))
        else:
            config._inference_config._max_calib = 0
        # Update the Pipeline cache with New Values after pre-processing
        pipeline_cache.set_val(qcc.PIPELINE_MAX_CALIB, config._inference_config._max_calib)
        # calculating infer output size
        platforms = config._inference_config._platforms
        output_dims = platforms[0]._output_info
        qaic_logger.debug('output_dims type{} value {}'.format(type(output_dims), output_dims))
        infer_size = 0
        batch_size = 1
        for out_node, val in output_dims.items():
            qaic_logger.debug('val {} for node {}'.format(val, out_node))
            if val[0] not in size_dict:
                qaic_logger.error('output type {} not supported in outputs_info '
                                  'in config'.format(val[0]))
            infer_size_per_out = 1
            for idx, v in enumerate(val[1]):
                if 0 == idx:
                    batch_size = v
                infer_size_per_out *= v
            infer_size += infer_size_per_out * size_dict[val[0]]

        infer_size = infer_size * num_platforms * (
                inputs / batch_size)  # (inputs/batch_size) --> num of infer files

        MB_divider = (1024 * 1024)
        total_req_sizes = [total_preproc_size / MB_divider, calib_size / MB_divider,
                           infer_size / MB_divider]
        return total_req_sizes

    def run_schedule_in_parallel(self, config, preproc_file, model_path, dataset,
                                 plat_idx, platform, device_id, pipeline_stages, compile_only_flag=False,
                                 load_compiled_binary_from_dir_flag=False, enable_perf_flag=False,
                                 perf_iter_count=100,cleanup='',cli_infer_file=None):
        """
        run in parallel
        """
        # platform_name = self.get_platform_name(plat_idx, platform._name)
        platform_name = platform.get_platform_name()
        qaic_logger.info('Pipeline Execution - Platform: {} running on device-id: {}'
                         .format(platform_name, device_id if not device_id == -1 else 'Not AIC'))

        err_status, infer_fail_stage, infer_file, execution_time = self.infer(model_path, config, preproc_file,
                                                            platform, dataset, device_id,
                                                            platform_name,
                                                            compile_only=compile_only_flag,
                                                            load_binary_from_dir=load_compiled_binary_from_dir_flag,
                                                            enable_perf=enable_perf_flag,
                                                            perf_iter_count=perf_iter_count)

        if err_status:
            logging.error('({}) inference failed'.format(platform_name))
            self.platform_run_status[platform_name] = {'status': qcc.PLAT_INFER_FAIL}
            self.platform_run_status[platform_name]['infer_stage_status'] = infer_fail_stage
            # exit the  thread
            return 1
        else:
            self.platform_run_status[platform_name] = {'status': qcc.PLAT_INFER_SUCCESS}
            self.platform_run_status[platform_name]['infer_stage_status'] = infer_fail_stage

        dir_name = self.get_output_path(config._work_dir, qcc.STAGE_INFER, platform_name)
        if platform._name == 'aic' and enable_perf_flag and not compile_only_flag:
            infer_runlog_path = self.get_output_path(dir=dir_name, type=qcc.INFER_RESULTS_FILE)
            profiling_result = self.parse_inference_runlog(infer_runlog_path)
            self.platform_run_status[platform_name]['throughput'] = profiling_result['Inf/Sec']
            self.platform_run_status[platform_name]['latency'] = profiling_result['avg_latency']
            # set quantization, compilation and infer time
        pipeline_cache.set_val(qcc.INTERNAL_QUANTIZATION_TIME, execution_time[0], platform_name)
        pipeline_cache.set_val(qcc.INTERNAL_COMPILATION_TIME, execution_time[1], platform_name)
        pipeline_cache.set_val(qcc.INTERNAL_INFER_TIME, execution_time[2], platform_name)

        # Post processing
        # capturing start time of post processing
        start_time = time.time()
        if qcc.STAGE_POSTPROC in pipeline_stages:
            if infer_file is None:
                if cli_infer_file:
                    infer_file = cli_infer_file
                else:
                    logging.error('infer-file needed if inference stage is skipped')
                    return 1
            err_status, postproc_file = self.postprocess(plat_idx, config, dataset, infer_file,
                                                         platform_name)
            if err_status:
                logging.error('({}) post processing failed'.format(platform_name))
                self.platform_run_status[platform_name]['status'] = qcc.PLAT_POSTPROC_FAIL
                return 1
            else:
                self.platform_run_status[platform_name]['status'] = qcc.PLAT_POSTPROC_SUCCESS

            # set post processing time
            self.capture_time(qcc.INTERNAL_POSTPROC_TIME, start_time, platform_name)
            start_time = None  # reset start time

            # delete intermediate inference output files if configured.
            if qcc.CLEANUP_INTERMEDIATE == cleanup and config._postprocessing_config:
                self.cleanup_files(config, qcc.STAGE_INFER, platform._name, plat_idx)

        # Metrics
        # capturing start time of infer
        start_time = time.time()
        if qcc.STAGE_METRIC in pipeline_stages:
            ret_status = self.evaluate_metrics(plat_idx, config, dataset, postproc_file,
                                                platform)
            if ret_status:
                logging.error('({}) Metrics evaluation failed. See qacc.log for more details.'
                                .format(platform_name))

            # delete postprocessed output files if configured.
            if qcc.CLEANUP_INTERMEDIATE == cleanup:
                if config._postprocessing_config:
                    self.cleanup_files(config, qcc.STAGE_POSTPROC, platform._name, plat_idx)
                else:
                    self.cleanup_files(config, qcc.STAGE_INFER, platform._name, plat_idx)

        # set metric time
        self.capture_time(qcc.INTERNAL_METRIC_TIME, start_time, platform_name)
        start_time = None  # reset start time

    def cleanup_files(self, config, stage, plat_name=None, plat_idx=None):
        """
        Cleanup output files generated during various stages of the pipeline
        """
        # check if cleaning all stages
        cleanup_all = ('all' == stage)

        # cleanup preproc outputs
        if qcc.STAGE_PREPROC == stage or cleanup_all:
            logging.info('Cleaning up pre-processed outputs')
            shutil.rmtree(self.get_output_path(config._work_dir, qcc.STAGE_PREPROC),
                          ignore_errors=True)
            shutil.rmtree(self.get_output_path(config._work_dir, qcc.STAGE_PREPROC_CALIB),
                          ignore_errors=True)

        # cleanup infer outputs
        if qcc.STAGE_INFER == stage or cleanup_all:
            logging.info('Cleaning up inference outputs')
            if plat_name is None and plat_idx is None:
                dir = self.get_output_path(config._work_dir, qcc.STAGE_INFER)
            else:
                dir = self.get_output_path(config._work_dir, qcc.STAGE_INFER,
                                           self.get_platform_name(plat_idx, plat_name))

            infer_files = []
            file_types = defaults.get_value('qacc.file_type.' + qcc.STAGE_INFER)
            file_types = [type.strip() for type in file_types.split(',')]
            for file_type in file_types:
                infer_files.extend(glob.glob(dir + '/**/*.' + file_type, recursive=True))
            for file in infer_files:
                if qcc.INFER_SKIP_CLEANUP in file:
                    continue
                os.remove(file)

        # cleanup postproc outputs
        if qcc.STAGE_POSTPROC == stage or cleanup_all:
            logging.info('Cleaning up post-processed outputs')
            if plat_name is None:
                shutil.rmtree(self.get_output_path(config._work_dir, qcc.STAGE_POSTPROC),
                              ignore_errors=True)
            else:
                shutil.rmtree(self.get_output_path(config._work_dir, qcc.STAGE_POSTPROC, plat_name),
                              ignore_errors=True)

    def validate_pipeline_stage(self, stage, config):
        """
        Performs validation on the pipeline stage results

        Returns:
             True: if the results are valid, False otherwise
        """
        exit_execution = False
        # if not enabled only show warning
        if defaults.get_value('qacc.zero_output_check'):
            exit_execution = True

        file_types = defaults.get_value('qacc.file_type.' + stage)
        file_types = [type.strip() for type in file_types.split(',')]

        dir = os.path.join(config._work_dir, stage)
        if os.path.exists(dir):
            files = []

            # fetch all files based on extension
            for file_type in file_types:
                files.extend(glob.glob(dir + '/**/' + '*.' + file_type, recursive=True))

            # if no files generated mark validation failed
            if 0 == len(files):
                qaic_logger.warning('No files found to validate')
                if exit_execution:
                    return False

            # check all files
            for file in files:
                # if file size zero mark validation failed
                if os.path.getsize(file) == 0:
                    qaic_logger.warning('File size zero: {}'.format(file))
                    if exit_execution:
                        return False

        # if didn't return False till this point means validation passed
        return True

    def capture_time(self, key, start_time, nested_key=None):
        pipeline_cache.set_val(key, time.time() - start_time, nested_key)

    def copy_pipeline_stage_execution_time(self, platforms, pipeline_stages):
        def get_time_from_dict(key, nested_key=None):
            if pipeline_cache.get_val(key, nested_key) is None:
                return 0
            else:
                return pipeline_cache.get_val(key, nested_key)

        # common execution time
        qaic_logger.info('Preprocessing Time Summary:')
        preproc_time = get_time_from_dict(qcc.INTERNAL_CALIB_TIME) + get_time_from_dict(
            qcc.INTERNAL_PREPROC_TIME)
        summary = [['Preprocessing', str(datetime.timedelta(seconds=preproc_time))]]

        table = tabulate(summary, headers=['Preprocessing', 'Time (hh:mm:ss)'])
        console(table)
        qaic_logger.info(table)

        if qcc.STAGE_INFER in pipeline_stages:
            qaic_logger.info('Platform Wise Time Summary (hh:mm:ss):')
            summary = []
            for plat_idx, platform in enumerate(platforms):
                entry = []
                total_time = 0
                #platform_name = self.get_platform_name(plat_idx, platform._name)
                platform_name = platform.get_platform_name()
                entry.append(platform_name)
                entry.append(str(datetime.timedelta(
                    seconds=get_time_from_dict(qcc.INTERNAL_QUANTIZATION_TIME, platform_name))))
                entry.append(str(datetime.timedelta(
                    seconds=get_time_from_dict(qcc.INTERNAL_COMPILATION_TIME, platform_name))))
                entry.append(str(datetime.timedelta(
                    seconds=get_time_from_dict(qcc.INTERNAL_INFER_TIME, platform_name))))
                entry.append(str(datetime.timedelta(
                    seconds=get_time_from_dict(qcc.INTERNAL_POSTPROC_TIME, platform_name))))
                entry.append(str(datetime.timedelta(
                    seconds=get_time_from_dict(qcc.INTERNAL_METRIC_TIME, platform_name))))
                phases = [qcc.INTERNAL_QUANTIZATION_TIME, qcc.INTERNAL_COMPILATION_TIME,
                          qcc.INTERNAL_INFER_TIME, qcc.INTERNAL_POSTPROC_TIME,
                          qcc.INTERNAL_METRIC_TIME]
                for phase in phases:
                    total_time += get_time_from_dict(phase, platform_name)
                entry.append(str(datetime.timedelta(seconds=total_time)))
                summary.append(entry)
            headers = ['Platform', 'Quantization', 'Compilation', 'Inference',
                       'Postprocessing', 'Metrics', 'Total']
            table = tabulate(summary, headers=headers)
            config = qc.Configuration.getInstance()
            profile_csv_path = self.get_output_path(config._work_dir, qcc.PROFILING_TABLE_CSV)
            self.write2csv(profile_csv_path, summary, header=headers)
            console(table)
            qaic_logger.info(table)

    def get_output_path(self, dir, type, plat_name=None):
        '''
        Returns the output directory for various stages of the pipeline
        '''
        # preprocessing or infer file or metric file
        if type in [qcc.STAGE_PREPROC, qcc.INFER_OUTFILE, qcc.PROCESSED_OUTFILE, qcc.QNN_PROCESSED_OUTFILE,
                    qcc.STAGE_PREPROC_CALIB, qcc.STAGE_INFER, qcc.STAGE_POSTPROC, qcc.STAGE_METRIC,
                    qcc.PROFILING_TABLE_CSV, qcc.RESULTS_TABLE_CSV, qcc.INFER_RESULTS_FILE,
                    qcc.DATASET_DIR, qcc.INPUT_LIST_FILE, qcc.CALIB_FILE] and plat_name is None:
            return os.path.join(dir, type)

        # inference or postprocessing
        elif type in [qcc.STAGE_INFER, qcc.STAGE_POSTPROC, qcc.STAGE_METRIC]:
            return os.path.join(dir, type, plat_name)

        # binary
        elif type == qcc.BINARY_PATH:
            return os.path.join(dir, qcc.STAGE_INFER, plat_name, 'temp')

    def get_platform_name(self, plat_idx, plat_name):
        return 'plat' + str(plat_idx) + '_' + plat_name

    def filter_platforms(self, platforms, platform_name=None, platform_tag=None):
        # select platform based on supplied args
        if platform_name:
            platforms = [p for p in platforms if p._name == platform_name]
            if len(platforms) == 0:
                logging.error('Invalid platform name in -platform option')
                sys.exit(1)
        if platform_tag:
            platforms = [p for p in platforms if
                         p._tag is not None and platform_tag in p._tag]
            if len(platforms) == 0:
                logging.error('Invalid platform tag in -platform_tag option')
                sys.exit(1)
        return platforms

    def update_platform_params(self, platform_tag_params, platforms):
        '''dict of dict: keys are platform_tags and values are dict containing platform_params'''
        plat_tags = platform_tag_params.keys()  # list of 1st Value passed in cli args
        for plat_tag in plat_tags:
            for plat in platforms:
                # For each platform in the config check if any platform params is passed
                if plat_tag in plat._tag and platform_tag_params[plat_tag]:
                    plat._params.update(platform_tag_params[plat_tag])
                    # Remove Keys with Empty value: Delete Support
                    for p_key, p_val in plat._params.items():
                        if p_val == '':
                            del plat._params[p_key]
                    qaic_logger.info(
                        "Updating the platform params of {} using supplied params via "
                        "cli with ".format(plat_tag))
                    qaic_logger.info(platform_tag_params[plat_tag])
        return platforms

    def parse_platform_tag_params(self, platform_tag_params_lists=[]):
        # Takes a list of list containing strings
        # return dict of dict containing parameter values of platforms based on tag_name
        # ["aic_int8, quantization-calibration: Percentile, xyz:200",
        # "aic_int8_2, load-profile:/path/to/file.yaml"]
        platform_tag_params = {}
        for param_string in platform_tag_params_lists:
            # cli based  platform_tag_params: List of list containing str (Sample below)
            # "aic_int8, quantization-calibration: Percentile, xyz:200"
            plat_tag_name, *cli_params = param_string[0].strip().split(',')
            if plat_tag_name == param_string[0]:  # No Splitting due to lack of delimiter
                logging.error(
                    "No Delimiter(,) found in the -platform-tag-params string passed. Check "
                    "delimiter in "
                    "-platform-tag-params")
                sys.exit(1)
            if plat_tag_name == '':  # Empty platform-tag passed (Currently not supported)
                logging.error(
                    "platform tag not provided in the -platform-tag-params passed. Got empty "
                    "platform tag")
                sys.exit(1)
            if len(cli_params) == 0:  # Only platform-tag passed.
                logging.error(
                    "-platform-tag-params passed has no parameters to update.")
                sys.exit(1)
            params_dict = {}
            for cli_param in cli_params:
                try:
                    param_name, param_value = cli_param.split(":")
                    if param_name == '':  # check param_name is not empty
                        logging.error(
                            "parameter name within platform-tag-param cannot be empty.")
                        sys.exit(1)
                except:
                    logging.error(
                        "{} supplied within platform-tag-param could not be split using ':' "
                        "delimiter".format(cli_param))
                    sys.exit(1)
                params_dict[param_name.strip()] = param_value.strip()
            platform_tag_params[plat_tag_name.strip()] = params_dict
        return platform_tag_params

    def write2csv(self, fname, rows, header):
        # check all rows have same length
        assert len(header) == len(rows[0])
        with open(fname, 'w') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(header)
            writer.writerows(rows)

    def parse_inference_runlog(self, runlog_path):
        with open(runlog_path) as file:
            lines = [line.strip() for line in file.readlines()]
        result_dict = {}
        inf_result_line = lines[-1]
        inf_result_split = inf_result_line.split(" ")
        for i in range(len(inf_result_split) // 2):
            result_dict[inf_result_split[i * 2]] = inf_result_split[i * 2 + 1]
        result_dict['TotalDuration'] = result_dict['TotalDuration'].rstrip('us')
        inf_rate = float(result_dict['Inf/Sec'])
        total_inference_time = float(result_dict['TotalDuration'])
        batch_size = float(result_dict['BatchSize'])
        result_dict['avg_latency'] = self.get_avg_latency(inf_rate, total_inference_time,
                                                          batch_size)
        return result_dict

    def get_avg_latency(self, inf_rate, total_inference_time, batch_size):
        '''
        Returns average latency or failure message.

        Finds
        1. total inferences = total inf time * inf rate
        2. total transactions = total inferences/batch size
        3. avg latency = total inf time/total transactions
        '''
        total_inferences = inf_rate * total_inference_time * pow(10, -6)
        total_transactions = total_inferences / batch_size
        avg_time_taken = float(total_inference_time) / total_transactions
        return round(avg_time_taken, 4)

    def get_out_names(self, out_file):
        """ Returns the names of the outputs from the out_file
        """
        out_names = []
        with open(out_file) as ref_file:
            outputs = ref_file.readline().split(',')
        for op in outputs:
            file_name, _ = os.path.splitext(op.split('/')[-1])
            out_names.append(file_name)

        return out_names

    def update_relative_paths(self, preproc_file, work_dir):
        """
        Create a new preproc file and modify the relative paths to absolute paths
        """
        updated_preproc_file = os.path.join(work_dir, "updated_input_list.txt")
        original_list_dir = os.path.dirname(os.path.abspath(preproc_file))
        with open(updated_preproc_file, "w") as write_file, \
             open(preproc_file, "r") as read_file:

            for line in read_file:
                write_file.write(os.path.join(original_list_dir, line))

        return updated_preproc_file

    def get_pipeline_stages_from_config(self, config):
        pipeline_stages = [qcc.STAGE_PREPROC, qcc.STAGE_COMPILE, qcc.STAGE_INFER,
                           qcc.STAGE_POSTPROC, qcc.STAGE_METRIC]
        pipeline_start = 'infer'
        pipeline_end = 'infer'
        if config._preprocessing_config:
            pipeline_start = qcc.STAGE_PREPROC
            pipeline_end = qcc.STAGE_PREPROC
        if config._inference_config:
            pipeline_end = qcc.STAGE_INFER
        if config._postprocessing_config:
            pipeline_end = qcc.STAGE_POSTPROC
        if config._evaluator_config:
            pipeline_end = qcc.STAGE_METRIC
        pipeline_stages = pipeline_stages[pipeline_stages.index(pipeline_start):
                                              pipeline_stages.index(pipeline_end) + 1]

        return pipeline_stages, pipeline_start, pipeline_end
