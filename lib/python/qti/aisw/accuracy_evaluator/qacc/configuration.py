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
import ast
import os
import re
import shutil
import sys
import yaml
import copy
import json

import qti.aisw.accuracy_evaluator.common.exceptions as ce
import qti.aisw.accuracy_evaluator.qacc.plugin as pl
from qti.aisw.accuracy_evaluator.qacc.utils import convert_npi_to_json
from qti.aisw.accuracy_evaluator.common.transforms import ModelHelper
from qti.aisw.accuracy_evaluator.common.utilities import Helper, ModelType
from qti.aisw.accuracy_evaluator.qacc import defaults
from qti.aisw.accuracy_evaluator.qacc import qaic_logger
from qti.aisw.accuracy_evaluator.qacc.constants import Constants as qcc

DATASETS_YAML_PATH = os.path.join(os.path.dirname(__file__), '..', 'configs', 'datasets.yaml')

class Configuration:
    """
    QACC configuration class having all the configurations supplied by the user.

    To use:
    >>> config = Configuration()
    >>> config.load_config_from_yaml()

    >>> dataset = DatasetConfiguration(name, path, inputlist_file,
                                      annotation_file, calibration_file, max_inputs)

    >>> config.set_config(dataset_config=dataset)
    """
    __instance = None

    def __init__(self):
        if Configuration.__instance != None:
            raise ce.ConfigurationException('instance of config already exists')
        else:
            Configuration.__instance = self

    @classmethod
    def getInstance(cls):
        if Configuration.__instance == None:
            Configuration()
        return cls.__instance

    def load_config_from_yaml(self, config_path,work_dir,set_global=None,batchsize=None,
                             dataset_config_yaml=DATASETS_YAML_PATH, model_path=None):
        """
        loads the config from yaml file

        Args:
            config_path: path to yaml configuration
            work_dir: Work directory to be used to store results and other artifacts
            set_global: Globa constants values supplied via cli
            batchsize: batchsize to be updated from value passed from cli

        Raises:
            ConfigurationException:
                - if incorrect configuration file provided
                - configuration file empty
        """

        # Copy config file in work dir.
        self._work_dir = work_dir

        if not os.path.exists(self._work_dir):
            os.makedirs(self._work_dir)

        _, config_file_name = os.path.split(config_path)
        cur_path = os.path.join(self._work_dir, config_file_name)
        shutil.copyfile(config_path, cur_path)

        with open(cur_path, 'r') as stream:
            try:
                config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                raise ce.ConfigurationException('incorrect configuration file', exc)

        if config == None:
            raise ce.ConfigurationException('configuration file empty')
        elif config['model'] == None:
            raise ce.ConfigurationException('model key not found in configuration file')

        is_config_updated = False

        #Updating the model_path to dir
        if model_path is not None:
            config['model']['inference-engine']['model_path'] = model_path
        #Write back to the file
        with open(cur_path, 'w') as stream:
            yaml.dump(config, stream)

        if 'globals' in config['model'] and config['model']['globals']:
            gconfig = config['model']['globals']

            if len(gconfig) > 0:
                is_config_updated = True
                # replace globals with cmd line args -global
                cmd_gconfig = {}
                if set_global:
                    for g in set_global:
                        elems = g[0].split(':')
                        cmd_gconfig[elems[0]] = elems[1]
                    gconfig.update(cmd_gconfig)

                # update config file with globals.
                with open(cur_path, 'r') as stream:
                    file_data = stream.read()

                with open(cur_path, 'w') as stream:
                    for k, v in gconfig.items():
                        file_data = file_data.replace('$' + k, str(v))
                    stream.write(file_data)
        bs_key = qcc.MODEL_INFO_BATCH_SIZE
        if batchsize:
            bs = batchsize
        elif 'info' in config['model'] and 'batchsize' in config['model']['info']:
            bs = config['model']['info'][bs_key]
        else:
            bs = 1
        bs = str(bs)
        with open(cur_path, 'r') as stream:
            file_data = stream.read()
        # modify the batchsize in the config file
        file_data = re.sub('[\"\']\s*\*\s*[\"\']', bs, file_data)
        file_data = re.sub(bs_key + ':\s+\d+', bs_key + ': ' + bs, file_data)
        file_data = re.sub("-\s\'\*\'", '- '+bs, file_data)

        # write again so that config can be reloaded
        with open(cur_path, 'w') as stream:
            stream.write(file_data)

        with open(cur_path, 'r') as stream:
            try:
                config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                raise ce.ConfigurationException('incorrect configuration file', exc)

        self.set_config_from_yaml(config['model'], dataset_config_yaml=dataset_config_yaml)
        if 'info' in config['model']:
            qaic_logger.info('Model Info : {}'.format(config['model']['info']))
        if 'globals' in config['model']:
            qaic_logger.info('Global vars : {}'.format(config['model']['globals']))

    def set_config_from_yaml(self, config,dataset_config_yaml=DATASETS_YAML_PATH):
        """
        Set dataset, processing, inference and evaluator config using yaml config.

        Args:
            config: config as dictionary from yaml

        Raises:
            ConfigurationException
        """

        if 'info' in config:
            info = InfoConfiguration(**config['info'])
        else:
            info = InfoConfiguration()
        if 'dataset' in config:
            dataset = DatasetConfiguration(**config['dataset'],dataset_config_yaml=dataset_config_yaml)
        else:
            dataset = None
            qaic_logger.info('No dataset section found')

        if self.is_sub_section_valid(config, 'processing', 'preprocessing'):
            preprocessing = ProcessingConfiguration(**config['processing']['preprocessing'])
            # Squash is disabled for preprocessing.
            preprocessing.squash_results = False
        else:
            preprocessing = None
            qaic_logger.info('No preprocessing section found')

        if self.is_sub_section_valid(config, 'processing', 'postprocessing'):
            postprocessing = ProcessingConfiguration(**config['processing']['postprocessing'])
        else:
            postprocessing = None
            qaic_logger.info('No postprocessing section found')

        if ('inference-engine' in config):
            infer_engines = InferenceEngineConfiguration(**config['inference-engine'])
        else:
            infer_engines = None
            qaic_logger.info('No inference-engine section found')

        if ('evaluator' in config) and ('metrics' in config['evaluator']):
            evaluator = EvaluatorConfiguration(**config['evaluator'])
        else:
            evaluator = None
            qaic_logger.info('No evaluator section found')

        self.set_config(info_config=info,
                        dataset_config=dataset,
                        preprocessing_config=preprocessing,
                        postprocessing_config=postprocessing,
                        inference_config=infer_engines,
                        evaluator_config=evaluator)

    def is_sub_section_valid(self, config, section, subsection):
        """
        Returns true if the subsection is configured in the model config
        """
        if (section in config) and (config[section]) and \
                (subsection in config[section]):
            return True
        else:
            return False

    def set_config(self, info_config=None, dataset_config=None, preprocessing_config=None,
                   postprocessing_config=None, inference_config=None, evaluator_config=None):
        """
        Setter for config

        Args:
            info_config
            dataset_config
            preprocessing_config
            postprocessing_config
            inference_config
            evaluator_config
        """
        self._info_config = info_config
        self._dataset_config = dataset_config
        self._preprocessing_config = preprocessing_config
        self._postprocessing_config = postprocessing_config
        self._inference_config = inference_config
        self._evaluator_config = evaluator_config

    def get_ref_platform(self):
        ref_found = False
        ref_platforms = []
        for platform in self._inference_config._platforms:
            if platform._is_ref:
                ref_found = True
                qaic_logger.info('[configuration] plat'+ str(platform._idx) + '_' + platform._name + '[is_ref=True]')
                ref_platforms.append(platform)

        if not ref_found:
            qaic_logger.info('is_ref is not set for any platform, tool is using first platform as reference platform')
            qaic_logger.info('reference platform name=plat0_' + self._inference_config._platforms[0]._name)
            return self._inference_config._platforms[0]
        else:
            if len(ref_platforms) > 1 :
                qaic_logger.info('is_ref is set to True for multiple platforms')
                qaic_logger.info('tool is using first configured platform as reference platform')

            qaic_logger.info('reference platform name=plat'+ str(ref_platforms[0]._idx) + '_' + ref_platforms[0]._name)
            return ref_platforms[0]

class InfoConfiguration:

    def __init__(self, desc=None, batchsize=1):
        self._desc = desc
        self._batchsize = batchsize if batchsize is not None else 1
        if batchsize is None:
            qaic_logger.error(
                '{} not present in info section of model config. Using {} = 1.'.format(
                    qcc.MODEL_INFO_BATCH_SIZE, qcc.MODEL_INFO_BATCH_SIZE))


class DatasetConfiguration:
    """
    QACC dataset configuration class

    To use:
    >>> dataset_config = DatasetConfiguration(name, path, inputlist_file,
                                                    annotation_file, calibration, max_inputs)
    """

    def __init__(self, name=None, path=None, inputlist_file=None, annotation_file=None,
                 calibration=None, max_inputs=None, transformations=None,dataset_config_yaml='datasets.yaml'):
        self._name = name
        self._dataset_config_yaml = dataset_config_yaml

        if self._dataset_config_yaml and (not os.path.exists(self._dataset_config_yaml)):
            raise ce.ConfigurationException(
                'datasets.yaml does not exist. Invalid path for datasets.yaml file')

        self._read_dataset_config(name)

        # Override if provided in config
        if path:
            self._path = path
            self._inputlist_path = path
            self._calibration_path = path
        if inputlist_file:
            self._inputlist_file = os.path.join(self._inputlist_path, inputlist_file)
        if annotation_file:
            self._annotation_file = os.path.join(self._path, annotation_file)
        if calibration:
            self._calibration_file = os.path.join(self._calibration_path, calibration['path'])
            self._calibration_type = calibration['type']
        self._max_inputs = max_inputs
        self._update_max_inputs()

        # instantiate transformations
        self._transformations = TransformationsConfiguration(transformations)

    def _update_max_inputs(self):
        max_count = sum(1 for input in open(self._inputlist_file))
        #if self._max_inputs is None or -1 == self._max_inputs:
        self._max_inputs = max_count

    def __str__(self):
        return self._name

    def _read_dataset_config(self, ds_name):
        with open(self._dataset_config_yaml, 'r') as stream:
            try:
                config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                raise ce.ConfigurationException('incorrect dataset configuration file', exc)

        if ds_name not in config['Datasets']:
            raise ce.ConfigurationException(
                'Invalid Dataset Key. Checks datasets.yaml for configured datasets')

        try:
            self._path = config['Datasets'][ds_name]['path']
            self._calibration_path = self._path
            self._inputlist_path = self._path
            self._inputlist_file = os.path.join(self._path,
                                                config['Datasets'][ds_name]['inputlist_file'])
            self._annotation_file = None
            if 'annotation_file' in config['Datasets'][ds_name]:
                self._annotation_file = config['Datasets'][ds_name]['annotation_file']
                self._annotation_file = os.path.join(self._path, self._annotation_file)

            _calibration = config['Datasets'][ds_name]['calibration']
            if _calibration:
                self._calibration_file = os.path.join(self._path, _calibration['file'])
                self._calibration_type = _calibration['type']
            else:
                self._calibration_file = None
                self._calibration_type = None

        except Exception as exc:
            raise ce.ConfigurationException(
                'incorrect dataset configuration file for dataset: {}'.format(ds_name))

        if not os.path.exists(self._inputlist_file):
            raise ce.ConfigurationException(
                'Invalid path for dataset {} inputlist_file'.format(ds_name))

        if self._annotation_file and (not os.path.exists(self._annotation_file)):
            raise ce.ConfigurationException(
                'Invalid path for dataset {} annotation_file'.format(ds_name))

        if _calibration and (not os.path.exists(self._calibration_file)):
            raise ce.ConfigurationException(
                'Invalid path for dataset {} calibration file'.format(ds_name))
        if _calibration and (self._calibration_type not in [qcc.CALIBRATION_TYPE_INDEX,
                                                            qcc.CALIBRATION_TYPE_RAW,
                                                            qcc.CALIBRATION_TYPE_DATASET]):
            raise ce.ConfigurationException(
                'Invalid type for dataset {} calibration. Can be index|raw|dataset'.format(ds_name))

    def validate(self):
        # TODO: check file exists or not
        """
        Validates the dataset config

        Returns:
            path: true if the dataset config is valid and false otherwise.
        """
        pass


class ProcessingConfiguration:
    """
    QACC processing configuration class handling both preprocessing and postprocessing.

    To use:
    >>> preprocessing_config = ProcessingConfiguration(name, path, generate_annotation,
    inputlist_file,
                                                        annotation_file, calibration_file,
                                                        max_inputs)
    """

    def __init__(self, transformations, path=None, target=None, enable=True, squash_results=False,
                 save_outputs=False, defaults=None):
        self._transformations = TransformationsConfiguration(transformations)
        self._path = path
        # squash only used for post processors.
        self._squash_results = squash_results
        if target == None:
            # TODO set target from default config
            pass
        else:
            self._target = target
        self._enable = enable


class PluginConfiguration:
    """
    QACC plugin configuration class
    """

    def __init__(self, name, input_info=None, output_info=None, env=None, indexes=None,
                 params=None):
        self._name = name
        if name in pl.PluginManager.registered_plugins:
            self._cls = pl.PluginManager.registered_plugins[name]
            self._input_info = self.get_info_dict(input_info, type='in')
            self._output_info = self.get_info_dict(output_info, type='out')
        elif name in pl.PluginManager.registered_metric_plugins:
            # metric plugins dont need input and output info.
            self._cls = pl.PluginManager.registered_metric_plugins[name]
            self._input_info = None
            self._output_info = None
        elif name in pl.PluginManager.registered_dataset_plugins:
            # dataset plugins don't need input and output info.
            self._cls = pl.PluginManager.registered_dataset_plugins[name]
            self._input_info = None
            self._output_info = None
        else:
            raise ce.ConfigurationException('Configured plugin {} is not registered'.format(name))

        self._env = env
        self._indexes = indexes.split(',') if indexes else None
        self._params = params

    def get_info_dict(self, info, type):
        """
        type=mem|path|dir, dtype=float32, format=cv2
        """
        info_dict = {}
        if info:
            info = info.split(',')
            for i in info:
                kv = i.strip().split('=')
                info_dict[kv[0]] = kv[1]
        else:
            # use default defined in Plugin class
            if type == 'in':
                info_dict = self._cls.default_inp_info
            else:
                info_dict = self._cls.default_out_info

        return info_dict

    def __str__(self):
        return '\nPlugin Info::\nName: {}\nInput:{}\nOutput:{}\nEnv:{}\nIndex:{}\n' \
               '\nParams:{}' \
            .format(self._name, self._input_info, self._output_info, self._env, self._indexes,
                    self._params)


class TransformationsConfiguration:
    """
    QACC transformations configuration class
    """

    def __init__(self, transformations):
        self._plugin_config_list = self.get_plugin_list_from_dict(
            transformations) if transformations else []

    def get_plugin_list_from_dict(self, transformations):
        """
        Returns a list of plugin objects from the dictionary
        of plugin objects

        Args:
            transformations: transformations as dictionary

        Returns:
            plugin_config_list: list of plugin config objects
        """
        plugin_config_list = []
        for plugin in transformations:
            plugin_config_list.append(PluginConfiguration(**plugin['plugin']))

        # validate plugin list
        self.validate_plugin_config_list(plugin_config_list)
        return plugin_config_list

    def validate_plugin_config_list(self, plugin_config_list):
        """
        Directory plugins can't have output configured as path/mem
        """
        for idx, plugin in enumerate(plugin_config_list):
            if plugin._input_info and plugin._input_info[qcc.IO_TYPE] == qcc.PLUG_INFO_TYPE_DIR:
                if plugin._output_info[qcc.IO_TYPE] != qcc.PLUG_INFO_TYPE_DIR:
                    raise ce.ConfigurationException(
                        '{} : Directory plugin with Path or Mem output type '
                        'not supported'.format(plugin._name))


class InferenceEngineConfiguration:

    def __init__(self, model_path, clean_model=True, platforms=None, inputs_info=None,
                 outputs_info=None, comparator=None, aic_device_ids=[0], max_calib=-1,
                 check_model=True, onnx_define_symbol=None, auto_platform=None, platform_common_params=None):
        self._model_object = False # this flag is to check and skip model cleaning if model_path is a tf.session or nn.module
        self._platforms = []
        self._clean_model = True  # Setting this to true everytime
        self._model_path = os.path.join(defaults.get_value('common.model_zoo_path'), model_path)
        qaic_logger.warning(
            'max_calib in inference-engine section is deprecated, use max_calib in dataset plugin'
            'inside model configuration yaml')
        self._max_calib = max_calib
        self._aic_device_ids = aic_device_ids
        # used to perform calibration if int8 platform available
        self._is_calib_req = False
        # Skip framework specific Validations if any
        self._check_model = check_model
        # Any Symbols used internally by onnx model
        self._onnx_define_symbol = onnx_define_symbol
        #  prams that are common across all aic platforms: eg custom-op
        self._platform_common_params = platform_common_params
        #Save input names in order
        self._input_names = []
        # comparator is enabled by default. If not provided, create default config.
        if comparator is None:
            self._comparator = {'enabled': True, 'fetch_top': 1, 'type': 'avg', 'tol': 0.001}
        else:
            self._comparator = {}
            self._comparator['enabled'] = comparator['enabled'] if 'enabled' in comparator else True
            self._comparator['fetch_top'] = comparator[
                'fetch-top'] if 'fetch-top' in comparator else 1
            self._comparator['type'] = comparator['type'] if 'type' in comparator else 'avg'
            self._comparator['tol'] = comparator['tol'] if 'tol' in comparator else 0.001
            self._comparator['box_input_json'] = comparator['box_input_json'] if 'box_input_json' in comparator else None

        # Formatting the input_info as needed by inference engine.
        fmt_input_info = None
        if inputs_info:
            fmt_input_info = {}
            for input_info in inputs_info:
                for k, m in input_info.items():
                    assert len(
                        m.values()) == 2, 'Invalid format for input info. Should have type and shape keys'
                    if isinstance(m['shape'], list):
                        fmt_input_info[ModelHelper._replace_special_chars(str(k))] = [m['type'],m['shape']]
                        self._input_names.append(ModelHelper._replace_special_chars(str(k)))
                    else:
                        raise ce.ConfigurationException('Invalid shape in input info :{}.'
                                                        ' usage e.g: [1,224,224,3]'.format(m['shape']))
        fmt_output_info = None
        if outputs_info:
            fmt_output_info = {}
            for out_info in outputs_info:
                for k, m in out_info.items():
                    assert len(
                        m.values()) >= 2, 'Invalid format for output info. Must have type and shape ' \
                                    'keys'
                    if isinstance(m['shape'], list):
                        fmt_output_info[ModelHelper._replace_special_chars(str(k))] = [m['type'],m['shape']]
                    else:
                        raise ce.ConfigurationException('Invalid shape in output info :{}.'
                                                        ' usage e.g: [1,224,224,3]'.format(m['shape']))

        #Making the output info to float32 in qnn platform
        platform_names = []
        #  Auto Platform
        if isinstance(auto_platform, dict):
            self._auto_platform = auto_platform.get('enabled', False)
            run_native_platform = auto_platform.get('run_native_platform', True)
        else:  # for bool and None Case
            self._auto_platform = auto_platform
            run_native_platform = True
        if self._auto_platform:
            platforms = self.add_auto_platform(run_native_platform=run_native_platform)
            for idx, plat in enumerate(platforms):
                plat._input_info = fmt_input_info
                plat._output_info = fmt_output_info
                plat._model_path = model_path
                plat._idx = idx
                platform_names.append(plat._name)
                self._platforms.append(plat)
        elif platforms and len(platforms) > 0:
            auto_quantization_flag = False
            #TODO: Enable auto quantization for QNN
            for idx, platform in enumerate(platforms):
                if (platform['platform']['precision'] == qcc.PRECISION_QUANT and
                    'converter_params' not in platform['platform']):
                    platform['platform']['converter_params'] = defaults.get_value(
                            'qacc.auto_quantization')
                    auto_quantization_flag = True
                if self._platform_common_params:
                    platform = self.update_platform_common_params(platform)
                plat = QnnPlatformConfiguration(**platform['platform'])
                plat._input_info = fmt_input_info
                plat._output_info = fmt_output_info
                plat._model_path = model_path
                plat._idx = idx
                self._platforms.append(plat)
                if auto_quantization_flag:
                    break  # Stop adding other combinations
                qaic_logger.debug(f"Platform name : {plat._name}")
                platform_names.append(plat._name)
        else:
            raise ce.ConfigurationException("No Platforms to run. Check the config file.")

        #qnn-net-run outputs only in float32 ?
        if plat._output_info:
            for idx, plat in enumerate(self._platforms):
                qaic_logger.info(f"{plat._name} {plat._idx}")
                qaic_logger.info(plat._output_info)
                if plat._name == "qnn":
                    op_info = copy.deepcopy(plat._output_info)
                    for op, info in op_info.items():
                        if info != "float32":
                            op_info[op][0]  = "float32"
                    plat._output_info = op_info
                qaic_logger.info(plat._output_info)

        #Add int64 flags if input type is int64
        if fmt_input_info:
            is_inp_int64 = False
            for inp, info in fmt_input_info.items():
                if info[0] == "int64":
                    is_inp_int64 = True

            if is_inp_int64:
                for idx, plat in enumerate(self._platforms):
                    if plat._name == "qnn":
                        converter_params =copy.deepcopy(plat._converter_params)
                        converter_params["keep_int64_inputs"] = "True"
                        converter_params["use_native_dtype"] = "True"
                        plat._converter_params = converter_params


    def update_platform_common_params(self, platform):
        for plat_param in self._platform_common_params:
            for plat_type, params in plat_param.items():
                if plat_type == platform['platform']['name']:
                    # add params section if not already present
                    if 'params' not in platform['platform']:
                        platform['platform']['params'] = {}

                    for k, v in params.items():
                        if k in platform['platform']['params']:  # if key:value given within platform ignore
                            continue
                        else:  # If not given update
                            platform['platform']['params'].update({k: v})
        return platform

    def add_auto_platform(self,run_native_platform=True):
        ''' Adds a Reference(Native) platform, fp16 and int8 aic platforms'''
        platforms = []
        if run_native_platform:
            reference_platforms = [qcc.INFER_ENGINE_ONNXRT, qcc.INFER_ENGINE_TFRT,
                                   qcc.INFER_ENGINE_TORCHSCRIPTRT]
            ref_plat_type_idx= Helper.get_model_type(self._model_path).value
            ref_name = reference_platforms[ref_plat_type_idx]
            reference_plat = PlatformConfiguration(name=ref_name, precision='fp32', is_ref=True)
            platforms.append(reference_plat)

        aic_common_params = {}
        if self._platform_common_params:
            aic_common_params = [plat_param for plat_param in self._platform_common_params for
                                 plat_type, params in plat_param.items() if plat_type == 'aic']
            if len(aic_common_params) != 0:
                aic_common_params = aic_common_params[0][
                    'aic']  # Always the 1st item [Only Distinct platform types]

        if self._platform_common_params is None or len(aic_common_params) == 0:
            aic_common_params = {}

        # fp16 aic platform
        fp16_platform = PlatformConfiguration(name='aic', precision='fp16',
                                              params=aic_common_params)
        platforms.append(fp16_platform)

        auto_quant_params = defaults.get_value('qacc.auto_quantization')
        auto_quant_params.update(aic_common_params)

        auto_quant_platform = PlatformConfiguration(name='aic', precision='int8',
                                                    params=auto_quant_params)
        platforms.append(auto_quant_platform)

        return platforms


class PlatformConfiguration:

    def __init__(self, name, env=None, params={}, binary_path=None,
                 multithreaded=True, precision='fp32', use_precompiled=None, reuse_pgq=True,
                 pgq_group=None, tag='', is_ref=False):

        self._name = name
        self._idx =None
        self._env = env
        self._params = params if params else {}
        # batchsize is added in manager based on exec support of providing
        # batchsize with a model having multiple inputs.
        # This filed is added while setting pipeline cache with key INTERNAL_EXEC_BATCH_SIZE

        # onnxrt specific
        self._multithreaded = multithreaded

        # AIC specific params
        self._use_precompiled = use_precompiled
        self._binary_path = binary_path
        if precision == 'default':
            self._precision = qcc.PRECISION_FP32
        else:
            self._precision = precision
        self._input_info = None
        self._output_info = None
        self._precompiled_path = use_precompiled
        self._is_ref = is_ref

        # to specify if PGQ profile to be reused across
        # the generated platforms after search space scan
        self._reuse_pgq = reuse_pgq

        # to specify if PGQ profile can be used across multiple
        # platforms. All the platforms having the pgq_group tag will
        # use the same generated PGQ profile. If reuse_pgq is False the
        # PGQ profile will be regenerated for all the configured platforms
        # in one platform section.
        self._pgq_group = pgq_group

        # platform tag used to filter a platform from multiple same
        # type of platforms.
        # Example: Used to distinguish between multiple AIC platforms
        self._tag = [t.strip() for t in tag.split(',')]
        self.validate()

    def validate(self):
        if self._name is None:
            raise ce.ConfigurationException('inference-engine platform name is mandatory!')
        if self._name not in [qcc.INFER_ENGINE_AIC,
                              qcc.INFER_ENGINE_AIC_SIM,
                              qcc.INFER_ENGINE_ONNXRT,
                              qcc.INFER_ENGINE_TFRT,
                              qcc.INFER_ENGINE_TFRT_SESSION,
                              qcc.INFER_ENGINE_TORCHSCRIPTRT]:
            raise ce.ConfigurationException(
                'Invalid or Unsupported platform: {}'.format(self._name))

        if self._precision not in [qcc.PRECISION_FP16,
                                   qcc.PRECISION_FP32,
                                   qcc.PRECISION_INT8]:
            raise ce.ConfigurationException(
                'Invalid precision in inference engine: {}'.format(self._precision))

        if 'external-quantization' in self._params:
            profile_path = self._params.get('external-quantization', None)
            if profile_path is None or not os.path.exists(profile_path):
                raise ce.ConfigurationException(
                    'External Quantization profile supplied is invalid or doesnt exists: {}'.format(
                        profile_path))

        if self._is_ref is True and self._params is not None:
            is_permutational = False
            for key, val in self._params.items():
                val = str(val)
                vals = [v.strip() for v in val.split(qcc.SEARCH_SPACE_DELIMITER)]
                if len(vals) > 1:
                    is_permutational = True
                    break
                for v_idx, v in enumerate(vals):
                    if v.startswith(qcc.RANGE_BASED_SWEEP_PREFIX) and v.endswith(')'):
                        is_permutational = True
            if is_permutational:
                qaic_logger.error("is_ref is set to True for a configuration which generates multiple platforms.")
                qaic_logger.error("Please set a single platform as reference platform")
                raise ce.ConfigurationException(
                'platform={}, is_ref is set to True for a configuration which generates multiple platforms.'.format(self._name))

    def __str__(self):
        print('name: {}, params: {}'.format(self._name, self._params))

    def __eq__(self, other_platform):

        if self._name == other_platform._name and self._precision == other_platform._precision \
                and self._precompiled_path == other_platform._precompiled_path and self._params \
                == other_platform._params:
            return True
        else:
            return False

class QnnPlatformConfiguration:

    def __init__(self, name, target_arch="x86_64-linux-clang", backend="cpu",
                 env=None, compiler_params={}, runtime_params={}, converter_params={},
                 binary_path=None, multithreaded=True, precision='fp32', use_precompiled=None,
                 reuse_pgq=True, pgq_group=None, tag='', is_ref=False, convert_nchw=False,
                 compiler_params_json=None, runtime_params_json=None):

        self._name = name
        self._idx =None
        self._env = env

        self._target_arch = target_arch
        self._backend = backend
        self._compiler_params = compiler_params if compiler_params else {}
        self._runtime_params = runtime_params if runtime_params else {}
        self._converter_params = converter_params if converter_params else {}
        self._compiler_params_json = compiler_params_json
        self._runtime_params_json = runtime_params_json
        qaic_logger.debug(f"{self._compiler_params}\n {self._runtime_params}\n" +
                          f"{self._converter_params}")
        # batchsize is added in manager based on exec support of providing
        # batchsize with a model having multiple inputs.
        # This filed is added while setting pipeline cache with key INTERNAL_EXEC_BATCH_SIZE

        # onnxrt specific
        self._multithreaded = multithreaded

        # AIC specific params
        self._use_precompiled = use_precompiled
        self._binary_path = binary_path
        if precision == 'default':
            self._precision = qcc.PRECISION_FP32
        else:
            self._precision = precision
        self._input_info = None
        self._output_info = None
        self._precompiled_path = use_precompiled
        self._is_ref = is_ref

        # to specify if PGQ profile to be reused across
        # the generated platforms after search space scan
        self._reuse_pgq = reuse_pgq

        # to specify if PGQ profile can be used across multiple
        # platforms. All the platforms having the pgq_group tag will
        # use the same generated PGQ profile. If reuse_pgq is False the
        # PGQ profile will be regenerated for all the configured platforms
        # in one platform section.
        self._pgq_group = pgq_group

        # platform tag used to filter a platform from multiple same
        # type of platforms.
        # Example: Used to distinguish between multiple AIC platforms
        self._tag = [t.strip() for t in tag.split(',')]

        # Modify input layout from NHWC to NCHW for onnxrt platform,
        # Since we pass NHWC inputs to the QNN.
        self._convert_nchw = convert_nchw

        #Convert npi yaml to json
        if "quantization_overrides" in self._converter_params:
            _, extn = os.path.splitext(self._converter_params['quantization_overrides'])
            if extn == ".yaml":
                output_json = os.path.join(os.getcwd(), "output.json")
                convert_npi_to_json(self._converter_params['quantization_overrides'],
                                    output_json)
                self._converter_params['quantization_overrides'] = output_json

        self.validate()

        if self._compiler_params_json:
            self._compiler_params = self._read_config_json(self._compiler_params_json)
        if self._runtime_params_json:
            self._runtime_params = self._read_config_json(self._runtime_params_json)

    def _read_config_json(self, config_json):
        """Read param values from json and returns a dict
        """
        with open(config_json, "r") as f:
            params = json.load(f)
        return params

    def get_platform_name(self):
        """
        Returns a platform name based on its params
        """
        plat_name =  'plat' + str(self._idx) + '_' + self._name
        if self._precision in [qcc.PRECISION_FP16, qcc.PRECISION_FP32]:
            plat_name += "_" + self._precision
        elif self._converter_params:
            if "param_quantizer" in self._converter_params:
                plat_name += "_" + self._converter_params["param_quantizer"]
            if "act_quantizer" in self._converter_params:
                plat_name += "_" + self._converter_params["act_quantizer"]
            if "algorithms" in self._converter_params and self._converter_params["algorithms"] != "default":
                plat_name += "_" + self._converter_params["algorithms"]
            if "use_per_channel_quantization" in self._converter_params:
                plat_name += "_pcq" if self._converter_params["use_per_channel_quantization"]=="True" else ""

        return plat_name

    def validate(self):
        if self._name is None:
            raise ce.ConfigurationException('inference-engine platform name is mandatory!')
        if self._name not in [qcc.INFER_ENGINE_QNN,
                              qcc.INFER_ENGINE_AIC,
                              qcc.INFER_ENGINE_AIC_SIM,
                              qcc.INFER_ENGINE_ONNXRT,
                              qcc.INFER_ENGINE_TFRT,
                              qcc.INFER_ENGINE_TFRT_SESSION,
                              qcc.INFER_ENGINE_TORCHSCRIPTRT]:
            raise ce.ConfigurationException(
                'Invalid or Unsupported platform: {}'.format(self._name))

        if self._precision not in [qcc.PRECISION_FP16,
                                   qcc.PRECISION_FP32,
                                   qcc.PRECISION_INT8,
                                   qcc.PRECISION_QUANT]:
            raise ce.ConfigurationException(
                'Invalid precision in inference engine: {}'.format(self._precision))

        if 'quantization_overrides' in self._converter_params:
            profile_path = self._converter_params.get('quantization_overrides', None)
            if profile_path is None or not os.path.exists(profile_path):
                raise ce.ConfigurationException(
                    'External Quantization profile supplied is invalid or doesnt exists: {}'.format(
                        profile_path))

        if self._compiler_params and self._compiler_params_json is not None:
            raise ce.ConfigurationException("compiler_params_json and compiler_params cannot be used together")

        if self._runtime_params and self._runtime_params_json is not None:
            raise ce.ConfigurationException("runtime_params_json and runtime_params cannot be used together")

        if self._is_ref is True and self._converter_params is not None:
            is_permutational = False
            for key, val in self._converter_params.items():
                val = str(val)
                vals = [v.strip() for v in val.split(qcc.SEARCH_SPACE_DELIMITER)]
                if len(vals) > 1:
                    is_permutational = True
                    break
                for v_idx, v in enumerate(vals):
                    if v.startswith(qcc.RANGE_BASED_SWEEP_PREFIX) and v.endswith(')'):
                        is_permutational = True
            if is_permutational:
                qaic_logger.error("is_ref is set to True for a configuration which generates multiple platforms.")
                qaic_logger.error("Please set a single platform as reference platform")
                raise ce.ConfigurationException(
                'platform={}, is_ref is set to True for a configuration which generates multiple platforms.'.format(self._name))

    def __str__(self):
        print('name: {}, params: {}'.format(self._name, self._converter_params))

    def __eq__(self, other_platform):

        if (self._name == other_platform._name and self._precision == other_platform._precision
            and self._precompiled_path == other_platform._precompiled_path and self._compiler_params
            == other_platform._compiler_params and self._runtime_params == other_platform._runtime_params and
            self._converter_params == other_platform._converter_params):
            return True
        else:
            return False


class EvaluatorConfiguration:
    """
    QACC Evaluator configuration class
    """

    def __init__(self, metrics):
        self._metrics_plugin_list = self.get_plugin_list_from_dict(metrics)

    def get_plugin_list_from_dict(self, metrics):
        """
        Returns a list of plugin objects from the dictionary
        of plugin objects

        Args:
            transformations: metrics as dictionary

        Returns:
            plugin_config_list: list of plugin config objects
        """
        plugin_config_list = []
        for plugin in metrics:
            plugin_config_list.append(PluginConfiguration(**plugin['plugin']))
        return plugin_config_list


class PipelineCache:
    """
    Class acting as global pipeline_cache for the entire pipeline and plugins to share
    relevant information between the plugins or stages of the E2E pipeline
    """
    __instance = None

    def __init__(self):
        if PipelineCache.__instance != None:
            pass
        else:
            PipelineCache.__instance = self
            self._pipeline_cache = {}
            self._nested_keys = [qcc.PIPELINE_POSTPROC_DIR, qcc.PIPELINE_POSTPROC_FILE,
                                 qcc.PIPELINE_INFER_DIR, qcc.PIPELINE_INFER_FILE,
                                 qcc.PIPELINE_NETWORK_DESC, qcc.PIPELINE_NETWORK_BIN_DIR,
                                 qcc.PIPELINE_PROGRAM_QPC, qcc.INTERNAL_INFER_TIME,
                                 qcc.INTERNAL_POSTPROC_TIME, qcc.INTERNAL_METRIC_TIME,
                                 qcc.INTERNAL_QUANTIZATION_TIME, qcc.INTERNAL_COMPILATION_TIME]
            # init empty nested keys
            for key in self._nested_keys:
                self._pipeline_cache[key] = {}

    @classmethod
    def getInstance(cls):
        if PipelineCache.__instance == None:
            PipelineCache()
        return cls.__instance

    def set_val(self, key, val, nested_key=None):
        """
        stores the key and value in the global dictionary
        """
        if nested_key is None:
            self._pipeline_cache[key] = val
            qaic_logger.debug('Pipeline pipeline_cache - storing key {} value {}'.format(key, val))
        else:
            self._pipeline_cache[key][nested_key] = val
            qaic_logger.debug(
                'Pipeline pipeline_cache - storing key {}:{}  value {}'.format(key, nested_key,
                                                                               val))

    def get_val(self, key, nested_key=None):
        """
        returns value from information stored in dictionary during various stages
        of the pipeline.

        Args:
            key_string: nested keys in string format eg key.key.key

        Returns:
            value: value associated to the key, None otherwise
        """
        if key not in self._nested_keys:
            if key in self._pipeline_cache:
                return self._pipeline_cache[key]
            else:
                qaic_logger.warning('Pipeline pipeline_cache key {} incorrect'.format(key))
        else:
            if key in self._pipeline_cache and nested_key in self._pipeline_cache[key]:
                return self._pipeline_cache[key][nested_key]
            else:
                qaic_logger.warning(
                    'Pipeline pipeline_cache key {}:{} incorrect'.format(key, nested_key))
        return None