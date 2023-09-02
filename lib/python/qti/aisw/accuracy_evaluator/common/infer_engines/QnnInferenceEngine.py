##############################################################################
#
# Copyright (c) 2022, 2023 Qualcomm Technologies, Inc.
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
import os
import subprocess
import time
import yaml
import copy
import json
import shutil
from collections import OrderedDict

import qti.aisw.accuracy_evaluator.common.defaults as df
import qti.aisw.accuracy_evaluator.common.exceptions as ce
from qti.aisw.accuracy_evaluator.qacc.constants import Constants as qcc
from qti.aisw.accuracy_evaluator.common.defaults import qaic_logger
from qti.aisw.accuracy_evaluator.common.utilities import Helper, ModelType
from qti.aisw.accuracy_evaluator.common.transforms import ModelHelper
from qti.aisw.accuracy_evaluator.common.infer_engines.executors import LocalExecutor, AdbExecutor
from qti.aisw.accuracy_evaluator.common.infer_engines.converters import OnnxConverter, TensorflowConverter,\
                                                                         TFLiteConverter, PytorchConverter

defaults = df.Defaults.getInstance()

class QnnInferenceEngine():

    def __init__(self, model, inputlistfile, calibration_file, output_path, multithread=False,
                 input_info=None, output_info=None, gen_out_file=None, compiler_params=None,
                 runtime_params=None, converter_params=None, binary_path=None, backend="cpu",
                 target_arch="x86_64-linux-clang", qnn_sdk_dir="", device_id=None):
        self.model_path = model
        self.input_path = inputlistfile
        self.calibration_file = calibration_file
        self.input_info = input_info
        self.output_info = output_info
        self.output_path = output_path
        self.compiler_params = compiler_params
        self.runtime_params = runtime_params
        self.converter_params = converter_params
        self.gen_out_file = gen_out_file
        self.binary_path = self.output_path + '/temp'
        self.debug_log_path = self.output_path + '/debug_'
        self.backend = backend #cpu/aic/htp
        self.target_arch = target_arch
        self.engine_path = qnn_sdk_dir
        self.device_id = device_id

        self.is_adb = (backend == "htp" and target_arch == "aarch64-android")
        if self.is_adb:
            self.compile_arch = "x86_64-linux-clang"
        else:
            self.compile_arch = self.target_arch
        #Initialize all the executable paths
        self._setup()

        #Check for creating the config file in fp16 and quantization modes
        if self._is_config_needed():
            self._create_config_files()

        self.env = defaults.get_env()

        # Status for each of inference stage: converter, model_lib_generator,
        # context_binary_generator, net_run
        self.stage_status = OrderedDict([('qnn-converter', True), ('qnn-model-lib-generator', True),
                              ('qnn-context-binary-generator', True), ('qnn-net-run', True)])

    def _is_config_needed(self):
        """Return if config file needs to be generated for AIC/HTP"""

        is_config = False
        if self.backend == "aic":
            if self.converter_params:
                is_config = True
            if "compiler_convert_to_FP16" in self.compiler_params and \
                 self.compiler_params["compiler_convert_to_FP16"]:
                is_config = True
        elif self.backend == "htp":
            if self.converter_params:
                is_config = True

        return is_config

    def _create_config_json(self, params, is_compiler_config=False):
        """Create json config file with compiler params for AIC"""

        #Add the graph_names to the json.
        if self.backend == "aic" and is_compiler_config:
            params["graph_names"] = [self.model_name]
        qaic_logger.debug(f"Setting the compiler options : {params}")
        if is_compiler_config:
            out_file = self.compiler_config_json
        else:
            out_file = self.runtime_config_json

        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(params, f, ensure_ascii=False, indent=4)

        return out_file

    def _create_top_json(self,net_run_extension_path, config_json, is_compiler_config=False):
        """Create top level json with extension path and config json path"""

        data = {}
        if self.is_adb:
            data["backend_extensions"] = {"shared_library_path": net_run_extension_path.split("/")[-1],
                                          "config_file_path": config_json.split("/")[-1],
                                         }
        else:
            data["backend_extensions"] = {"shared_library_path": net_run_extension_path,
                                          "config_file_path": config_json,
                                         }
        if is_compiler_config:
            out_file = self.context_config_json
        else:
            out_file = self.netrun_config_json

        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)


    def _create_config_files(self):
        """Create the json files with the compiler options and path to the extension binary"""

        if self.compiler_params:
            compiler_params = copy.deepcopy(self.compiler_params)
            compiler_config_json = self._create_config_json(compiler_params, is_compiler_config=True)
            self._create_top_json(self.net_run_extension_path, compiler_config_json,
                              is_compiler_config=True)

        if self.runtime_params:
            runtime_params = copy.deepcopy(self.runtime_params)
            runtime_config_json = self._create_config_json(runtime_params)
            self._create_top_json(self.net_run_extension_path, runtime_config_json)

    def qnn_converter(self):
        """
        Converts the reference model to QNN IR
        """
        model_type = Helper.get_model_type(self.model_path)
        if model_type == ModelType.ONNX:
            converter_cls = OnnxConverter
        elif model_type == ModelType.TENSORFLOW:
            converter_cls = TensorflowConverter
        elif model_type == ModelType.TFLITE:
            converter_cls =TFLiteConverter
        elif model_type == ModelType.TORCHSCRIPT:
            converter_cls = PytorchConverter

        if model_type == ModelType.TORCHSCRIPT:
            converter = converter_cls(sdkPath=self.engine_path,
                                      inputNetwork=self.model_path,
                                      outputPath=self.converter_output,
                                      inputList=self.calibration_file,
                                      converterParams=self.converter_params,
                                      input_info=self.input_info)
        else:
            converter = converter_cls(sdkPath=self.engine_path,
                                      inputNetwork=self.model_path,
                                      outputPath=self.converter_output,
                                      inputList=self.calibration_file,
                                      converterParams=self.converter_params)
        try:
            debug_log_file = self.debug_log_path + 'converter.log'
            converter.convert(self.env, debug_log_file)
        except Exception as e:
            self.stage_status['qnn-converter'] = False
            Helper.dump_stage_error_log(debug_log_file)

    def qnn_model_lib_generator(self):
        """
        Compiles the QNN IR to .so for a specific target architecture.
        """
        model_lib_gen_command = [self.MODEL_LIB_GENERATOR,
                                 f"-c {self.converter_output}",
                                 f"-o {self.model_binaries}",
                                 f"-b {self.converter_bin}",
                                 f"-l {self.model_so}",
                                 f"-t {self.compile_arch}",
                                 ]
        cmd = ' '.join(model_lib_gen_command)
        qaic_logger.info(cmd)
        executor = LocalExecutor()
        debug_log_file = self.debug_log_path + 'model_lib_gen.log'
        status = executor.run(cmd, self.env, debug_log_file)
        self.stage_status['qnn-model-lib-generator'] = not bool(status)
        if status != 0:
            Helper.dump_stage_error_log(debug_log_file)
            qaic_logger.error("qnn-model-lib-generator failed to run succesfully")
            raise ce.QnnModelLibGeneratorException("model-lib-generator failed to run succesfully.")

    def qnn_context_binary_generator(self):
        """
        Creates the compiled binary for AIC backend.
        """

        context_bin_command = [self.CONTEXT_BINARY_GENERATOR,
                               f"--model {os.path.join(self.model_binaries, self.compile_arch, self.model_so)}",
                               f"--backend {self.compiler_backend_path}",
                               f"--binary_file {self.context_binary}",
                               f"--output_dir {self.output_path}"
                              ]

        if self.backend == "aic" and self._is_config_needed():
            context_bin_command.append(f'--config_file {self.context_config_json}')

        cmd = ' '.join(context_bin_command)
        qaic_logger.info(cmd)
        executor = LocalExecutor()
        debug_log_file = self.debug_log_path + 'context_bin_gen.log'
        status = executor.run(cmd, self.env, debug_log_file)
        self.stage_status['qnn-context-binary-generator'] = not bool(status)
        if status != 0:
            Helper.dump_stage_error_log(debug_log_file)
            qaic_logger.error("qnn-context-binary-generator failed to run succesfully")
            raise ce.QnnContextBinaryGeneratorException("context-binary-generator failed to run succesfully.")

    def qnn_net_run(self):
        """
        Inference on the device.
        """
        if self.is_adb:
            self._run_on_target_device()
            return

        net_run_cmd = [self.NET_RUN,
                        f"--backend {self.backend_path}",
                        f"--input_list {self.input_path}",
                        f"--output_dir {self.output_path}",
                        f" --output_data_type float_only"
                       ]

        if "use_native_dtype" in self.converter_params or "keep_int64_inputs" in self.converter_params:
            net_run_cmd.append(f"--input_data_type native")
        else:
            if self.input_info: # For config mode
                native_tensor_list = []
                for iname, values in self.input_info.items():
                    if values[0] != "float32":
                        native_tensor_list.append(iname)
                if native_tensor_list:
                    native_tensors = ','.join(native_tensor_list)
                    net_run_cmd.append(f"--native_input_tensor_names model:{native_tensors}")
                else:
                    net_run_cmd.append(f"--input_data_type float")
            else: # For minimal mode
                net_run_cmd.append(f"--input_data_type float")

        if self.backend == "aic":
            net_run_cmd.append( f"--retrieve_context {self.output_path}/{self.context_binary}.bin")
        else:
            net_run_cmd.append(f"--model {self.model_binaries}/{self.target_arch}/{self.model_so}")

        if self.backend == "htp" and self.converter_params:
            net_run_cmd.append(f"--config_file {self.netrun_config_json}")
            net_run_cmd.append(f"--perf_profile HIGH_PERFORMANCE")

        if "extra_args" in self.runtime_params:
            net_run_cmd.append(f' {self.runtime_params["extra_args"]}')

        cmd = ' '.join(net_run_cmd)
        qaic_logger.info(cmd)
        executor = LocalExecutor()
        debug_log_file = self.debug_log_path + 'netrun.log'
        status = executor.run(cmd, self.env, debug_log_file)
        self.stage_status['qnn-net-run'] = not bool(status)
        if status != 0:
            Helper.dump_stage_error_log(debug_log_file)
            qaic_logger.error("qnn-net-run failed to run succesfully")
            raise ce.QnnNetRunException("qnn-net-run failed to run succesfully.")

    def _run_on_target_device(self):
        """Runs the qnn-net-run inference on the target device through adb"""

        ADB_PATH = defaults.get_value("common.adb_path")
        inputDataPath = self.get_data_folder(self.input_path)
        configFilename = self.netrun_config_json.split("/")[-1]
        settingsFilename = self.runtime_config_json.split("/")[-1]

        adbDevice = AdbExecutor(pathToAdbBinary=ADB_PATH, deviceSerialNumber=self.device_id,
                               inputList=self.input_path, inputDataPath=inputDataPath,
                               graphDir=self.model_binaries, sdkDir=self.engine_path,
                               outputDir=self.work_dir, configFilename=configFilename,
                               settingsFilename=settingsFilename)

        adbDevice.buildQnnNetRunArgs()
        if adbDevice.pushArtifacts() == -1:
            logging.info('Error pushing artifacts to target device (adb). Please check the console or logs for details.')
            return -1
        else:
            if adbDevice.runModel() == -1:
                logging.info('Error running model on target (adb). Please check the console or logs for details.')
                return -1
            else:
                if adbDevice.pullOutput() == -1:
                    logging.info('Error pulling output files from target device (adb). Please check the console or logs for details.')
                    return -1
                else:
                    if adbDevice.cleanup() == -1:
                        logging.info('Error cleaning up target device (adb). Please check the console or logs for details.')
                        return -1

        #Move all the files to one level above in th output directory for comparision
        source_dir = os.path.join(self.work_dir, "output")
        dest_dir = self.work_dir
        all_outputs = os.listdir(source_dir)
        for op_folder in all_outputs:
            shutil.move(os.path.join(source_dir, op_folder), os.path.join(dest_dir, op_folder))

    def _setup(self):
        """
        This function sets up the working directory and environment to execute QNN inferences
        It should:
        - Setup the QNN execution environment on host x86 device
        """

        # Setting the paths to the executables
        self.MODEL_LIB_GENERATOR = os.path.join(self.engine_path, "bin",
                                                self.compile_arch, qcc.MODEL_LIB_GENERATOR)
        self.CONTEXT_BINARY_GENERATOR = os.path.join(self.engine_path, "bin",
                                                     self.compile_arch, qcc.CONTEXT_BINARY_GENERATOR)
        self.NET_RUN = os.path.join(self.engine_path, "bin",
                                    self.target_arch, qcc.NET_RUN)

        work_dir = os.path.join(os.getcwd(), self.output_path)
        os.makedirs(work_dir, exist_ok=True)
        self.work_dir = work_dir

        self.model_name = qcc.MODEL_IR_FILE
        self.converter_output = os.path.join(self.work_dir, qcc.MODEL_IR_FOLDER, self.model_name+".cpp")
        self.converter_bin = os.path.join(self.work_dir, qcc.MODEL_IR_FOLDER, self.model_name+".bin")
        self.model_binaries = os.path.join(self.work_dir, qcc.MODEL_IR_FOLDER, qcc.MODEL_BINARIES_FOLDER)
        self.model_so = "lib"+ self.model_name + ".so"
        self.context_binary = qcc.CONTEXT_BINARY_FILE
        self.compiler_config_json = os.path.join(self.work_dir, qcc.COMPILER_CONFIG)
        self.runtime_config_json = os.path.join(self.work_dir, qcc.RUNTIME_CONFIG)
        self.context_config_json = os.path.join(self.work_dir, qcc.CONTEXT_CONFIG)
        self.netrun_config_json = os.path.join(self.work_dir, qcc.NETRUN_CONFIG)

        #setup backend_path
        if self.backend == "aic":
            compiler_backend = qcc.AIC_COMPILER_BACKEND
            runtime_backend = qcc.AIC_RUNTIME_BACKEND
            netrun_extension = qcc.AIC_NETRUN_EXTENSION
        elif self.backend == "htp":
            compiler_backend = qcc.HTP_BACKEND
            runtime_backend = qcc.HTP_BACKEND
            netrun_extension = qcc.HTP_NETRUN_EXTENSION
        elif self.backend == "cpu":
            compiler_backend = qcc.CPU_BACKEND
            runtime_backend = qcc.CPU_BACKEND
            netrun_extension = ""

        self.compiler_backend_path = os.path.join(self.engine_path, "lib", self.compile_arch, compiler_backend)
        self.backend_path = os.path.join(self.engine_path, "lib", self.target_arch, runtime_backend)
        self.net_run_extension_path = os.path.join(self.engine_path, "lib", self.target_arch, netrun_extension)

    def get_output_names(self):

        output_names = []
        for root, dirs, files in os.walk(os.path.join(self.work_dir, "Result_0")):
            for file in files:
                filename, fileExt = os.path.splitext(file)
                if fileExt == '.raw':
                    output_names.append(filename)
        output_names.sort()

        return output_names

    def gen_output_file(self):
        # Create the output file if requested.
        qaic_logger.debug(f"Generating output file {self.gen_out_file}")
        out_list_file = open(self.gen_out_file, 'w')
        # Output file names
        #assert self.output_info, 'Output names is mandatory'
        if self.output_info:
            output_names = list(self.output_info.keys())
        else:
            output_names = self.get_output_names()

        self.num_inputs = sum(1 for line in open(self.input_path))
        with open(self.gen_out_file, 'w') as F:
            for i in range(self.num_inputs):
                paths = []
                for out_name in output_names:
                    _path = os.path.join(self.output_path, f"Result_{i}/{out_name}.raw")
                    paths.append(_path)
                F.write(','.join(paths) + '\n')

    def execute(self):
        """
        Executes the QNN workflow in sequence
        TODO: Seperate the compile and execution stages.
        """

        self.qnn_converter()
        self.qnn_model_lib_generator()
        if self.backend == "aic" or self.is_adb:
            self.qnn_context_binary_generator()
            self.qnn_net_run()
        else:
            self.qnn_net_run()

        #Generate the infer output file to compare
        self.gen_output_file()

    def get_data_folder(self, input_path):
        """ Returns the folder of the input raw files.
        """
        cwd = os.getcwd()
        with open(input_path, "r") as F:
            rel_path = "/".join(F.readline().split("/")[:-1])

        return os.path.join(cwd, rel_path)

    @classmethod
    def get_calibration_skip_params(self):
        return ['quantization_overrides']
