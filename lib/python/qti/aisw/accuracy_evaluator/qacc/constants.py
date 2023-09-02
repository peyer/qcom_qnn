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

class Constants:
    # Plugin constants.
    PLUG_INFO_TYPE_MEM = 'mem'
    PLUG_INFO_TYPE_PATH = 'path'
    PLUG_INFO_TYPE_DIR = 'dir'

    # Inference Engine constants
    INFER_ENGINE_QNN = "qnn"
    INFER_ENGINE_AIC = 'aic'
    INFER_ENGINE_AIC_SIM = 'aic-sim'
    INFER_ENGINE_ONNXRT = 'onnxrt'
    INFER_ENGINE_TFRT = 'tensorflow'
    INFER_ENGINE_TORCHSCRIPTRT = 'torchscript'
    INFER_ENGINE_PYTORCH_MODULE = 'pytorch-module'
    INFER_ENGINE_TFRT_SESSION = 'tensorflow-session'

    PRECISION_FP16 = 'fp16'
    PRECISION_FP32 = 'fp32'
    PRECISION_INT8 = 'int8'
    PRECISION_QUANT = "quant"

    # IO info keys
    IO_TYPE = 'type'
    IO_DTYPE = 'dtype'
    IO_FORMAT = 'format'

    # Datatypes
    DTYPE_FLOAT16 = 'float16'
    DTYPE_FLOAT32 = 'float32'
    DTYPE_FLOAT64 = 'float64'
    DTYPE_INT8 = 'int8'
    DTYPE_INT16 = 'int16'
    DTYPE_INT32 = 'int32'
    DTYPE_INT64 = 'int64'

    # Formats
    FMT_NPY = 'np'
    FMT_CV2 = 'cv2'
    FMT_PIL = 'pil'

    # Plugin status
    STATUS_SUCCESS = 0
    STATUS_ERROR = 1
    STATUS_REMOVE = 2

    # Calibration type
    CALIBRATION_TYPE_INDEX = 'index'
    CALIBRATION_TYPE_RAW = 'raw'
    CALIBRATION_TYPE_DATASET = 'dataset'

    # Pipeline stage names
    STAGE_PREPROC_CALIB = 'calibration'
    STAGE_PREPROC = 'preproc'
    STAGE_POSTPROC = 'postproc'
    STAGE_COMPILE='compiled'
    STAGE_INFER = 'infer'
    STAGE_METRIC = 'metric'

    # File type
    BINARY_PATH = 'binary'

    # output file names
    PROCESSED_OUTFILE = 'processed-outputs.txt'
    QNN_PROCESSED_OUTFILE = 'qnn-processed-outputs.txt'
    INFER_OUTFILE = 'infer-outlist.txt'
    PROFILE_YAML = 'profile.yaml'
    RESULTS_TABLE_CSV = 'metrics-info.csv'
    PROFILING_TABLE_CSV = 'profiling-info.csv'
    INPUT_LIST_FILE = 'processed-inputlist.txt'
    CALIB_FILE = 'processed-calibration.txt'
    INFER_RESULTS_FILE = 'runlog_inf.txt'

    # output directory names
    DATASET_DIR = 'dataset'

    # qacc platform runstatus
    PLAT_INFER_SUCCESS = 0
    PLAT_INFER_FAIL = 1
    PLAT_POSTPROC_SUCCESS = 2
    PLAT_POSTPROC_FAIL = 3
    PLAT_METRIC_SUCCESS = 4
    PLAT_METRIC_FAIL = 5

    def get_plat_status(code):
       if code == Constants.PLAT_INFER_FAIL:
           return 'Inference failed'
       elif code == Constants.PLAT_POSTPROC_FAIL:
           return 'PostProcess failed'
       elif code == Constants.PLAT_METRIC_FAIL:
           return 'Failed'
       else:
           return 'Success'

    # search space delimiter
    SEARCH_SPACE_DELIMITER = '|'
    RANGE_BASED_DELIMITER = '-'
    RANGE_BASED_SWEEP_PREFIX= 'range=('

    # cleanup options
    CLEANUP_AT_END = 'end'
    CLEANUP_INTERMEDIATE = 'intermediate'
    INFER_SKIP_CLEANUP = '/temp/'

    # aic platforms
    AIC_PLATFORM_EXEC = 'qaic-exec'
    AIC_PLATFORM_PYTHON = 'qaic-python'

    # config info
    MODEL_INFO_BATCH_SIZE = 'batchsize'

    # pipeline pipeline_cache keys
    PIPELINE_BATCH_SIZE = 'config.info.batchsize'
    PIPELINE_WORK_DIR = 'qacc.work_dir'
    PIPELINE_MAX_INPUTS = 'qacc.dataset.max_inputs'
    PIPELINE_MAX_CALIB = 'qacc.dataset.max_calib'

    # preproc
    PIPELINE_PREPROC_DIR = 'qacc.preproc_dir'
    PIPELINE_PREPROC_FILE = 'qacc.preproc_file'
    # calib
    PIPELINE_CALIB_DIR = 'qacc.calib_dir'
    PIPELINE_CALIB_FILE = 'qacc.calib_file'
    PIPELINE_PREPROC_IS_CALIB = 'qacc.preproc_is_calib'
    # postproc
    PIPELINE_POSTPROC_DIR = 'qacc.postproc_dir' # contains nested structure
    PIPELINE_POSTPROC_FILE = 'qacc.postproc_file' # contains nested structure
    # infer
    PIPELINE_INFER_DIR = 'qacc.infer_dir' # contains nested structure
    PIPELINE_INFER_FILE = 'qacc.infer_file' # contains nested structure
    PIPELINE_INFER_INPUT_INFO = 'qacc.infer_input_info'
    PIPELINE_INFER_OUTPUT_INFO = 'qacc.infer_output_info'
    PIPELINE_NETWORK_BIN_DIR = 'qacc.network_bin_dir' # contains nested structure
    PIPELINE_NETWORK_DESC = 'qacc.network_desc' # contains nested structure
    PIPELINE_PROGRAM_QPC = 'qacc.program_qpc'  # contains nested structure

    # internal pipeline cache keys
    INTERNAL_CALIB_TIME = 'qacc.calib_time'
    INTERNAL_PREPROC_TIME = 'qacc.preproc_time'
    INTERNAL_QUANTIZATION_TIME = 'qacc.quantization_time' # contains nested structure
    INTERNAL_COMPILATION_TIME = 'qacc.compilation_time'  # contains nested structure
    INTERNAL_INFER_TIME = 'qacc.infer_time'  # contains nested structure
    INTERNAL_POSTPROC_TIME = 'qacc.postproc_time'  # contains nested structure
    INTERNAL_METRIC_TIME = 'qacc.metric_time'  # contains nested structure
    INTERNAL_EXEC_BATCH_SIZE = 'qacc.exec_batch_size'

    # file naming convention
    NETWORK_DESC_FILE = 'networkdesc.bin'
    PROGRAM_QPC_FILE = 'programqpc.bin'

    # options for get orig file paths API
    LAST_BATCH_TRUNCATE = 1
    LAST_BATCH_REPEAT_LAST_RECORD = 2
    LAST_BATCH_NO_CHANGE = 3

    # dataset filter plugin keys
    DATASET_FILTER_PLUGIN_NAME = 'filter_dataset'
    DATASET_FILTER_PLUGIN_PARAM_RANDOM = 'random'
    DATASET_FILTER_PLUGIN_PARAM_MAX_INPUTS = 'max_inputs'
    DATASET_FILTER_PLUGIN_PARAM_MAX_CALIB = 'max_calib'

    #model configurator artifacts
    MODEL_CONFIGURATOR_RESULT_FILE = 'results.csv'
    MODEL_CONFIGURATOR_DIR = 'model_configurator'
    MODEL_SETTING_FILE = 'model_settings.yaml'

    # Infernce Toolkit Stages: Placeholder values
    STAGE_EVALUATOR = 'accuracy_evaluator'
    STAGE_MODEL_CONFIGURATOR = 'model_configurator'
    STAGE_FILTER_PLATFORMS = 'filter_platforms'

    #QNN related cache keys
    QNN_SDK_DIR = "qnn_sdk_dir"

    #QNN executables
    MODEL_LIB_GENERATOR = "qnn-model-lib-generator"
    CONTEXT_BINARY_GENERATOR = "qnn-context-binary-generator"
    NET_RUN = "qnn-net-run"

    #QNN binaries
    AIC_COMPILER_BACKEND = "libQnnAicCC.so"
    AIC_RUNTIME_BACKEND = "libQnnAicRt.so"
    AIC_NETRUN_EXTENSION = "libQnnAicNetRunExtensions.so"
    HTP_BACKEND = "libQnnHtp.so"
    HTP_NETRUN_EXTENSION = "libQnnHtpNetRunExtensions.so"
    CPU_BACKEND = "libQnnCpu.so"

    #QNN intermediate files
    MODEL_IR_FILE = "model"
    MODEL_IR_FOLDER = "qnn_ir"
    MODEL_BINARIES_FOLDER = "model_binaries"
    CONTEXT_BINARY_FILE = "context_binary"
    COMPILER_CONFIG = "compiler_config.json"
    RUNTIME_CONFIG = "runtime_config.json"
    CONTEXT_CONFIG = "context_config.json"
    NETRUN_CONFIG = "netrun_config.json"

    #Backed libraries
    DSPV73_BACKEND_LIBRARIES = [
                                "lib/hexagon-v73/unsigned/libQnnHtpV73Skel.so",
                                "lib/aarch64-android/libQnnHtpV73Stub.so",
                                "lib/aarch64-android/libQnnHtp.so",
                                "lib/aarch64-android/libQnnHtpPrepare.so",
                                "lib/aarch64-android/libQnnHtpNetRunExtensions.so"
                                ]