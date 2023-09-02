#
# Copyright (c) 2017-2023 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#

'''
Helper script to download artifacts to run inception_v3 model with QNN SDK.
'''

import os
import subprocess
import shutil
import hashlib
import argparse
import sys

INCEPTION_V3_ARCHIVE_CHECKSUM = 'a904ddf15593d03c7dd786d552e22d73'
INCEPTION_V3_ARCHIVE_FILE = 'inception_v3_2016_08_28_frozen.pb.tar.gz'
INCEPTION_V3_ARCHIVE_URL = 'https://storage.googleapis.com/download.tensorflow.org/models/' + INCEPTION_V3_ARCHIVE_FILE
INCEPTION_V3_PB_FILENAME = 'inception_v3_2016_08_28_frozen.pb'
INCEPTION_V3_PB_OPT_FILENAME = 'inception_v3_2016_08_28_frozen_opt.pb'
INCEPTION_V3_LBL_FILENAME = 'imagenet_slim_labels.txt'
OPT_4_INFERENCE_SCRIPT = 'optimize_for_inference.py'
RAW_LIST_FILE = 'raw_list.txt'
TARGET_RAW_LIST_FILE = 'target_raw_list.txt'

QNN_ROOT = ""
OP_PACKAGE_DIR = ""
OP_PACKAGE_GEN_DIR = ""
CUSTOM_OP_PACKAGE_NAME = "ReluOpPackage"
CPU_OP_PACKAGE_LIB_PATH = ""
OP_PACKAGE_OUT_DIR = ""


def wget(download_dir, file_url):
    cmd = ['wget', '-N', '--directory-prefix={}'.format(download_dir), file_url]
    subprocess.call(cmd)


def generateMd5(path):
    checksum = hashlib.md5()
    with open(path, 'rb') as data_file:
        while True:
            block = data_file.read(checksum.block_size)
            if not block:
                break
            checksum.update(block)
    return checksum.hexdigest()


def checkResource(inception_v3_data_dir, filename, md5):
    filepath = os.path.join(inception_v3_data_dir, filename)
    if not os.path.isfile(filepath):
        raise RuntimeError(filename + ' not found at the location specified by ' \
                           + inception_v3_data_dir + '. Re-run with download option.')
    else:
        checksum = generateMd5(filepath)
        if not checksum == md5:
            raise RuntimeError('Checksum of ' + filename + ' : ' + checksum + \
                               ' does not match checksum of file ' + md5)


def find_optimize_for_inference():
    tensorflow_root = os.path.abspath(os.environ['TENSORFLOW_HOME'])
    for root, dirs, files in os.walk(tensorflow_root):
        if OPT_4_INFERENCE_SCRIPT in files:
            return os.path.join(root, OPT_4_INFERENCE_SCRIPT)


def optimize_for_inference(model_dir, tensorflow_dir):
    # Try to optimize the inception v3 PB for inference
    opt_4_inference_file = find_optimize_for_inference()

    pb_filename = ""

    if not opt_4_inference_file:
        print("\nWARNING: cannot find " + OPT_4_INFERENCE_SCRIPT + \
              " script. Skipping inference optimization.\n")
        pb_filename = INCEPTION_V3_PB_FILENAME
    else:
        print('INFO: Optimizing for inference Inception v3 using ' + opt_4_inference_file)
        print('      Please wait. It could take a while...')
        cmd = ['python3', opt_4_inference_file,
               '--input', os.path.join(tensorflow_dir, INCEPTION_V3_PB_FILENAME),
               '--output', os.path.join(tensorflow_dir, INCEPTION_V3_PB_OPT_FILENAME),
               '--input_names', 'input',
               '--output_names', 'InceptionV3/Predictions/Reshape_1']
        subprocess.call(cmd, stdout=subprocess.DEVNULL)
        pb_filename = INCEPTION_V3_PB_OPT_FILENAME

    return pb_filename


def prepare_data_images(model_dir, tensorflow_dir):
    data_dir = os.path.join(model_dir, 'data')
    if not os.path.isdir(data_dir + '/cropped'):
        os.makedirs(data_dir + '/cropped')

    # copy the labels file to the data directory
    src_label_file = os.path.join(tensorflow_dir, INCEPTION_V3_LBL_FILENAME)
    shutil.copy(src_label_file, data_dir)

    print('INFO: Creating QNN inception_v3 raw data')
    scripts_dir = os.path.join(model_dir, 'scripts')
    create_raws_script = os.path.join(scripts_dir, 'create_inceptionv3_raws.py')
    data_cropped_dir = os.path.join(data_dir, 'cropped')
    cmd = ['python3', create_raws_script,
           '-i', data_dir,
           '-d', data_cropped_dir]
    subprocess.call(cmd)

    print('INFO: Creating image list data files')
    create_file_list_script = os.path.join(scripts_dir, 'create_file_list.py')
    cmd = ['python3', create_file_list_script,
           '-i', data_cropped_dir,
           '-o', os.path.join(data_cropped_dir, RAW_LIST_FILE),
           '-e', '*.raw']
    subprocess.call(cmd)
    cmd = ['python3', create_file_list_script,
           '-i', data_cropped_dir,
           '-o', os.path.join(data_dir, TARGET_RAW_LIST_FILE),
           '-e', '*.raw',
           '-r']
    subprocess.call(cmd)


def generate_compile_packages(backend="CPU"):
    package_gen_tool = os.path.join(QNN_ROOT, 'bin', 'x86_64-linux-clang', 'qnn-op-package-generator')

    if not os.path.exists(package_gen_tool) or not os.path.isfile(package_gen_tool):
        raise RuntimeError('Could not generate packages as package generator tool was not found. '
                           'Path: {} is not valid'.format(package_gen_tool))

    if not os.path.exists(OP_PACKAGE_OUT_DIR):
        os.makedirs(OP_PACKAGE_OUT_DIR, exist_ok=True)

    if backend == "cpu":
        _generate_and_compile_cpu(package_gen_tool)
    elif backend == "htp" or backend == "dsp":
        # check hexagon path
        if 'HEXAGON_SDK_ROOT' not in os.environ:
            raise RuntimeError('Hexagon-sdk path is not set. Please see HTP/DSP example README for details on hexagon sdk setup')

        if backend == "htp":
            _generate_and_compile_htp(package_gen_tool)
        elif backend == "dsp":
            _generate_and_compile_dsp(package_gen_tool)
    else:
        raise TypeError("Unsupported backend type: {} specified for package generation".format(backend))


# TODO: Add generation capability for GPU
def _generate_and_compile_cpu(package_gen_tool):
    # check args
    cpu_xml_path = os.path.join(OP_PACKAGE_GEN_DIR, 'ReluOpPackageCpu.xml')
    cpu_op_package_dir = os.path.join(OP_PACKAGE_DIR, 'CPU')
    cpu_out_dir = os.path.join(OP_PACKAGE_OUT_DIR, 'CPU')
    custom_op_package_path = os.path.join(cpu_out_dir, CUSTOM_OP_PACKAGE_NAME)

    if not os.path.exists(cpu_xml_path) or not os.path.isfile(cpu_xml_path):
        raise RuntimeError('Could not generate cpu package due to file error: {}'.format(cpu_xml_path))

    if not os.path.isdir(cpu_op_package_dir):
        raise IOError('Cpu example package directory does not exist: {}'.format(cpu_op_package_dir))

    print("INFO: Generating Relu CPU package")
    cmd = [package_gen_tool,
           '-p', cpu_xml_path,
           '-o', cpu_out_dir]

    subprocess.check_call(cmd)

    # copy example source code for CPU
    example_cpu_src = os.path.join(cpu_op_package_dir, 'Relu.cpp')
    cpu_src_out_dir = os.path.join(custom_op_package_path, 'src', 'ops')
    if not os.path.exists(example_cpu_src) or not os.path.isfile(example_cpu_src):
        raise RuntimeError('Cannot retrieve CPU example source code needed for compilation')

    print("INFO: Replacing skeleton CPU op package source code: Relu.cpp with completed example")
    shutil.copy(example_cpu_src, cpu_src_out_dir)

    # compile for CPU
    print("INFO: Compiling Relu Op Package for CPU targets")
    subprocess.check_call(['make', 'cpu'], cwd=custom_op_package_path, stdout=subprocess.DEVNULL)

    global CPU_OP_PACKAGE_LIB_PATH
    CPU_OP_PACKAGE_LIB_PATH = os.path.join(custom_op_package_path, 'libs', 'x86_64-linux-clang', 'libReluOpPackage.so')
    print("INFO: CPU Op Package compilation done")


def _generate_and_compile_htp(package_gen_tool):
    # check args
    htp_xml_path = os.path.join(OP_PACKAGE_GEN_DIR, 'ReluOpPackageHtp.xml')
    htp_op_package_dir = os.path.join(OP_PACKAGE_DIR, 'HTP')
    htp_out_dir = os.path.join(OP_PACKAGE_OUT_DIR, 'HTP')
    custom_op_package_path = os.path.join(htp_out_dir, CUSTOM_OP_PACKAGE_NAME)

    hexagon_sdk_root = os.environ.get('HEXAGON_SDK_ROOT', "")

    if not os.path.exists(htp_xml_path) or not os.path.isfile(htp_xml_path):
        raise RuntimeError('Could not generate htp package due to file error: {}'.format(htp_xml_path))

    if not os.path.isdir(htp_op_package_dir):
        raise IOError('HTP example package directory does not exist: {}'.format(htp_op_package_dir))

    print("INFO: Generating Relu HTP package")
    cmd = [package_gen_tool,
           '-p', htp_xml_path,
           '-o', htp_out_dir]
    subprocess.check_call(cmd)

    # copy example source code for HTP
    example_htp_src = os.path.join(htp_op_package_dir, 'Relu.cpp')
    htp_src_out_dir = os.path.join(custom_op_package_path, 'src', 'ops')
    if not os.path.exists(example_htp_src) or not os.path.isfile(example_htp_src):
        raise RuntimeError('Cannot retrieve HTP example source code needed for compilation')

    print("INFO: Replacing skeleton HTP op package source code: Relu.cpp with completed example")
    shutil.copy(example_htp_src, htp_src_out_dir)

    # compile for HTP
    hexagon_version = '4.2.0'
    if not hexagon_sdk_root or hexagon_version not in hexagon_sdk_root:
        raise RuntimeError('Hexagon sdk root is set to the wrong version. '
                           'Expected: hexagon-sdk-{}, instead got {}'.format(hexagon_version, str(hexagon_sdk_root)))

    print("INFO: Compiling Relu Op Package for HTP targets")
    subprocess.check_call(['make', 'all'], cwd=custom_op_package_path, stdout=subprocess.DEVNULL)
    print("INFO: HTP Op Package compilation done")


def _generate_and_compile_dsp(package_gen_tool):
    # check args
    dsp_xml_path = os.path.join(OP_PACKAGE_GEN_DIR, 'ReluOpPackageDsp.xml')
    dsp_op_package_dir = os.path.join(OP_PACKAGE_DIR, 'DSP')
    dsp_out_dir = os.path.join(OP_PACKAGE_OUT_DIR, 'DSP')  # does this really need to be in another directory?
    custom_op_package_path = os.path.join(dsp_out_dir, CUSTOM_OP_PACKAGE_NAME)

    hexagon_sdk_root = os.getenv('HEXAGON_SDK_ROOT', "")

    if not os.path.exists(dsp_xml_path) or not os.path.isfile(dsp_xml_path):
        raise RuntimeError('Could not generate dsp package due to file error: {}'.format(dsp_xml_path))

    if not os.path.isdir(dsp_op_package_dir):
        raise IOError('DSP example package directory does not exist: {}'.format(dsp_op_package_dir))

    print("INFO: Generating Relu DSP package")
    cmd = [package_gen_tool,
           '-p', dsp_xml_path,
           '-o', dsp_out_dir]
    subprocess.check_call(cmd)

    # copy example source code for DSP
    example_dsp_src = os.path.join(dsp_op_package_dir, 'Relu.cpp')
    example_dsp_include = os.path.join(dsp_op_package_dir, 'DspOps.hpp')
    dsp_src_out_dir = os.path.join(custom_op_package_path, 'src', 'ops')
    dsp_include_dir = os.path.join(custom_op_package_path, 'include')

    if not os.path.exists(example_dsp_src) or not os.path.isfile(example_dsp_src):
        raise RuntimeError('Cannot retrieve DSP example source code needed for compilation')

    print("INFO: Replacing skeleton DSP op package source code: Relu.cpp with completed example")
    shutil.copy(example_dsp_src, dsp_src_out_dir)

    if not os.path.exists(example_dsp_include) or not os.path.isfile(example_dsp_include):
        raise RuntimeError('Cannot retrieve DSP example include file needed for compilation')

    print("INFO: Replacing skeleton DSP op package include: DspOps.hpp with completed example")
    shutil.copy(example_dsp_include, dsp_include_dir)

    hexagon_version = '3.5.1'

    if not hexagon_sdk_root or hexagon_version not in hexagon_sdk_root:
        raise RuntimeError('Hexagon sdk root is set to the wrong version. '
                           'Expected: hexagon-sdk-3.5.1, instead got {}'.format(str(hexagon_sdk_root)))

    print("INFO: Compiling Relu Op Package for DSP targets")
    subprocess.check_call(['make', 'all'], cwd=custom_op_package_path, stdout=subprocess.DEVNULL)
    print("INFO: DSP Op Package compilation done")


def convert_model(pb_filename, model_dir, tensorflow_dir, quantize, custom):
    print('INFO: Converting ' + pb_filename + ' to QNN API calls')
    out_dir = os.path.join(model_dir, 'model')
    model_name = "Inception_v3_quantized" if quantize else "Inception_v3"
    tools_dir = os.path.join(QNN_ROOT, 'bin', 'x86_64-linux-clang')
    cmd = [os.path.join(tools_dir, 'qnn-tensorflow-converter'),
           '--input_network', os.path.join(tensorflow_dir, pb_filename),
           '--input_dim', 'input', '1,299,299,3',
           '--out_node', 'InceptionV3/Predictions/Reshape_1',
           '--output_path', os.path.join(out_dir, model_name + ".cpp")]
    if quantize:
        cmd.append('--input_list')
        cmd.append(os.path.join(model_dir, 'data', 'cropped', 'raw_list.txt'))

    if custom:
        print('INFO: Using custom op config: ReluOpPackageCpu.xml')
        cmd.append('--op_package_config')
        cmd.append(os.path.join(OP_PACKAGE_GEN_DIR, 'ReluOpPackageCpu.xml'))

        if quantize:
            # sanity check op package lib path
            if not CPU_OP_PACKAGE_LIB_PATH or not os.path.isfile(CPU_OP_PACKAGE_LIB_PATH):
                raise RuntimeError('Could not retrieve op package library: {}'.format(CPU_OP_PACKAGE_LIB_PATH))
            print('INFO: Using custom op package library: {} for quantization'.format(CPU_OP_PACKAGE_LIB_PATH))
            cmd.append('--op_package_lib')
            cmd.append(':'.join([CPU_OP_PACKAGE_LIB_PATH, 'ReluOpPackageInterfaceProvider']))
    subprocess.call(cmd)

    if 'ANDROID_NDK_ROOT' not in os.environ:
        raise RuntimeError('ANDROID_NDK_ROOT not setup.  Please run the SDK env setup script.')

    print('INFO: Compiling model artifacts into shared libraries at: {}'.format(os.path.join(model_dir, 'model_libs')))
    for t in ['aarch64-android', 'x86_64-linux-clang']:
        cmd = [os.path.join(tools_dir, 'qnn-model-lib-generator'),
               '-c', os.path.join(out_dir, model_name + ".cpp"),
               '-b', os.path.join(out_dir, model_name + ".bin"),
               '-o', os.path.join(model_dir, 'model_libs'), '-t', t]
        subprocess.call(cmd, stdout=subprocess.DEVNULL)  # only print errors or warnings


def setup_assets(inception_v3_data_dir, download, convert, quantize, custom, generate):
    if 'QNN_SDK_ROOT' not in os.environ:
        raise RuntimeError('QNN_SDK_ROOT not setup.  Please run the SDK env setup script.')

    global QNN_ROOT
    QNN_ROOT = os.path.abspath(os.environ['QNN_SDK_ROOT'])
    if not os.path.isdir(QNN_ROOT):
        raise RuntimeError('QNN_SDK_ROOT (%s) is not a dir' % QNN_ROOT)

    global OP_PACKAGE_GEN_DIR
    OP_PACKAGE_GEN_DIR = os.path.join(QNN_ROOT, 'examples', 'QNN', 'OpPackageGenerator')
    if not os.path.isdir(OP_PACKAGE_GEN_DIR):
        raise RuntimeError('{} does not exist.  Your SDK may be faulty.'.format(OP_PACKAGE_GEN_DIR))

    global OP_PACKAGE_DIR
    OP_PACKAGE_DIR = os.path.join(OP_PACKAGE_GEN_DIR, 'generated')

    global OP_PACKAGE_OUT_DIR
    OP_PACKAGE_OUT_DIR = os.path.abspath(os.path.join(QNN_ROOT, 'examples', 'Models', 'InceptionV3', 'InceptionV3OpPackage'))

    if not os.path.isdir(OP_PACKAGE_DIR):
        raise RuntimeError('{} does not exist.  Your SDK may be faulty.'.format(OP_PACKAGE_DIR))

    if generate is not None:
        generate_compile_packages(backend=generate)
    elif quantize and custom:
        print('INFO: Package generation is not enabled but CPU package will be generated due to quantize option being set')
        generate_compile_packages(backend="cpu")

    if quantize and not convert:
        raise RuntimeError("ERROR: --quantize_model option must be run with --convert_model option.")

    if download:
        url_path = INCEPTION_V3_ARCHIVE_URL
        print("INFO: Downloading inception_v3 TensorFlow model...")
        wget(inception_v3_data_dir, url_path)

    try:
        checkResource(inception_v3_data_dir, INCEPTION_V3_ARCHIVE_FILE, INCEPTION_V3_ARCHIVE_CHECKSUM)
    except Exception as err:
        sys.stderr.write('ERROR: %s\n' % str(err))
        sys.exit(0)

    model_dir = os.path.join(QNN_ROOT, 'examples', 'Models', 'InceptionV3')

    if not os.path.isdir(model_dir):
        raise RuntimeError('%s does not exist.  Your SDK may be faulty.' % model_dir)

    print('INFO: Extracting TensorFlow model')
    tensorflow_dir = os.path.join(model_dir, 'tensorflow')
    if not os.path.isdir(tensorflow_dir):
        os.makedirs(tensorflow_dir)
    cmd = ['tar', '-xzf', os.path.join(inception_v3_data_dir, INCEPTION_V3_ARCHIVE_FILE), '-C', tensorflow_dir]
    subprocess.call(cmd, stdout=subprocess.DEVNULL)

    pb_filename = optimize_for_inference(model_dir, tensorflow_dir)

    prepare_data_images(model_dir, tensorflow_dir)

    model_output_dir = os.path.join(model_dir, "model")  # should this be customizable?

    if not os.path.isdir(model_output_dir):
        os.makedirs(model_output_dir)

    if convert:
        convert_model(pb_filename, model_dir, tensorflow_dir, quantize, custom)

    print('INFO: Setup inception_v3 completed.')


def getArgs():
    parser = argparse.ArgumentParser(
        prog=str(__file__),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Prepares the inception_v3 assets for tutorial examples.")

    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    required.add_argument('-a', '--assets_dir', type=str, required=True,
                          help='directory containing the inception_v3 assets')
    optional.add_argument('-c', '--convert_model', action="store_true", required=False,
                          help='Convert and compile model once acquired.')
    optional.add_argument('-cu', '--custom', action="store_true", required=False,
                          help='Convert the model using Relu as a custom operation. Only available if --c or --convert_model option is chosen')
    optional.add_argument('-d', '--download', action="store_true", required=False,
                          help='Download inception_v3 assets to inception_v3 example directory')
    optional.add_argument('-g', '--generate_packages', type=str, choices=['cpu', 'dsp', 'htp'], required=False,
                          help='Generate and compile custom op packages for HTP, CPU and DSP')
    optional.add_argument('-q', '--quantize_model', action="store_true", required=False,
                          help='Quantize the model during conversion. Only available if --c or --convert_model option is chosen')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = getArgs()

    try:
        setup_assets(args.assets_dir, args.download, args.convert_model, args.quantize_model, args.custom, args.generate_packages)
    except Exception as err:
        sys.stderr.write('ERROR: %s\n' % str(err))
