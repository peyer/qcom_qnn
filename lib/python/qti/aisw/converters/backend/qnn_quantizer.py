# =============================================================================
#
#  Copyright (c) 2020-2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import sys

try:
    from . import ir_quantizer
except ImportError as ie:
    print("Failed to find necessary quantization packages:")
    print(str(ie))
    print("Please ensure that $QNN_SDK_ROOT/lib/python is in your PYTHONPATH")
    sys.exit(1)

from qti.aisw.converters.common.utils.converter_utils import log_info, log_warning
from qti.aisw.converters.common.utils.argparser_util import ArgParserWrapper
from qti.aisw.converters.common.utils import validation_utils


class QnnQuantizer(object):
    class ArgParser(ArgParserWrapper):
        def __init__(self, **kwargs):
            super(QnnQuantizer.ArgParser, self).__init__(**kwargs)
            q_group = self.add_argument_group(title='Quantizer Options')

            q_group.add_argument('--input_list', type=str,
                                 action=validation_utils.validate_filename_arg(must_exist=True),
                                 help='Path to a file specifying the input data. This file should be a plain text '
                                      'file, containing one or more absolute file paths per line. Each path is '
                                      'expected to point to a binary file containing one input in the "raw" format, '
                                      'ready to be consumed by the quantizer without any further preprocessing. '
                                      'Multiple files per line separated by spaces indicate multiple inputs to the '
                                      'network. See documentation for more details. Must be specified for quantization. '
                                      'All subsequent quantization options are ignored when this is not provided.')
            q_group.add_argument('--param_quantizer', type=str, default='tf',
                                 help='Optional parameter to indicate the weight/bias quantizer to use. Must be followed '
                                      'by one of the following options: '
                                      '"tf": Uses the real min/max of the data and specified bitwidth (default) '
                                      '"enhanced": Uses an algorithm useful for quantizing models with long tails present '
                                      'in the weight distribution '
                                      '"adjusted": Uses an adjusted min/max for computing the range, particularly good for '
                                      'denoise models '
                                      '"symmetric": Ensures min and max have the same absolute values about zero. Data '
                                      'will be stored as int#_t data such that the offset is always 0. ')
            q_group.add_argument('--act_quantizer', type=str, default='tf',
                                 help='Optional parameter to indicate the activation quantizer to use. Must be followed by '
                                      'one of the following options: '
                                      '"tf": Uses the real min/max of the data and specified bitwidth (default) '
                                      '"enhanced": Uses an algorithm useful for quantizing models with long tails present '
                                      'in the weight distribution '
                                      '"adjusted": Uses an adjusted min/max for computing the range, particularly good for '
                                      'denoise models '
                                      '"symmetric": Ensures min and max have the same absolute values about zero. Data '
                                      'will be stored as int#_t data such that the offset is always 0. ')
            q_group.add_argument('--algorithms', type=str, nargs='+', default=[],
                                 help='Use this option to enable new optimization algorithms. Usage is: '
                                      '--algorithms <algo_name1> ... '
                                      'The available optimization algorithms are: '
                                      '"cle" - Cross layer equalization includes a number of methods for equalizing '
                                      'weights and biases across layers in order to rectify imbalances that cause '
                                      'quantization errors. '
                                      '"bc" - Bias correction adjusts biases to offset activation quantization errors. '
                                      'Typically used in conjunction with "cle" to improve quantization accuracy. ')
            q_group.add_argument('--bias_bw', type=int, default=8,
                                 help='Use the --bias_bw option to select the bitwidth to use when quantizing the biases, '
                                      'either 8 (default) or 32.')
            q_group.add_argument('--act_bw', type=int, default=8,
                                 help='Use the --act_bw option to select the bitwidth to use when quantizing the '
                                      'activations, either 8 (default) or 16.')
            q_group.add_argument('--weight_bw', type=int, default=8,
                                 help='Use the --weight_bw option to select the bitwidth to use when quantizing the '
                                      'weights, currently only 8 bit (default) supported.')
            q_group.add_argument('--float_bw', type=int, default=32,
                                 help='Use the --float_bw option to select the bitwidth to use for float tensors, '
                                      'either 32 (default) or 16.')
            q_group.add_argument('--float_bias_bw', type=int, default=0,
                                 help='Use the --float_bias_bw option to select the bitwidth to use when biases are in float, '
                                      'either 32 or 16.')
            q_group.add_argument('--ignore_encodings', action='store_true', default=False,
                                 help='Use only quantizer generated encodings, ignoring any user or model provided '
                                      'encodings.\n'
                                      'Note: Cannot use --ignore_encodings with --quantization_overrides')

            q_group.add_argument('--use_per_row_quantization', action='store_true', default=False,
                                 help='Use this option to enable rowwise quantization of Matmul and FullyConnected ops.'
                                      )

            q_group.add_argument('--use_per_channel_quantization', default=[False], nargs='*', type=lambda x: str(x).lower() in ['true','1'],
                                 help='Use per-channel quantization for convolution-based op weights. \n'
                                      'Note: This will replace built-in model QAT encodings when used for a given weight.'
                                      'Usage \"--use_per_channel_quantization\" to enable or \"--use_per_channel_quantization false\" (default) to disable')

            q_group.add_argument('--use_native_input_files', action='store_true', default=False,
                                 help='Boolean flag to indicate how to read input files:\n'
                                      '1. float (default): reads inputs as floats and quantizes if necessary based on quantization parameters in the model.\n'
                                      '2. native:          reads inputs assuming the data type to be native to the model. For ex., uint8_t.\n')

            q_group.add_argument('--use_native_dtype', action='store_true', default=False,
                                 help='Note: This option is deprecated, use --use_native_input_files option in future.\n'
                                      'Boolean flag to indicate how to read input files:\n'
                                      '1. float (default): reads inputs as floats and quantizes if necessary based on quantization parameters in the model.\n'
                                      '2. native:          reads inputs assuming the data type to be native to the model. For ex., uint8_t.\n')

            q_group.add_argument('--use_native_output_files', action='store_true', default=False,
                                 help='Use this option to indicate the data type of the output files\n'
                                      '1. float (default): output the file as floats.\n'
                                      '2. native:          outputs the file that is native to the model. For ex., uint8_t.\n')

            q_group.add_argument('--disable_relu_squashing', action='store_true', default=False,
                                  help="Disables squashing of Relu against Convolution based ops for "
                                            "quantized models")

            q_group.add_argument('--restrict_quantization_steps', type=validation_utils.two_hex, action = "store",
                                 help='Specifies the number of steps to use for computing quantization encodings such that '
                                      'scale = (max - min) / number of quantization steps.\n'
                                      'The option should be passed as a space separated pair of hexadecimal string minimum and maximum values'
                                      'i.e. --restrict_quantization_steps "MIN MAX".  \n Please note that this is a hexadecimal string literal'
                                      ' and not a signed integer, to supply a negative value an explicit minus sign is required.\n'
                                      'E.g.--restrict_quantization_steps "-0x80 0x7F" indicates an example 8 bit range,\n'
                                      '    --restrict_quantization_steps "-0x8000 0x7F7F" indicates an example 16 bit range.\n',
                                 metavar="ENCODING_MIN, ENCODING_MAX", default=[])

    def __init__(self, args):
        self.opts = ir_quantizer.IrQuantizerOpts()
        if (args.input_list is None):
            self.should_quantize = False
        else:
            self.should_quantize = True
            self.opts.input_list = args.input_list

        if not self.should_quantize:
            return

        # TODO: Resolve dependency on quantization_overrides which is defined in different file
        if args.ignore_encodings and args.quantization_overrides:
            raise Exception("Invalid combination: --quantization_overrides and "
                            "--ignore_encodings cannot be provided at the same time.")

        if args.use_native_dtype:
            log_warning("--use_native_dtype option is deprecated, use --use_native_input_files option in future.")

        self.opts.param_quantizer = args.param_quantizer
        self.opts.act_quantizer = args.act_quantizer
        self.opts.algorithms = args.algorithms
        self.opts.bias_bw = args.bias_bw
        self.opts.act_bw = args.act_bw
        self.opts.weight_bw = args.weight_bw
        self.opts.float_bw = args.float_bw
        self.opts.float_bias_bw = args.float_bias_bw
        self.opts.optimizations = True
        self.opts.op_package_lib = args.op_package_lib
        self.opts.ignore_encodings = args.ignore_encodings
        self.opts.use_per_row_quantization = args.use_per_row_quantization
        self.opts.use_per_channel_quantization = True if not args.use_per_channel_quantization else args.use_per_channel_quantization[0]
        self.opts.use_native_input_dtype = args.use_native_input_files or args.use_native_dtype
        self.opts.use_native_output_dtype = args.use_native_output_files
        self.opts.reset_irgraph_maps = True
        self.opts.enable_qnn_quantizer = True
        self.opts.disable_relu_squashing = args.disable_relu_squashing

        if args.restrict_quantization_steps:
            if self.opts.param_quantizer == "symmetric" or self.opts.use_per_channel_quantization or self.opts.use_per_row_quantization:
                self.opts.quantization_step_min = args.restrict_quantization_steps[0]
                self.opts.quantization_step_max = args.restrict_quantization_steps[1]
                log_info("Restricting number of quantization steps to: min: {} - max: {}".format(self.opts.quantization_step_min,
                                                                                                 self.opts.quantization_step_max))
            else:
                log_warning("Restrict_quantization_steps is only supported for --param_quantizer = symmetric"
                            " or per channel/row quantization. Value will be ignored.")

    def get_opts(self):
        return self.opts

    def quantize(self, ir_graph, converter_backend, user_custom_io=False):
        self.graph = ir_graph
        self.converter_backend = converter_backend

        if user_custom_io:
            self.opts.user_custom_io = user_custom_io

        if not self.should_quantize:
            log_info('Skipping quantization, no input_list provided')
            return

        if not converter_backend.op_package_lib:
            if self.opts.input_list and converter_backend.custom_op_config_paths:
                log_warning('OP_PACKAGE_LIB_NOT_FOUND: Custom op configs were provided with no '
                            'custom op package libraries. '
                            'Note: Custom op packages may be required to '
                            'correctly quantize custom ops')

        # Quantize and store as float as QNN CPU BE only supports float data
        quantizer = ir_quantizer.IrQuantizer(self.get_opts(), ir_graph)
        quantizer.quantize_params(True)  # True indicates that it should be stored as floats
        quantizer.generate_activations()

        # Quantize "for real"
        quantizer.quantize_params(False)  # False indicates it should be stored as normal quantized data
        converter_backend.c_ir_graph = ir_graph

        quantizer.mixed_precision_processing()

    def construct_model(self, modelgen_backend, modelgen_interface, context, graph_configs_info, num_graph_configs_info):
        model = self.converter_backend.construct_model(self.graph, modelgen_backend, modelgen_interface,context,
                                                       graph_configs_info, num_graph_configs_info)
        self.tensor_map = self.converter_backend.get_tensor_map()
        return model

    def get_tensor_map(self):
        return self.tensor_map
