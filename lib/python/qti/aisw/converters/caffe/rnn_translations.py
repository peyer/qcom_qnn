# ==============================================================================
#
#  Copyright (c) 2019-2020, 2022-2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from .caffe_base_translation import CaffeTranslationBase, CaffeTranslations
from qti.aisw.converters.common.converter_ir import op_adapter
from qti.aisw.converters.common.utils.converter_utils import *
from qti.aisw.converters.common import ir_graph
from qti.aisw.converters.common.utils.converter_utils import log_warning


# -----------------------------------------------------------------
# Converter translations
# -----------------------------------------------------------------


class CaffeLstmTranslation(CaffeTranslationBase):
    def __init__(self):
        CaffeTranslationBase.__init__(self)
        self.input_names = []
        self.cell_state_weights = None
        self.no_of_gates = 4
        self.optional_inputs = {'initial_h':'', 'initial_c':'', 'norm_weights':'', 'cell_state_weights':'', 'proj_weights':'', 'proj_bias':''}

    def extract_input_names(self, src_op, converter_context):

        # RolledLstm expects input_weights, rec_weights, and bias to be inputs and not parameters.
        # Inputs have a name, here we assign names to input_weights, rec_weights, and bias.
        input_weights_name = src_op.name + "_input_weights"
        rec_weights_name = src_op.name + "_rec_weights"
        bias_name = src_op.name + "_input_bias"

        # The input names are in sequence as expected by the RolledLstm Op.
        formatted_input_names = [self.input_names[0], self.optional_inputs['initial_h'], self.optional_inputs['initial_c'],
                                 input_weights_name, rec_weights_name, bias_name,
                                 self.optional_inputs['norm_weights'], self.optional_inputs['cell_state_weights'],
                                 self.optional_inputs['proj_weights'], self.optional_inputs['proj_bias']]
        return formatted_input_names

    def prepare_params_as_constants(self, graph, params):
        for param_name, tensor in params.items():
            if not graph.has_buffer(param_name):
                graph.add(op_adapter.ConstantOp(name=param_name, tensor=tensor),
                          input_names=[], output_names=[param_name])
                graph.add_src_op_info(param_name, [], [param_name])
            elif graph.get_producer_op(param_name).type != op_adapter.ConstantOp.TRANSLATION_KEY:
                raise ValueError("lstm requires weights and bias to be constant, got dynamic tensor from {}".format(
                    graph.get_producer_op(param_name).name))

    def extract_parameters(self, layer, converter_context):
        graph = converter_context.ir_graph
        self.input_names = graph.naming_policy.get_input_names(layer, layer.bottom)
        sequence_continuation_name = ''
        if len(self.input_names) > 1:
            sequence_continuation_name = self.input_names[1]

        x_static_name = ''
        if len(self.input_names) in (3, 5):
            x_static_name = self.input_names[2]

        if sequence_continuation_name:
            log_warning("SNPE/QNN does not support sequence_continuation input.")

        if x_static_name:
            log_warning("SNPE/QNN does not support x_static input.")

        c_0_input_name = ''
        h_0_input_name = ''
        if len(self.input_names) > 3:
            c_0_input_name = self.input_names[-2]
            h_0_input_name = self.input_names[-1]
            self.optional_inputs['initial_h'] = h_0_input_name
            self.optional_inputs['initial_c'] = c_0_input_name

        x_weights, bias, h_weights = converter_context.weights.get_lstm_weights(layer)
        rolled_lstm_input_name_list = self.extract_input_names(layer, converter_context)
        input_weights_name, rec_weights_name, bias_name = rolled_lstm_input_name_list[3:6]
        norm_weights_name, cell_state_weights_name, proj_weights_name, proj_bias_name = rolled_lstm_input_name_list[6:]

        params_dict = {}
        params_dict[input_weights_name] = x_weights
        params_dict[rec_weights_name] = h_weights
        params_dict[bias_name] = bias
        if cell_state_weights_name:
            params_dict[cell_state_weights_name] = self.cell_state_weights

        # We need to make input_weights, rec_weights, and bias as inputs. We do this by using ConstantOp.
        self.prepare_params_as_constants(graph, params_dict)

        return op_adapter.RolledLstmOp(layer.name,
                                       direction=ir_graph.QNN_OP_LSTM_DIRECTION_FORWARD,
                                       c_0_input_name=c_0_input_name,
                                       h_0_input_name=h_0_input_name,
                                       # if c_0 and h_0 exist, reset_state_at_time_step_0 will be False
                                       reset_state_at_time_step_0=True if
                                       not c_0_input_name and not h_0_input_name else False,
                                       hidden_size=int(bias.size / self.no_of_gates))

    def extract_output_names(self, layer, converter_context):
        graph = converter_context.ir_graph
        name = str(layer.name)
        input_names = graph.naming_policy.get_input_names(layer, layer.bottom)
        output_names = [name]
        if len(input_names) > 3:
            output_names.append('{}_c_T'.format(name))
            output_names.append('{}_h_T'.format(name))

        return output_names

    def infer_output_shapes(self, op, input_shapes):
        time_steps = 1
        streams = 1
        output_dims = []
        # TNF for shape since axes_to_spatial_first not done yet
        if len(input_shapes[0]) == 3:
            time_steps = input_shapes[0][0]
            streams = input_shapes[0][1]
        output_channel = op.hidden_state_weights.shape[1]  # this gets us recurrent_param.num_output
        output_dims.append([time_steps, streams, output_channel])

        if op.c_0_input_name and op.h_0_input_name:
            # Layer has exposed recurrent inputs, therefore we need to add c_T and h_T outputs
            output_dims.append([streams, output_channel])
            output_dims.append([streams, output_channel])

        return output_dims


CaffeTranslations.register_translation(CaffeLstmTranslation(),
                                       converter_type('lstm', 'caffe'),
                                       op_adapter.LstmOp.TRANSLATION_KEY)
