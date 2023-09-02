# ==============================================================================
#
#  Copyright (c) 2019-2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from .onnx_translations import *
from qti.aisw.converters.backend.custom_ops.op_factory import OpFactory
from qti.aisw.converters.backend.custom_ops.core import get_internal_dtype
import numpy as np


# ------------------------------------------------------------------------------
#   Custom Op
# ------------------------------------------------------------------------------
class OnnxCustomOpTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.custom_op = None

    def add_op(self, src_op, context, **kwargs):
        graph = context.ir_graph
        op = self.extract_parameters(src_op, context, **kwargs)
        if op is None:
            return
        input_names = self.extract_input_names(src_op, context)
        output_names = self.extract_output_names(src_op, context)
        node = graph.add(op, input_names, output_names)
        self.add_src_op_info(node.op.name, src_op, graph)
        return node

    def extract_input_names(self, src_op, converter_context):
        return self.custom_op.input_names

    def extract_output_names(self, src_op, converter_context):
        return [str(output.name) for output in self.custom_op.outputs]

    def extract_parameters(self, src_op, converter_context, **kwargs):
        graph = converter_context.ir_graph

        dynamic = kwargs.get("dynamic", False)
        # If dynamic flag is set to True, then create a new custom onnx op for this source op, otherwise directly retrieve
        # the custom onnx op from the custom op collection
        if dynamic:
            model = kwargs['model']
            converter_type = 'onnx'
            # find the corresponding operator which is defined in the XML config for this op type
            operator = None
            for op in OpFactory.custom_opdefs:
                if op.type_name == src_op.op_type:
                    operator = op
            custom_op = OpFactory.create_op_from_operator(operator, src_op, model, converter_type)
        else:
            custom_op = OpFactory.op_collection.get_first_of(src_op.op_type)

        package_name = OpFactory.get_package_name(custom_op.op_type)
        self.custom_op = custom_op

        converter_op_package_lib = None
        if 'converter_op_package_libs' in OpFactory.package_resolver:
            converter_op_package_lib = OpFactory.package_resolver['converter_op_package_libs'][package_name]

        for name, custom_param in custom_op.params.items():
            param = custom_param.param
            if param.data is None:
                if not param.static:
                    raise ValueError(
                        code_to_message.get_error_message("ERROR_CUSTOM_OP_PARAM_NO_DATA")
                        (name, custom_op.op_type))
                elif converter_context.weights.has(name):
                    param.data = np.asarray(converter_context.weights.weight_map[str(name)].weights)
                    param.data_type = get_internal_dtype(param.data, param)
                    param.dimensions = param.data.shape
                    param.rank = len(param.data.shape)
                    converter_context.weights.weight_map[str(name)].consumed = True
                elif param.default_value:
                    param.data = param.default_value
                    param.data_type = get_internal_dtype(param.data, param)
                    param.dimensions = np.asarray(param.data).shape
                    param.rank = len(param.data)
                else:
                    raise LookupError(code_to_message.get_error_message("ERROR_CANNOT"
                                                                        "_INGEST_STATIC_INPUT")
                                      (str(name)))

        inputs, outputs, scalar_params, tensor_params = custom_op.as_dict(graph)
        # adds input_names to custom op to access the updated inputs in extract_input_names
        # after the buffers for static inputs are added to the graph since updated input names
        # cannot be accessed from src_op and custom_op.inputs and custom_op.input_tensor_infos
        self.custom_op.input_names = list(inputs.keys())
        return op_adapter.CustomOp(name=src_op.name,
                                   package_name=package_name,
                                   custom_type=src_op.op_type,
                                   axis_orders=custom_op.axis_orders,
                                   inputs=inputs,
                                   outputs=outputs,
                                   output_dims=custom_op.output_dims,
                                   tensor_params=tensor_params,
                                   scalar_params=scalar_params,
                                   converter_op_package_lib=converter_op_package_lib)

    def add_op(self, src_op, context, **kwargs):
        def add_static_tensor_as_constant_op(node_name):
            if not graph.has_node(node_name):
                constant_op = self.fetch_constant_op(node_name, context)
                graph.add(
                    op=constant_op,
                    input_names=[],
                    output_names=[node_name])

        def is_static_input(input_name):
            return context.weights.has(input_name)

        def add_static_inputs_to_graph(input_names):
            static_input_names = [name for name in input_names if is_static_input(name)]
            for static_input_name in static_input_names:
                add_static_tensor_as_constant_op(static_input_name)

        graph = context.ir_graph
        op = self.extract_parameters(src_op, context)
        if op is None:
            return
        input_names = self.extract_input_names(src_op, context)
        output_names = self.extract_output_names(src_op, context)
        add_static_inputs_to_graph(input_names)
        node = graph.add(op, input_names, output_names)
        self.add_src_op_info(node.op.name, src_op, graph)
        return node

OnnxTranslations.register_translation(OnnxCustomOpTranslation(),
                                      converter_type('custom', 'onnx'),
                                      op_adapter.CustomOp.TRANSLATION_KEY)
