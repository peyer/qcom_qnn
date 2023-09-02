#=============================================================================
#
#  Copyright (c) 2022, 2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#=============================================================================
import os
import tensorflow as tf
import onnx
import logging

import qti.aisw.accuracy_evaluator.common.exceptions as ce
from qti.aisw.accuracy_evaluator.common.defaults import qaic_logger
from qti.aisw.accuracy_evaluator.common.infer_engines.executors import LocalExecutor, AdbExecutor

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class TensorflowConverter:
    def __init__(self, sdkPath, inputNetwork, outputPath, inputList=None, converterParams=None):
        self.__sdkPath = sdkPath
        self.__inputNetwork = inputNetwork
        self.__output_path = outputPath
        self.__inputList = inputList
        self.__converter_params = converterParams
        inputsAndShapes, outputNames = self.__getTfGraphInputsAndOutputs__()
        self.__inputArgs = inputsAndShapes
        self.__outputArgs = outputNames

    def convert(self, env=None, debug_log_path=None):
        qnnTensorflowConverterBinaryPath = os.path.join(self.__sdkPath, 'bin', 'x86_64-linux-clang', 'qnn-tensorflow-converter')
        inputArgsWithSwitches = ' -d '.join(self.__inputArgs)
        outputArgsWithSwitches = ' --out_node '.join(self.__outputArgs)
        baseArgs = ' -d ' + inputArgsWithSwitches + ' --out_node ' + outputArgsWithSwitches + ' -i ' + self.__inputNetwork
        baseArgs += ' -o ' + self.__output_path

        if self.__converter_params:
            if self.__inputList is not None:
                baseArgs += ' --input_list ' + self.__inputList
            if "param_quantizer" in self.__converter_params:
                baseArgs += ' --param_quantizer ' + self.__converter_params["param_quantizer"]
            if "act_quantizer" in self.__converter_params:
                baseArgs += ' --act_quantizer ' + self.__converter_params["act_quantizer"]
            if "algorithms" in self.__converter_params and self.__converter_params["algorithms"] != "default":
                baseArgs += ' --algorithms ' + self.__converter_params["algorithms"]
            if "use_per_channel_quantization" in self.__converter_params and \
                 self.__converter_params["use_per_channel_quantization"]=="True":
                baseArgs += ' --use_per_channel_quantization '
            if "act_bw" in self.__converter_params:
                baseArgs += ' --act_bw ' + self.__converter_params["act_bw"]
            if "bias_bw" in self.__converter_params:
                baseArgs += ' --bias_bw ' + self.__converter_params["bias_bw"]
            if "quantization_overrides" in self.__converter_params:
                baseArgs += ' --quantization_overrides ' + self.__converter_params["quantization_overrides"]
            if "float_bw" in self.__converter_params:
                baseArgs += ' --float_bw ' + self.__converter_params["float_bw"]
            if "custom_io" in self.__converter_params:
                baseArgs += ' --custom_io ' + self.__converter_params["custom_io"]
            if "extra_args" in self.__converter_params:
                    baseArgs += f' {self.__converter_params["extra_args"]}'

        converter_cmd = qnnTensorflowConverterBinaryPath + baseArgs

        executor = LocalExecutor()
        qaic_logger.info(converter_cmd)
        status = executor.run(converter_cmd, env=env, log_file=debug_log_path)
        if status != 0:
            qaic_logger.error("Converter failed to run succesfully")
            raise ce.QnnConverterException("Converter failed to run succesfully.")

    def __getTfGraphInputsNameAndShape__(self, graph_def):
        inputTensors = []
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name='')
        for op in graph.get_operations():
            if op.type == "Placeholder":
                for output in op.outputs:
                    inputTensors.append([op.name, output.get_shape().as_list()])

        inputsAndShapes = []
        for inputTensor in inputTensors:
            if inputTensor[1] is not None and inputTensor[1][0] == None:
                inputTensor[1][0] = 1
            if None in inputTensor[1]:
                inputTensor = promptUserForInputDims(inputTensor)
            listToStr = ','.join(map(str, inputTensor[1]))
            inputsAndShapes.append(inputTensor[0] + ' ' + listToStr)

        return inputsAndShapes

    def __getTfGraphInputsAndOutputs__(self):
        tfGraph = self.__getTfGraph__(self.__inputNetwork)
        inputsAndShapes = self.__getTfGraphInputsNameAndShape__(tfGraph)
        outputNames = self.__getTfGraphOutputsName__(tfGraph)
        return (inputsAndShapes, outputNames)

    def __getTfGraphOutputsName__(self, graph_def):
        outputs = []

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name='')
            ops = tf.compat.v1.get_default_graph().get_operations()
            outputs_set = set(ops)
            for op in ops:
                if len(op.inputs) == 0 and op.type != 'Const':#network input nodes detected
                    continue
                else:
                    for input_tensor in op.inputs:
                        if input_tensor.op in outputs_set:
                            outputs_set.remove(input_tensor.op)

        for op in outputs_set:
            outputs.append(op.node_def.name)
        return outputs


    def __getTfGraph__(self, pbFile):
        session = tf.compat.v1.Session(graph=tf.Graph())
        with session.graph.as_default():
            graph_def = tf.compat.v1.GraphDef()
            with open(pbFile, "rb") as f:
                graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name="")
        return graph_def

class TFLiteConverter:
    def __init__(self, sdkPath, inputNetwork, outputPath, inputList=None, converterParams=None):
        self.__sdkPath = sdkPath
        self.__inputNetwork = inputNetwork
        self.__output_path = outputPath
        self.__inputList = inputList
        self.__converter_params = converterParams
        inputsAndShapes = self.__getTFLiteGraphInputsNameAndShape(self.__inputNetwork)
        self.__inputArgs = inputsAndShapes

    def convert(self, env=None, debug_log_path=None):
        qnnTFLiteConverterBinaryPath = os.path.join(self.__sdkPath, 'bin', 'x86_64-linux-clang', 'qnn-tflite-converter')

        inputArgsWithSwitches = ' -d '.join(self.__inputArgs)
        baseArgs = ' -d ' + inputArgsWithSwitches + ' -i ' + self.__inputNetwork
        baseArgs += ' -o ' + self.__output_path

        if self.__converter_params:
            if self.__inputList is not None:
                baseArgs += ' --input_list ' + self.__inputList
            if "param_quantizer" in self.__converter_params:
                baseArgs += ' --param_quantizer ' + self.__converter_params["param_quantizer"]
            if "act_quantizer" in self.__converter_params:
                baseArgs += ' --act_quantizer ' + self.__converter_params["act_quantizer"]
            if "algorithms" in self.__converter_params and self.__converter_params["algorithms"] != "default":
                baseArgs += ' --algorithms ' + self.__converter_params["algorithms"]
            if "use_per_channel_quantization" in self.__converter_params and \
                 self.__converter_params["use_per_channel_quantization"]=="True":
                baseArgs += ' --use_per_channel_quantization '
            if "act_bw" in self.__converter_params:
                baseArgs += ' --act_bw ' + self.__converter_params["act_bw"]
            if "bias_bw" in self.__converter_params:
                baseArgs += ' --bias_bw ' + self.__converter_params["bias_bw"]
            if "quantization_overrides" in self.__converter_params:
                baseArgs += ' --quantization_overrides ' + self.__converter_params["quantization_overrides"]
            if "float_bw" in self.__converter_params:
                baseArgs += ' --float_bw ' + self.__converter_params["float_bw"]
            if "custom_io" in self.__converter_params:
                baseArgs += ' --custom_io ' + self.__converter_params["custom_io"]
            if "extra_args" in self.__converter_params:
                    baseArgs += f' {self.__converter_params["extra_args"]}'

        converter_cmd = qnnTFLiteConverterBinaryPath + baseArgs

        executor = LocalExecutor()
        qaic_logger.info(converter_cmd)
        status = executor.run(converter_cmd, env=env, log_file=debug_log_path)
        if status != 0:
            qaic_logger.error("Converter failed to run succesfully")
            raise ce.QnnConverterException("Converter failed to run succesfully.")

    def __getTFLiteGraphInputsNameAndShape(self, model_path):
        inputTensors = []
        tflite_interpreter = tf.lite.Interpreter(model_path)
        inputs_info = tflite_interpreter.get_input_details()
        for inp in inputs_info:
            inputTensors.append([inp["name"], inp["shape"].tolist()])

        inputsAndShapes = []
        for inputTensor in inputTensors:
            if inputTensor[1] is not None and inputTensor[1][0] == None:
                inputTensor[1][0] = 1
            if None in inputTensor[1]:
                inputTensor = promptUserForInputDims(inputTensor)
            listToStr = ','.join(map(str, inputTensor[1]))
            inputsAndShapes.append(inputTensor[0] + ' ' + listToStr)

        return inputsAndShapes

class PytorchConverter:
    def __init__(self, sdkPath, inputNetwork, outputPath, input_info, inputList=None, converterParams=None):
        self.__sdkPath = sdkPath
        self.__inputNetwork = inputNetwork
        self.__output_path = outputPath
        self.__inputList = inputList
        self.__converter_params = converterParams
        self.__inputArgs = self.__getPytorchGraphInputsNameAndShape(input_info)

    def convert(self, env=None, debug_log_path=None):
        qnnPytorchConverterBinaryPath = os.path.join(self.__sdkPath, 'bin', 'x86_64-linux-clang', 'qnn-pytorch-converter')

        inputArgsWithSwitches = ' -d '.join(self.__inputArgs)
        baseArgs = ' -d ' + inputArgsWithSwitches + ' -i ' + self.__inputNetwork
        baseArgs += ' -o ' + self.__output_path

        if self.__converter_params:
            if self.__inputList is not None:
                baseArgs += ' --input_list ' + self.__inputList
            if "param_quantizer" in self.__converter_params:
                baseArgs += ' --param_quantizer ' + self.__converter_params["param_quantizer"]
            if "act_quantizer" in self.__converter_params:
                baseArgs += ' --act_quantizer ' + self.__converter_params["act_quantizer"]
            if "algorithms" in self.__converter_params and self.__converter_params["algorithms"] != "default":
                baseArgs += ' --algorithms ' + self.__converter_params["algorithms"]
            if "use_per_channel_quantization" in self.__converter_params and \
                 self.__converter_params["use_per_channel_quantization"]=="True":
                baseArgs += ' --use_per_channel_quantization '
            if "act_bw" in self.__converter_params:
                baseArgs += ' --act_bw ' + self.__converter_params["act_bw"]
            if "bias_bw" in self.__converter_params:
                baseArgs += ' --bias_bw ' + self.__converter_params["bias_bw"]
            if "quantization_overrides" in self.__converter_params:
                baseArgs += ' --quantization_overrides ' + self.__converter_params["quantization_overrides"]
            if "float_bw" in self.__converter_params:
                baseArgs += ' --float_bw ' + self.__converter_params["float_bw"]
            if "custom_io" in self.__converter_params:
                baseArgs += ' --custom_io ' + self.__converter_params["custom_io"]
            if "extra_args" in self.__converter_params:
                    baseArgs += f' {self.__converter_params["extra_args"]}'

        converter_cmd = qnnPytorchConverterBinaryPath + baseArgs

        executor = LocalExecutor()
        qaic_logger.info(converter_cmd)
        status = executor.run(converter_cmd, env=env, log_file=debug_log_path)
        if status != 0:
            qaic_logger.error("Converter failed to run succesfully")
            raise ce.QnnConverterException("Converter failed to run succesfully.")

    def __getPytorchGraphInputsNameAndShape(self, inputs_info):
        inputTensors = []
        for inp, shape in inputs_info.items():
            inputTensors.append([inp, shape[1]])

        inputsAndShapes = []
        for inputTensor in inputTensors:
            if inputTensor[1] is not None and inputTensor[1][0] == None:
                inputTensor[1][0] = 1
            if None in inputTensor[1]:
                inputTensor = promptUserForInputDims(inputTensor)
            listToStr = ','.join(map(str, inputTensor[1]))
            inputsAndShapes.append(inputTensor[0] + ' ' + listToStr)

        return inputsAndShapes

class OnnxConverter:
    def __init__(self, sdkPath, inputNetwork, outputPath, inputList=None, converterParams=None):
        self.__sdkPath = sdkPath
        self.__inputNetwork = inputNetwork
        self.__output_path = outputPath
        #self.__inputArgs = self.__getOnnxGraphInputs__()
        self.__inputList = inputList
        self.__converter_params = converterParams

    def convert(self, env=None, debug_log_path=None):
        qnnOnnxConverterBinaryPath = os.path.join(self.__sdkPath, 'bin', 'x86_64-linux-clang', 'qnn-onnx-converter')

        #inputArgsWithSwitches = ' -d '.join(self.__inputArgs)
        #baseArgs = ' -d ' + inputArgsWithSwitches + ' -i ' + self.__inputNetwork
        baseArgs = ' -i ' + self.__inputNetwork
        baseArgs += ' -o ' + self.__output_path

        if self.__converter_params:
            if self.__inputList is not None:
                baseArgs += ' --input_list ' + self.__inputList
            if "param_quantizer" in self.__converter_params:
                baseArgs += ' --param_quantizer ' + self.__converter_params["param_quantizer"]
            if "act_quantizer" in self.__converter_params:
                baseArgs += ' --act_quantizer ' + self.__converter_params["act_quantizer"]
            if "algorithms" in self.__converter_params and self.__converter_params["algorithms"] != "default":
                baseArgs += ' --algorithms ' + self.__converter_params["algorithms"]
            if "use_per_channel_quantization" in self.__converter_params and \
                 self.__converter_params["use_per_channel_quantization"]=="True":
                baseArgs += ' --use_per_channel_quantization '
            if "act_bw" in self.__converter_params:
                baseArgs += ' --act_bw ' + self.__converter_params["act_bw"]
            if "bias_bw" in self.__converter_params:
                baseArgs += ' --bias_bw ' + self.__converter_params["bias_bw"]
            if "quantization_overrides" in self.__converter_params:
                baseArgs += ' --quantization_overrides ' + self.__converter_params["quantization_overrides"]
            if "float_bw" in self.__converter_params:
                baseArgs += ' --float_bw ' + self.__converter_params["float_bw"]
            if "custom_io" in self.__converter_params:
                baseArgs += ' --custom_io ' + self.__converter_params["custom_io"]
            if "use_native_dtype" in self.__converter_params:
                if self.__converter_params["use_native_dtype"]=="True":
                    baseArgs += ' --use_native_dtype'
            if "keep_int64_inputs" in self.__converter_params:
                if self.__converter_params["keep_int64_inputs"]=="True":
                    baseArgs += ' --keep_int64_inputs'
            if "extra_args" in self.__converter_params:
                    baseArgs += f' {self.__converter_params["extra_args"]}'

        converter_cmd = qnnOnnxConverterBinaryPath + baseArgs

        executor = LocalExecutor()
        qaic_logger.info(converter_cmd)
        status = executor.run(converter_cmd, env=env, log_file=debug_log_path)
        if status != 0:
            qaic_logger.error("Converter failed to run succesfully")
            raise ce.QnnConverterException("Converter failed to run succesfully.")

    def __getOnnxGraphInputs__(self):
        model = onnx.load(self.__inputNetwork)

        parameterNames = set()
        for tensor in model.graph.initializer:
            parameterNames.add(str(tensor.name))

        inputTensors = []
        for input in model.graph.input:
            inputInfo = []
            name = str(input.name)
            if name in parameterNames:
                continue
            dims = []
            tensorType = input.type.tensor_type
            if (tensorType.HasField("shape")):
                for dim in tensorType.shape.dim:
                    if (dim.HasField("dim_value")):
                        dims.append(dim.dim_value)
                    elif (dim.HasField("dim_param")):
                        dims.append(dim.dim_param)
                    else:
                        dims.append('?')
            else:
                print ("ERROR: Unknown input shape")
            inputInfo = [input.name, dims]
            if not self.__checkOnnxForUnknownDims__(dims):
                inputInfo = promptUserForInputDims(inputInfo)
            listToStr = ','.join(map(str, inputInfo[1]))
            inputTensors.append(inputInfo[0] + ' ' + listToStr)

        return inputTensors

    def __checkOnnxForUnknownDims__(self, dims):
        return all(isinstance(dim, int) for dim in dims)

def promptUserForInputDims(inputTensor):
    print('Input found with unknown dimensions...', flush=True)
    print('Please enter the dimensions for the following input: ' + inputTensor[0] + ' ' + ','.join(map(str, inputTensor[1])) + ', in the format B,H,W,C: ', flush=True)
    dimensions = input()
    strToList = dimensions.split(',')
    if len(strToList) != 4:
        print('Error parsing input dimensions. Exiting generator tool.', flush=True)
        exit()
    inputTensor[1] = strToList
    return inputTensor
