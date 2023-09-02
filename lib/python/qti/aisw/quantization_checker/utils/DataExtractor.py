import os
import re
import numpy as np
from qti.aisw.converters.common import ir_graph
from qti.aisw.dlc_utils.snpe_dlc_utils import OpRow, ModelInfo
from qti.aisw.quantization_checker.utils.Op import Op
from qti.aisw.quantization_checker.utils.Logger import PrintOptions
import qti.aisw.quantization_checker.utils.Constants as Constants

def getDataTypeBasedOnBitWidth(bitWidth):
    dataType = np.uint8
    if bitWidth == 16:
        dataType = np.uint16
    elif bitWidth == 32:
        dataType = np.uint32
    return dataType

class DataExtractor:
    def __init__(self, quantizationVariations, inputList, model, outputDir, logger):
        self.quantizationVariations = quantizationVariations
        self.inputList = inputList
        self.inputFileNames = []
        self.outputDir = outputDir
        self.model = model
        self.opMap = {}
        self.inputData = {}
        self.logger = logger

    def getAllOps(self):
        return self.opMap

    def __extractWeights(self, op, quantizationVariation):
        if op.getWeightName() not in (None, ''):
            weights = op.getNode().inputs()[1]
            dataType = np.uint8
            if quantizationVariation != Constants.UNQUANTIZED:
                quantEncoding = weights.get_encoding().encInfo
                # quantEncoding format:
                # bw, min, max, scale, offset : uint32_t, float, float, float, int32_t
                op.setIsQuantizedPerChannel(False)
                weightsScaleOffset = {}
                weightsScaleOffset['scale'] = quantEncoding.scale
                weightsScaleOffset['offset'] = quantEncoding.offset
                op.setWeightsScaleOffset(weightsScaleOffset)
                bitWidth = quantEncoding.bw
                dataType = getDataTypeBasedOnBitWidth(bitWidth)
                Op.setWeightWidth(str(bitWidth))
            else:
                Op.setWeightWidth('32')
                dataType = np.float32
            weightsData = ir_graph.PyIrStaticTensor(weights)
            op.setWeights(np.frombuffer(weightsData.data().flatten(), dtype=dataType))
            op.setWeightsDims(weightsData.data().shape)

    def __extractBiases(self, op, quantizationVariation):
        if op.getBiasName() not in (None, ''):
            dataType = np.uint8
            biases = op.getNode().inputs()[2]
            if quantizationVariation != Constants.UNQUANTIZED:
                quantEncoding = biases.get_encoding().encInfo
                # quantEncoding format:
                # bw, min, max, scale, offset : uint32_t, float, float, float, int32_t
                op.setBiasScale(quantEncoding.scale)
                op.setBiasOffset(quantEncoding.offset)
                bitWidth = quantEncoding.bw
                dataType = getDataTypeBasedOnBitWidth(bitWidth)
                Op.setBiasWidth(str(bitWidth))
            else:
                Op.setBiasWidth('32')
                dataType = np.float32
            biasesData = ir_graph.PyIrStaticTensor(biases)
            op.setBiases(np.frombuffer(biasesData.data().flatten(), dtype=dataType))

    def extractActivations(self):
        activationPath = os.path.join(self.outputDir, Constants.NET_RUN_OUTPUT_DIR, Constants.UNQUANTIZED)
        for quantizationVariation in self.quantizationVariations:
            # skip quantized models which are failed to get converted correctly
            if quantizationVariation not in self.opMap:
                continue

            for key in self.opMap[quantizationVariation].keys():
                op = self.opMap[quantizationVariation][key]
                activationNodeName = op.getActivationNodeName()
                if activationNodeName is None:
                    continue
                if quantizationVariation == Constants.UNQUANTIZED:
                    resultCount = 0
                    with os.scandir(activationPath) as items:
                        activationList = []
                        for entry in items:
                            if entry.is_dir():
                                for root, _, files in os.walk(entry):
                                    for file in files:
                                        if str(os.path.join(root, file)) == str(os.path.join(activationPath, entry.name, activationNodeName + '.raw')):
                                            activationList.append((self.inputFileNames[resultCount], np.fromfile(os.path.join(root, file), dtype='float32')))
                                resultCount += 1
                        op.setActivations(activationList)
                self.opMap[quantizationVariation][key] = op

    def extract(self):
        self.logger.print('Extracting weights and biases from dlc.')
        self.__parseOpsFromDlc()
        self.logger.print('Extracting input file names and input file data.')
        self.__extractInputData()

    def __parseOpsFromDlc(self):
        for quantizationVariation in self.quantizationVariations:
            opsInfo = self.__getOpsFromDlcForQuantizationVariation(quantizationVariation)
            if opsInfo is not None:
                self.opMap[quantizationVariation] = opsInfo

    def __getOpsFromDlcForQuantizationVariation(self, quantizationVariation):
        dlcFilePath = os.path.join(self.outputDir, quantizationVariation, quantizationVariation + '.dlc')
        if not os.path.exists(dlcFilePath):
            return
        return self.__parseOpDataFromDlc(self.__loadSnpeModel(dlcFilePath), quantizationVariation)

    def __loadSnpeModel(self, dlcFilePath):
        self.logger.print('Loading the following model: ' + dlcFilePath)
        model = ModelInfo()
        model.load(dlcFilePath)
        return model

    def __parseOpDataFromDlc(self, model, quantizationVariation):
        opMap = {}
        graph = model.model_reader.get_ir_graph()
        nodes = graph.get_ops()
        for node in nodes:
            layer = OpRow(node, [])
            if layer.type == 'data':
                continue
            op = Op(layer.name)
            op.setActivationNodeName(node.outputs()[0].name())
            if quantizationVariation != Constants.UNQUANTIZED:
                activationEncoding = node.outputs()[0].get_encoding().encInfo
                op.setActivationScale(activationEncoding.scale)
                op.setActivationOffset(activationEncoding.offset)
                Op.setActivationWidth(activationEncoding.bw)
            if layer.get_input_list():
                inputNames = layer.get_input_list()
                if layer.type.upper() in (type.upper() for type in Op.getOpTypesWithWeightsBiases()):
                    op.setInputNodeName(inputNames[0])
                    op.setWeightName(layer.name + '_weight')
                    op.setBiasName(layer.name + '_bias')
                    op.setNode(node)
                    self.__extractWeights(op, quantizationVariation)
                    self.__extractBiases(op, quantizationVariation)
            opMap[layer.name] = op
        return opMap

    def __extractInputData(self):
        try:
            with open(self.inputList) as file:
                inputDirPath = os.path.dirname(self.model)
                inputFileNames = file.readlines()
                for line in inputFileNames:
                    filenames = line.rstrip()
                    self.inputFileNames.append(filenames)
                    for file in filenames.split():
                        if file:
                            file = re.split('=|:', file)[-1]
                            filePath = os.path.join(inputDirPath, file)
                            if not os.path.exists(filePath):
                                self.logger.print('The following file from the input list (' + filePath + ') could not be found. Exiting...')
                                exit(-1)
                            self.inputData[file] = np.fromfile(filePath, dtype=np.float32)
        except Exception as e:
            self.logger.print("Unable to open input list file, please check the file path! Exiting...")
            self.logger.print(e, PrintOptions.LOGFILE)
            exit(-1)

    def getInputFiles(self):
        return self.inputFileNames

    def getInputData(self):
        return self.inputData
