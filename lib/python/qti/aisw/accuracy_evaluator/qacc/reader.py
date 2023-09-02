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
import cv2
from PIL import Image
import numpy as np

import qti.aisw.accuracy_evaluator.common.exceptions as ce
from qti.aisw.accuracy_evaluator.qacc.constants import Constants as qcc
from qti.aisw.accuracy_evaluator.qacc import qaic_logger


class Reader:
    def __init__(self):
        pass

    def read(self, input_path, dtype, format):
        """
        Use a specific reader to read the input.
        The read methods needs to be thread safe.
        """
        if format == qcc.FMT_CV2:
            return CV2Reader.read(input_path)
        elif format == qcc.FMT_PIL:
            return PILReader.read(input_path)
        elif format == qcc.FMT_NPY:
            return RawReader.read(input_path, dtype)

        raise ce.UnsupportedException('Invalid Reader type : ' + format)


class CV2Reader:
    @classmethod
    def read(self, input_path):
        image = cv2.imread(input_path)
        if image is None:
            raise RuntimeError('CV2 failed to read image :' + input_path)

        return image


class PILReader:
    @classmethod
    def read(self, input_path):
        image = Image.open(input_path)
        if image is None:
            raise RuntimeError('PIL failed to read image :' + input_path)

        return image


class RawReader:
    @classmethod
    def read(self, inp_path, dtype):
        inp = np.fromfile(inp_path, dtype=np.float32)
        return inp
