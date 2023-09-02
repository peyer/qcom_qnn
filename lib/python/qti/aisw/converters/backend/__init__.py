# ==============================================================================
#
#  Copyright (c) 2020,2022-2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
import sys

try:
    if sys.version_info[0] == 3 and sys.version_info[1] == 6:
        from qti.aisw.converters.common import libPyIrGraph36 as ir_graph
        from . import libPyQnnModelTools36 as qnn_modeltools
        from qti.aisw.converters.common import libPyIrSerializer36 as qnn_ir
        from . import libPyIrQuantizer36 as ir_quantizer
    else:
        from qti.aisw.converters.common import libPyIrGraph as ir_graph
        from . import libPyQnnModelTools as qnn_modeltools
        from qti.aisw.converters.common import libPyIrSerializer as qnn_ir
        from . import libPyIrQuantizer as ir_quantizer
    from qti.aisw.converters.backend import ir_to_qnn as ir_to_native_backend
except ImportError as e:
    try:
        if sys.version_info[0] == 3 and sys.version_info[1] == 6:
            import libPyQnnModelTools36 as qnn_modeltools
        else:
            import libPyQnnModelTools as qnn_modeltools
    except ImportError:
        raise e
