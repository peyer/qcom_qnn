//=============================================================================
//
//  Copyright (c) 2020-2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//============================================================================

#include <cmath>

#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "HTP/core/simple_reg.h"

/*
 * Relevant information on writing HTP op packages can be found in
 * "Op Writing Guidelines" section in QNN SDK docs/general/backend.html
 */

/* Add BEGIN_PKG_OP_DEFINITION(<name>), where <name> is a C++ identifier that
uniquely IDs the source file
NOTE: You must also append DECLARE_OPS_OPTS_LIST(<name>) to the list
defined in ExampleOpPackageInterface.cpp
*/
BEGIN_PKG_OP_DEFINITION(PKG_Softmax);

// op execute function declarations
template <typename T_Ttype>
int softmaxWithbetaWrapper(T_Ttype &out, const T_Ttype &in, const Tensor &beta);

template <typename T_OutType, typename T_InTtype>
int softmaxWithbetaFastWrapper(T_OutType &out, const T_InTtype &in, const Tensor &beta);

template <typename T_OutType, typename T_InTtype>
int softmaxD2Impl(T_OutType &out, const T_InTtype &in, const float beta);

template <typename T_OutType, typename T_InTtype>
int softmaxD2WithbetaFastWrapper(T_OutType &out,
                                 const T_InTtype &in,
                                 const Tensor &beta,
                                 const TensorContiguous<Tdefs::QuantUint8> &lookupTable);

GraphStatus softmaxD2TablegenImpl(TensorContiguous<Tdefs::QuantUint8> &out,
                                  const Tensor &inStepsize,
                                  const Tensor &beta);

template <typename T_OutType, typename T_InTtype>
int softmaxCroutonImpl(T_OutType &out, const T_InTtype &in, const Tensor &beta);

static float softmaxCost(const Op *op);

/*
 * op definitions
 * need to be global in the package
 * one definition per op
 */

/*
 * method 1 for defining op, using default cost value (i.e. GLACIAL) and default flag
 * (Flags::RESOURCE_HVX) syntax: DEF_PACKAGE_OP(F,OP) e.g.
 * DEF_PACKAGE_OP((softmaxWithbetaWrapper<Tensor>), "Softmax")
 */
DEF_PACKAGE_OP(softmaxWithbetaWrapper<Tensor>, "Softmax")
DEF_PACKAGE_OP((softmaxWithbetaWrapper<QuantUint8Tensor>), "Softmax")

/*
 * method 2 for defining op with specified cost value (one of GLACIAL, SNAIL,
 * FAST, FREE) and provided flags syntax:
 * DEF_PACKAGE_OP_AND_COST_AND_FLAGS(F,OP,COST,...) can use zero or
 * more flags, FLAG options are IS_CONST, INHIBIT_CONST_PROP, RESOURCE_HVX,
 * RESOURCE_HMX(not supported in external op packages) e.g.
 * DEF_PACKAGE_OP_AND_COST_AND_FLAGS((softmaxWithbetaWrapper<PlainFloatTensor>),
 * "Softmax", SNAIL, Flags::IS_CONST)
 */
DEF_PACKAGE_OP_AND_COST_AND_FLAGS(softmaxD2TablegenImpl,
                                  "Softmax.d2.TableGen",
                                  FREE,
                                  Flags::RESOURCE_HVX)

DEF_PACKAGE_OP_AND_COST_AND_FLAGS((softmaxWithbetaWrapper<PlainFloatTensor>),
                                  "Softmax",
                                  SNAIL,
                                  Flags::RESOURCE_HVX)

DEF_PACKAGE_OP_AND_COST_AND_FLAGS((softmaxCroutonImpl<QUint8CroutonTensor, QUint8CroutonTensor>),
                                  "Softmax_Crouton",
                                  GLACIAL,
                                  Flags::RESOURCE_HVX)

DEF_PACKAGE_OP_AND_COST_AND_FLAGS(
    (softmaxCroutonImpl<QUint8CroutonTensor_TCM, QUint8CroutonTensor_TCM>),
    "Softmax_Crouton",
    GLACIAL,
    Flags::RESOURCE_HVX)

/*
 * method 3 for defining op with cost function pointer and provided flags
 * cost function pointer type: typedef float (*cost_function) (const Op * op);
 * syntax: DEF_PACKAGE_OP_AND_COST_F_AND_FLAGS(F,OP,COST_F,...)
 * e.g.
 * DEF_PACKAGE_OP_AND_COST_F_AND_FLAGS((softmaxWithbetaWrapper<PlainFloatTensor>),
 * "Softmax", softmaxCostFunc, Flags::IS_CONST)
 */

DEF_PACKAGE_OP_AND_COST_F_AND_FLAGS(
    (softmaxWithbetaFastWrapper<PlainFloatTensor, QuantUint8Tensor>),
    "Softmax",
    softmaxCost,
    Flags::RESOURCE_HVX)

DEF_PACKAGE_OP_AND_COST_F_AND_FLAGS(
    (softmaxWithbetaFastWrapper<PlainFloatTensor_TCM, QuantUint8Tensor_TCM>),
    "Softmax",
    softmaxCost,
    Flags::RESOURCE_HVX)

DEF_PACKAGE_OP_AND_COST_F_AND_FLAGS(
    (softmaxWithbetaFastWrapper<PlainFloatTensor, QuantUint16Tensor>),
    "Softmax",
    softmaxCost,
    Flags::RESOURCE_HVX)

DEF_PACKAGE_OP_AND_COST_F_AND_FLAGS(
    (softmaxWithbetaFastWrapper<PlainFloatTensor_TCM, QuantUint16Tensor_TCM>),
    "Softmax",
    softmaxCost,
    Flags::RESOURCE_HVX)

DEF_PACKAGE_OP_AND_COST_F_AND_FLAGS(
    (softmaxD2WithbetaFastWrapper<QuantUint8Tensor, QuantUint8Tensor>),
    "Softmax.d2",
    softmaxCost,
    Flags::RESOURCE_HVX)

DEF_PACKAGE_OP_AND_COST_F_AND_FLAGS(
    (softmaxD2WithbetaFastWrapper<QuantUint8Tensor_TCM, QuantUint8Tensor_TCM>),
    "Softmax.d2",
    softmaxCost,
    Flags::RESOURCE_HVX)

/*
 * optimization definitions
 * need to be global in the package
 * one definition per optimization
 * syntax:
 * DEF_PACKAGE_OPTIMIZATION(PRIORITY,MATCHCODE,CONSTRAINTCODE,REPLACECODE)
 * PRIORITY predefined values include EARLY(2000), MIDDLE(3000), LATE(4000)
 * HTP core provides some replacement functions for op package to use, e.g.
 * gen_ConstScalar_f32 for more information about optimization rules, please
 * refer to documentation located at QNN SDK docs/HTP/optimization_grammar.html
 */

/* the following optimization rule utilizes HTP core ops in its replacement ops
 * that are not defined or registered in this package, it is incorporated as
 * part of HTP core as of now
 */
DEF_PACKAGE_OPTIMIZATION(QNN,
                         Op("Softmax", "In", "Beta", "Axis"),
                         EQ(CONSTVAL_INT("Axis", 0), 3),
                         Op("Softmax", "In", "Beta"))

// Split on Height. Height 1 special case. Needs to happen at same time for
// other ops (EARLY-2)
DEF_PACKAGE_OPTIMIZATION(
    EARLY - 2,
    Op("Softmax", Op(FROM_DEFAULT_PACKAGE("height1_sequence"), "In"), "Beta"),
    OK,
    Op(FROM_DEFAULT_PACKAGE("height1_sequence"),
       AUTOSPLIT(1,
                 "I",
                 1,
                 Op(FROM_DEFAULT_PACKAGE("Reshape_to_h1"),
                    WITH_SIZE(gen_Shape(DIM_BATCHES("*"),
                                        TILE_HEIGHT,
                                        DIV(DIM_WIDTH("*"), TILE_HEIGHT),
                                        DIM_DEPTH("*")),
                              Op("Softmax",
                                 WITH_TYPE("In",
                                           WITH_SIZE(gen_Shape(DIM_BATCHES("In"),
                                                               TILE_HEIGHT,
                                                               DIV(DIM_WIDTH("In"), TILE_HEIGHT),
                                                               DIM_DEPTH("In")),
                                                     Op(FROM_DEFAULT_PACKAGE("Reshape_to_h8"),
                                                        TYPICAL_SLICE("In", "I")))),
                                 "Beta"))))))

// Split on Height. Height 1 special case. Needs to happen at same time for
// other ops (EARLY-2)
DEF_PACKAGE_OPTIMIZATION(
    EARLY - 2,
    Op("Softmax", Op(FROM_DEFAULT_PACKAGE("height1_sequence"), "In")),
    OK,
    Op(FROM_DEFAULT_PACKAGE("height1_sequence"),
       AUTOSPLIT(1,
                 "I",
                 1,
                 Op(FROM_DEFAULT_PACKAGE("Reshape_to_h1"),
                    WITH_SIZE(gen_Shape(DIM_BATCHES("*"),
                                        TILE_HEIGHT,
                                        DIV(DIM_WIDTH("*"), TILE_HEIGHT),
                                        DIM_DEPTH("*")),
                              Op("Softmax",
                                 WITH_TYPE("In",
                                           WITH_SIZE(gen_Shape(DIM_BATCHES("In"),
                                                               TILE_HEIGHT,
                                                               DIV(DIM_WIDTH("In"), TILE_HEIGHT),
                                                               DIM_DEPTH("In")),
                                                     Op(FROM_DEFAULT_PACKAGE("Reshape_to_h8"),
                                                        TYPICAL_SLICE("In", "I"))))))))))

DEF_PACKAGE_OPTIMIZATION(EARLY,
                         Op("Softmax", "In"),
                         OK,
                         Op("Softmax", "In", gen_ConstScalar_f32(1.0f)))

// A graph may have reshapes before and after a softmax to
// bring all the non depth dimensions into one axis for the
// softmax's sake. This isn't necessary and can mess with the tiling
DEF_PACKAGE_OPTIMIZATION(EARLY + 1,
                         Op(FROM_DEFAULT_PACKAGE("Reshape"),
                            Op("Softmax", Op(FROM_DEFAULT_PACKAGE("Reshape"), "In"), "Beta")),
                         SAME_SHAPE("In", "*"),
                         Op("Softmax", "In", "Beta"))

// this is the case for a reshape followed by a softmax
// since if the softmax is along depth, there is no need to collapse
// the other dimensions into a single one, i.e., (b1,h1,w1,d) ->
// (1,1,b1xh1xw2,d) NOTE: can only do this if the input depth = output depth
DEF_PACKAGE_OPTIMIZATION(
    EARLY + 2,
    Op("Softmax", LET("ReshapeOut", Op(FROM_DEFAULT_PACKAGE("Reshape"), "In")), "Beta"),
    EQ(DIM_DEPTH("In"), DIM_DEPTH("ReshapeOut")),
    Op(FROM_DEFAULT_PACKAGE("Reshape"), WITH_SIZE("In", Op("Softmax", "In", "Beta"))))

DEF_PACKAGE_OPTIMIZATION(
    EARLY + 3,
    Op("Softmax", "In", "Beta"),
    AND(IS_QUINT8("In"),
        EQ(DIM_DEPTH("In"), 2),
        OR(LT(DIV(STEPSIZE_OF("*"), DIV(1, 256.0f)), 0.995f),
           GT(DIV(STEPSIZE_OF("*"), DIV(1, 256.0f)), 1.005f),
           NOT(EQ(ZERO_OFFSET_OF("*"), 0)))),
    Op(FROM_DEFAULT_PACKAGE("Requantize"),
       WITH_OUTPUT_TYPE(DType::QUInt8, 0, DIV(1, 256.0f), Op("Softmax", "In", "Beta"))))

// Split on batch
DEF_PACKAGE_OPTIMIZATION(EARLY + 4,
                         Op("Softmax", "In", "Beta"),
                         GT(DIM_BATCHES("*"), 1),
                         AUTOSPLIT(0, "I", 1, Op("Softmax", TYPICAL_SLICE("In", "I"), "Beta")))

// Split on Height
DEF_PACKAGE_OPTIMIZATION(
    EARLY + 4,
    Op("Softmax", "In", "Beta"),
    GT(DIM_HEIGHT("*"), TILE_HEIGHT),
    AUTOSPLIT(1, "I", TILE_HEIGHT, Op("Softmax", TYPICAL_SLICE("In", "I"), "Beta")))

// Split on Width
DEF_PACKAGE_OPTIMIZATION(EARLY + 5,
                         Op("Softmax", "In", "Beta"),
                         AND(GT(DIM_WIDTH("*"), 256), LT(DIM_DEPTH("*"), 32)),
                         AUTOSPLIT(2, "I", 256, Op("Softmax", TYPICAL_SLICE("In", "I"), "Beta")))

DEF_PACKAGE_OPTIMIZATION(EARLY + 5,
                         Op("Softmax", "In", "Beta"),
                         AND(GT(DIM_WIDTH("*"), 32), GT(DIM_DEPTH("*"), 256)),
                         AUTOSPLIT(2, "I", 32, Op("Softmax", TYPICAL_SLICE("In", "I"), "Beta")))

DEF_PACKAGE_OPTIMIZATION(
    EARLY + 6,
    Op("Softmax", "In", "Beta"),
    AND(IS_QUINT8("In"),
        EQ(DIM_DEPTH("In"), 2),
        EQ(ZERO_OFFSET_OF("*"), 0),
        GT(DIV(STEPSIZE_OF("*"), DIV(1, 256.0f)), 0.995f),
        LT(DIV(STEPSIZE_OF("*"), DIV(1, 256.0f)), 1.005f)),
    Op("Softmax.d2",
       WITH_SAME_OUTPUT("In", Op(FROM_DEFAULT_PACKAGE("ForceFormat_Flat"), "In")),
       "Beta",
       WITH_TYPE(
           "In",
           WITH_SIZE(gen_Shape(1, 1, 1, 256),
                     Op("Softmax.d2.TableGen", gen_ConstScalar_f32(STEPSIZE_OF("In")), "Beta")))))

DEF_PACKAGE_OPTIMIZATION(EARLY + 7,
                         Op("Softmax", "In", "Beta"),
                         OR(EQ(DTYPE_OF("*"), DType::QUInt8), EQ(DTYPE_OF("*"), DType::QUInt16)),
                         Op(FROM_DEFAULT_PACKAGE("Quantize"),
                            WITH_OUTPUT_TYPE(DType::Float32, 0, 1.0f, Op("Softmax", "In", "Beta"))))

DEF_PACKAGE_OPTIMIZATION(
    EARLY + 8,
    Op("Softmax", "In", "Beta"),
    NOT(AND(
        IS_FLOAT16("In"), LE(DIM_DEPTH("In"), 4), GT(DIM_HEIGHT("In"), 1), GT(DIM_WIDTH("In"), 1))),
    Op("Softmax.tmp",
       WITH_SAME_OUTPUT("In", Op(FROM_DEFAULT_PACKAGE("ForceFormat_Flat"), "In")),
       "Beta"))

DEF_PACKAGE_OPTIMIZATION(EARLY + 9,
                         Op("Softmax.tmp", "In", "Beta"),
                         OK,
                         Op("Softmax", "In", "Beta"))

DEF_PACKAGE_OPTIMIZATION(
    LATE + 10,
    Op("Softmax", "In", "Beta"),
    AND(IS_FLOAT16("In"), LE(DIM_DEPTH("In"), 4), GT(DIM_HEIGHT("In"), 1), GT(DIM_WIDTH("In"), 1)),
    Op(FROM_DEFAULT_PACKAGE("ConvLayer.opt.activations_from_vtcm"),
       Op("Softmax.tcm",
          WITH_SAME_OUTPUT("In",
                           Op(FROM_DEFAULT_PACKAGE("ConvLayer.opt.activations_to_vtcm"), "In")),
          "Beta")))

DEF_PACKAGE_OPTIMIZATION(LATE + 11,
                         Op("Softmax.tcm", "In", "Beta"),
                         OK,
                         Op("Softmax", "In", "Beta"))

DEF_PACKAGE_OPTIMIZATION(LATE + 20,
                         Op(FROM_DEFAULT_PACKAGE("flat_to_vtcm"), Op("Softmax", "In", "Beta")),
                         OK,
                         Op("Softmax",
                            WITH_SAME_OUTPUT("In", Op(FROM_DEFAULT_PACKAGE("flat_to_vtcm"), "In")),
                            "Beta"))

DEF_PACKAGE_OPTIMIZATION(LATE + 20,
                         Op("Softmax", Op(FROM_DEFAULT_PACKAGE("flat_from_vtcm"), "In"), "Beta"),
                         OK,
                         Op(FROM_DEFAULT_PACKAGE("flat_from_vtcm"), Op("Softmax", "In", "Beta")))

DEF_PACKAGE_OPTIMIZATION(LATE + 20,
                         Op("Softmax",
                            Op(FROM_DEFAULT_PACKAGE("ConvLayer.opt.activations_from_vtcm"), "In"),
                            "Beta"),
                         OK,
                         Op("Softmax", "In", "Beta"))

DEF_PACKAGE_OPTIMIZATION(
    LATE + 20,
    Op("Softmax.d2",
       Op(FROM_DEFAULT_PACKAGE("ForceFormat_Flat"),
          Op(FROM_DEFAULT_PACKAGE("ConvLayer.opt.activations_from_vtcm"), "In")),
       "Beta",
       "Table"),
    OK,
    Op("Softmax.d2",
       WITH_SAME_OUTPUT("In", Op(FROM_DEFAULT_PACKAGE("ForceFormat_Flat"), "In")),
       "Beta",
       "Table"))

DEF_PACKAGE_OPTIMIZATION(LATE + 20,
                         Op(FROM_DEFAULT_PACKAGE("flat_to_vtcm"),
                            Op("Softmax.d2", "In", "Beta", "Table")),
                         OK,
                         Op("Softmax.d2",
                            WITH_SAME_OUTPUT("In", Op(FROM_DEFAULT_PACKAGE("flat_to_vtcm"), "In")),
                            "Beta",
                            "Table"))

DEF_PACKAGE_OPTIMIZATION(
    LATE + 20,
    Op("Softmax.d2", Op(FROM_DEFAULT_PACKAGE("flat_from_vtcm"), "In"), "Beta", "Table"),
    OK,
    Op(FROM_DEFAULT_PACKAGE("flat_from_vtcm"), Op("Softmax.d2", "In", "Beta", "Table")))

DEF_PACKAGE_OPTIMIZATION(
    LATE + 10,
    Op(FROM_DEFAULT_PACKAGE("ConvLayer.opt.activations_to_vtcm"),
       Op("Softmax", Op(FROM_DEFAULT_PACKAGE("ForceFormat_Flat"), "In"), "Beta")),
    OK,
    Op(FROM_DEFAULT_PACKAGE("ForceFormat_Crouton"),
       Op("Softmax",
          WITH_SAME_OUTPUT(
              "In",
              Op(FROM_DEFAULT_PACKAGE("ForceFormat_Flat"),
                 WITH_SAME_OUTPUT(
                     "In", Op(FROM_DEFAULT_PACKAGE("ConvLayer.opt.activations_to_vtcm"), "In")))),
          "Beta")))

DEF_PACKAGE_OPTIMIZATION(
    LATE + 20,
    Op(FROM_DEFAULT_PACKAGE("SlicePad_shape_inplace.tcm"),
       LET("SOFTMAXQUANTIZE",
           Op(FROM_DEFAULT_PACKAGE("Quantize.tcm"), Op("Softmax", "In", "Beta"))),
       "Before",
       "Start",
       "Out",
       "Zero"),
    OK,
    Op(FROM_DEFAULT_PACKAGE("SlicePad_shape_inplace.tcm"),
       WITH_SAME_OUTPUT("SOFTMAXQUANTIZE",
                        Op(FROM_DEFAULT_PACKAGE("ForceFormat_Crouton"), "SOFTMAXQUANTIZE")),
       "Before",
       "Start",
       "Out",
       "Zero"))

DEF_PACKAGE_OPTIMIZATION(
    LATE + 20,
    Op(FROM_DEFAULT_PACKAGE("SlicePad_shape_inplace.tcm"),
       LET("SOFTMAXQUANTIZE",
           OpVarIn(FROM_DEFAULT_PACKAGE("Concat"),
                   "axis",
                   Op(FROM_DEFAULT_PACKAGE("Quantize.tcm"), Op("Softmax", "In", "Beta")))),
       "Before",
       "Start",
       "Out",
       "Zero"),
    OK,
    Op(FROM_DEFAULT_PACKAGE("SlicePad_shape_inplace.tcm"),
       WITH_SAME_OUTPUT("SOFTMAXQUANTIZE",
                        Op(FROM_DEFAULT_PACKAGE("ForceFormat_Crouton"), "SOFTMAXQUANTIZE")),
       "Before",
       "Start",
       "Out",
       "Zero"))

DEF_PACKAGE_OPTIMIZATION(
    LATE + 10,
    Op(FROM_DEFAULT_PACKAGE("ConvLayer.opt.activations_to_vtcm"),
       Op("Softmax.d2", "In", "Beta", "Table")),
    OK,
    Op(FROM_DEFAULT_PACKAGE("ForceFormat_Crouton"),
       Op("Softmax.d2",
          WITH_SAME_OUTPUT("In",
                           Op(FROM_DEFAULT_PACKAGE("ForceFormat_Flat"),
                              Op(FROM_DEFAULT_PACKAGE("ConvLayer.opt.activations_to_vtcm"), "In"))),
          "Beta",
          "Table")))

DEF_PACKAGE_OPTIMIZATION(
    LATE + 1000,
    Op(FROM_DEFAULT_PACKAGE("ForceFormat_Crouton"),
       Op(FROM_DEFAULT_PACKAGE("Quantize"),
          Op("Softmax", Op(FROM_DEFAULT_PACKAGE("ForceFormat_Flat"), "In"), "Beta"))),
    AND(LE(DIM_HEIGHT("In"), 8), GE(DIM_DEPTH("In"), 128), LE(DIM_DEPTH("In"), 2048)),
    Op("Softmax_Crouton", "In", "Beta"))

/* execute functions for ops */

union UVecFlts {
  HVX_Vector m_hvxVector;
  float m_floatArray32[32];
};

inline size_t flatToVlut(size_t index) {
  return ((index & 63) << 1) | ((index >> 6) & 1) | (index & -128);
}

[[gnu::noinline]] static void finishSoftmax(float *pout, int length, UVecFlts const *vsumf);

void softmaxApprox(float *pout, const uint8_t *pin, float scale, int length) {
  scale /= float(log(2.0));
  int bscale = -((flt_getfrac(-scale) + 1) >> 1);
  int brsh   = min_i32(31, max_i32(flt_getexp(-scale) - 23 + 14 + 1, -17));
  HVX_Vector *iptr, *optr;

  iptr            = (HVX_Vector *)pin;
  HVX_Vector xmax = Q6_V_vzero();

  // find max
  for (int d = length; d > 127; d -= 128) {
    HVX_Vector xinval = vmemu(iptr);
    iptr++;
    xmax = Q6_Vub_vmax_VubVub(xmax, xinval);
  }
  if ((length & 127) != 0) {
    HVX_Vector xinval         = vmemu(iptr);
    HVX_VectorPred qfinalmask = Q6_Q_vsetq2_R(length);
    xinval                    = Q6_V_vmux_QVV(qfinalmask, xinval, xmax);
    xmax                      = Q6_Vub_vmax_VubVub(xmax, xinval);
  }
  int nshift = 1;
  for (int i = 0; i < 7; i++) {
    HVX_VectorPair temps = Q6_W_vshuff_VVR(xmax, xmax, nshift);
    xmax                 = Q6_Vub_vmax_VubVub(Q6_V_lo_W(temps), Q6_V_hi_W(temps));
    nshift <<= 1;
  }

  // calculate sum
  HVX_Vector const0x7FFE = Q6_Vh_vsplat_R(0x7FFE);
  HVX_Vector const0x007F = Q6_Vh_vsplat_R(0x007F);
  HVX_Vector f0          = Q6_Vh_vsplat_R(0x8000);
  HVX_Vector f1          = Q6_Vh_vsplat_R(0x5863);
  HVX_Vector f2          = Q6_Vh_vsplat_R(0x1E75);
  HVX_Vector f3          = Q6_Vh_vsplat_R(0x0928);
  HVX_Vector vbeta       = Q6_V_vsplat_R(bscale);
  HVX_Vector vbrshf      = Q6_V_vsplat_R(brsh);
  HVX_Vector vbrshe      = Q6_V_vsplat_R(brsh - 15);
  HVX_Vector vsumf       = Q6_V_vzero();
  iptr                   = (HVX_Vector *)pin;
  optr                   = (HVX_Vector *)pout;

  for (int d = length; d > 0; d -= 128) {
    HVX_Vector x, x0, x1, x2, x3, xd02, xd13, p02, p13;
    HVX_Vector xd0, xd1, xd2, xd3, xe0, xe1, xe2, xe3, xe02, xe13;
    HVX_Vector poly0, poly1, poly2, poly3;
    HVX_VectorPair xdiff, xf02, xf13, xe13Xe02, poly01, poly23;

    x = vmemu(iptr);
    iptr++;
    xdiff = Q6_Wh_vsub_VubVub(xmax, x);
    x0    = Q6_Vw_vmpyie_VwVuh(vbeta, Q6_V_lo_W(xdiff));
    x2    = Q6_Vw_vmpyio_VwVh(vbeta, Q6_V_lo_W(xdiff));
    x1    = Q6_Vw_vmpyie_VwVuh(vbeta, Q6_V_hi_W(xdiff));
    x3    = Q6_Vw_vmpyio_VwVh(vbeta, Q6_V_hi_W(xdiff));

    xd0 = Q6_Vw_vasl_VwVw(x0, vbrshf);
    xd1 = Q6_Vw_vasl_VwVw(x1, vbrshf);
    xd2 = Q6_Vw_vasl_VwVw(x2, vbrshf);
    xd3 = Q6_Vw_vasl_VwVw(x3, vbrshf);

    xd02 = Q6_V_vand_VV(Q6_Vh_vshuffe_VhVh(xd2, xd0), const0x7FFE);
    xd13 = Q6_V_vand_VV(Q6_Vh_vshuffe_VhVh(xd3, xd1), const0x7FFE);

    xe0 = Q6_Vw_vasl_VwVw(x0, vbrshe);
    xe1 = Q6_Vw_vasl_VwVw(x1, vbrshe);
    xe2 = Q6_Vw_vasl_VwVw(x2, vbrshe);
    xe3 = Q6_Vw_vasl_VwVw(x3, vbrshe);

    xe02     = Q6_Vh_vadd_VhVh_sat(Q6_Vh_vsat_VwVw(xe2, xe0), const0x007F);
    xe13     = Q6_Vh_vadd_VhVh_sat(Q6_Vh_vsat_VwVw(xe3, xe1), const0x007F);
    xe13Xe02 = Q6_Wuh_vzxt_Vub(Q6_Vub_vsat_VhVh(xe13, xe02));
    xe02     = Q6_V_lo_W(xe13Xe02);
    xe13     = Q6_V_hi_W(xe13Xe02);

    p02 = Q6_Vh_vadd_VhVh(Q6_Vh_vmpy_VhVh_s1_rnd_sat(xd02, f3), f2);
    p02 = Q6_Vh_vadd_VhVh(Q6_Vh_vmpy_VhVh_s1_rnd_sat(xd02, p02), f1);
    p02 = Q6_Vh_vadd_VhVh(Q6_Vh_vmpy_VhVh_s1_rnd_sat(xd02, p02), f0);
    p02 = Q6_Vh_vadd_VhVh(p02, p02);

    p13 = Q6_Vh_vadd_VhVh(Q6_Vh_vmpy_VhVh_s1_rnd_sat(xd13, f3), f2);
    p13 = Q6_Vh_vadd_VhVh(Q6_Vh_vmpy_VhVh_s1_rnd_sat(xd13, p13), f1);
    p13 = Q6_Vh_vadd_VhVh(Q6_Vh_vmpy_VhVh_s1_rnd_sat(xd13, p13), f0);
    p13 = Q6_Vh_vadd_VhVh(p13, p13);

    xf02 = Q6_W_vshuff_VVR(xe02, p02, -2);
    xf13 = Q6_W_vshuff_VVR(xe13, p13, -2);

    poly01 = Q6_W_vshuff_VVR(Q6_V_lo_W(xf13), Q6_V_lo_W(xf02), -4);
    poly23 = Q6_W_vshuff_VVR(Q6_V_hi_W(xf13), Q6_V_hi_W(xf02), -4);

    poly0 = Q6_Vw_vasl_VwR(Q6_V_lo_W(poly01), 7);
    poly1 = Q6_Vw_vasl_VwR(Q6_V_hi_W(poly01), 7);

    poly2 = Q6_Vw_vasl_VwR(Q6_V_lo_W(poly23), 7);
    poly3 = Q6_Vw_vasl_VwR(Q6_V_hi_W(poly23), 7);

    if (d >= 128) {
      q6op_vstu_AV(optr, poly0);
      optr++;
      q6op_vstu_AV(optr, poly1);
      optr++;
      q6op_vstu_AV(optr, poly2);
      optr++;
      q6op_vstu_AV(optr, poly3);
      optr++;

      vsumf = Q6_Vqf32_vadd_Vqf32Vsf(vsumf, poly0);
      vsumf = Q6_Vqf32_vadd_Vqf32Vsf(vsumf, poly1);
      vsumf = Q6_Vqf32_vadd_Vqf32Vsf(vsumf, poly2);
      vsumf = Q6_Vqf32_vadd_Vqf32Vsf(vsumf, poly3);
    } else {
      int dremain = d;
      if (dremain >= 64) {  // process poly0 & poly1 entirely...
        vsumf = Q6_Vqf32_vadd_Vqf32Vsf(vsumf, poly0);
        vsumf = Q6_Vqf32_vadd_Vqf32Vsf(vsumf, poly1);
        q6op_vstu_AV(optr, poly0);
        optr++;
        q6op_vstu_AV(optr, poly1);
        optr++;
        poly0 = poly2;
        poly1 = poly3;
        dremain -= 64;
      }
      if (dremain >= 32) {  // process poly0 entirely.
        vsumf = Q6_Vqf32_vadd_Vqf32Vsf(vsumf, poly0);
        q6op_vstu_AV(optr, poly0);
        optr++;
        poly0 = poly1;
        dremain -= 32;
      }
      if (dremain) {  // -> dremain is 1..31
        HVX_VectorPred q0   = Q6_Q_vsetq2_R(4 * dremain);
        HVX_Vector newvSumf = Q6_Vqf32_vadd_Vqf32Vsf(vsumf, poly0);
        vsumf               = Q6_V_vmux_QVV(q0, newvSumf, vsumf);  // don't update the extra lanes
        q6op_vstu_variable_ARV(optr, dremain * 4, poly0);
      }
    }
  }
  nshift = 4;
  for (int i = 0; i < 5; i++) {
    HVX_VectorPair temps = Q6_W_vshuff_VVR(vsumf, vsumf, nshift);
    vsumf                = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(temps), Q6_V_hi_W(temps));
    nshift <<= 1;
  }
  UVecFlts vsumfVar = {Q6_Vsf_equals_Vqf32(vsumf)};

  finishSoftmax(pout, length, &vsumfVar);
}

void softmaxHApprox(float *pout, const uint16_t *pin, float scale, int length) {
  scale /= float(log(2.0));
  int brsh   = flt_getexp(-scale) - 24 + 31 + (16 - 7);
  int bscale = -flt_getfrac(-scale);

  HVX_Vector *iptr = (HVX_Vector *)pin;
  HVX_Vector xmax  = Q6_V_vzero();

  // find max
  for (int d = length; d > 63; d -= 64) {
    HVX_Vector xinval = vmemu(iptr);
    iptr++;
    xmax = Q6_Vuh_vmax_VuhVuh(xmax, xinval);
  }
  if ((length & 63) != 0) {
    HVX_Vector xinval         = vmemu(iptr);
    HVX_VectorPred qfinalmask = Q6_Q_vsetq2_R(length * 2);
    xinval                    = Q6_V_vmux_QVV(qfinalmask, xinval, xmax);
    xmax                      = Q6_Vuh_vmax_VuhVuh(xmax, xinval);
  }
  int nshift = 2;
  for (int i = 0; i < 6; i++) {
    HVX_VectorPair temps = Q6_W_vshuff_VVR(xmax, xmax, nshift);
    xmax                 = Q6_Vuh_vmax_VuhVuh(Q6_V_lo_W(temps), Q6_V_hi_W(temps));
    nshift <<= 1;
  }

  // calculate sum
  HVX_Vector const0x7FFFFFFF = Q6_V_vsplat_R(0x7FFFFFFF);
  HVX_Vector const0x007F     = Q6_V_vsplat_R(0x007F);
  HVX_Vector f1              = Q6_V_vsplat_R(0x5862D2C9);
  HVX_Vector f2              = Q6_V_vsplat_R(0x1E74B4C8);
  HVX_Vector f3              = Q6_V_vsplat_R(0x0928786F);
  HVX_Vector vbeta           = Q6_V_vsplat_R(bscale << 7);
  HVX_Vector vbrshf          = Q6_V_vsplat_R(brsh);
  HVX_Vector vbrshe          = Q6_V_vsplat_R(brsh - 31);
  HVX_Vector vsumf           = Q6_V_vzero();
  HVX_Vector *optr           = (HVX_Vector *)pout;
  iptr                       = (HVX_Vector *)pin;

  for (int d = length; d > 0; d -= 64) {
    HVX_Vector x, x0, x1, xd0, xd1, xe0, xe1, p0, p1;
    HVX_VectorPair xdiff, p10;

    x = vmemu(iptr);
    iptr++;
    xdiff = Q6_Ww_vsub_VuhVuh(xmax, x);
    x0    = Q6_Vw_vmpye_VwVuh(vbeta, Q6_V_lo_W(xdiff));
    x1    = Q6_Vw_vmpye_VwVuh(vbeta, Q6_V_hi_W(xdiff));

    xd0 = Q6_Vw_vasl_VwVw(x0, vbrshf);
    xd1 = Q6_Vw_vasl_VwVw(x1, vbrshf);

    xd0 = Q6_V_vand_VV(xd0, const0x7FFFFFFF);
    xd1 = Q6_V_vand_VV(xd1, const0x7FFFFFFF);

    xe0 = Q6_Vw_vasl_VwVw(x0, vbrshe);
    xe1 = Q6_Vw_vasl_VwVw(x1, vbrshe);
    xe0 = Q6_Vw_vasl_VwR(Q6_Vw_vadd_VwVw(xe0, const0x007F), 23);
    xe1 = Q6_Vw_vasl_VwR(Q6_Vw_vadd_VwVw(xe1, const0x007F), 23);

    p0 = Q6_Vw_vadd_VwVw(
        Q6_Vw_vmpyoacc_VwVwVh_s1_rnd_sat_shift(Q6_Vw_vmpye_VwVuh(xd0, f3), xd0, f3), f2);
    p0 = Q6_Vw_vadd_VwVw(
        Q6_Vw_vmpyoacc_VwVwVh_s1_rnd_sat_shift(Q6_Vw_vmpye_VwVuh(xd0, p0), xd0, p0), f1);
    p0 = Q6_Vw_vmpyoacc_VwVwVh_s1_rnd_sat_shift(Q6_Vw_vmpye_VwVuh(xd0, p0), xd0, p0);
    p0 = Q6_Vw_vasracc_VwVwR(xe0, p0, 8);

    p1 = Q6_Vw_vadd_VwVw(
        Q6_Vw_vmpyoacc_VwVwVh_s1_rnd_sat_shift(Q6_Vw_vmpye_VwVuh(xd1, f3), xd1, f3), f2);
    p1 = Q6_Vw_vadd_VwVw(
        Q6_Vw_vmpyoacc_VwVwVh_s1_rnd_sat_shift(Q6_Vw_vmpye_VwVuh(xd1, p1), xd1, p1), f1);
    p1 = Q6_Vw_vmpyoacc_VwVwVh_s1_rnd_sat_shift(Q6_Vw_vmpye_VwVuh(xd1, p1), xd1, p1);
    p1 = Q6_Vw_vasracc_VwVwR(xe1, p1, 8);

    p10 = Q6_W_vshuff_VVR(p1, p0, -4);

    if (d >= 64) {
      q6op_vstu_AV(optr, Q6_V_lo_W(p10));
      optr++;
      q6op_vstu_AV(optr, Q6_V_hi_W(p10));
      optr++;

      vsumf = Q6_Vqf32_vadd_Vqf32Vsf(vsumf, p0);
      vsumf = Q6_Vqf32_vadd_Vqf32Vsf(vsumf, p1);
    } else {
      HVX_Vector vx0 = Q6_V_lo_W(p10);
      int dremain    = d;
      if (dremain >= 32) {
        vsumf = Q6_Vqf32_vadd_Vqf32Vsf(vsumf, vx0);
        q6op_vstu_AV(optr, vx0);
        optr++;
        vx0 = Q6_V_hi_W(p10);
        dremain -= 32;
      }
      if (dremain) {  // is 1..31
        HVX_Vector vsumfNew = Q6_Vqf32_vadd_Vqf32Vsf(vsumf, vx0);
        HVX_VectorPred q0   = Q6_Q_vsetq2_R(4 * dremain);
        vsumf               = Q6_V_vmux_QVV(q0, vsumfNew, vsumf);
        q6op_vstu_variable_ARV(optr, dremain * 4, vx0);
      }
    }
  }

  for (int i = 0, nshift = 4; i < 5; i++) {
    HVX_VectorPair temps = Q6_W_vshuff_VVR(vsumf, vsumf, nshift);
    vsumf                = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(temps), Q6_V_hi_W(temps));
    nshift <<= 1;
  }
  UVecFlts vsumfVar = {Q6_Vsf_equals_Vqf32(vsumf)};

  finishSoftmax(pout, length, &vsumfVar);
}

static void finishSoftmax(float *pout, int length, UVecFlts const *vsumf) {
  // scale output
  // read float from first element of *vsumf, 1/x, splat back to vector
  float sumF        = vsumf->m_floatArray32[0];
  HVX_Vector vRecip = Q6_V_vsplat_R((image_convert<uint32_t, float>(1.0f / sumF)));
  HVX_Vector *ioptr = (HVX_Vector *)pout;

  HVX_Vector x;
  int d;
  for (d = length; d > 31; d -= 32) {
    x = vmemu(&ioptr[0]);
    x = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(x, vRecip));
    q6op_vstu_AV(&ioptr[0], x);
    ioptr++;
  }
  if (d != 0) {
    x = vmemu(&ioptr[0]);
    x = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(x, vRecip));
    q6op_vstu_variable_ARV(&ioptr[0], d * 4, x);
  }
}

void softmaxHD2Approx(float *pout, const uint16_t *pin, float scale, int32_t length) {
  float reciprocalSum[32] __attribute__((aligned(128)));
  scale /= float(log(2.0));
  int brsh   = flt_getexp(-scale) - 24 + 31 + (16 - 7);
  int bscale = -flt_getfrac(-scale);

  HVX_Vector *iptr = (HVX_Vector *)pin;
  HVX_Vector *optr = (HVX_Vector *)pout;
  HVX_Vector xmax;
  HVX_VectorPair xdiff, p10;

  HVX_Vector const0x7FFFFFFF = Q6_V_vsplat_R(0x7FFFFFFF);
  HVX_Vector const0x007F     = Q6_V_vsplat_R(0x007F);
  HVX_Vector f1              = Q6_V_vsplat_R(0x5862D2C9);
  HVX_Vector f2              = Q6_V_vsplat_R(0x1E74B4C8);
  HVX_Vector f3              = Q6_V_vsplat_R(0x0928786F);
  HVX_Vector vzero           = Q6_V_vzero();
  HVX_Vector vbeta           = Q6_V_vsplat_R(bscale << 7);
  HVX_Vector vbrshf          = Q6_V_vsplat_R(brsh);
  HVX_Vector vbrshe          = Q6_V_vsplat_R(brsh - 31);
  p10                        = Q6_W_vcombine_VV(vzero, vzero);

  int32_t i;
  for (i = 0; i < length; i += 64) {
    vmemu(optr) = Q6_V_lo_W(p10);
    if (i) optr++;
    vmemu(optr) = Q6_V_hi_W(p10);
    if (i) optr++;

    // find max
    HVX_Vector xinval = vmemu(iptr);
    iptr++;
    HVX_Vector xinval2 = Q6_Vh_vshuffo_VhVh(xinval, xinval);
    xmax               = Q6_Vuh_vmax_VuhVuh(xinval2, xinval);
    xmax               = Q6_Vh_vshuffe_VhVh(xmax, xmax);

    // calculate sum
    HVX_Vector vsumf = Q6_V_vzero();

    HVX_Vector x0, x1, xd0, xd1, xe0, xe1, p0, p1;

    xdiff = Q6_Ww_vsub_VuhVuh(xmax, xinval);
    x0    = Q6_Vw_vmpye_VwVuh(vbeta, Q6_V_lo_W(xdiff));
    x1    = Q6_Vw_vmpye_VwVuh(vbeta, Q6_V_hi_W(xdiff));

    xd0 = Q6_Vw_vasl_VwVw(x0, vbrshf);
    xd1 = Q6_Vw_vasl_VwVw(x1, vbrshf);

    xd0 = Q6_V_vand_VV(xd0, const0x7FFFFFFF);
    xd1 = Q6_V_vand_VV(xd1, const0x7FFFFFFF);

    xe0 = Q6_Vw_vasl_VwVw(x0, vbrshe);
    xe1 = Q6_Vw_vasl_VwVw(x1, vbrshe);
    xe0 = Q6_Vw_vasl_VwR(Q6_Vw_vadd_VwVw(xe0, const0x007F), 23);
    xe1 = Q6_Vw_vasl_VwR(Q6_Vw_vadd_VwVw(xe1, const0x007F), 23);

    p0 = Q6_Vw_vadd_VwVw(
        Q6_Vw_vmpyoacc_VwVwVh_s1_rnd_sat_shift(Q6_Vw_vmpye_VwVuh(xd0, f3), xd0, f3), f2);
    p0 = Q6_Vw_vadd_VwVw(
        Q6_Vw_vmpyoacc_VwVwVh_s1_rnd_sat_shift(Q6_Vw_vmpye_VwVuh(xd0, p0), xd0, p0), f1);
    p0 = Q6_Vw_vmpyoacc_VwVwVh_s1_rnd_sat_shift(Q6_Vw_vmpye_VwVuh(xd0, p0), xd0, p0);
    p0 = Q6_Vw_vasracc_VwVwR(xe0, p0, 8);

    p1 = Q6_Vw_vadd_VwVw(
        Q6_Vw_vmpyoacc_VwVwVh_s1_rnd_sat_shift(Q6_Vw_vmpye_VwVuh(xd1, f3), xd1, f3), f2);
    p1 = Q6_Vw_vadd_VwVw(
        Q6_Vw_vmpyoacc_VwVwVh_s1_rnd_sat_shift(Q6_Vw_vmpye_VwVuh(xd1, p1), xd1, p1), f1);
    p1 = Q6_Vw_vmpyoacc_VwVwVh_s1_rnd_sat_shift(Q6_Vw_vmpye_VwVuh(xd1, p1), xd1, p1);
    p1 = Q6_Vw_vasracc_VwVwR(xe1, p1, 8);

    vsumf                              = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(p1, p0));
    *(HVX_Vector *)(&reciprocalSum[0]) = vsumf;

    for (int j = 0; j < 32; j++) reciprocalSum[j] = 1 / reciprocalSum[j];

    vsumf = *(HVX_Vector *)(&reciprocalSum[0]);
    p0    = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(p0, vsumf));
    p1    = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(p1, vsumf));
    p10   = Q6_W_vshuff_VVR(p1, p0, -4);
  }
  int remain = length - (i - 64);  // 1..64
  hvx_store_vec_x2_unaligned_inline(optr, Q6_V_lo_W(p10), Q6_V_hi_W(p10), remain * 4);
}

/////////////////// depth = 2 special case /////////////////////////////////
//    (1) find d0-d1   (from quantized values, range -255...255)
//    (2) using abs(d0-d1), do a table lookup. result is 128..255
//        Table depends on beta & scaling; the result is correct for
//        output 0 if d0>d1.
//    (3) find the second result by subtracting the first from 255;
//    (4) if d0<d1, swap the results.
//     - result is correct for output range 0..1.0
HVX_VectorPair softmaxD2U8Hvx(const HVX_Vector &in0,
                              const HVX_Vector &in1,
                              const HVX_Vector &tbl0,
                              const HVX_Vector &tbl1) {
  // HVX_VectorPair inShuff = Q6_W_vdeal_VVR(in1, in0, 1);
  HVX_VectorPair inShuff = Q6_Wb_vshuffoe_VbVb(in1, in0);

  // abs(d0 - d1)
  HVX_Vector dabs = Q6_Vub_vabsdiff_VubVub(Q6_V_lo_W(inShuff), Q6_V_hi_W(inShuff));

  HVX_VectorPred qv = Q6_Q_vcmp_gt_VubVub(Q6_V_lo_W(inShuff), Q6_V_hi_W(inShuff));
  // HVX_VectorPred qv_not = Q6_Q_not_Q(qv);

  // u8->u8 lookup on abs(d0 - d1)
  HVX_Vector vout = Q6_Vb_vlut32_VbVbI(dabs, tbl0, 0);
  vout            = Q6_Vb_vlut32or_VbVbVbI(vout, dabs, tbl0, 1);
  vout            = Q6_Vb_vlut32or_VbVbVbI(vout, dabs, tbl0, 2);
  vout            = Q6_Vb_vlut32or_VbVbVbI(vout, dabs, tbl0, 3);
  vout            = Q6_Vb_vlut32or_VbVbVbI(vout, dabs, tbl1, 4);
  vout            = Q6_Vb_vlut32or_VbVbVbI(vout, dabs, tbl1, 5);
  vout            = Q6_Vb_vlut32or_VbVbVbI(vout, dabs, tbl1, 6);
  vout            = Q6_Vb_vlut32or_VbVbVbI(vout, dabs, tbl1, 7);

  // compute 255 - vout to get out value of the smaller element
  HVX_Vector voutComp = Q6_V_vxor_VV(vout, Q6_Vb_vsplat_R(255));

  // use comparison d0 > d1 to decide between vout and 255 - vout
  // HVX_Vector vout0 = Q6_V_vmux_QVV(qv, vout, voutComp);
  // HVX_Vector vout1 = Q6_V_vmux_QVV(qv_not, vout, voutComp);
  HVX_VectorPair vouts = Q6_W_vswap_QVV(qv, vout, voutComp);

  // and then combine them
  // return Q6_Wb_vshuffoe_VbVb( vout1, vout0);
  return Q6_Wb_vshuffoe_VbVb(Q6_V_hi_W(vouts), Q6_V_lo_W(vouts));
}

void softmaxBD2Approx(uint8_t *out, const uint8_t *in, const uint8_t *lut, size_t inSize) {
  const size_t numVecPairs = inSize >> 8;
  const size_t leftover    = inSize & 255;

  const HVX_Vector *iptr = (const HVX_Vector *)in;
  HVX_Vector *optr       = (HVX_Vector *)out;

  HVX_Vector tbl0 = vmemu(lut);
  HVX_Vector tbl1 = vmemu(lut + 128);

  bool useUnalign = (((size_t)in) & 0x7f) != 0 || (((size_t)out) & 0x7f) != 0;

  HVX_Vector in0, in1;
  HVX_VectorPair results;

  if (useUnalign) {
    for (int i = 0; i < numVecPairs; i++) {
      in0     = vmemu(iptr + 0);
      in1     = vmemu(iptr + 1);
      results = softmaxD2U8Hvx(in0, in1, tbl0, tbl1);
      q6op_vstu_AV(optr + 0, Q6_V_lo_W(results));
      q6op_vstu_AV(optr + 1, Q6_V_hi_W(results));
      iptr += 2;
      optr += 2;
    }
  } else {
    HVX_Vector vout, voutComp;
    HVX_VectorPair vouts;

    in0                    = *iptr++;
    in1                    = *iptr++;
    HVX_VectorPair inShuff = Q6_Wb_vshuffoe_VbVb(in1, in0);

    HVX_Vector dabs   = Q6_Vub_vabsdiff_VubVub(Q6_V_lo_W(inShuff), Q6_V_hi_W(inShuff));
    HVX_VectorPred qv = Q6_Q_vcmp_gt_VubVub(Q6_V_lo_W(inShuff), Q6_V_hi_W(inShuff));

    for (int i = 1; i < numVecPairs; i++) {
      in0     = *iptr++;
      in1     = *iptr++;                        // for next iteration
      inShuff = Q6_Wb_vshuffoe_VbVb(in1, in0);  // for next iteration

      vout = Q6_Vb_vlut32_VbVbI(dabs, tbl0, 0);
      vout = Q6_Vb_vlut32or_VbVbVbI(vout, dabs, tbl0, 1);
      vout = Q6_Vb_vlut32or_VbVbVbI(vout, dabs, tbl0, 2);
      vout = Q6_Vb_vlut32or_VbVbVbI(vout, dabs, tbl0, 3);
      vout = Q6_Vb_vlut32or_VbVbVbI(vout, dabs, tbl1, 4);
      vout = Q6_Vb_vlut32or_VbVbVbI(vout, dabs, tbl1, 5);
      vout = Q6_Vb_vlut32or_VbVbVbI(vout, dabs, tbl1, 6);
      vout = Q6_Vb_vlut32or_VbVbVbI(vout, dabs, tbl1, 7);

      voutComp = Q6_V_vxor_VV(vout, Q6_Vb_vsplat_R(255));

      vouts = Q6_W_vswap_QVV(qv, vout, voutComp);

      HVX_VectorPair results = Q6_Wb_vshuffoe_VbVb(Q6_V_hi_W(vouts), Q6_V_lo_W(vouts));

      *optr++ = Q6_V_lo_W(results);
      *optr++ = Q6_V_hi_W(results);

      // for next iteration
      dabs = Q6_Vub_vabsdiff_VubVub(Q6_V_lo_W(inShuff), Q6_V_hi_W(inShuff));
      qv   = Q6_Q_vcmp_gt_VubVub(Q6_V_lo_W(inShuff), Q6_V_hi_W(inShuff));
    }

    vout = Q6_Vb_vlut32_VbVbI(dabs, tbl0, 0);
    vout = Q6_Vb_vlut32or_VbVbVbI(vout, dabs, tbl0, 1);
    vout = Q6_Vb_vlut32or_VbVbVbI(vout, dabs, tbl0, 2);
    vout = Q6_Vb_vlut32or_VbVbVbI(vout, dabs, tbl0, 3);
    vout = Q6_Vb_vlut32or_VbVbVbI(vout, dabs, tbl1, 4);
    vout = Q6_Vb_vlut32or_VbVbVbI(vout, dabs, tbl1, 5);
    vout = Q6_Vb_vlut32or_VbVbVbI(vout, dabs, tbl1, 6);
    vout = Q6_Vb_vlut32or_VbVbVbI(vout, dabs, tbl1, 7);

    voutComp = Q6_V_vxor_VV(vout, Q6_Vb_vsplat_R(255));

    vouts = Q6_W_vswap_QVV(qv, vout, voutComp);

    results = Q6_Wb_vshuffoe_VbVb(Q6_V_hi_W(vouts), Q6_V_lo_W(vouts));

    *optr++ = Q6_V_lo_W(results);
    *optr++ = Q6_V_hi_W(results);
  }

  if (leftover) {
    in0     = vmemu(iptr + 0);
    in1     = leftover > 128 ? vmemu(iptr + 1) : in0;
    results = softmaxD2U8Hvx(in0, in1, tbl0, tbl1);
    hvx_store_vec_x2_unaligned_inline(optr, Q6_V_lo_W(results), Q6_V_hi_W(results), leftover);
  }
}

/* Coefficients in float representation */
static const float sg_c0Coeffs[32] __attribute__((aligned(128))) = {
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    3.882601794814435,
    3.6625422144222575,
    3.464451548227971,
    3.2869700047974098,
    3.126105117815294,
    2.9797652947122333,
    2.846287833147896,
    2.7247270166228237,
    2.614282526778659,
    2.5119448279766914,
    2.4168240690138916,
    2.3287715099556494,
    2.2470044371606255,
    2.1705097010458525,
    2.0993232550771013,
    2.032425103348979,
};
static const float sg_c1Coeffs[32] __attribute__((aligned(128))) = {
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    -5.65213466274883,
    -5.029649818173625,
    -4.500359068222728,
    -4.051125252469975,
    -3.6643282495304743,
    -3.3293252513210945,
    -3.0377500909629918,
    -2.78384542029156,
    -2.562751394984757,
    -2.3660481944625364,
    -2.1902579830702398,
    -2.033579850063907,
    -1.8932880190031018,
    -1.7665817851802996,
    -1.6526109646324616,
    -1.5489652830974667,
};
static const float sg_c2Coeffs[32] __attribute__((aligned(128))) = {
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    3.6564123863772062,
    3.0693863078484034,
    2.5979108429264546,
    2.2188401136904137,
    1.90879196515026,
    1.6531365145318937,
    1.4408072849395228,
    1.2640160009581791,
    1.1164726565567085,
    0.9904366133906549,
    0.8821387892416702,
    0.7892039810345458,
    0.7089644931002874,
    0.6390020714403465,
    0.5781761255999769,
    0.5246475096790261,
};
static const float sg_c3Coeffs[32] __attribute__((aligned(128))) = {
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    -0.8868796162009371,
    -0.7023245532864408,
    -0.5623148115716742,
    -0.45568061400557225,
    -0.3728293181808119,
    -0.30778916969628956,
    -0.25624427383670373,
    -0.21520836864975557,
    -0.18238585316003267,
    -0.1554651987039696,
    -0.133224398745864,
    -0.11484835534787588,
    -0.09954996553138899,
    -0.08667244996867919,
    -0.07585106425203664,
    -0.06663557250850614,
};

static constexpr unsigned sg_nv = 128;

union HvxVectorConstant {
  uint8_t ub[sg_nv];
  HVX_Vector v;
};

template <bool REVERSE>
inline constexpr HvxVectorConstant vectorDeltaFromMapping(HvxVectorConstant mapVec) {
  HvxVectorConstant result = {{
      0,
  }};
  struct Mp {
    uint8_t arr[sg_nv];
  } mapping     = {{
      0,
  }},
    mappingNext = {{
        0,
    }};

  for (unsigned i = 0; i < sg_nv; i++) {
    unsigned val   = mapVec.ub[i];
    mapping.arr[i] = (val < sg_nv) ? val : 0xFF;
  }

  unsigned wend  = REVERSE ? 0 : sg_nv;
  bool conflicts = false;
  for (unsigned w = (REVERSE ? sg_nv / 2 : 1); w != wend; w = REVERSE ? (w >> 1) : (w << 1)) {
    for (unsigned i = 0; i < sg_nv; i++) mappingNext.arr[i] = 0xFF;  // init all "don't care"
    for (unsigned i0 = 0; i0 < sg_nv - 1; i0++) {
      unsigned i1 = i0 ^ w;
      if (i1 > i0) {
        unsigned m0 = mapping.arr[i0];
        unsigned m1 = mapping.arr[i1];
        unsigned s0 = i0 | (m0 & w);  // = i0 or i1
        unsigned s1 = i0 | (m1 & w);
        if (m0 != 0xFF) {  // route m0 s0->i0
          mappingNext.arr[s0] = m0;
          result.ub[i0] |= (i0 ^ s0);  // 0 or w
        }
        if (m1 != 0xFF) {  // route m1 s1->i1
          // are we about to overwrite m0 with a different m1?
          if (s0 == s1 && m0 != m1 && m0 != 0xFF) conflicts = true;
          mappingNext.arr[s1] = m1;
          result.ub[i1] |= (i1 ^ s1);  // 0 or w
        }
      }
    }
    mapping = mappingNext;
  }
  if (conflicts) {
    debuglog("There is conflict in the mapping arrays");
  }
  return result;
}

template <typename T_GENFUNC>
inline constexpr HvxVectorConstant vectorFromFuncB(T_GENFUNC f) {
  HvxVectorConstant v = {{
      0,
  }};
  for (int i = 0; i < sg_nv; i++) v.ub[i] = f(i);
  return v;
}
//
// generate a 'delta' or 'rdelta' control vector from a mapping function. The
// mapping function is called with i = 0...127, once for each output lane; it
// returns the index 0...127 of the input lane which is to be routed to that
// output. To indicate a 'don't-care', the function can return -1, or anything
// in range 128...255.
//
template <bool REVERSE, typename T_GENFUNC>
inline constexpr HvxVectorConstant vectorDeltaFromFunctionB(T_GENFUNC f) {
  return vectorDeltaFromMapping<REVERSE>(vectorFromFuncB(f));
}

template <int D>
inline constexpr int patternIm2colU8(int pos) {
  return ((pos % 32) + (pos / 32) * D);
}

template <int D>
inline constexpr int packDown32(int pos) {
  return pos < D * 4 ? (pos % D + (pos / D) * 32) : -1;
}

template <int D>
inline constexpr int packUp32(int pos) {
  pos -= (D <= 16) ? D * 4 : (128 - D * 4);
  if (pos >= 0 && pos < D * 4) return pos % D + (pos / D) * 32;
  return -1;
}

const HvxVectorConstant g_im2colU8Controls[32] = {
    vectorDeltaFromFunctionB<false>(patternIm2colU8<1>),
    vectorDeltaFromFunctionB<false>(patternIm2colU8<2>),
    vectorDeltaFromFunctionB<false>(patternIm2colU8<3>),
    vectorDeltaFromFunctionB<false>(patternIm2colU8<4>),
    vectorDeltaFromFunctionB<false>(patternIm2colU8<5>),
    vectorDeltaFromFunctionB<false>(patternIm2colU8<6>),
    vectorDeltaFromFunctionB<false>(patternIm2colU8<7>),
    vectorDeltaFromFunctionB<false>(patternIm2colU8<8>),
    vectorDeltaFromFunctionB<false>(patternIm2colU8<9>),
    vectorDeltaFromFunctionB<false>(patternIm2colU8<10>),
    vectorDeltaFromFunctionB<false>(patternIm2colU8<11>),
    vectorDeltaFromFunctionB<false>(patternIm2colU8<12>),
    vectorDeltaFromFunctionB<false>(patternIm2colU8<13>),
    vectorDeltaFromFunctionB<false>(patternIm2colU8<14>),
    vectorDeltaFromFunctionB<false>(patternIm2colU8<15>),
    vectorDeltaFromFunctionB<false>(patternIm2colU8<16>),
    vectorDeltaFromFunctionB<false>(patternIm2colU8<17>),
    vectorDeltaFromFunctionB<false>(patternIm2colU8<18>),
    vectorDeltaFromFunctionB<false>(patternIm2colU8<19>),
    vectorDeltaFromFunctionB<false>(patternIm2colU8<20>),
    vectorDeltaFromFunctionB<false>(patternIm2colU8<21>),
    vectorDeltaFromFunctionB<false>(patternIm2colU8<22>),
    vectorDeltaFromFunctionB<false>(patternIm2colU8<23>),
    vectorDeltaFromFunctionB<false>(patternIm2colU8<24>),
    vectorDeltaFromFunctionB<false>(patternIm2colU8<25>),
    vectorDeltaFromFunctionB<false>(patternIm2colU8<26>),
    vectorDeltaFromFunctionB<false>(patternIm2colU8<27>),
    vectorDeltaFromFunctionB<false>(patternIm2colU8<28>),
    vectorDeltaFromFunctionB<false>(patternIm2colU8<29>),
    vectorDeltaFromFunctionB<false>(patternIm2colU8<30>),
    vectorDeltaFromFunctionB<false>(patternIm2colU8<31>),
    vectorDeltaFromFunctionB<false>(patternIm2colU8<32>)};

const HvxVectorConstant g_tileU8PackDownControls[62] = {
    vectorDeltaFromFunctionB<true>(packDown32<1>),  vectorDeltaFromFunctionB<true>(packUp32<1>),
    vectorDeltaFromFunctionB<true>(packDown32<2>),  vectorDeltaFromFunctionB<true>(packUp32<2>),
    vectorDeltaFromFunctionB<true>(packDown32<3>),  vectorDeltaFromFunctionB<true>(packUp32<3>),
    vectorDeltaFromFunctionB<true>(packDown32<4>),  vectorDeltaFromFunctionB<true>(packUp32<4>),
    vectorDeltaFromFunctionB<true>(packDown32<5>),  vectorDeltaFromFunctionB<true>(packUp32<5>),
    vectorDeltaFromFunctionB<true>(packDown32<6>),  vectorDeltaFromFunctionB<true>(packUp32<6>),
    vectorDeltaFromFunctionB<true>(packDown32<7>),  vectorDeltaFromFunctionB<true>(packUp32<7>),
    vectorDeltaFromFunctionB<true>(packDown32<8>),  vectorDeltaFromFunctionB<true>(packUp32<8>),
    vectorDeltaFromFunctionB<true>(packDown32<9>),  vectorDeltaFromFunctionB<true>(packUp32<9>),
    vectorDeltaFromFunctionB<true>(packDown32<10>), vectorDeltaFromFunctionB<true>(packUp32<10>),
    vectorDeltaFromFunctionB<true>(packDown32<11>), vectorDeltaFromFunctionB<true>(packUp32<11>),
    vectorDeltaFromFunctionB<true>(packDown32<12>), vectorDeltaFromFunctionB<true>(packUp32<12>),
    vectorDeltaFromFunctionB<true>(packDown32<13>), vectorDeltaFromFunctionB<true>(packUp32<13>),
    vectorDeltaFromFunctionB<true>(packDown32<14>), vectorDeltaFromFunctionB<true>(packUp32<14>),
    vectorDeltaFromFunctionB<true>(packDown32<15>), vectorDeltaFromFunctionB<true>(packUp32<15>),
    vectorDeltaFromFunctionB<true>(packDown32<16>), vectorDeltaFromFunctionB<true>(packUp32<16>),
    vectorDeltaFromFunctionB<true>(packDown32<17>), vectorDeltaFromFunctionB<true>(packUp32<17>),
    vectorDeltaFromFunctionB<true>(packDown32<18>), vectorDeltaFromFunctionB<true>(packUp32<18>),
    vectorDeltaFromFunctionB<true>(packDown32<19>), vectorDeltaFromFunctionB<true>(packUp32<19>),
    vectorDeltaFromFunctionB<true>(packDown32<20>), vectorDeltaFromFunctionB<true>(packUp32<20>),
    vectorDeltaFromFunctionB<true>(packDown32<21>), vectorDeltaFromFunctionB<true>(packUp32<21>),
    vectorDeltaFromFunctionB<true>(packDown32<22>), vectorDeltaFromFunctionB<true>(packUp32<22>),
    vectorDeltaFromFunctionB<true>(packDown32<23>), vectorDeltaFromFunctionB<true>(packUp32<23>),
    vectorDeltaFromFunctionB<true>(packDown32<24>), vectorDeltaFromFunctionB<true>(packUp32<24>),
    vectorDeltaFromFunctionB<true>(packDown32<25>), vectorDeltaFromFunctionB<true>(packUp32<25>),
    vectorDeltaFromFunctionB<true>(packDown32<26>), vectorDeltaFromFunctionB<true>(packUp32<26>),
    vectorDeltaFromFunctionB<true>(packDown32<27>), vectorDeltaFromFunctionB<true>(packUp32<27>),
    vectorDeltaFromFunctionB<true>(packDown32<28>), vectorDeltaFromFunctionB<true>(packUp32<28>),
    vectorDeltaFromFunctionB<true>(packDown32<29>), vectorDeltaFromFunctionB<true>(packUp32<29>),
    vectorDeltaFromFunctionB<true>(packDown32<30>), vectorDeltaFromFunctionB<true>(packUp32<30>),
    vectorDeltaFromFunctionB<true>(packDown32<31>), vectorDeltaFromFunctionB<true>(packUp32<31>)};

void softmaxApproxShortd(float *pout, const uint8_t *pin, float scale, int depth, int nsamples) {
  HVX_Vector tmpbuf[9];
  int block = 128 / depth;
  scale /= log(2.0);
  int bscale = -((flt_getfrac(-scale) + 1) >> 1);
  int brsh   = min_i32(31, max_i32(flt_getexp(-scale) - 23 + 14 + 1, -17));
  //
  // constants for finding max
  HVX_Vector const permctrl0 =
      (depth < 32) ? *(HVX_Vector const *)&g_im2colU8Controls[depth - 1] : Q6_V_vzero();
  HVX_Vector const permctrl1 =
      (depth < 32) ? *(HVX_Vector const *)&g_tileU8PackDownControls[2 * (depth - 1)] : Q6_V_vzero();

  HVX_Vector vzero      = Q6_V_vzero();
  HVX_Vector mask       = Q6_V_vand_QR(Q6_Q_vsetq_R(depth), -1);
  mask                  = Q6_V_vor_VV(Q6_V_vror_VR(mask, 96), mask);
  mask                  = Q6_V_vor_VV(Q6_V_vror_VR(mask, 64), mask);
  HVX_VectorPred qdepth = Q6_Q_vcmp_gt_VubVub(mask, vzero);
  //
  // constatns for calculating exp and sum
  HVX_Vector const0x7FFE = Q6_Vh_vsplat_R(0x7FFE);
  HVX_Vector const0x007F = Q6_Vh_vsplat_R(0x007F);
  HVX_Vector f0          = Q6_Vh_vsplat_R(0x8000);
  HVX_Vector f1          = Q6_Vh_vsplat_R(0x5863);
  HVX_Vector f2          = Q6_Vh_vsplat_R(0x1E75);
  HVX_Vector f3          = Q6_Vh_vsplat_R(0x0928);
  HVX_Vector vbeta       = Q6_V_vsplat_R(bscale);
  HVX_Vector vbrshf      = Q6_V_vsplat_R(brsh);
  HVX_Vector vbrshe      = Q6_V_vsplat_R(brsh - 15);
  //
  // (copied from SDK)
  // constants for reciprocal
  /*
   * Splat scale factor in order to be used later for finding indexes of
   * coefficients. Scale factor is represented in IEEE 16-bit floating-point
   * format and it is calculated using the following formula: scale_factor =
   * (16.0 / (b0 - a0)) NOTE: Calculated value is slightly decreased in order to
   * avoid out of bound indexes during VLUT lookup.
   */
  HVX_Vector scaleV = Q6_V_vsplat_R(0x417ffffe);

  /*
   * Vector of zeroes used as neutral element in sf to qf32 conversions.
   * NOTE: Some of conversions (i.e conversion of scale factor and coefficients)
   *       can be avoided in real-time, but this is not done in order to don't
   *       sacrify code readibility in expense of insignificant performance
   * improvement.
   */
  HVX_Vector zeroVSf = Q6_V_vzero();

  /* Set sign = 0, exp = 254, mant = 0 */
  HVX_Vector exp = Q6_V_vsplat_R(0x7F000000);

  /* Set mask for sign and exponent */
  HVX_Vector signexpMask = Q6_V_vsplat_R(0xFF800000);

  /* Mask for extracting only 4 bits of mantissa */
  HVX_Vector maskIdx1V = Q6_V_vsplat_R(0x0000000F);
  HVX_Vector maskIdx2V = Q6_V_vsplat_R(0x00000010);

  /* 16.0 in IEEE 16-bit floating-point representation */
  HVX_Vector const160VSf = Q6_V_vsplat_R(0x41800000);
  /*
   * Prepare vector of input_min values, that is used later in shifting input
   * range. input_min is low boundary of specified input range.
   */
  HVX_Vector inputMinVF = Q6_V_vsplat_R(0x3f800000);

  /* Convert scale factor from sf to q32. Use the same vector for both formats
   */
  scaleV = Q6_Vqf32_vadd_VsfVsf(scaleV, zeroVSf);

  /* Load coefficients */
  HVX_Vector c0CoeffV = *((HVX_Vector *)(sg_c0Coeffs));
  HVX_Vector c1CoeffV = *((HVX_Vector *)(sg_c1Coeffs));
  HVX_Vector c2CoeffV = *((HVX_Vector *)(sg_c2Coeffs));
  HVX_Vector c3CoeffV = *((HVX_Vector *)(sg_c3Coeffs));

  /* Convert coefficients from sf to qf32 format. Use the same vector for both
   * representations */
  c0CoeffV = Q6_Vqf32_vadd_VsfVsf(c0CoeffV, zeroVSf);
  c1CoeffV = Q6_Vqf32_vadd_VsfVsf(c1CoeffV, zeroVSf);
  c2CoeffV = Q6_Vqf32_vadd_VsfVsf(c2CoeffV, zeroVSf);
  c3CoeffV = Q6_Vqf32_vadd_VsfVsf(c3CoeffV, zeroVSf);

  /* Split 32-bit coefficients to lower and upper part in order to obtain them
   * later with VLUT16. */
  HVX_VectorPair c0CoeffDv = Q6_Wuw_vzxt_Vuh(c0CoeffV);
  HVX_VectorPair c1CoeffDv = Q6_Wuw_vzxt_Vuh(c1CoeffV);
  HVX_VectorPair c2CoeffDv = Q6_Wuw_vzxt_Vuh(c2CoeffV);
  HVX_VectorPair c3CoeffDv = Q6_Wuw_vzxt_Vuh(c3CoeffV);

  for (int n = nsamples; n > 0; n -= block) {
    int blockValid = min_i32(n, block);
    HVX_Vector x   = vmemu(pin);
    pin += block * depth;
    /*
     * Find max. Process up to 4 depth channels per iteration and load into xmax
     * until filled with max values for entire block.
     */
    HVX_Vector xmax;
    HVX_Vector xt = x;
    int nbytes    = min_i32(block, 4) * depth;
    for (int k = block; k > 0; k -= 4) {
      if (k < 4) nbytes = k * depth;
      // Only processing up to 4 * depth elements. Permute these so that each
      // depth is in 32 bytes, with unused bytes zeroed out.
      HVX_Vector xmax4 = Q6_V_vmux_QVV(qdepth, Q6_V_vdelta_VV(xt, permctrl0), vzero);
      // Prep xt for next iteration
      xt = Q6_V_vror_VR(xt, nbytes);
      // Find max for each depth channel. xmax4 should have max val of each
      // depth channel in the first (depth+1)/2 elements of its 32 bytes.
      HVX_VectorPair xx = Q6_Wb_vshuffoe_VbVb(xmax4, xmax4);
      xmax4             = Q6_Vub_vmax_VubVub(Q6_V_lo_W(xx), Q6_V_hi_W(xx));
      for (int d = 2; d < depth; d <<= 1) {
        xx    = Q6_W_vshuff_VVR(xmax4, xmax4, d);
        xmax4 = Q6_Vub_vmax_VubVub(Q6_V_lo_W(xx), Q6_V_hi_W(xx));
      }
      // Group max of all 4 depth channels together and load them into xmax
      xmax4 = Q6_V_vrdelta_VV(xmax4, permctrl1);
      xmax  = Q6_V_vlalign_VVR(xmax4, xmax, -nbytes);
    }
    xmax = Q6_V_valign_VVR(vzero, xmax, (-block * depth) & 127);
    //
    // calculate exp and sum
    HVX_Vector p02, p13;
    HVX_VectorPair xdiff = Q6_Wh_vsub_VubVub(xmax, x);
    HVX_Vector x0        = Q6_Vw_vmpyie_VwVuh(vbeta, Q6_V_lo_W(xdiff));
    HVX_Vector x2        = Q6_Vw_vmpyio_VwVh(vbeta, Q6_V_lo_W(xdiff));
    HVX_Vector x1        = Q6_Vw_vmpyie_VwVuh(vbeta, Q6_V_hi_W(xdiff));
    HVX_Vector x3        = Q6_Vw_vmpyio_VwVh(vbeta, Q6_V_hi_W(xdiff));

    HVX_Vector xd0 = Q6_Vw_vasl_VwVw(x0, vbrshf);
    HVX_Vector xd1 = Q6_Vw_vasl_VwVw(x1, vbrshf);
    HVX_Vector xd2 = Q6_Vw_vasl_VwVw(x2, vbrshf);
    HVX_Vector xd3 = Q6_Vw_vasl_VwVw(x3, vbrshf);

    HVX_Vector xd02 = Q6_V_vand_VV(Q6_Vh_vshuffe_VhVh(xd2, xd0), const0x7FFE);
    HVX_Vector xd13 = Q6_V_vand_VV(Q6_Vh_vshuffe_VhVh(xd3, xd1), const0x7FFE);

    HVX_Vector xe0 = Q6_Vw_vasl_VwVw(x0, vbrshe);
    HVX_Vector xe1 = Q6_Vw_vasl_VwVw(x1, vbrshe);
    HVX_Vector xe2 = Q6_Vw_vasl_VwVw(x2, vbrshe);
    HVX_Vector xe3 = Q6_Vw_vasl_VwVw(x3, vbrshe);

    HVX_Vector xe02         = Q6_Vh_vadd_VhVh_sat(Q6_Vh_vsat_VwVw(xe2, xe0), const0x007F);
    HVX_Vector xe13         = Q6_Vh_vadd_VhVh_sat(Q6_Vh_vsat_VwVw(xe3, xe1), const0x007F);
    HVX_VectorPair xe13Xe02 = Q6_Wuh_vzxt_Vub(Q6_Vub_vsat_VhVh(xe13, xe02));
    xe02                    = Q6_V_lo_W(xe13Xe02);
    xe13                    = Q6_V_hi_W(xe13Xe02);

    p02 = Q6_Vh_vadd_VhVh(Q6_Vh_vmpy_VhVh_s1_rnd_sat(xd02, f3), f2);
    p02 = Q6_Vh_vadd_VhVh(Q6_Vh_vmpy_VhVh_s1_rnd_sat(xd02, p02), f1);
    p02 = Q6_Vh_vadd_VhVh(Q6_Vh_vmpy_VhVh_s1_rnd_sat(xd02, p02), f0);
    p02 = Q6_Vh_vadd_VhVh(p02, p02);

    p13 = Q6_Vh_vadd_VhVh(Q6_Vh_vmpy_VhVh_s1_rnd_sat(xd13, f3), f2);
    p13 = Q6_Vh_vadd_VhVh(Q6_Vh_vmpy_VhVh_s1_rnd_sat(xd13, p13), f1);
    p13 = Q6_Vh_vadd_VhVh(Q6_Vh_vmpy_VhVh_s1_rnd_sat(xd13, p13), f0);
    p13 = Q6_Vh_vadd_VhVh(p13, p13);

    HVX_VectorPair xf02 = Q6_W_vshuff_VVR(xe02, p02, -2);
    HVX_VectorPair xf13 = Q6_W_vshuff_VVR(xe13, p13, -2);

    HVX_VectorPair poly01 = Q6_W_vshuff_VVR(Q6_V_lo_W(xf13), Q6_V_lo_W(xf02), -4);
    HVX_VectorPair poly23 = Q6_W_vshuff_VVR(Q6_V_hi_W(xf13), Q6_V_hi_W(xf02), -4);

    HVX_Vector poly0 = Q6_Vw_vasl_VwR(Q6_V_lo_W(poly01), 7);
    HVX_Vector poly1 = Q6_Vw_vasl_VwR(Q6_V_hi_W(poly01), 7);

    HVX_Vector poly2 = Q6_Vw_vasl_VwR(Q6_V_lo_W(poly23), 7);
    HVX_Vector poly3 = Q6_Vw_vasl_VwR(Q6_V_hi_W(poly23), 7);

    tmpbuf[0] = poly0;
    tmpbuf[1] = poly1;
    tmpbuf[2] = poly2;
    tmpbuf[3] = poly3;

    HVX_VectorPred ql = Q6_Q_vsetq2_R(depth * sizeof(float));
    float *ptrInp     = (float *)&tmpbuf[0];
    float *ptrSum     = (float *)&tmpbuf[4];

    for (int k = 0; k < block; k++) {
      HVX_Vector y = vmemu(ptrInp);
      ptrInp += depth;
      y                 = Q6_V_vmux_QVV(ql, y, vzero);
      HVX_VectorPair yy = Q6_W_vshuff_VVR(y, y, 4);
      HVX_Vector vsumf  = Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(yy), Q6_V_hi_W(yy));

      for (int d = 2; d < depth; d <<= 1) {
        yy    = Q6_W_vshuff_VVR(vsumf, vsumf, 4 * d);
        vsumf = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(yy), Q6_V_hi_W(yy));
      }
      vsumf = Q6_Vsf_equals_Vqf32(vsumf);
      q6op_vstu_AV(ptrSum, vsumf);
      ptrSum += depth;
    }
    //
    // scale outputs
    HVX_Vector *iptr = tmpbuf;
    HVX_Vector *sptr = tmpbuf + 4;
    /* Process one vector at a time */
    for (int k = blockValid * depth; k > 0; k -= 32) {
      HVX_Vector sline = *sptr++;
      /* Calculate normalization factor */
      HVX_Vector normFactor = Q6_V_vand_VV(sline, signexpMask);
      normFactor            = Q6_Vw_vsub_VwVw(exp, normFactor);
      /* Normalize input */
      sline = Q6_Vqf32_vmpy_VsfVsf(sline, normFactor);
      /* Convert normalization factor to qf32 */
      normFactor = Q6_Vqf32_vadd_VsfVsf(normFactor, zeroVSf);
      /* Shift input range from [input_min, input_max] to [0, input_max -
       * input_min] */
      HVX_Vector inputShiftedVQf32 = Q6_Vqf32_vsub_Vqf32Vsf(sline, inputMinVF);
      /*
       * Scale shifted input range from [0, input_max - input_min] to [0,16.0)
       * in order to get corresponding coefficient indexes
       */
      HVX_Vector inputScaledVQf32 = Q6_Vqf32_vmpy_Vqf32Vqf32(inputShiftedVQf32, scaleV);
      /*
       * VLUT 16 requires integer indexes. Shift scaled input range from
       * [0,16.0) to [16.0,32.0) in order to convert float indexes to integer
       * values. Float values, represented in IEEE 754, in range [16.0,32.0]
       * have the same exponent, which means 4 MSB of mantissa carry information
       * about integer index.
       */
      inputScaledVQf32 = Q6_Vqf32_vadd_Vqf32Vsf(inputScaledVQf32, const160VSf);
      /* Convert back from qf32 to sf in order to extract integer index */
      HVX_Vector tmpV = Q6_Vsf_equals_Vqf32(inputScaledVQf32);
      /* Only 4 MSB bits of mantissa represent segment index */
      HVX_Vector idx1V = Q6_Vuw_vlsr_VuwR(tmpV, 19);
      idx1V            = Q6_V_vand_VV(idx1V, maskIdx1V);
      idx1V            = Q6_V_vor_VV(idx1V, maskIdx2V);
      HVX_Vector idx2V = Q6_Vw_vasl_VwR(idx1V, 16);

      /* Obtain the polynomial coefficients from lookup table */
      HVX_VectorPair c0CoeffVp = Q6_Wh_vlut16_VbVhR(idx1V, Q6_V_lo_W(c0CoeffDv), 1);
      c0CoeffVp                = Q6_Wh_vlut16or_WhVbVhR(c0CoeffVp, idx2V, Q6_V_hi_W(c0CoeffDv), 1);
      HVX_VectorPair c1CoeffVp = Q6_Wh_vlut16_VbVhR(idx1V, Q6_V_lo_W(c1CoeffDv), 1);
      c1CoeffVp                = Q6_Wh_vlut16or_WhVbVhR(c1CoeffVp, idx2V, Q6_V_hi_W(c1CoeffDv), 1);
      HVX_VectorPair c2CoeffVp = Q6_Wh_vlut16_VbVhR(idx1V, Q6_V_lo_W(c2CoeffDv), 1);
      c2CoeffVp                = Q6_Wh_vlut16or_WhVbVhR(c2CoeffVp, idx2V, Q6_V_hi_W(c2CoeffDv), 1);
      HVX_VectorPair c3CoeffVp = Q6_Wh_vlut16_VbVhR(idx1V, Q6_V_lo_W(c3CoeffDv), 1);
      c3CoeffVp                = Q6_Wh_vlut16or_WhVbVhR(c3CoeffVp, idx2V, Q6_V_hi_W(c3CoeffDv), 1);
      /* Perform evaluation of polynomial using Horner's method */
      HVX_Vector vRecip;
      vRecip = Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(c3CoeffVp), sline);
      vRecip = Q6_Vqf32_vadd_Vqf32Vqf32(vRecip, Q6_V_lo_W(c2CoeffVp));
      vRecip = Q6_Vqf32_vmpy_Vqf32Vqf32(vRecip, sline);
      vRecip = Q6_Vqf32_vadd_Vqf32Vqf32(vRecip, Q6_V_lo_W(c1CoeffVp));
      vRecip = Q6_Vqf32_vmpy_Vqf32Vqf32(vRecip, sline);
      vRecip = Q6_Vqf32_vadd_Vqf32Vqf32(vRecip, Q6_V_lo_W(c0CoeffVp));
      /* Multiply result by same normalization factor applied to input earlier
       */
      vRecip = Q6_Vqf32_vmpy_Vqf32Vqf32(vRecip, normFactor);

      /* Convert from qf32 to sf */
      vRecip = Q6_Vsf_equals_Vqf32(vRecip);
      /* compute/store output */
      HVX_Vector sout = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(*iptr++, vRecip));
      if (k > 32) {
        q6op_vstu_AV(pout, sout);
        pout += 32;
      } else {
        q6op_vstu_variable_ARV(pout, 4 * k, sout);
        pout += k;
      }
    }
  }
}

// ====================================================
// FUNCTION: (2**-~x)     y(0) = 0.5,  y(0.5) = 0.7071, y(1) = 1
// Order:2; continuity: True; Ends forced: True
// Mode: unsigned;   Result fractional bits: 15
// Peak Error: 6.9827e-05  Rms Error: 2.4918e-05   Mean Error: 1.2308e-07
//      32769   22652    4300
//      32876   21816    5119
//      33371   19857    6087
//      34675   16404    7230
static inline HVX_Vector softmaxU8Exp2Frac(HVX_Vector vx) {
  // input is 0..0xffff representing 0.0  .. 1.0
  vx = Q6_V_vnot_V(vx);  // poly built for this
  HVX_Vector p;
  p = Q6_Vh_vlut4_VuhPh(vx, 0x1C3E17C713FF10CCull);
  p = Q6_Vh_vmpa_VhVhVuhPuh_sat(p, vx, 0x40144D915538587Cull);
  p = Q6_Vh_vmpa_VhVhVuhPuh_sat(p, vx, 0x8773825B806C8001ull);
  return p;  // signed result, 15 fractional bits
}

// Find normalized 16 bit reciprocal for 32 bit sum
// Return reciprocal and shift amount to apply to product
static inline HVX_VectorPair softmaxFixedVecRecip(HVX_Vector vSum) {
  HVX_Vector vMulback;
  HVX_Vector verr;
  HVX_Vector vcorr;
  HVX_Vector vcorrShift;
  HVX_Vector vleadingZeroes = Q6_Vuw_vcl0_Vuw(vSum);
  HVX_Vector vOneShamt      = Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(15), vleadingZeroes);
  HVX_Vector vOne           = Q6_Vw_vasl_VwVw(Q6_V_vsplat_R(0x10000), vOneShamt);
  HVX_Vector vRecip         = Q6_V_vsplat_R(0x8000);
  for (int i = 0; i < 5; i++) {
    vMulback   = Q6_Vw_vmpye_VwVuh(vSum, vRecip);  // >>16
    verr       = Q6_Vw_vsub_VwVw(vOne, vMulback);
    vcorr      = Q6_Vw_vmpye_VwVuh(verr, vRecip);  // >>16
    vcorrShift = Q6_Vw_vasr_VwVw(vcorr, vOneShamt);
    vRecip     = Q6_Vw_vadd_VwVw(vRecip, vcorrShift);
  }
  return Q6_W_vcombine_VV(vRecip, vOneShamt);
}

void softmaxApproxCrouton(uint8_t **outtab,
                          uint8_t **intab,
                          uint32_t fixedScale,
                          uint32_t fracBits,
                          uint32_t sumscale,
                          int depth,
                          int vecOffset,
                          uint16_t *intermed) {
  HVX_Vector **vouttab      = (HVX_Vector **)outtab;
  HVX_Vector const **vintab = (HVX_Vector const **)intab;
  int depthBlocks           = ((unsigned int)depth + 0x1f) >> 5;
  int depthLeftovers        = depth & 0x1f;
  HVX_Vector vSelect        = Q6_V_vand_QR(Q6_Q_vsetq2_R(4 * depthLeftovers), -1);
  // We have the right number of FF and 00, but not in the right place.
  // Pack them together twice
  vSelect = Q6_Vb_vpacke_VhVh(vSelect, vSelect);
  vSelect = Q6_Vb_vpacke_VhVh(vSelect, vSelect);

  HVX_Vector vmax = Q6_V_vzero();
  // Find maxima elementwise
  HVX_Vector const *pvin = vintab[0] + vecOffset;
  for (int d = 1; d < depthBlocks; d++) {
    vmax = Q6_Vub_vmax_VubVub(vmax, *pvin);
    pvin = vintab[d] + vecOffset;
  }
  vmax = Q6_Vub_vmax_VubVub(vmax, Q6_V_vand_VV(*pvin, vSelect));

  // Reduce maxima and splat
  // Implementation note: vshuffeb/eh might be good for two of these rounds
  for (int j = 0; j <= 4; j++) {
    HVX_VectorPair wTmp = Q6_W_vshuff_VVR(vmax, vmax, 1 << j);
    vmax                = Q6_Vub_vmax_VubVub(Q6_V_hi_W(wTmp), Q6_V_lo_W(wTmp));
  }
  // Sum of exponentials
  uint32_t rScales     = Q6_R_vsplatb_R(fixedScale);
  HVX_Vector dmask0h   = Q6_V_vand_QR(Q6_Q_vsetq2_R(4 * ((depthLeftovers + 1) & ~1)), -1);
  HVX_Vector dmask1h   = Q6_V_vand_QR(Q6_Q_vsetq2_R(4 * ((depthLeftovers + 0) & ~1)), -1);
  dmask0h              = Q6_Vb_vpacke_VhVh(dmask0h, dmask0h);
  dmask0h              = Q6_Vb_vpacke_VhVh(dmask0h, dmask0h);
  dmask1h              = Q6_Vb_vpacke_VhVh(dmask1h, dmask1h);
  dmask1h              = Q6_Vb_vpacke_VhVh(dmask1h, dmask1h);
  HVX_Vector vMaxshift = Q6_Vh_vsplat_R(15);
  HVX_Vector *optr     = (HVX_Vector *)intermed;
  HVX_Vector vacc      = Q6_V_vzero();

  HVX_Vector vin  = vintab[0][vecOffset];
  HVX_Vector vals = Q6_Vb_vsub_VbVb(vmax, vin);

  HVX_VectorPair wScaled = Q6_Wuh_vmpy_VubRub(vals, rScales);
  HVX_Vector vScaledLo   = Q6_V_lo_W(wScaled);
  HVX_Vector vscaledHi   = Q6_V_hi_W(wScaled);
  HVX_Vector vfrac0      = Q6_Vh_vasl_VhR(vScaledLo, 16 - fracBits);
  HVX_Vector vFrac1      = Q6_Vh_vasl_VhR(vscaledHi, 16 - fracBits);

  for (int d = 1; d < depthBlocks; d++) {
    HVX_Vector vshamtHi = Q6_Vuh_vmin_VuhVuh(vMaxshift, Q6_Vuh_vlsr_VuhR(vscaledHi, fracBits));
    HVX_Vector vshamtLo = Q6_Vuh_vmin_VuhVuh(vMaxshift, Q6_Vuh_vlsr_VuhR(vScaledLo, fracBits));

    HVX_Vector vpow0 = softmaxU8Exp2Frac(vfrac0);
    HVX_Vector vpow1 = softmaxU8Exp2Frac(vFrac1);

    vin  = vintab[d][vecOffset];
    vals = Q6_Vb_vsub_VbVb(vmax, vin);

    HVX_VectorPair wScaled = Q6_Wuh_vmpy_VubRub(vals, rScales);
    vScaledLo              = Q6_V_lo_W(wScaled);
    vscaledHi              = Q6_V_hi_W(wScaled);
    vfrac0                 = Q6_Vh_vasl_VhR(vScaledLo, 16 - fracBits);
    vFrac1                 = Q6_Vh_vasl_VhR(vscaledHi, 16 - fracBits);

    HVX_Vector vExp0 = Q6_Vh_vlsr_VhVh(vpow0, vshamtLo);
    HVX_Vector vExp1 = Q6_Vh_vlsr_VhVh(vpow1, vshamtHi);
    *optr++          = vExp0;
    *optr++          = vExp1;

    // Accumulate 2x16 16b exps into 2x8 32b exps
    HVX_VectorPair wSum = Q6_Ww_vadd_VuhVuh(vExp0, vExp1);
    // Accumulate 2x8 32b exps into 1x8 32b exp, add to accumulator
    vacc = Q6_Vw_vadd_VwVw(vacc, Q6_Vw_vadd_VwVw(Q6_V_hi_W(wSum), Q6_V_lo_W(wSum)));
  }

  HVX_Vector vshamtLo = Q6_Vuh_vmin_VuhVuh(vMaxshift, Q6_Vuh_vlsr_VuhR(vScaledLo, fracBits));
  HVX_Vector vshamtHi = Q6_Vuh_vmin_VuhVuh(vMaxshift, Q6_Vuh_vlsr_VuhR(vscaledHi, fracBits));

  HVX_Vector vpow0    = softmaxU8Exp2Frac(vfrac0);
  HVX_Vector vpow1    = softmaxU8Exp2Frac(vFrac1);
  HVX_Vector vExp0    = Q6_Vh_vlsr_VhVh(vpow0, vshamtLo);
  HVX_Vector vExp1    = Q6_Vh_vlsr_VhVh(vpow1, vshamtHi);
  *optr++             = vExp0;
  *optr++             = vExp1;
  vExp0               = Q6_V_vand_VV(vExp0, dmask0h);
  vExp0               = Q6_V_vand_VV(vExp1, dmask1h);
  HVX_VectorPair wSum = Q6_Ww_vadd_VuhVuh(vExp0, vExp1);
  vacc                = Q6_Vw_vadd_VwVw(vacc, Q6_Vw_vadd_VwVw(Q6_V_hi_W(wSum), Q6_V_lo_W(wSum)));
  // Reduce Sums
  // Implementation Note: if we have lots of accumulators from unrolling this,
  // We will want to shrink here so we can do fewer reciprocals below.
  for (int i = 2; i <= 4; i++) {
    HVX_VectorPair wTmp = Q6_W_vdeal_VVR(vacc, vacc, 1 << i);
    vacc                = Q6_Vw_vadd_VwVw(Q6_V_hi_W(wTmp), Q6_V_lo_W(wTmp));
  }
  // Scale the sums by sumscale
  HVX_Vector vsumscale = Q6_V_vsplat_R(sumscale);
  HVX_Vector vaccSc    = Q6_Vuw_vlsr_VuwR(Q6_Vw_vmpyie_VwVuh(vacc, vsumscale), 8);

  // OK, we have 4 sums, but they're repeated 8 times.  Or at least, they should
  // be... Do reciprocals
  HVX_VectorPair wrecipShift = softmaxFixedVecRecip(vaccSc);
  HVX_Vector vRecip          = Q6_V_hi_W(wrecipShift);
  HVX_Vector vShift          = Q6_V_lo_W(wrecipShift);
  // OK, these are 2x 32b values, but they're both really 16b values
  // Duplicate them
  // We're going to shift right by 16, and want 8 bits left ... so make some
  // adjustments...
  vShift = Q6_Vh_vadd_VhVh(vShift, Q6_V_vsplat_R(7));
  vRecip = Q6_Vh_vshuffe_VhVh(vRecip, vRecip);
  vShift = Q6_Vh_vshuffe_VhVh(vShift, vShift);
  // Apply to exponentials
  HVX_Vector *iptr = (HVX_Vector *)intermed;

  HVX_Vector vexp0        = *iptr++;
  HVX_Vector vexp1        = *iptr++;
  HVX_VectorPair wprodsLo = Q6_Wuw_vmpy_VuhVuh(vexp0, vRecip);
  HVX_VectorPair wprodsHi = Q6_Wuw_vmpy_VuhVuh(vexp1, vRecip);
  HVX_Vector vprodsLo     = Q6_Vh_vshuffo_VhVh(Q6_V_hi_W(wprodsLo), Q6_V_lo_W(wprodsLo));
  HVX_Vector vprodsHi     = Q6_Vh_vshuffo_VhVh(Q6_V_hi_W(wprodsHi), Q6_V_lo_W(wprodsHi));

  for (int d = 0; d < depthBlocks - 1; d++) {
    HVX_Vector vprodsHiSh = Q6_Vh_vlsr_VhVh(vprodsHi, vShift);  // needs to be bidirectional...
    HVX_Vector vprodsLoSh = Q6_Vh_vlsr_VhVh(vprodsLo, vShift);

    HVX_Vector vexp0        = *iptr++;
    HVX_Vector vexp1        = *iptr++;
    HVX_VectorPair wprodsLo = Q6_Wuw_vmpy_VuhVuh(vexp0, vRecip);
    HVX_VectorPair wprodsHi = Q6_Wuw_vmpy_VuhVuh(vexp1, vRecip);
    vprodsLo                = Q6_Vh_vshuffo_VhVh(Q6_V_hi_W(wprodsLo), Q6_V_lo_W(wprodsLo));
    vprodsHi                = Q6_Vh_vshuffo_VhVh(Q6_V_hi_W(wprodsHi), Q6_V_lo_W(wprodsHi));

    HVX_Vector vout       = Q6_Vub_vasr_VuhVuhR_rnd_sat(vprodsHiSh, vprodsLoSh, 1);
    vouttab[d][vecOffset] = vout;
  }
  HVX_Vector vprodsLoSh = Q6_Vh_vlsr_VhVh(vprodsLo, vShift);
  HVX_Vector vprodsHiSh = Q6_Vh_vlsr_VhVh(vprodsHi, vShift);  // needs to be bidirectional...
  HVX_Vector vout       = Q6_Vub_vasr_VuhVuhR_rnd_sat(vprodsHiSh, vprodsLoSh, 1);
  vouttab[depthBlocks - 1][vecOffset] = Q6_V_vand_VV(vout, vSelect);
}

template <typename T_OutType, typename T_InTtype>
int softmaxImpl(T_OutType &out, const T_InTtype &in, const float beta) {
  debuglog("reference softmax (%s)", __PRETTY_FUNCTION__);
  out.set_dims(in);
  auto [b_in, h_in, w_in, d_in] = in.dims();
  for (Idx b = 0; b < b_in; b++) {
    for (Idx h = 0; h < h_in; h++) {
      for (Idx w = 0; w < w_in; w++) {
        float max = in(b, h, w, 0);
        for (Idx d = 0; d < d_in; d++) {
          float inval = in(b, h, w, d);
          max         = fmaxf(inval, max);
        }
        float sum = 0;
        for (Idx d = 0; d < d_in; d++) {
          float inval = in(b, h, w, d);
          sum += (out(b, h, w, d) = expf(beta * (inval - max)));
        }
        float sum_recip = 1.0f / sum;
        for (Idx d = 0; d < d_in; d++) {
          float outval    = out(b, h, w, d);
          out(b, h, w, d) = outval * sum_recip;
        }
      }
    }
  }
  return GraphStatus::Success;
}

template <typename T_Ttype>
int softmaxWithbetaWrapper(T_Ttype &out, const T_Ttype &in, const Tensor &beta) {
  return softmaxImpl<T_Ttype>(out, in, beta(0, 0, 0, 0));
}

template <typename T_OutType, typename T_InTtype>
int softmaxFastImpl(T_OutType &out, const T_InTtype &in, const float beta) {
  debuglog("fast softmax (%s)", __PRETTY_FUNCTION__);
  out.set_dims(in);
  auto [b_in, h_in, w_in, d_in] = in.dims();

  using inT   = typename T_InTtype::element_type;
  using outT  = typename T_OutType::element_type;
  float scale = in.interface_scale() * beta;

  if constexpr (std::is_same<T_InTtype, QuantUint8Tensor_TCM>::value ||
                std::is_same<T_InTtype, QuantUint8Tensor>::value) {
    if (d_in > 2 && d_in <= 32 && (h_in > 1 || w_in > 1)) {
      for (Idx b = 0; b < b_in; b++) {
        const inT *pin = &in.get_raw(b, 0, 0, 0);
        outT *pout     = &out.get_raw(b, 0, 0, 0);
        softmaxApproxShortd(pout, pin, scale, d_in, h_in * w_in);
      }
      return GraphStatus::Success;
    }
  }

  for (Idx b = 0; b < b_in; b++) {
    for (Idx h = 0; h < h_in; h++) {
      for (Idx w = 0; w < w_in; w++) {
        const inT *pin = &in.get_raw(b, h, w, 0);
        outT *pout     = &out.get_raw(b, h, w, 0);
        if constexpr (std::is_same<T_InTtype, QuantUint8Tensor_TCM>::value ||
                      std::is_same<T_InTtype, QuantUint8Tensor>::value) {
          softmaxApprox(pout, pin, scale, d_in);
        } else if constexpr (std::is_same<T_InTtype, QuantUint16Tensor_TCM>::value ||
                             std::is_same<T_InTtype, QuantUint16Tensor>::value) {
          softmaxHApprox(pout, pin, scale, d_in);
        }
      }
    }
  }
  return GraphStatus::Success;
}

template <typename T_OutType, typename T_InTtype>
int softmaxCroutonImpl(T_OutType &out, const T_InTtype &in, const Tensor &beta) {
  debuglog("FAST softmax approximation");
  out.set_dims(in);
  auto [b_in, h_in, w_in, d_in] = in.dims();

  const float betaval  = beta(0, 0, 0, 0);
  const float inScale  = in.get_interface_scale();   // input stepsize
  const float outScale = out.get_interface_scale();  // output stepsize

  const float inscale     = inScale * 1.4426950408889634f;  // Total scale factor for input values
  const uint32_t sumscale = (1 << 16) * outScale;           // How to apply output scale to sum

  const float scalep  = inscale * betaval;
  uint32_t fixedScale = uint32_t(flt_getfrac(scalep) >> 16);
  uint32_t fracBits   = uint32_t(8 - flt_getexp(scalep));
  if (fracBits > 14) {
    fixedScale >>= fracBits - 14;
    fracBits = 14;
  }

  uint8_t **iAdrTab = (uint8_t **)in.blocktab_ptr();
  uint8_t **oAdrTab = (uint8_t **)out.blocktab_ptr();
  int tOffCol       = (d_in + 31) >> 5;

  tileExt::tile_buffers<8> tile_bufs;  // support depth up to 8*(8*32) = 2048
  uint16_t *intermed = (uint16_t *)tile_bufs.buf(0);

  for (size_t b = 0; b < b_in; b++) {
    for (size_t h = 0; h < h_in; h += 8) {
      int ht = min_i32(h_in - h, 8);
      for (size_t w = 0; w < w_in; w += 8) {
        int wt          = min_i32(w_in - w, 8);
        int step        = wt > 4 ? 1 : 2;
        uint8_t **ppin  = iAdrTab;
        uint8_t **ppout = oAdrTab;
        for (int v = 0; v < 2 * ht; v += step) {
          softmaxApproxCrouton(ppout, ppin, fixedScale, fracBits, sumscale, d_in, v, intermed);
        }
        iAdrTab += tOffCol;
        oAdrTab += tOffCol;
      }
    }
  }

  return GraphStatus::Success;
}

template <typename T_OutType, typename T_InTtype>
int softmaxD2Impl(T_OutType &out, const T_InTtype &in, const float beta) {
  debuglog("fast softmax (%s)", __PRETTY_FUNCTION__);
  out.set_dims(in);
  auto [b_in, h_in, w_in, d_in] = in.dims();
  float scale                   = in.interface_scale() * beta;

  void softmaxHD2Approx(float *pout, const uint16_t *pin, float scale, int32_t length);
  softmaxHD2Approx(
      out.get_raw_addr(0, 0, 0, 0), in.get_raw_addr(0, 0, 0, 0), scale, b_in * h_in * w_in * d_in);

  return GraphStatus::Success;
}

template <typename T_OutType, typename T_InTtype>
int softmaxWithbetaFastWrapper(T_OutType &out, const T_InTtype &in, const Tensor &beta) {
  if constexpr (std::is_same<T_InTtype, QuantUint16Tensor_TCM>::value ||
                std::is_same<T_InTtype, QuantUint16Tensor>::value) {
    if (in.dim(3) == 2) return softmaxD2Impl(out, in, beta(0, 0, 0, 0));
  }
  return softmaxFastImpl<T_OutType, T_InTtype>(out, in, beta(0, 0, 0, 0));
}

template <typename T_OutType, typename T_InTtype>
int softmaxD2WithbetaFastWrapper(T_OutType &out,
                                 const T_InTtype &in,
                                 const Tensor &beta,
                                 const TensorContiguous<Tdefs::QuantUint8> &lookupTable) {
  out.set_dims(in);

  softmaxBD2Approx((uint8_t *)&out.get_raw(0, 0, 0, 0),
                   (const uint8_t *)&in.get_raw(0, 0, 0, 0),
                   (const uint8_t *)&lookupTable.get_raw(0, 0, 0, 0),
                   out.total_storage_elements());

  return GraphStatus::Success;
}

GraphStatus softmaxD2TablegenImpl(TensorContiguous<Tdefs::QuantUint8> &out,
                                  const Tensor &inStepsize,
                                  const Tensor &beta) {
  const float inStepsizeVal = inStepsize(0, 0, 0, 0);
  const float betaVal       = beta(0, 0, 0, 0);
  const float k             = inStepsizeVal * betaVal;
  out.get_raw(0, 0, 0, 0)   = 128;
  int vali                  = 0;
  for (int i = 1; i < 256; i++) {
    if (vali < 255) {
      vali = roundf_i32(255.0f / (1.0f + exp(-k * (float)i)));
    }
    out.get_raw(0, 0, 0, flatToVlut(i)) = (uint8_t)vali;
  }
  return GraphStatus::Success;
}

static float softmaxCost(const Op *op) {
  auto [b, h, w, d] = op->get_output(0)->dims();

  float cost = float(b * h * w * d);
  debuglog("Calculating cost=%f", cost);
  return cost;
}

/* At the bottom of the op file, call END_PKG_OP_DEFINITION(<name>),
   where <name> is as BEGIN_PKG_OP_DEFINITION
*/
END_PKG_OP_DEFINITION(PKG_Softmax);