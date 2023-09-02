//==============================================================================
//
// Copyright (c) 2018 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef AFUNCS_H
#define AFUNCS_H 1

#include <algorithm>
#include <cmath>
#include "dtype.h"
#ifndef __hexagon__
#include <string.h> // for memcpy etc
#endif
// #include "asm_define.h"

struct tile_data {
    uint8_t **addr;
    uint32_t offset_t_col;
    uint32_t offset_t_row;
    uint32_t width;
    uint32_t height;
    uint32_t depth;
};

#define TILEDATA(adrtab, next_tab_col, next_tab_row, h, w, d)                                                          \
    {                                                                                                                  \
        .addr = (uint8_t **)(adrtab), .offset_t_col = static_cast<uint32_t>(next_tab_col),                             \
        .offset_t_row = static_cast<uint32_t>(next_tab_row), .width = static_cast<uint32_t>(w),                        \
        .height = static_cast<uint32_t>(h), .depth = static_cast<uint32_t>(d)                                          \
    }

/*=======================================*/
/* Auxiliary functions                   */
/*=======================================*/
#if defined(__hexagon__)
inline int32_t max_i32(int32_t a, int32_t b)
{
    return Q6_R_max_RR(a, b);
}
inline int32_t min_i32(int32_t a, int32_t b)
{
    return Q6_R_min_RR(a, b);
}
inline uint32_t max_u32(uint32_t a, uint32_t b)
{
    return Q6_R_maxu_RR(a, b);
}
inline uint32_t min_u32(uint32_t a, uint32_t b)
{
    return Q6_R_minu_RR(a, b);
}
#else
inline int32_t max_i32(int32_t a, int32_t b)
{
    return (a < b) ? b : a;
}
inline int32_t min_i32(int32_t a, int32_t b)
{
    return (a < b) ? a : b;
}
inline uint32_t max_u32(uint32_t a, uint32_t b)
{
    return (a < b) ? b : a;
}
inline uint32_t min_u32(uint32_t a, uint32_t b)
{
    return (a < b) ? a : b;
}
#endif

inline __attribute__((unused, always_inline)) int64_t roundf_i64(float val)
{
    // add 0.5 (with same sign as val) and then conversion to int truncates toward 0.
    // values exactly halfway will round away from 0 (like roundf).

    return (int64_t)(val + copysignf(0.5f, val));
}

inline __attribute__((unused, always_inline)) NN_INT32_T roundf_i32(float val)
{
    // add 0.5 (with same sign as val) and then conversion to int truncates toward 0.
    // values exactly halfway will round away from 0 (like roundf).

    return (int)(val + copysignf(0.5f, val));
}
// same thing for rounding to unsigned range; -ve inputs will give 0.
//
inline __attribute__((unused, always_inline)) uint32_t roundf_u32(float val)
{
    // add 0.5f and then convert to uint (trunc towards 0; -ve values are clipped to 0).
#ifdef __hexagon__
    // use intrinsic since conv of -ve float to unsigned is 'undefined behaviour' in C.
    return Q6_R_convert_sf2uw_R_chop(val + 0.5f);
#else
    return (val < 0.5f) ? 0 : (uint32_t)(val + 0.5f);
#endif
}

inline __attribute__((unused, always_inline)) NN_INT32_T saturate_u8(NN_INT32_T val)
{
#ifdef __hexagon__
    return Q6_R_satub_R(val);
#else
    return (val < 0) ? 0 : ((val > 255) ? 255 : val);
#endif
}

inline __attribute__((unused, always_inline)) NN_INT32_T saturate_u16(NN_INT32_T val)
{
#ifdef __hexagon__
    return Q6_R_satuh_R(val);
#else
    return (val < 0) ? 0 : ((val > 65535) ? 65535 : val);
#endif
}

static inline __attribute__((unused, always_inline)) NN_INT32_T saturate_i16(NN_INT32_T val)
{
#ifdef __hexagon__
    return Q6_R_sath_R(val);
#else
    return (val < -32768) ? -32768 : ((val > 32767) ? 32767 : val);
#endif
}

/**
 * @brief low-cost frexpf (but only the exponent result);
 * Generates only a few instructions on hexagon.
 *
 * Input must not be inf,nan, zero, or denormal.
 *
 * returns:
 *        -1 if abs(x) is in range 0.25 ... 0.249999
 *         0 if abs(x) is in range 0.5 ... 0.99999
 *         1 if abs(x) is in range 1.0 .. 1.9999
 *  etc
 *
 *  If the value -126 is returned, x is a zero or denormal;
 *  129 is returned for inf or NaN. for other cases the value is the same
 *  as what frexpf  (in math.h) generates for the exponent.
 */
inline __attribute__((unused, always_inline)) constexpr int flt_getexp(float x)
{
    union {
        float f;
        int32_t u32;
    } const uu = {x};
    return ((uu.u32 >> 23) & 0xFF) - 126;
}
/**
 * @brief low-cost frexpf (but only the 'fraction' result);
 * Generates only a few instructions on hexagon.
 *
 * Input must not be inf,nan, zero, or denormal.
 *
 * returns a value in the range [0.5, 1.0)  (or in (-1.0,-0.5] when x < 0)
 * such that x = flt_getmant(x) * powf2(2.0, flt_getexp(x))
 *
 */
inline __attribute__((unused, always_inline)) constexpr float flt_getmant(float x)
{
    union {
        float f;
        int32_t u32;
    } uu = {x};
    uu.u32 = (uu.u32 & 0x807fffffu) | (126 << 23); // force exponent = 126
    return uu.f;
}

/**
 * @brief returns the mantissa of x, as a 24-bit number
 * in the range 0x800000 .. 0xFFFFFF
 *
 * Input must not be inf,nan, zero, or denormal.
 *
 * Sign is discarded. same as powf(2,24) * flt_getmant(fabsf(x)).
 */
inline __attribute__((unused, always_inline)) constexpr int32_t flt_getfrac(float x)
{
    union {
        float f;
        int32_t u32;
    } const uu = {x};
    int32_t const m = (uu.u32 & 0x007fffffu) | (1 << 23);
    return m;
}

//
// This 'normalizes' a float to 0.5 .. 0.9999  (sign is retained)
// Same result as the return value from frexpf, without using a function call
// Results are not valid if x is 0, denormal, or inf/nan
//
inline __attribute__((unused, always_inline)) float flt_getfrac_norm(float x)
{
    union {
        float f;
        int32_t u32;
    } uu = {x};
    uu.u32 = (uu.u32 & 0x807fffffu) | (126 << 23); // force exponent = 126
    return uu.f;
}
/**
 * @brief low-cost 2.0*n for integer n.
 * Same as powf(2.0f, iexpo) without a function call;
 *
 * Constraint: iexpo must be in range -126..127
 */
inline __attribute__((unused, always_inline)) constexpr float flt_power2(int iexpo)
{
    int const a = (iexpo + 127) & 0xFF;
    union {
        int32_t u32;
        float f;
    } const uu = {a << 23};
    return uu.f;
}
/**
 * @brief low-cost ldexpf
 * Same as ldexpf(val, iexpo) without a function call;
 *
 * Constraint: iexpo must be in range -126..127
 */
inline __attribute__((unused, always_inline)) constexpr float flt_ldexp(float val, int iexpo)
{
    return val * flt_power2(iexpo);
}

/**
 * @brief returns the exponent and mantissa of x, as a n-bit number
 *
 * Constraint: iexpo must be in range -126..127
 * Input must not be negative, inf,nan, zero, or denormal.
 */
template <int MBITS> inline constexpr std::pair<int32_t, uint32_t> get_scalefactor(float x)
{
    union {
        float f;
        int32_t u32;
    } const uu = {x};

    uint32_t inval = uu.u32;
    uint32_t const mask = (1 << MBITS) - 1;
    inval = (inval + (1 << (24 - MBITS - 1))) >> (24 - MBITS); // possibly overflows into exponent, but that's OK.
    uint32_t const m = ((inval & mask) | (1 << (MBITS - 1)));
    int32_t const e = int32_t((inval >> (MBITS - 1)) & 0xFF) - 126;
    return {e, m};
}

/**
 * @brief returns the parameters for scaling.
 * bit 31-24: left shift amount
 * bit 23-16: right shift amout
 * bit 15- 0: scale factor
 *
 * Input must not be inf,nan, zero, negative or denormal.
 *
 */
inline __attribute__((unused, always_inline)) constexpr uint32_t get_scaling_params(float x, int max_sl, int max_sr)
{
    auto [e, m] = get_scalefactor<15>(x);
    int sl = (e > 0) ? e : 0;
    int sr = (e > 0) ? 0 : -e;

    if (sl == 0 && sr > 0) {
        sl = min_i32(max_sl, max_i32(max_sr - sr, 0));
        sr = sr + sl;
    }
    return ((sl & 0x0FF) << 24) | ((sr & 0x0FF) << 16) | m;
}

/**
 * @brief given a scale in float and a recip shift amount
 *  return a quantized scale multiplier and change recip shamt inplace
 *
 */
inline uint32_t get_quantized_multipiler(const float scale_f, int &recip_shamt)
{
    recip_shamt = (scale_f <= 1.0f) ? 0 : flt_getexp(scale_f);
    uint32_t scale = roundf(flt_ldexp(scale_f, (31 - recip_shamt)));
    scale = (scale < 0x7fffffffu) ? scale : 0x7FFFFFFFu;
    return scale;
}

/**
 * @brief given a scale in float and a recip shift amount
 *  return a quantized scale multiplier and change recip shamt inplace
 *
 */
//Now with corrected spelling
inline uint32_t get_quantized_multiplier(const float scale_f, int &recip_shamt)
{
    return get_quantized_multipiler(scale_f, recip_shamt);
}
#endif /*AFUNCS_H*/
