//==============================================================================
//
// Copyright (c) 2023 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

// Compiler builtin intrinsic functions should be specified in this file

#ifndef BUILTIN_INTRINSICS_H_
#define BUILTIN_INTRINSICS_H_

// Branch prediction
#if defined(__clang__)

#define HEX_LIKELY(x)   __builtin_expect(!!(x), 1)
#define HEX_UNLIKELY(x) __builtin_expect(!!(x), 0)

#define HEX_ASSUME      __builtin_assume
#define HEX_UNREACHABLE __builtin_unreachable

#elif defined(_MSC_VER)

#define HEX_LIKELY(x)   (x)
#define HEX_UNLIKELY(x) (x)

#define HEX_ASSUME        __assume
#define HEX_UNREACHABLE() __assume(0)

#endif // defined(__clang__)

// Overflow detection
#if defined(__clang__)

#define HEX_ADD_OVERFLOW __builtin_add_overflow
#define HEX_MUL_OVERFLOW __builtin_mul_overflow

#elif defined(_MSC_VER)

#include <limits>

template <typename _T> static inline bool HEX_ADD_OVERFLOW(_T a, _T b, _T *out)
{
    *out = a + b;
    return ((b > 0) && (a > std::numeric_limits<_T>::max() - b)) ||
           ((b < 0) && (a < std::numeric_limits<_T>::min() - b));
}

template <typename _T> static inline bool HEX_MUL_OVERFLOW(_T a, _T b, _T *out)
{
    *out = a * b;
    return ((b > 0) && (a > std::numeric_limits<_T>::max() / b || a < std::numeric_limits<_T>::min() / b)) ||
           ((b < 0) && (a > std::numeric_limits<_T>::min() / b || a < std::numeric_limits<_T>::max() / b));
}

#endif // __clang__

// Count bits
#if defined(__clang__)

#define HEX_COUNT_LEADING_ZERO     __builtin_clz
#define HEX_COUNT_LEADING_ZERO_UL  __builtin_clzl
#define HEX_COUNT_LEADING_ZERO_ULL __builtin_clzll

#define HEX_COUNT_TRAILING_ZERO     __builtin_ctz
#define HEX_COUNT_TRAILING_ZERO_UL  __builtin_ctzl
#define HEX_COUNT_TRAILING_ZERO_ULL __builtin_ctzll

#define HEX_COUNT_ONE_BIT     __builtin_popcount
#define HEX_COUNT_ONE_BIT_UL  __builtin_popcountl
#define HEX_COUNT_ONE_BIT_ULL __builtin_popcountll

#elif defined(_MSC_VER)

#include <intrin.h>

// Returns the number of leading 0-bits in x, starting at the most significant
// bit position. If x is 0, the result is undefined.
static inline int HEX_COUNT_LEADING_ZERO_ULL(unsigned long long x)
{
    unsigned long where;
    if (_BitScanReverse64(&where, x)) return static_cast<int>(63 - where);
    return 64; // Undefined behavior
}

static inline int HEX_COUNT_LEADING_ZERO(unsigned int x)
{
    unsigned long where;
    if (_BitScanReverse(&where, x)) return static_cast<int>(31 - where);
    return 32; // Undefined Behavior.
}

static inline int HEX_COUNT_LEADING_ZERO_UL(unsigned long x)
{
    return sizeof(x) == 8 ? HEX_COUNT_LEADING_ZERO_ULL(x) : HEX_COUNT_LEADING_ZERO(static_cast<unsigned int>(x));
}

// Returns the number of trailing 0-bits in x, starting at the least significant
// bit position. If x is 0, the result is undefined.
static inline int HEX_COUNT_TRAILING_ZERO_ULL(unsigned long long x)
{
    unsigned long where;
    if (_BitScanForward64(&where, x)) return static_cast<int>(where);
    return 64; // Undefined Behavior.
}

static inline int HEX_COUNT_TRAILING_ZERO(unsigned int x)
{
    unsigned long where;
    if (_BitScanForward(&where, x)) return static_cast<int>(where);
    return 32; // Undefined Behavior.
}

static inline int HEX_COUNT_TRAILING_ZERO_UL(unsigned long x)
{
    return sizeof(x) == 8 ? HEX_COUNT_TRAILING_ZERO_ULL(x) : HEX_COUNT_TRAILING_ZERO(static_cast<unsigned int>(x));
}

static inline int HEX_COUNT_ONE_BIT(unsigned int x)
{
    // Binary: 0101...
    static const unsigned int m1 = 0x55555555;
    // Binary: 00110011..
    static const unsigned int m2 = 0x33333333;
    // Binary:  4 zeros,  4 ones ...
    static const unsigned int m4 = 0x0f0f0f0f;
    // The sum of 256 to the power of 0,1,2,3...
    static const unsigned int h01 = 0x01010101;
    // Put count of each 2 bits into those 2 bits.
    x -= (x >> 1) & m1;
    // Put count of each 4 bits into those 4 bits.
    x = (x & m2) + ((x >> 2) & m2);
    // Put count of each 8 bits into those 8 bits.
    x = (x + (x >> 4)) & m4;
    // Returns left 8 bits of x + (x<<8) + (x<<16) + (x<<24).
    return (x * h01) >> 24;
}

static inline int HEX_COUNT_ONE_BIT_ULL(unsigned long long x)
{
    // Binary: 0101...
    static const unsigned long long m1 = 0x5555555555555555;
    // Binary: 00110011..
    static const unsigned long long m2 = 0x3333333333333333;
    // Binary:  4 zeros,  4 ones ...
    static const unsigned long long m4 = 0x0f0f0f0f0f0f0f0f;
    // The sum of 256 to the power of 0,1,2,3...
    static const unsigned long long h01 = 0x0101010101010101;
    // Put count of each 2 bits into those 2 bits.
    x -= (x >> 1) & m1;
    // Put count of each 4 bits into those 4 bits.
    x = (x & m2) + ((x >> 2) & m2);
    // Put count of each 8 bits into those 8 bits.
    x = (x + (x >> 4)) & m4;
    // Returns left 8 bits of x + (x<<8) + (x<<16) + (x<<24) + ...
    return static_cast<int>((x * h01) >> 56);
}

static inline int HEX_COUNT_ONE_BIT_UL(unsigned long x)
{
    return sizeof(x) == 8 ? HEX_COUNT_ONE_BIT_ULL(x) : HEX_COUNT_ONE_BIT(static_cast<unsigned int>(x));
}

#endif // defined(__clang__)

// Atomic operation

#if defined(__clang__)

#define HEX_ATOMIC_ADD_AND_FETCH __sync_add_and_fetch
#define HEX_ATOMIC_FETCH_AND_ADD __sync_fetch_and_add

#define HEX_ATOMIC_AND_AND_FETCH __sync_and_and_fetch
#define HEX_ATOMIC_OR_AND_FETCH  __sync_or_and_fetch

#define HEX_ATOMIC_VAL_COMPARE_AND_SWAP  __sync_val_compare_and_swap
#define HEX_ATOMIC_BOOL_COMPARE_AND_SWAP __sync_bool_compare_and_swap

#elif defined(_MSC_VER)

#include <intrin.h>

#define HEX_ATOMIC_ADD_AND_FETCH(_p, _v)                                                                               \
    (sizeof *(_p) == sizeof(__int64) ? _InterlockedAdd64((__int64 *)_p, (__int64)_v)                                   \
                                     : _InterlockedAdd((long *)_p, (long)_v))
#define HEX_ATOMIC_FETCH_AND_ADD(_p, _v)                                                                               \
    (sizeof *(_p) == sizeof(__int64) ? _InterlockedExchangeAdd64((__int64 *)_p, (__int64)_v)                           \
                                     : _InterlockedExchangeAdd((long *)_p, (long)v))

template <typename _T> static inline _T HEX_ATOMIC_AND_AND_FETCH(_T volatile *_p, _T _v)
{
    _InterlockedAnd((long *)_p, (long)_v);
    return static_cast<_T>(*_p);
}

template <typename _T> static inline _T HEX_ATOMIC_OR_AND_FETCH(_T volatile *_p, _T _v)
{
    _InterlockedOr((long *)_p, (long)_v);
    return static_cast<_T>(*_p);
}

#define HEX_ATOMIC_VAL_COMPARE_AND_SWAP(_p, _old, _new)                                                                \
    (sizeof *(_p) == sizeof(__int64) ? _InterlockedCompareExchange64((__int64 *)_p, (__int64)_new, (__int64)_old)      \
                                     : _InterlockedCompareExchange((long *)_p, (long)_new, (long)_old))

#define HEX_ATOMIC_BOOL_COMPARE_AND_SWAP(_p, _old, _new) (HEX_ATOMIC_VAL_COMPARE_AND_SWAP(_p, _old, _new) == _old)

#endif // defined(__clang__)

#endif /* BUILTIN_INTRINSICS_H_ */
