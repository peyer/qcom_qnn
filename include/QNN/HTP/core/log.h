//==============================================================================
//
// Copyright (c) 2020 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef LOG_H
#define LOG_H 1

#include <cstdio>
#include <string>
#include "c_tricks.h"
#include "graph_status.h"
#include <utility>

// Constexpr that will get the offset of a path's basename for logging purposes
constexpr size_t fileNameOffset(const char *path)
{
    const char *const start = path;
    size_t filename_offset = 0;
    while (*path) {
        if (*path++ == '/') {
            filename_offset = path - start;
        }
    }
    return filename_offset;
}

// If log level or the dynamic logging flag are defined but don't have a value,
// then consider them to be undefined.
#if ~(~NN_LOG_MAXLVL + 0) == 0 && ~(~NN_LOG_MAXLVL + 1) == 1
#undef NN_LOG_MAXLVL
#endif

#if ~(~NN_LOG_DYNLVL + 0) == 0 && ~(~NN_LOG_DYNLVL + 1) == 1
#undef NN_LOG_DYNLVL
#endif

/*
 * We have migrated using C++ features like iostream to printf strings.
 * Why?
 * * C++ iostream makes it more difficult to use mixed decimal/hex
 * * C++ iostream isn't easily compatible with on-target logging facilities
 * * C++ iostream is bad for code size, printf is much better
 */

typedef void (*DspLogCallbackFunc)(int level, const char *fmt, va_list args);

// Dynamically set the logging priority level.
#ifndef _MSC_VER
#pragma GCC visibility push(default)
#endif
extern "C" {
void SetLogPriorityLevel(int level);
int GetLogPriorityLevel();
void SetLogCallbackFunc(DspLogCallbackFunc fn);
DspLogCallbackFunc GetLogCallbackFunc();
}
#ifndef _MSC_VER
#pragma GCC visibility pop
#endif

extern "C" {
// special log message for x86 that will log regardless logging level
void qnndsp_x86_log(const char *fmt, ...);
void progress_log(const char *info);
}

/**
 * @brief Logger declares all of its logging member functions once.
 * Alternate implementations for these functions are provided according to the
 * definitions of ENABLE_QNNDSP_LOG, NN_LOG_MAXLVL, and NN_LOG_DYNLVL.
 */
class Logger {
  public:
    template <typename... Args> static void qnndsp_base_log(int prio, char const *fmt, Args &&...args);
    template <typename... Args> static GraphStatus rawlog(char const *fmt, Args &&...args);
    template <typename... Args> static GraphStatus okaylog(char const *fmt, Args &&...args);
    template <typename... Args> static GraphStatus logmsgl(int prio, char const *fmt, Args &&...args);
    template <typename... Args> static GraphStatus logmsgraw(int prio, char const *fmt, Args &&...args);
    template <typename... Args> static GraphStatus logmsg(int prio, char const *fmt, Args &&...args);
    template <typename... Args> static GraphStatus verboselog(char const *fmt, Args &&...args);
    template <typename... Args> static GraphStatus infolog(char const *fmt, Args &&...args);
    template <typename... Args> static GraphStatus i_infolog(char const *fmt, Args &&...args);
    template <typename...>
    static GraphStatus statslog(char const *fmt, char const *file, char const *statname, char const *statvalue);
    template <typename...>
    static GraphStatus i_statslog(char const *fmt, char const *file, char const *statname, char const *statvalue);
    template <typename...>
    static GraphStatus statlog(char const *fmt, char const *file, char const *statname, long long statvalue);
    template <typename...>
    static GraphStatus i_statlog(char const *fmt, char const *file, char const *statname, long long statvalue);
    template <typename... Args> static GraphStatus warnlog(char const *fmt, Args &&...args);
    template <typename... Args> static GraphStatus errlog(char const *fmt, Args &&...args);
    template <typename... Args> static GraphStatus debuglog(char const *fmt, Args &&...args);
    template <typename... Args> static GraphStatus errlogl(char const *fmt, Args &&...args);
};

/**
 * @brief Logging levels. Note that the verbose log level
 * is from 4-10, while the debug log level is > 10.
 */
enum NNLogLevel {
    NN_LOG_ERRORLVL = 0,
    NN_LOG_WARNLVL = 1,
    NN_LOG_STATLVL = 2,
    NN_LOG_INFOLVL = 3,
    NN_LOG_VERBOSELVL = 4,
    NN_LOG_STATLVL_INTERNAL = 8,
    NN_LOG_INFOLVL_INTERNAL = 9,
    NN_LOG_DEBUGLVL = 11
};

/////////////////////////ENABLE_QNN_LOG
#ifdef ENABLE_QNNDSP_LOG

#ifndef _MSC_VER
#pragma GCC visibility push(default)
#endif
#include "weak_linkage.h"

API_C_FUNC void API_FUNC_NAME(SetLogCallback)(DspLogCallbackFunc cbFn, int logPriority);

extern "C" {
void qnndsp_log(int prio, const char *FMT, ...);

API_FUNC_EXPORT void hv3_load_log_functions(decltype(SetLogCallback) **SetLogCallback_f);
}
#ifndef _MSC_VER
#pragma GCC visibility pop
#endif

#ifdef NN_LOG_MAXLVL
template <typename... Args> void Logger::qnndsp_base_log(int prio, char const *fmt, Args &&...args)
{
    if (prio <= NN_LOG_MAXLVL) {
        qnndsp_log(prio, fmt, std::forward<Args>(args)...);
    }
}
#else
template <typename... Args> void Logger::qnndsp_base_log(int prio, char const *fmt, Args &&...args)
{
    qnndsp_log(prio, fmt, std::forward<Args>(args)...);
}
#endif

template <typename... Args> GraphStatus Logger::rawlog(char const *fmt, Args &&...args)
{
    qnndsp_base_log(NN_LOG_VERBOSELVL, fmt, std::forward<Args>(args)...);
    return GraphStatus::ErrorFatal;
}

template <typename... Args> GraphStatus Logger::okaylog(char const *fmt, Args &&...args)
{
    qnndsp_base_log(NN_LOG_ERRORLVL, fmt, std::forward<Args>(args)...);
    return GraphStatus::ErrorFatal;
}

template <typename... Args> GraphStatus Logger::errlog(char const *fmt, Args &&...args)
{
    qnndsp_base_log(NN_LOG_ERRORLVL, fmt, std::forward<Args>(args)...);
    return GraphStatus::ErrorFatal;
}

template <typename... Args> GraphStatus Logger::warnlog(char const *fmt, Args &&...args)
{
    qnndsp_base_log(NN_LOG_WARNLVL, fmt, std::forward<Args>(args)...);
    return GraphStatus::ErrorFatal;
}

template <typename...>
GraphStatus Logger::statlog(char const *fmt, char const *file, char const *statname, long long statvalue)
{
    qnndsp_base_log(NN_LOG_STATLVL, fmt, file, statname, statvalue);
    return GraphStatus::ErrorFatal;
}

template <typename...>
GraphStatus Logger::i_statlog(char const *fmt, char const *file, char const *statname, long long statvalue)
{
    qnndsp_base_log(NN_LOG_STATLVL_INTERNAL, fmt, file, statname, statvalue);
    return GraphStatus::ErrorFatal;
}

template <typename...>
GraphStatus Logger::statslog(char const *fmt, char const *file, char const *statname, char const *statvalue)
{
    qnndsp_base_log(NN_LOG_STATLVL, fmt, file, statname, statvalue);
    return GraphStatus::ErrorFatal;
}

template <typename...>
GraphStatus Logger::i_statslog(char const *fmt, char const *file, char const *statname, char const *statvalue)
{
    qnndsp_base_log(NN_LOG_STATLVL_INTERNAL, fmt, file, statname, statvalue);
    return GraphStatus::ErrorFatal;
}

template <typename... Args> GraphStatus Logger::infolog(char const *fmt, Args &&...args)
{
    qnndsp_base_log(NN_LOG_INFOLVL, fmt, std::forward<Args>(args)...);
    return GraphStatus::ErrorFatal;
}

template <typename... Args> GraphStatus Logger::i_infolog(char const *fmt, Args &&...args)
{
    qnndsp_base_log(NN_LOG_INFOLVL_INTERNAL, fmt, std::forward<Args>(args)...);
    return GraphStatus::ErrorFatal;
}

template <typename... Args> GraphStatus Logger::debuglog(char const *fmt, Args &&...args)
{
    qnndsp_base_log(NN_LOG_DEBUGLVL, fmt, std::forward<Args>(args)...);
    return GraphStatus::ErrorFatal;
}

template <typename... Args> GraphStatus Logger::verboselog(char const *fmt, Args &&...args)
{
    qnndsp_base_log(NN_LOG_VERBOSELVL, fmt, std::forward<Args>(args)...);
    return GraphStatus::ErrorFatal;
}

template <typename... Args> GraphStatus Logger::logmsgraw(int prio, char const *fmt, Args &&...args)
{
    qnndsp_base_log(prio, fmt, std::forward<Args>(args)...);
    return GraphStatus::ErrorFatal;
}

template <typename... Args> GraphStatus Logger::logmsg(int prio, char const *fmt, Args &&...args)
{
    qnndsp_base_log(prio, fmt, std::forward<Args>(args)...);
    return GraphStatus::ErrorFatal;
}

template <typename... Args> GraphStatus Logger::logmsgl(int prio, char const *fmt, Args &&...args)
{
    qnndsp_base_log(prio, fmt, std::forward<Args>(args)...);
    return GraphStatus::ErrorFatal;
}

#else //Hexagon default log
template <typename... Args> GraphStatus Logger::rawlog(char const *fmt, Args &&...args)
{
    (void)printf(fmt, std::forward<Args>(args)...);
    (void)fflush(stdout);
    return GraphStatus::ErrorFatal;
}

template <typename... Args> GraphStatus Logger::okaylog(char const *fmt, Args &&...args)
{
    (void)printf(fmt, std::forward<Args>(args)...);
    (void)fflush(stdout);
    return GraphStatus::ErrorFatal;
}

template <typename... Args> GraphStatus Logger::errlog(char const *fmt, Args &&...args)
{
    (void)printf(fmt, std::forward<Args>(args)...);
    (void)fflush(stdout);
    return GraphStatus::ErrorFatal;
}

template <typename... Args> GraphStatus Logger::errlogl(char const *fmt, Args &&...args)
{
    (void)printf(fmt, std::forward<Args>(args)...);
    return GraphStatus::ErrorFatal;
}
#if defined(NN_LOG_DYNLVL) && (NN_LOG_DYNLVL > 0)
template <typename... Args> GraphStatus Logger::logmsgraw(int prio, char const *fmt, Args &&...args)
{
    if (prio <= GetLogPriorityLevel()) {
        (void)rawlog(fmt, std::forward<Args>(args)...);
    }
    return GraphStatus::ErrorFatal;
}

template <typename... Args> GraphStatus Logger::logmsg(int prio, char const *fmt, Args &&...args)
{
    if (prio <= GetLogPriorityLevel()) {
        (void)okaylog(fmt, std::forward<Args>(args)...);
    }
    return GraphStatus::ErrorFatal;
}

template <typename... Args> GraphStatus Logger::logmsgl(int prio, char const *fmt, Args &&...args)
{
    if (prio <= GetLogPriorityLevel()) {
        (void)errlogl(fmt, std::forward<Args>(args)...);
    }
    return GraphStatus::ErrorFatal;
}
#elif defined(NN_LOG_MAXLVL)
template <typename... Args> GraphStatus Logger::logmsgraw(int prio, char const *fmt, Args &&...args)
{
    if (prio <= NN_LOG_MAXLVL) {
        (void)rawlog(fmt, std::forward<Args>(args)...);
    }
    return GraphStatus::ErrorFatal;
}

template <typename... Args> GraphStatus Logger::logmsg(int prio, char const *fmt, Args &&...args)
{
    if (prio <= NN_LOG_MAXLVL) {
        (void)okaylog(fmt, std::forward<Args>(args)...);
    }
    return GraphStatus::ErrorFatal;
}

template <typename... Args> GraphStatus Logger::logmsgl(int prio, char const *fmt, Args &&...args)
{
    if (prio <= NN_LOG_MAXLVL) {
        (void)errlogl(fmt, std::forward<Args>(args)...);
    }
    return GraphStatus::ErrorFatal;
}
#else
template <typename... Args> GraphStatus Logger::logmsgl(int prio, char const *fmt, Args &&...args)
{
    (void)prio;
    return errlogl(fmt, std::forward<Args>(args)...);
}

template <typename... Args> GraphStatus Logger::logmsgraw(int prio, char const *fmt, Args &&...args)
{
    (void)prio;
    return rawlog(fmt, std::forward<Args>(args)...);
}

template <typename... Args> GraphStatus Logger::logmsg(int prio, char const *fmt, Args &&...args)
{
    (void)prio;
    return okaylog(fmt, std::forward<Args>(args)...);
}
#endif
template <typename... Args> GraphStatus Logger::warnlog(char const *fmt, Args &&...args)
{
    return logmsg(NN_LOG_WARNLVL, fmt, std::forward<Args>(args)...);
}

template <typename...>
GraphStatus Logger::statlog(char const *fmt, char const *file, char const *statname, long long statvalue)
{
    return logmsg(NN_LOG_STATLVL, fmt, file, statname, statvalue);
}

template <typename...>
GraphStatus Logger::i_statlog(char const *fmt, char const *file, char const *statname, long long statvalue)
{
    return logmsg(NN_LOG_STATLVL_INTERNAL, fmt, file, statname, statvalue);
}

template <typename...>
GraphStatus Logger::statslog(char const *fmt, char const *file, char const *statname, char const *statvalue)
{
    return logmsg(NN_LOG_STATLVL, fmt, file, statname, statvalue);
}

template <typename...>
GraphStatus Logger::i_statslog(char const *fmt, char const *file, char const *statname, char const *statvalue)
{
    return logmsg(NN_LOG_STATLVL_INTERNAL, fmt, file, statname, statvalue);
}

template <typename... Args> GraphStatus Logger::infolog(char const *fmt, Args &&...args)
{
    return logmsg(NN_LOG_INFOLVL, fmt, std::forward<Args>(args)...);
}

template <typename... Args> GraphStatus Logger::i_infolog(char const *fmt, Args &&...args)
{
    return logmsg(NN_LOG_INFOLVL_INTERNAL, fmt, std::forward<Args>(args)...);
}

template <typename... Args> GraphStatus Logger::debuglog(char const *fmt, Args &&...args)
{
    return logmsg(NN_LOG_DEBUGLVL, fmt, std::forward<Args>(args)...);
}

template <typename... Args> GraphStatus Logger::verboselog(char const *fmt, Args &&...args)
{
    return logmsg(NN_LOG_VERBOSELVL, fmt, std::forward<Args>(args)...);
}
#endif

#ifdef NN_LOG_MAXLVL
#define LOG_STAT()    ((NN_LOG_MAXLVL) >= NN_LOG_STATLVL)
#define LOG_INFO()    ((NN_LOG_MAXLVL) >= NN_LOG_INFOLVL)
#define LOG_DEBUG()   ((NN_LOG_MAXLVL) >= NN_LOG_DEBUGLVL)
#define LOG_VERBOSE() ((NN_LOG_MAXLVL) >= NN_LOG_VERBOSELVL)
#else
#define LOG_STAT()    (1)
#define LOG_INFO()    (1)
#define LOG_DEBUG()   (1)
#define LOG_VERBOSE() (1)
#endif //#ifdef NN_LOG_MAXLVL

#define STRIP_DIR(file) &file[fileNameOffset(file)]

#define logmsgl2(priority, fmt, ...)   Logger::logmsgl(priority, fmt "%s", __VA_ARGS__)
#define logmsgraw2(priority, fmt, ...) Logger::logmsgraw(priority, fmt "%s", __VA_ARGS__)
#define logmsg2(priority, fmt, ...)                                                                                    \
    Logger::logmsg(priority, "%s:" TOSTRING(__LINE__) ":" fmt "%s\n", STRIP_DIR(__FILE__), __VA_ARGS__)
#define verboselog2(fmt, ...)                                                                                          \
    Logger::verboselog("%s:" TOSTRING(__LINE__) ":" fmt "%s\n", STRIP_DIR(__FILE__), __VA_ARGS__)
#define infolog2(fmt, ...) Logger::infolog("%s:" TOSTRING(__LINE__) ":" fmt "%s\n", STRIP_DIR(__FILE__), __VA_ARGS__)
#define i_infolog2(fmt, ...)                                                                                           \
    Logger::i_infolog("%s:" TOSTRING(__LINE__) ":" fmt "%s\n", STRIP_DIR(__FILE__), __VA_ARGS__)

#define warnlog2(fmt, ...)                                                                                             \
    Logger::warnlog("%s:" TOSTRING(__LINE__) ":WARNING: " fmt "%s\n", STRIP_DIR(__FILE__), __VA_ARGS__)
#define errlog2(fmt, ...)                                                                                              \
    Logger::errlog("%s:" TOSTRING(__LINE__) ":ERROR:" fmt "%s\n", STRIP_DIR(__FILE__), __VA_ARGS__)
#define debuglog2(fmt, ...) Logger::debuglog("%s:" TOSTRING(__LINE__) ":" fmt "%s\n", STRIP_DIR(__FILE__), __VA_ARGS__)

#define logmsgl(priority, ...)   logmsgl2(priority, __VA_ARGS__, "")
#define logmsgraw(priority, ...) logmsgraw2(priority, __VA_ARGS__, "")
#define logmsg(priority, ...)    logmsg2(priority, __VA_ARGS__, "")
#define verboselog(...)          verboselog2(__VA_ARGS__, "")
#define infolog(...)             infolog2(__VA_ARGS__, "")
#define i_infolog(...)           i_infolog2(__VA_ARGS__, "")
#define statslog(statname, statvalue)                                                                                  \
    Logger::statslog("%s:" TOSTRING(__LINE__) ":STAT: %s=%s\n", STRIP_DIR(__FILE__), statname, (statvalue))
#define i_statslog(statname, statvalue)                                                                                \
    Logger::i_statslog("%s:" TOSTRING(__LINE__) ":STAT: %s=%s\n", STRIP_DIR(__FILE__), statname, (statvalue))
#define statlog(statname, statvalue)                                                                                   \
    Logger::statlog("%s:" TOSTRING(__LINE__) ":STAT: %s=%lld\n", STRIP_DIR(__FILE__), statname, (long long)(statvalue))
#define i_statlog(statname, statvalue)                                                                                 \
    Logger::i_statlog("%s:" TOSTRING(__LINE__) ":STAT: %s=%lld\n", STRIP_DIR(__FILE__), statname,                      \
                      (long long)(statvalue))
#define warnlog(...)   warnlog2(__VA_ARGS__, "")
#define errlog(...)    errlog2(__VA_ARGS__, "")
#define _debuglog(...) debuglog2(__VA_ARGS__, "")

// Wrapper for debuglog, so certain debug contexts get conditionally #undef it
#define debuglog(...) _debuglog(__VA_ARGS__)

#endif
