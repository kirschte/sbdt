#ifndef CUSTOM_LOGGING_H
#define CUSTOM_LOGGING_H

#include "spdlog/spdlog.h"

// helper functions to build "var-arg" logging (max=8)
#define concat(one, two) ((std::string) one + (std::string) two).c_str()
#define LOG_INFO_ARG(msg, ...) spdlog::info(concat("[{0:>17s}] ", msg), __func__,  __VA_ARGS__)
#define LOG_INFO_NO_ARG(msg) spdlog::info(concat("[{0:>17s}] ", msg), __func__)
#define LOG_DEBUG_ARG(msg, ...) spdlog::debug(concat("[{0:>17s}] ", msg), __func__,  __VA_ARGS__)
#define LOG_DEBUG_NO_ARG(msg) spdlog::debug(concat("[{0:>17s}] ", msg), __func__)
#define GET_9TH_ARG(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, ...) arg9
#define LOG_INFO_MACRO_CHOOSER(...) \
    GET_9TH_ARG(__VA_ARGS__, LOG_INFO_ARG, LOG_INFO_ARG, LOG_INFO_ARG, LOG_INFO_ARG, LOG_INFO_ARG, LOG_INFO_ARG, LOG_INFO_ARG, LOG_INFO_NO_ARG, )
#define LOG_DEBUG_MACRO_CHOOSER(...) \
    GET_9TH_ARG(__VA_ARGS__, LOG_DEBUG_ARG, LOG_DEBUG_ARG, LOG_DEBUG_ARG, LOG_DEBUG_ARG, LOG_DEBUG_ARG, LOG_DEBUG_ARG, LOG_DEBUG_ARG, LOG_DEBUG_NO_ARG, )

// Logging functions
// - can call these two functions with variable number of args, as in python.
// - only difference is that you have to start enumerating at one, e.g.:
//   LOG_INFO("This {1} has {2:.0f} bugs", "logger", 0.42)
#define LOG_INFO(...) LOG_INFO_MACRO_CHOOSER(__VA_ARGS__)(__VA_ARGS__)
#define LOG_DEBUG(...) LOG_DEBUG_MACRO_CHOOSER(__VA_ARGS__)(__VA_ARGS__)

// highlight
#define YELLOW(words) "\033[0;40;33m" + words + "\033[0m"


inline void setup_logging(unsigned int verbosity) {
    spdlog::set_level(static_cast<spdlog::level::level_enum>( verbosity ));
    spdlog::set_pattern("[%H:%M:%S] [%^%5l%$] %v");
}


#endif /* CUSTOM_LOGGING_H */
