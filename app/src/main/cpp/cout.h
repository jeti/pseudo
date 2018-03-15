#ifndef COUT_HEADER
#define COUT_HEADER

/** The purpose of this header is to properly define "cout".
 * On most systems, this is simply an alias for std::cout.
 * However, on Android, we redirect everything to the error log.
 */
#include <iostream>
using std::endl;

/* Android-specific stuff */
#ifdef __ANDROID__

#include <jni.h>
#include <android/log.h>
#include <sstream>

#define LOG_TAG "Solution Info"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

class Log {
private:

    template<typename T>
    std::string to_string(T value) {
        std::ostringstream os;
        os << value;
        return os.str();
    }

public:

    Log() {};

    template<typename T>
    Log &operator<<(const T &value) {
        LOGE("%s", to_string(value).c_str());
        return *this;
    }

    /* Needed for handling endl */
    Log &operator<<(std::ostream &(*pf)(std::ostream &)) {
        LOGE("%s", to_string(pf).c_str());
        return *this;
    }
};

//Log cout;
using std::cout;

#else

using std::cout;

#endif

#endif /* COUT_HEADER */