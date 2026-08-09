#pragma once
#include "quant/socket_common.h"
#include <mutex>
#include <string>
#include <thread>

namespace quant {
namespace socket_helpers {

inline std::once_flag& get_wsa_flag() {
    static std::once_flag flag;
    return flag;
}

inline bool platform_init() {
#ifdef _WIN32
    bool ok = true;
    std::call_once(get_wsa_flag(), [&ok]() {
        WSADATA wsa;
        if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) ok = false;
    });
    return ok;
#else
    signal(SIGPIPE, SIG_IGN);
    return true;
#endif
}

inline void platform_cleanup() {
#ifdef _WIN32
    WSACleanup();
#endif
}

inline bool set_timeout(int fd, int seconds) {
#ifdef _WIN32
    DWORD timeout = seconds * 1000;
    return setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO,
                      (const char*)&timeout, sizeof(timeout)) == 0;
#else
    struct timeval tv;
    tv.tv_sec = seconds;
    tv.tv_usec = 0;
    return setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO,
                      &tv, sizeof(tv)) == 0;
#endif
}

inline bool close_socket(int fd) {
#ifdef _WIN32
    return closesocket(fd) == 0;
#else
    return close(fd) == 0;
#endif
}

} // namespace socket_helpers
} // namespace quant
