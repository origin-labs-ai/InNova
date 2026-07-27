#pragma once

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#define OILSOCKET_ERRNO WSAGetLastError()
#define OILSOCKET_EWOULDBLOCK WSAEWOULDBLOCK
#define OILSOCKET_ECONNRESET WSAECONNRESET
#define OILSOCKET_ETIMEDOUT WSAETIMEDOUT
#else
#include <sys/socket.h>
#include <sys/select.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <signal.h>
#include <errno.h>
#include <netdb.h>
#include <dlfcn.h>
#include <sys/stat.h>
#define OILSOCKET_ERRNO errno
#define OILSOCKET_EWOULDBLOCK EWOULDBLOCK
#define OILSOCKET_ECONNRESET ECONNRESET
#define OILSOCKET_ETIMEDOUT ETIMEDOUT
typedef int SOCKET;
#define INVALID_SOCKET (-1)
#define SOCKET_ERROR (-1)
#define closesocket(fd) close(fd)
#endif

#ifndef SOCKET_ERRNO
#define SOCKET_ERRNO OILSOCKET_ERRNO
#endif
#ifndef SOCKET_EWOULDBLOCK
#define SOCKET_EWOULDBLOCK OILSOCKET_EWOULDBLOCK
#endif
#ifndef SOCKET_ECONNRESET
#define SOCKET_ECONNRESET OILSOCKET_ECONNRESET
#endif
#ifndef SOCKET_ETIMEDOUT
#define SOCKET_ETIMEDOUT OILSOCKET_ETIMEDOUT
#endif
