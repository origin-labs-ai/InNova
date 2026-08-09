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
#define QUANTSOCKET_ERRNO WSAGetLastError()
#define QUANTSOCKET_EWOULDBLOCK WSAEWOULDBLOCK
#define QUANTSOCKET_ECONNRESET WSAECONNRESET
#define QUANTSOCKET_ETIMEDOUT WSAETIMEDOUT
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
#define QUANTSOCKET_ERRNO errno
#define QUANTSOCKET_EWOULDBLOCK EWOULDBLOCK
#define QUANTSOCKET_ECONNRESET ECONNRESET
#define QUANTSOCKET_ETIMEDOUT ETIMEDOUT
typedef int SOCKET;
#define INVALID_SOCKET (-1)
#define SOCKET_ERROR (-1)
#define closesocket(fd) close(fd)
#endif

#ifndef SOCKET_ERRNO
#define SOCKET_ERRNO QUANTSOCKET_ERRNO
#endif
#ifndef SOCKET_EWOULDBLOCK
#define SOCKET_EWOULDBLOCK QUANTSOCKET_EWOULDBLOCK
#endif
#ifndef SOCKET_ECONNRESET
#define SOCKET_ECONNRESET QUANTSOCKET_ECONNRESET
#endif
#ifndef SOCKET_ETIMEDOUT
#define SOCKET_ETIMEDOUT QUANTSOCKET_ETIMEDOUT
#endif
