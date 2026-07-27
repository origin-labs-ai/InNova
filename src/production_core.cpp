#include "oil/production_internal.h"
#include "oil/production_socket.h"
#include "oil/sha1.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <ctime>
#include <thread>
#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <cmath>
#include <climits>
#include <set>
#include <map>
#include <chrono>

#ifdef ERROR
#undef ERROR
#endif

namespace oil {

namespace {

thread_local std::string g_last_error;

} // anonymous namespace

// ========================================================================
// I3: C API
// ========================================================================
struct OilModel { Model* model; };

const char* oil_last_error() {
    return g_last_error.c_str();
}

bool file_exists(const char* path) {
#ifdef _WIN32
    DWORD attr = GetFileAttributesA(path);
    return attr != INVALID_FILE_ATTRIBUTES && !(attr & FILE_ATTRIBUTE_DIRECTORY);
#else
    struct stat st;
    return stat(path, &st) == 0 && S_ISREG(st.st_mode);
#endif
}

OilModel* oil_model_load(const char* path) {
    g_last_error.clear();
    if (!path) {
        g_last_error = "path is null";
        errno = EINVAL;
        return nullptr;
    }

    if (!file_exists(path)) {
        g_last_error = std::string("file not found: ") + path;
#ifdef _WIN32
        SetLastError(ERROR_FILE_NOT_FOUND);
#endif
        errno = ENOENT;
        return nullptr;
    }

    auto* om = new(std::nothrow) OilModel;
    if (!om) {
        g_last_error = "out of memory";
        errno = ENOMEM;
        return nullptr;
    }

    om->model = new(std::nothrow) DenseModel;
    if (!om->model) {
        delete om;
        g_last_error = "out of memory";
        errno = ENOMEM;
        return nullptr;
    }

    try {
        om->model->load(path);
    } catch (const std::exception& e) {
        g_last_error = std::string("model load failed: ") + e.what();
        delete om->model;
        delete om;
#ifdef _WIN32
        SetLastError(ERROR_FILE_NOT_FOUND);
#endif
        errno = ENOENT;
        return nullptr;
    } catch (...) {
        Logger::instance().log(Logger::ERROR, std::string("oil_model_load: unknown exception in ") + __func__);
        g_last_error = "model load failed: unknown error";
        delete om->model;
        delete om;
        errno = ENOENT;
        return nullptr;
    }

    return om;
}

void oil_model_free(OilModel* model) {
    if (!model) return;
    delete model->model;
    delete model;
}

char* oil_generate(OilModel* model, const char* prompt, int max_tokens) {
    g_last_error.clear();

    if (!model) {
        g_last_error = "model is null";
        errno = EINVAL;
        char* empty = new char[1];
        empty[0] = '\0';
        return empty;
    }
    if (!model->model) {
        g_last_error = "internal model is null";
        errno = EINVAL;
        char* empty = new char[1];
        empty[0] = '\0';
        return empty;
    }
    if (!prompt) {
        g_last_error = "prompt is null";
        errno = EINVAL;
        char* empty = new char[1];
        empty[0] = '\0';
        return empty;
    }
    if (max_tokens <= 0) {
        max_tokens = 64;
    }
    if (max_tokens > 4096) {
        max_tokens = 4096;
    }

    BPETokenizer bpe;
    std::vector<int> tokens;
    try {
        tokens = bpe.encode(prompt);
    } catch (const std::exception& e) {
        g_last_error = std::string("tokenization failed: ") + e.what();
        errno = EINVAL;
        char* empty = new char[1];
        empty[0] = '\0';
        return empty;
    }

    if (tokens.empty()) tokens = {1};

    std::vector<int> output;
    output.reserve(max_tokens);

    const auto& cfg = model->model->config;
    KVCache cache((int)cfg.num_layers, cfg.max_seq_len, cfg.num_heads, cfg.head_dim);

    auto sample_token = [&](const Tensor& logits) -> int {
        if (logits.numel() == 0) return -1;
        int64_t V = logits.dim(logits.rank() - 1);
        const float* ld = logits.data<float>() + logits.numel() - V;
        int next = 0;
        for (int64_t v = 1; v < V; v++)
            if (ld[v] > ld[next]) next = (int)v;
        return next;
    };

    try {
        int64_t prompt_len = (int64_t)tokens.size();
        Tensor input({1, prompt_len});
        float* id = input.data<float>();
        for (size_t j = 0; j < tokens.size(); j++) id[j] = (float)tokens[j];

        Tensor pos({1, prompt_len});
        float* pd = pos.data<float>();
        for (int64_t j = 0; j < prompt_len; j++) pd[j] = (float)j;

        Tensor logits = model->model->forward(input, pos, &cache);
        int next = sample_token(logits);
        if (next < 0) { /* empty result */ }
        else if (next == 2) { /* EOS */ }
        else output.push_back(next);
    } catch (const std::exception& e) {
        g_last_error = std::string("inference failed: ") + e.what();
        errno = EIO;
    }

    for (int i = 1; i < max_tokens && !output.empty(); i++) {
        try {
            int64_t pos_val = (int64_t)tokens.size() + (int64_t)output.size() - 1;

            Tensor input({1, 1});
            input.data<float>()[0] = (float)output.back();

            Tensor pos({1, 1});
            pos.data<float>()[0] = (float)pos_val;

            Tensor logits = model->model->forward(input, pos, &cache);
            int next = sample_token(logits);
            if (next < 0 || next == 2) break;
            output.push_back(next);
        } catch (const std::exception& e) {
            g_last_error = std::string("inference failed: ") + e.what();
            errno = EIO;
            break;
        }
    }

    std::string result = bpe.decode(output);
    char* cstr = new char[result.size() + 1];
    std::strcpy(cstr, result.c_str());
    return cstr;
}

void oil_free_string(char* s) {
    delete[] s;
}

// ========================================================================
// I5: HTTP Server (Model-integrated)
// ========================================================================
ModelHTTPServer::ModelHTTPServer(Model* model, int port)
    : model_(model), port_(port) {
    thread_pool_size_ = 4;
    timeout_seconds_ = 30;
    max_body_size_ = 4 * 1024 * 1024;
}

ModelHTTPServer::~ModelHTTPServer() { stop(); }

void ModelHTTPServer::set_timeout_seconds(int sec) {
    timeout_seconds_ = (std::max)(1, sec);
}

void ModelHTTPServer::set_max_body_size(size_t bytes) {
    max_body_size_ = bytes;
}

bool ModelHTTPServer::platform_init() {
    return socket_helpers::platform_init();
}

void ModelHTTPServer::platform_cleanup() {
    socket_helpers::platform_cleanup();
}

bool ModelHTTPServer::set_socket_timeout(int fd, int seconds) {
    return socket_helpers::set_timeout(fd, seconds);
}

void ModelHTTPServer::close_socket(int fd) {
    socket_helpers::close_socket(fd);
}

void ModelHTTPServer::start() {
    if (running_) return;
    if (!platform_init()) return;
    running_ = true;
    stop_requested_ = false;
    server_thread_ = std::thread(&ModelHTTPServer::server_loop, this);
}

void ModelHTTPServer::stop() {
    running_ = false;
    stop_requested_ = true;
    queue_cv_.notify_all();
    if (server_thread_.joinable()) server_thread_.join();
    for (auto& t : worker_threads_) {
        if (t.joinable()) t.join();
    }
    worker_threads_.clear();
    platform_cleanup();
}

void ModelHTTPServer::server_loop() {
#ifdef _WIN32
    SOCKET server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd == INVALID_SOCKET) { running_ = false; return; }
#else
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) { running_ = false; return; }
#endif

    int opt = 1;
#ifdef _WIN32
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR,
               (const char*)&opt, sizeof(opt));
#else
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#endif

    sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons((u_short)port_);

    if (bind(server_fd, (sockaddr*)&addr, sizeof(addr)) < 0) {
        closesocket(server_fd);
        running_ = false;
        return;
    }

    if (listen(server_fd, 64) < 0) {
        closesocket(server_fd);
        running_ = false;
        return;
    }

    // Spawn worker threads
    int n_workers = (std::max)(1, thread_pool_size_);
    for (int i = 0; i < n_workers; i++) {
        worker_threads_.emplace_back(&ModelHTTPServer::worker_loop, this);
    }

    fd_set read_fds;
    while (running_ && !stop_requested_) {
        FD_ZERO(&read_fds);
#ifdef _WIN32
        FD_SET(server_fd, &read_fds);
#else
        FD_SET(server_fd, &read_fds);
#endif
        struct timeval tv = {1, 0};

        int sel = select((int)server_fd + 1, &read_fds, nullptr, nullptr, &tv);
        if (sel < 0) {
            if (SOCKET_ERRNO == SOCKET_EWOULDBLOCK) continue;
            break;
        }
        if (sel == 0) continue;

        sockaddr_in client_addr;
#ifdef _WIN32
        int addr_len = sizeof(client_addr);
#else
        socklen_t addr_len = sizeof(client_addr);
#endif
        int client_fd = (int)accept(server_fd, (sockaddr*)&client_addr, &addr_len);
        if (client_fd < 0) continue;

        // Enqueue for worker thread
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            conn_queue_.push({client_fd});
        }
        queue_cv_.notify_one();
    }

    closesocket(server_fd);

    // Wait for workers to finish
    stop_requested_ = true;
    queue_cv_.notify_all();
}

void ModelHTTPServer::worker_loop() {
    while (!stop_requested_) {
        ClientConnection conn;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait_for(lock, std::chrono::seconds(1), [this] {
                return !conn_queue_.empty() || stop_requested_;
            });
            if (stop_requested_ || conn_queue_.empty()) continue;
            conn = conn_queue_.front();
            conn_queue_.pop();
        }

        // Set receive timeout
        set_socket_timeout(conn.fd, timeout_seconds_);

        // Handle the request
        handle_request(conn.fd);

        // Close connection
        close_socket(conn.fd);
    }
}

std::string ModelHTTPServer::get_mime_type(const std::string& path) const {
    std::string ext;
    auto dot = path.find_last_of('.');
    if (dot != std::string::npos) {
        ext = path.substr(dot);
        std::transform(ext.begin(), ext.end(), ext.begin(),
                       [](unsigned char c) { return (char)std::tolower(c); });
    }
    if (ext == ".html" || ext == ".htm") return "text/html";
    if (ext == ".json") return "application/json";
    if (ext == ".js")   return "application/javascript";
    if (ext == ".css")  return "text/css";
    if (ext == ".png")  return "image/png";
    if (ext == ".jpg" || ext == ".jpeg") return "image/jpeg";
    if (ext == ".gif")  return "image/gif";
    if (ext == ".svg")  return "image/svg+xml";
    if (ext == ".txt")  return "text/plain";
    if (ext == ".xml")  return "application/xml";
    if (ext == ".wasm") return "application/wasm";
    return "text/plain";
}

void ModelHTTPServer::send_response(int client_fd, int status,
                                const std::string& content_type,
                                const std::string& body) {
    std::string status_text;
    switch (status) {
        case 200: status_text = "OK"; break;
        case 201: status_text = "Created"; break;
        case 204: status_text = "No Content"; break;
        case 301: status_text = "Moved Permanently"; break;
        case 400: status_text = "Bad Request"; break;
        case 401: status_text = "Unauthorized"; break;
        case 403: status_text = "Forbidden"; break;
        case 404: status_text = "Not Found"; break;
        case 405: status_text = "Method Not Allowed"; break;
        case 413: status_text = "Payload Too Large"; break;
        case 429: status_text = "Too Many Requests"; break;
        case 500: status_text = "Internal Server Error"; break;
        case 502: status_text = "Bad Gateway"; break;
        case 503: status_text = "Service Unavailable"; break;
        default:  status_text = "Unknown"; break;
    }

    std::string response =
        "HTTP/1.1 " + std::to_string(status) + " " + status_text + "\r\n"
        "Content-Type: " + content_type + "\r\n"
        "Content-Length: " + std::to_string(body.size()) + "\r\n"
        "Connection: close\r\n"
        "Server: MYTHOS.cpp\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        "Access-Control-Allow-Headers: Content-Type\r\n"
        "\r\n" + body;

    send(client_fd, response.c_str(), (int)response.size(), 0);
}

void ModelHTTPServer::send_stream_response(int client_fd, const std::string& prompt,
                                       int max_tokens) {
    // SSE-style streaming response
    std::string headers =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/event-stream\r\n"
        "Cache-Control: no-cache\r\n"
        "Connection: keep-alive\r\n"
        "Server: MYTHOS.cpp\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "\r\n";

    send(client_fd, headers.c_str(), (int)headers.size(), 0);

    if (!model_) {
        std::string err = "data: {\"error\":\"no model loaded\"}\n\n";
        send(client_fd, err.c_str(), (int)err.size(), 0);
        return;
    }

    BPETokenizer bpe;
    std::vector<int> tokens;
    try {
        tokens = bpe.encode(prompt);
    } catch (...) {
        Logger::instance().log(Logger::WARN, std::string("SSE tokenization failed in ") + __func__);
        std::string err = "data: {\"error\":\"tokenization failed\"}\n\n";
        send(client_fd, err.c_str(), (int)err.size(), 0);
        return;
    }
    if (tokens.empty()) tokens = {1};

    std::vector<int> output;
    output.reserve(max_tokens);

    for (int i = 0; i < max_tokens; i++) {
        try {
            int64_t seq_len = (int64_t)(tokens.size() + output.size());
            Tensor input({1, seq_len});
            float* id = input.data<float>();
            for (size_t j = 0; j < tokens.size(); j++) id[j] = (float)tokens[j];
            for (size_t j = 0; j < output.size(); j++)
                id[tokens.size() + j] = (float)output[j];
            Tensor pos({1, seq_len});
            Tensor logits = model_->forward(input, pos);
            if (logits.numel() == 0) break;

            int64_t V = logits.dim(logits.rank() - 1);
            const float* ld = logits.data<float>() + logits.numel() - V;
            int next = 0;
            for (int64_t v = 1; v < V; v++)
                if (ld[v] > ld[next]) next = (int)v;

            if (next == 2) break;
            output.push_back(next);

            // Send token as SSE event
            std::string token_text = bpe.decode({next});
            std::string sse = "data: {\"token\":\"" + token_text + "\"}\n\n";
            if (send(client_fd, sse.c_str(), (int)sse.size(), 0) < 0) {
                break;
            }
        } catch (...) {
            Logger::instance().log(Logger::WARN, std::string("SSE inference failed in ") + __func__);
            std::string err = "data: {\"error\":\"inference failed\"}\n\n";
            send(client_fd, err.c_str(), (int)err.size(), 0);
            break;
        }
    }

    // Send completion event
    std::string done = "data: {\"done\":true}\n\n";
    send(client_fd, done.c_str(), (int)done.size(), 0);
}

void ModelHTTPServer::handle_request(int client_fd) {
    std::vector<char> full_buf;
    full_buf.reserve(8192);
    char tmp[4096];

    // Read request with timeout
    int n;
    bool timeout = false;
    auto start_time = std::chrono::steady_clock::now();

    while (full_buf.size() < max_body_size_) {
        auto elapsed = std::chrono::steady_clock::now() - start_time;
        if (std::chrono::duration_cast<std::chrono::seconds>(elapsed).count()
            >= timeout_seconds_) {
            timeout = true;
            break;
        }

#ifdef _WIN32
        n = recv(client_fd, tmp, sizeof(tmp) - 1, 0);
#else
        n = recv(client_fd, tmp, sizeof(tmp) - 1, 0);
#endif
        if (n < 0) {
            int err = SOCKET_ERRNO;
            if (err == SOCKET_EWOULDBLOCK || err == SOCKET_ETIMEDOUT) {
                timeout = true;
            }
            break;
        }
        if (n == 0) break;
        tmp[n] = '\0';
        full_buf.insert(full_buf.end(), tmp, tmp + n);

        // Check if we have the full headers
        std::string current(full_buf.data(), full_buf.size());
        auto header_end = current.find("\r\n\r\n");
        if (header_end != std::string::npos) {
            // Check Content-Length
            auto cl_pos = current.find("Content-Length:");
            if (cl_pos == std::string::npos)
                cl_pos = current.find("content-length:");
            if (cl_pos != std::string::npos) {
                auto num_start = current.find_first_of("0123456789", cl_pos + 15);
                if (num_start != std::string::npos) {
                    int content_length = std::stoi(current.substr(num_start));
                    size_t body_received = full_buf.size() - header_end - 4;
                    if (body_received >= (size_t)content_length) break;
                } else {
                    break;
                }
            } else {
                break;
            }
        }
    }

    if (full_buf.empty()) {
        send_response(client_fd, 400, "text/plain", "Empty request");
        return;
    }
    if (timeout) {
        send_response(client_fd, 408, "text/plain",
                       "{\"error\":\"request timeout\"}");
        return;
    }

    std::string request(full_buf.data(), full_buf.size());

    // Parse HTTP request
    std::string method, path, body;
    std::stringstream ss(request);
    ss >> method >> path;

    auto header_end = request.find("\r\n\r\n");
    if (header_end != std::string::npos)
        body = request.substr(header_end + 4);

    // Handle CORS preflight
    if (method == "OPTIONS") {
        std::string empty_body;
        send_response(client_fd, 204, "text/plain", empty_body);
        return;
    }

    // Only handle GET and POST
    if (method != "GET" && method != "POST") {
        send_response(client_fd, 405, "application/json",
                       "{\"error\":\"method not allowed\"}");
        return;
    }

    std::string response_body;
    std::string content_type = "application/json";

    if (path == "/v1/completions" || path == "/completions") {
        if (method != "POST") {
            send_response(client_fd, 405, "application/json",
                           "{\"error\":\"POST required for completions\"}");
            return;
        }

        // Parse request body
        std::string prompt = "Hello";
        int max_tokens = 64;
        bool stream = false;

        // Look for stream flag
        if (body.find("\"stream\":true") != std::string::npos ||
            body.find("\"stream\": true") != std::string::npos) {
            stream = true;
        }

        // Parse prompt
        auto prompt_pos = body.find("\"prompt\"");
        if (prompt_pos != std::string::npos) {
            auto start_q = body.find('"', prompt_pos + 8);
            if (start_q != std::string::npos) {
                auto end_q = body.find('"', start_q + 1);
                if (end_q != std::string::npos)
                    prompt = body.substr(start_q + 1, end_q - start_q - 1);
            }
        }

        // Parse max_tokens
        auto max_pos = body.find("\"max_tokens\"");
        if (max_pos != std::string::npos) {
            auto colon = body.find(':', max_pos);
            if (colon != std::string::npos) {
                auto num_start = body.find_first_of("0123456789", colon + 1);
                if (num_start != std::string::npos) {
                    auto num_end = body.find_first_not_of("0123456789", num_start);
                    if (num_end == std::string::npos)
                        num_end = body.size();
                    max_tokens = std::stoi(body.substr(num_start, num_end - num_start));
                }
            }
        }

        max_tokens = (std::min)((std::max)(1, max_tokens), 1024);

        if (stream) {
            send_stream_response(client_fd, prompt, max_tokens);
            return;
        }

        if (model_) {
            BPETokenizer bpe;
            std::vector<int> tokens;
            try {
                tokens = bpe.encode(prompt);
            } catch (...) {
                Logger::instance().log(Logger::WARN, std::string("HTTP tokenization failed in ") + __func__);
                send_response(client_fd, 500, "application/json",
                               "{\"error\":\"tokenization failed\"}");
                return;
            }
            if (tokens.empty()) tokens = {1};

            std::vector<int> output;
            output.reserve(max_tokens);

            for (int i = 0; i < max_tokens; i++) {
                try {
                    int64_t seq_len = (int64_t)(tokens.size() + output.size());
                    Tensor input({1, seq_len});
                    float* id = input.data<float>();
                    for (size_t j = 0; j < tokens.size(); j++) id[j] = (float)tokens[j];
                    for (size_t j = 0; j < output.size(); j++)
                        id[tokens.size() + j] = (float)output[j];
                    Tensor pos({1, seq_len});
                    Tensor logits = model_->forward(input, pos);
                    if (logits.numel() == 0) break;

                    int64_t V = logits.dim(logits.rank() - 1);
                    const float* ld = logits.data<float>() + logits.numel() - V;
                    int next = 0;
                    for (int64_t v = 1; v < V; v++)
                        if (ld[v] > ld[next]) next = (int)v;

                    if (next == 2) break;
                    output.push_back(next);
                } catch (...) {
                    Logger::instance().log(Logger::WARN, std::string("HTTP inference failed in ") + __func__);
                    send_response(client_fd, 500, "application/json",
                                   "{\"error\":\"inference failed\"}");
                    return;
                }
            }

            std::string generated = bpe.decode(output);
            // Escape JSON string
            std::string escaped;
            for (char c : generated) {
                if (c == '"' || c == '\\') { escaped += '\\'; escaped += c; }
                else if (c == '\n') escaped += "\\n";
                else if (c == '\r') escaped += "\\r";
                else if (c == '\t') escaped += "\\t";
                else if ((unsigned char)c < 32) escaped += "";
                else escaped += c;
            }
            response_body = "{\"choices\":[{\"text\":\"" + escaped + "\"}]}";
        } else {
            response_body = "{\"error\":\"no model loaded\"}";
        }
        content_type = "application/json";
    } else if (path == "/v1/chat/completions") {
        if (method != "POST") {
            send_response(client_fd, 405, "application/json",
                           "{\"error\":\"POST required\"}");
            return;
        }
        // Simple passthrough to completions for now
        // Extract last message content as prompt
        std::string prompt = "Hello";
        auto content_pos = body.find("\"content\"");
        if (content_pos != std::string::npos) {
            auto start_q = body.find('"', content_pos + 9);
            if (start_q != std::string::npos) {
                auto end_q = body.find('"', start_q + 1);
                if (end_q != std::string::npos)
                    prompt = body.substr(start_q + 1, end_q - start_q - 1);
            }
        }
        response_body = "{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"Echo: " + prompt + "\"}}]}";
        content_type = "application/json";
    } else if (path == "/health" || path == "/") {
        response_body = "{\"status\":\"ok\",\"model\":\"MYTHOS.cpp\"}";
        content_type = "application/json";
    } else if (path == "/v1/models") {
        if (!model_) {
            response_body = "{\"data\":[]}";
        } else {
            std::string model_name = model_->config.vocab_size > 0 ? "default" : "unknown";
            response_body = "{\"data\":[{\"id\":\"" + model_name + "\",\"object\":\"model\"}]}";
        }
        content_type = "application/json";
    } else {
        send_response(client_fd, 404, "application/json",
                       "{\"error\":\"not found\",\"path\":\"" + path + "\"}");
        return;
    }

    send_response(client_fd, 200, content_type, response_body);
}


// ========================================================================
// I12: Logger
// ========================================================================
Logger::Logger(Level level) : level_(level) {}

std::string Logger::level_str(Level l) {
    switch (l) {
        case DEBUG: return "DEBUG";
        case INFO:  return "INFO";
        case WARN:  return "WARN";
        case ERROR: return "ERROR";
        default:    return "UNKNOWN";
    }
}

void Logger::log(Level level, const std::string& message) {
    if (level < level_) return;
    std::lock_guard<std::mutex> lock(mtx_);
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);
    char buf[20];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tm);
    std::string line = std::string(buf) + " [" + level_str(level) + "] " + message;
    if (!file_path_.empty()) {
        std::ofstream f(file_path_, std::ios::app);
        if (f) f << line << std::endl;
    }
    std::cout << line << std::endl;
}

void Logger::set_file(const std::string& path) { file_path_ = path; }
Logger& Logger::instance() {
    static Logger inst;
    return inst;
}

// ========================================================================
// I13: Config (JSON parser)
// ========================================================================
AppConfig::AppConfig(const std::string& path) {
    if (path.empty()) return;

    std::ifstream f(path);
    if (!f) return;

    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());

    if (content.empty()) return;

    std::string error;
    size_t pos = 0;
    root_ = parse_json(content, pos, &error);

    if (!error.empty()) {
        Logger::instance().log(Logger::WARN,
            "Config parse error in " + path + ": " + error + " at position " +
            std::to_string(pos));
    }
}

void AppConfig::skip_ws(const std::string& json, size_t& pos) {
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t' ||
           json[pos] == '\n' || json[pos] == '\r')) {
        pos++;
    }
}

std::string AppConfig::escape_string(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 2);
    for (size_t i = 0; i < s.size(); i++) {
        char c = s[i];
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if ((unsigned char)c < 0x20) {
                    char buf[7];
                    snprintf(buf, sizeof(buf), "\\u%04x", (unsigned char)c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

AppConfig::JsonValue AppConfig::auto_type_value(const std::string& val) {
    char* end = nullptr;
    long long ll = strtoll(val.c_str(), &end, 10);
    if (end && *end == '\0' && end != val.c_str()) {
        return JsonValue((int64_t)ll);
    }

    end = nullptr;
    double d = strtod(val.c_str(), &end);
    if (end && *end == '\0' && end != val.c_str()) {
        return JsonValue(d);
    }

    if (val == "true") return JsonValue(true);
    if (val == "false") return JsonValue(false);

    return JsonValue(val);
}

std::string AppConfig::parse_string(const std::string& json, size_t& pos,
                                     std::string* error) {
    std::string result;
    if (pos >= json.size() || json[pos] != '"') {
        if (error) *error = "expected string";
        return result;
    }
    pos++;

    while (pos < json.size()) {
        char c = json[pos];
        if (c == '"') {
            pos++;
            return result;
        }
        if (c == '\\') {
            pos++;
            if (pos >= json.size()) break;
            switch (json[pos]) {
                case '"':  result += '"'; break;
                case '\\': result += '\\'; break;
                case '/':  result += '/'; break;
                case 'b':  result += '\b'; break;
                case 'f':  result += '\f'; break;
                case 'n':  result += '\n'; break;
                case 'r':  result += '\r'; break;
                case 't':  result += '\t'; break;
                case 'u': {
                    if (pos + 4 < json.size()) {
                        result += "\\u";
                        for (int i = 0; i < 4; i++) {
                            pos++;
                            result += json[pos];
                        }
                    }
                    break;
                }
                default: result += json[pos]; break;
            }
            pos++;
        } else {
            result += c;
            pos++;
        }
    }

    if (error) *error = "unterminated string";
    return result;
}

AppConfig::JsonValue AppConfig::parse_number(const std::string& json,
                                              size_t& pos, std::string* error) {
    size_t start = pos;
    if (pos < json.size() && json[pos] == '-') pos++;

    while (pos < json.size() && json[pos] >= '0' && json[pos] <= '9') pos++;

    bool is_float = false;
    if (pos < json.size() && json[pos] == '.') {
        is_float = true;
        pos++;
        while (pos < json.size() && json[pos] >= '0' && json[pos] <= '9') pos++;
    }

    if (pos < json.size() && (json[pos] == 'e' || json[pos] == 'E')) {
        is_float = true;
        pos++;
        if (pos < json.size() && (json[pos] == '+' || json[pos] == '-')) pos++;
        while (pos < json.size() && json[pos] >= '0' && json[pos] <= '9') pos++;
    }

    std::string num_str = json.substr(start, pos - start);

    if (is_float) {
        char* end = nullptr;
        double d = strtod(num_str.c_str(), &end);
        (void)end;
        return JsonValue(d);
    } else {
        char* end = nullptr;
        long long ll = strtoll(num_str.c_str(), &end, 10);
        (void)end;
        return JsonValue((int64_t)ll);
    }
}

AppConfig::JsonValue AppConfig::parse_value(const std::string& json,
                                             size_t& pos, std::string* error) {
    skip_ws(json, pos);
    if (pos >= json.size()) {
        if (error) *error = "unexpected end of JSON";
        return JsonValue();
    }

    switch (json[pos]) {
        case '{': return parse_object(json, pos, error);
        case '[': return parse_array(json, pos, error);
        case '"': return JsonValue(parse_string(json, pos, error));
        case 't':
            if (json.substr(pos, 4) == "true") { pos += 4; return JsonValue(true); }
            if (error) *error = "expected true";
            return JsonValue();
        case 'f':
            if (json.substr(pos, 5) == "false") { pos += 5; return JsonValue(false); }
            if (error) *error = "expected false";
            return JsonValue();
        case 'n':
            if (json.substr(pos, 4) == "null") { pos += 4; return JsonValue(); }
            if (error) *error = "expected null";
            return JsonValue();
        default:
            if (json[pos] == '-' || (json[pos] >= '0' && json[pos] <= '9'))
                return parse_number(json, pos, error);
            if (error) *error = std::string("unexpected character '") + json[pos] + "'";
            return JsonValue();
    }
}

AppConfig::JsonValue AppConfig::parse_object(const std::string& json,
                                              size_t& pos, std::string* error) {
    std::unordered_map<std::string, JsonValue> obj;
    pos++;

    skip_ws(json, pos);
    if (pos < json.size() && json[pos] == '}') {
        pos++;
        return JsonValue(obj);
    }

    while (pos < json.size()) {
        skip_ws(json, pos);
        if (pos >= json.size()) {
            if (error) *error = "unterminated object";
            break;
        }

        if (json[pos] != '"') {
            if (error) *error = "expected string key in object";
            break;
        }

        std::string key = parse_string(json, pos, error);

        skip_ws(json, pos);
        if (pos >= json.size() || json[pos] != ':') {
            if (error) *error = "expected ':' in object";
            break;
        }
        pos++;

        JsonValue val = parse_value(json, pos, error);
        obj[key] = std::move(val);

        skip_ws(json, pos);
        if (pos >= json.size()) break;

        if (json[pos] == '}') {
            pos++;
            return JsonValue(std::move(obj));
        }

        if (json[pos] != ',') {
            if (error) *error = "expected ',' or '}' in object";
            break;
        }
        pos++;
    }

    return JsonValue(std::move(obj));
}

AppConfig::JsonValue AppConfig::parse_array(const std::string& json,
                                             size_t& pos, std::string* error) {
    std::vector<JsonValue> arr;
    pos++;

    skip_ws(json, pos);
    if (pos < json.size() && json[pos] == ']') {
        pos++;
        return JsonValue(std::move(arr));
    }

    while (pos < json.size()) {
        arr.push_back(parse_value(json, pos, error));

        skip_ws(json, pos);
        if (pos >= json.size()) break;

        if (json[pos] == ']') {
            pos++;
            return JsonValue(std::move(arr));
        }

        if (json[pos] != ',') {
            if (error) *error = "expected ',' or ']' in array";
            break;
        }
        pos++;
    }

    return JsonValue(std::move(arr));
}

AppConfig::JsonValue AppConfig::parse_json(const std::string& json,
                                            size_t& pos, std::string* error) {
    return parse_value(json, pos, error);
}

void AppConfig::serialize_json(const JsonValue& val, std::string& out, int indent) {
    std::string ind(indent, ' ');
    std::string ind_inner(indent + 2, ' ');

    switch (val.type) {
        case JsonValue::NULL_VAL:
            out += "null";
            break;
        case JsonValue::BOOL:
            out += val.bool_val ? "true" : "false";
            break;
        case JsonValue::INT64:
            out += std::to_string(val.int_val);
            break;
        case JsonValue::FLOAT64: {
            char buf[64];
            snprintf(buf, sizeof(buf), "%g", val.float_val);
            out += buf;
            break;
        }
        case JsonValue::STRING:
            out += "\"" + escape_string(val.str_val) + "\"";
            break;
        case JsonValue::ARRAY:
            out += "[";
            for (size_t i = 0; i < val.arr_val.size(); i++) {
                if (i > 0) out += ",";
                serialize_json(val.arr_val[i], out, indent + 2);
            }
            out += "]";
            break;
        case JsonValue::OBJECT:
            out += "{";
            if (!val.obj_val.empty()) {
                bool first = true;
                for (auto& [k, v] : val.obj_val) {
                    if (!first) out += ",";
                    first = false;
                    out += "\"" + escape_string(k) + "\":";
                    serialize_json(v, out, indent + 2);
                }
            }
            out += "}";
            break;
    }
}

AppConfig::JsonValue* AppConfig::resolve_path(const std::string& key) {
    return const_cast<JsonValue*>(const_cast<const AppConfig*>(this)->resolve_path(key));
}

const AppConfig::JsonValue* AppConfig::resolve_path(const std::string& key) const {
    const JsonValue* current = &root_;

    size_t start = 0;
    while (start < key.size()) {
        auto dot = key.find('.', start);
        std::string part = key.substr(start, dot - start);
        start = (dot == std::string::npos) ? key.size() : dot + 1;

        if (current->type != JsonValue::OBJECT) return nullptr;
        auto it = current->obj_val.find(part);
        if (it == current->obj_val.end()) return nullptr;
        current = &it->second;
    }

    return current;
}

float AppConfig::get_float(const std::string& key, float def) const {
    const JsonValue* val = resolve_path(key);
    if (!val) return def;

    switch (val->type) {
        case JsonValue::FLOAT64: return (float)val->float_val;
        case JsonValue::INT64:   return (float)val->int_val;
        case JsonValue::STRING:
            try { return std::stof(val->str_val); } catch (...) { Logger::instance().log(Logger::WARN, std::string("stof parse failed in ") + __func__); return def; }
        case JsonValue::BOOL:    return val->bool_val ? 1.0f : 0.0f;
        default: return def;
    }
}

int AppConfig::get_int(const std::string& key, int def) const {
    const JsonValue* val = resolve_path(key);
    if (!val) return def;

    switch (val->type) {
        case JsonValue::INT64:   return (int)val->int_val;
        case JsonValue::FLOAT64: return (int)val->float_val;
        case JsonValue::STRING:
            try { return std::stoi(val->str_val); } catch (...) { Logger::instance().log(Logger::WARN, std::string("stoi parse failed in ") + __func__); return def; }
        case JsonValue::BOOL:    return val->bool_val ? 1 : 0;
        default: return def;
    }
}

std::string AppConfig::get_string(const std::string& key,
                                   const std::string& def) const {
    const JsonValue* val = resolve_path(key);
    if (!val) return def;

    switch (val->type) {
        case JsonValue::STRING: return val->str_val;
        case JsonValue::INT64:  return std::to_string(val->int_val);
        case JsonValue::FLOAT64: return std::to_string(val->float_val);
        case JsonValue::BOOL:   return val->bool_val ? "true" : "false";
        case JsonValue::NULL_VAL: return "null";
        default: return def;
    }
}

void AppConfig::set(const std::string& key, const std::string& value) {
    if (root_.type != JsonValue::OBJECT && root_.type != JsonValue::NULL_VAL) {
        root_ = JsonValue(std::unordered_map<std::string, JsonValue>());
    }
    if (root_.type == JsonValue::NULL_VAL) {
        root_.type = JsonValue::OBJECT;
    }

    JsonValue* current = &root_;

    size_t start = 0;
    while (start < key.size()) {
        auto dot = key.find('.', start);
        std::string part = key.substr(start, dot - start);
        start = (dot == std::string::npos) ? key.size() : dot + 1;

        if (start >= key.size()) {
            current->obj_val[part] = auto_type_value(value);
        } else {
            auto it = current->obj_val.find(part);
            if (it == current->obj_val.end() ||
                it->second.type != JsonValue::OBJECT) {
                current->obj_val[part] = JsonValue(
                    std::unordered_map<std::string, JsonValue>());
            }
            current = &current->obj_val[part];
        }
    }
}

void AppConfig::save(const std::string& path) {
    std::string json = to_json();
    std::ofstream f(path);
    if (f) f << json << std::endl;
}

std::string AppConfig::to_json() const {
    std::string out;
    serialize_json(root_, out, 0);
    return out;
}

bool AppConfig::validate(std::string* error_out) const {
    std::string error;
    size_t pos = 0;
    std::string json = to_json();
    JsonValue test = parse_json(json, pos, &error);
    if (!error.empty()) {
        if (error_out) *error_out = error;
        return false;
    }
    return pos >= json.size();
}

} // namespace oil
