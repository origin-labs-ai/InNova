#include "oil/format_registry.h"
#include <cstdio>

int main() {
    std::string table = oil::FormatRegistry::get_format_table();
    std::printf("%s", table.c_str());
    return 0;
}
