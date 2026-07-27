#include "oil/sops_integration.h"

namespace oil {

SopsGlobalState& sops_global() {
    static SopsGlobalState state;
    return state;
}

} // namespace oil
