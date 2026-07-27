#include "oil/oil_format.h"
#include "oil/tensor.h"
#include "oil/types.h"
#include "oil/codebook.h"
#include "oil/format_planner.h"

#include <iostream>
#include <cassert>
#include <cmath>
#include <cstring>
#include <vector>
#include <cstdio>

template<typename T>
static bool approx_equal(T a, T b, T eps = (T)1e-5) {
    return std::abs(a - b) < eps;
}

int main() {
    std::string tmp_path = "_test_format_tmp.oil";

    // Create test weights and write .oil
    {
        oil::Tensor weights = oil::Tensor::arange(256);
        weights = weights.reshape({16, 16});

        oil::OILWriter writer(tmp_path);

        oil::OILHeader hdr;
        memcpy(hdr.magic, "OIL1", 4);
        hdr.version = (1 << 22) | (0 << 12) | 0;
        hdr.flags = 0;
        hdr.config_size = 0;
        writer.write_header(hdr, nullptr);

        std::vector<oil::FormatBlockEntry> ft;
        {
            oil::FormatBlockEntry e;
            e.block_id = 0;
            e.format = (uint8_t)oil::Format::OIL32;
            e.cb_bytes = 0;
            ft.push_back(e);
        }
        writer.write_format_table(ft);

        std::vector<oil::TensorEntry> tensors;
        std::vector<std::string> names;
        {
            oil::TensorEntry e;
            e.name_len = (uint16_t)7;
            e.block_start = 0;
            e.num_blocks = 1;
            tensors.push_back(e);
            names.push_back("weights");
        }
        writer.write_tensor_table(tensors, names);

        oil::BlockData block;
        block.format = oil::Format::OIL32;
        block.num_weights = (uint32_t)weights.numel();
        block.indices.resize(weights.size_bytes());
        memcpy(block.indices.data(), weights.data(), weights.size_bytes());
        writer.write_block(block);

        writer.close();
    }

    // Read back and verify
    {
        oil::OILReader reader(tmp_path);

        // Verify header
        auto& hdr = reader.header(); (void)hdr;
        assert(memcmp(hdr.magic, "OIL1", 4) == 0);
        assert(hdr.version > 0);

        // Verify tensor names
        auto names = reader.tensor_names();
        assert(names.size() == 1);
        assert(names[0] == "weights");

        // Read tensor
        auto t = reader.read_tensor("weights");
        // OIL reader reconstructs as flat tensor (shape info not stored)
        assert(t.numel() == 256);
        assert(t.shape().rank == 1);
        assert(t.shape().dims[0] == 256);
        assert(t.dtype() == oil::DType::F32);

        // Verify data equality
        for (int64_t i = 0; i < t.numel(); i++)
            assert(approx_equal(t.data<float>()[i], (float)i));

        // Verify format table
        auto ft = reader.read_format_table();
        assert(!ft.empty());
        assert(ft[0].format == (uint8_t)oil::Format::OIL32);

        // Test tensor_formats
        auto formats = reader.tensor_formats("weights");
        assert(!formats.empty());
        assert(formats[0] == oil::Format::OIL32);

        // Verify config
        auto config = reader.read_config();
        (void)config;
    }

    // Clean up
    std::remove(tmp_path.c_str());

    // Test FormatPlanner
    {
        oil::FormatPlanner planner(1.58f); // SPARK target
        auto plan = planner.plan_for_model(1000000);
        assert(!plan.blocks.empty());
        assert(plan.achieved_bpw > 0.0f);
        assert(plan.target_bpw > 0.0f);

        auto bpw_est = oil::FormatPlanner::estimate_bpw(plan); (void)bpw_est;
        assert(bpw_est > 0.0f);
    }

    // Test optimal ratios for all bpw from 1.5 to 8.0 (3-regime model)
    {
        // Regime 1 [1.0-2.25]: OIL8+Spark, Regime 2 [2.25-4.0]: OIL4+Spark, Regime 3 [4.0-8.0]: OIL8+OIL4
        struct TestCase { float bpw; bool has_oil8; bool has_oil4; bool has_spark; };
        TestCase cases[] = {
            {1.5f,  true,  false, true},   // OIL8+Spark
            {2.0f,  true,  false, true},   // OIL8+Spark
            {2.3f,  false, true,  true},   // OIL4+Spark
            {3.0f,  false, true,  true},   // OIL4+Spark
            {3.5f,  false, true,  true},   // OIL4+Spark
            {4.0f,  false, true,  false},  // Pure OIL4
            {4.5f,  true,  true,  false},  // OIL8+OIL4
            {5.0f,  true,  true,  false},  // OIL8+OIL4
            {6.0f,  true,  true,  false},  // OIL8+OIL4
            {7.0f,  true,  true,  false},  // OIL8+OIL4
            {8.0f,  true,  false, false},  // Pure OIL8
        };

        for (auto& tc : cases) {
            oil::FormatPlanner p(tc.bpw);
            auto plan = p.plan_for_model(1000000);

            int cnt_oil8 = 0, cnt_oil4 = 0, cnt_spark_q0 = 0;
            for (auto& b : plan.blocks) {
                if (b.assigned_format == oil::Format::OIL8) cnt_oil8++;
                else if (b.assigned_format == oil::Format::OIL4) cnt_oil4++;
                else cnt_spark_q0++;
            }

            // Validate regime assignments
            if (tc.has_oil8) assert(cnt_oil8 > 0);
            if (!tc.has_oil8) assert(cnt_oil8 == 0);
            if (tc.has_oil4) assert(cnt_oil4 > 0);
            if (!tc.has_oil4) assert(cnt_oil4 == 0);
            if (tc.has_spark) assert(cnt_spark_q0 > 0);
            if (!tc.has_spark) assert(cnt_spark_q0 == 0);

            // Validate achieved bpw close to target
            float achieved = oil::FormatPlanner::estimate_bpw(plan);
            assert(std::abs(achieved - tc.bpw) < 1.0f);
        }
    }

    // Test OILHeader packing
    {
        oil::OILHeader hdr;
        memcpy(hdr.magic, "OIL1", 4);
        hdr.version = 1;
        hdr.flags = 0xFF;
        hdr.config_size = 1234;
        assert(memcmp(hdr.magic, "OIL1", 4) == 0);
        assert(hdr.config_size == 1234);
    }

    // Test FormatBlockEntry packing
    {
        oil::FormatBlockEntry e; (void)e;
        e.block_id = 42;
        e.format = (uint8_t)oil::Format::OIL8;
        e.cb_bytes = 256 * 4;
        assert(e.block_id == 42);
        assert(e.format == (uint8_t)oil::Format::OIL8);
        assert(e.cb_bytes == 1024);

        // Test enum values match
        assert((uint8_t)oil::Format::OIL32 == 6);
        assert((uint8_t)oil::Format::OIL8 == 4);
        assert((uint8_t)oil::Format::SPARK_Q0 == 1);
        assert((uint8_t)oil::Format::OIL1 == 0);
    }

    // Test format_name and format_bpw
    {
        assert(std::string(oil::format_name(oil::Format::OIL32)) == "OIL32");
        assert(std::string(oil::format_name(oil::Format::SPARK_Q0)) == "SPARK_Q0");
        assert(oil::format_bpw(oil::Format::OIL32) == 32.0f);
        assert(oil::format_bpw(oil::Format::SPARK_Q0) == 1.5f);
        assert(oil::format_bpw(oil::Format::OIL8) == 8.0f);
    }

    // Test format_to_dtype
    {
        assert(oil::format_to_dtype(oil::Format::OIL32) == oil::DType::F32);
        assert(oil::format_to_dtype(oil::Format::OIL16) == oil::DType::F16);
        assert(oil::format_to_dtype(oil::Format::OIL8) == oil::DType::U8);
        assert(oil::format_to_dtype(oil::Format::OIL4) == oil::DType::U4);
        assert(oil::format_to_dtype(oil::Format::SPARK_Q0) == oil::DType::U8);
    }

    std::cout << "All format tests passed!" << std::endl;
    return 0;
}
