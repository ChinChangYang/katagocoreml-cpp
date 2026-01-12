// katagocoreml - Standalone C++ KataGo to Core ML Converter
// Copyright (c) 2025

#include <gtest/gtest.h>
#include "katagocoreml/Options.hpp"

namespace katagocoreml {
namespace test {

TEST(BatchSizeTest, DefaultIsFixedBatchOne) {
    ConversionOptions opts;
    EXPECT_EQ(opts.min_batch_size, 1);
    EXPECT_EQ(opts.max_batch_size, 1);
    EXPECT_FALSE(opts.isDynamicBatch());
}

TEST(BatchSizeTest, DynamicBatchDetection) {
    ConversionOptions opts;
    opts.min_batch_size = 1;
    opts.max_batch_size = 8;
    EXPECT_TRUE(opts.isDynamicBatch());
}

TEST(BatchSizeTest, UnlimitedBatchDetection) {
    ConversionOptions opts;
    opts.min_batch_size = 1;
    opts.max_batch_size = -1;
    EXPECT_TRUE(opts.isDynamicBatch());
}

TEST(BatchSizeTest, FixedNonOneBatch) {
    ConversionOptions opts;
    opts.min_batch_size = 4;
    opts.max_batch_size = 4;
    EXPECT_FALSE(opts.isDynamicBatch());
}

TEST(BatchSizeTest, ZeroMaxBatchIsDynamic) {
    ConversionOptions opts;
    opts.min_batch_size = 1;
    opts.max_batch_size = 0;
    EXPECT_TRUE(opts.isDynamicBatch());
}

}  // namespace test
}  // namespace katagocoreml
