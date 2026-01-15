// katagocoreml - Standalone C++ KataGo to Core ML Converter
// Copyright (c) 2025, Chin-Chang Yang

#include <gtest/gtest.h>
#include "../src/parser/KataGoParser.hpp"

namespace katagocoreml {
namespace test {

// Test that version support check works correctly
TEST(KataGoParserTest, SupportsVersions8To16) {
    for (int v = 8; v <= 16; v++) {
        EXPECT_TRUE(KataGoParser::isVersionSupported(v));
    }
}

TEST(KataGoParserTest, RejectsUnsupportedVersions) {
    EXPECT_FALSE(KataGoParser::isVersionSupported(7));
    EXPECT_FALSE(KataGoParser::isVersionSupported(17));
    EXPECT_FALSE(KataGoParser::isVersionSupported(0));
    EXPECT_FALSE(KataGoParser::isVersionSupported(-1));
}

}  // namespace test
}  // namespace katagocoreml
