// katagocoreml - Standalone C++ KataGo to Core ML Converter
// Copyright (c) 2025, Chin-Chang Yang
//
// Command-line tool for converting KataGo models to Core ML format.

#include "katagocoreml/KataGoConverter.hpp"
#include "katagocoreml/Version.hpp"
#include <iostream>
#include <string>
#include <cstring>

void printUsage(const char* program) {
    std::cout << "katago2coreml - Convert KataGo models to Core ML format\n"
              << "\n"
              << "Usage: " << program << " [options] <input.bin[.gz]> <output.mlpackage>\n"
              << "\n"
              << "Options:\n"
              << "  -x, --board-x <size>     Board width (default: 19)\n"
              << "  -y, --board-y <size>     Board height (default: 19)\n"
              << "  --optimize-identity-mask Optimize for full board (skip mask ops)\n"
              << "  --float16                Use FLOAT16 compute precision\n"
              << "  --float16-io             Use FLOAT16 for inputs/outputs (requires --float16)\n"
              << "  --dynamic-batch <min,max> Enable dynamic batch (e.g. 1,8 or 1,-1 for unlimited)\n"
              << "  --author <name>          Set model author in metadata\n"
              << "  --license <license>      Set model license in metadata (e.g. MIT, CC0)\n"
              << "  --info                   Show model info and exit\n"
              << "  -v, --verbose            Enable verbose output\n"
              << "  -h, --help               Show this help\n"
              << "\n"
              << "Examples:\n"
              << "  " << program << " kata1-b40c256.bin.gz KataGo.mlpackage\n"
              << "  " << program << " -x 9 -y 9 model.bin.gz KataGo-9x9.mlpackage\n"
              << "  " << program << " --optimize-identity-mask model.bin.gz KataGo-opt.mlpackage\n"
              << "  " << program << " --dynamic-batch 1,8 model.bin.gz KataGo-batch.mlpackage\n"
              << "  " << program << " --info model.bin.gz\n"
              << "\n"
              << "Supported KataGo model versions: 8-16\n"
              << "\n"
              << "katagocoreml version " << katagocoreml::VERSION << "\n";
}

void printModelInfo(const katagocoreml::ModelInfo& info) {
    std::cout << "Model Information:\n"
              << "  Name: " << info.name << "\n"
              << "  Version: " << info.version << "\n"
              << "  Input channels: " << info.num_input_channels << "\n"
              << "  Global input channels: " << info.num_input_global_channels << "\n"
              << "  Residual blocks: " << info.num_blocks << "\n"
              << "  Trunk channels: " << info.trunk_channels << "\n"
              << "  Policy channels: " << info.num_policy_channels << "\n"
              << "  Score value channels: " << info.num_score_value_channels << "\n"
              << "  Has metadata encoder: " << (info.has_metadata_encoder ? "yes" : "no") << "\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }

    katagocoreml::ConversionOptions options;
    bool verbose = false;
    bool info_only = false;
    std::string input_path;
    std::string output_path;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "-v" || arg == "--verbose") {
            verbose = true;
        } else if (arg == "--info") {
            info_only = true;
        } else if (arg == "--optimize-identity-mask") {
            options.optimize_identity_mask = true;
        } else if (arg == "--float16") {
            options.compute_precision = "FLOAT16";
        } else if (arg == "--float16-io") {
            options.use_fp16_io = true;
        } else if (arg == "-x" || arg == "--board-x") {
            if (i + 1 >= argc) {
                std::cerr << "Error: " << arg << " requires a value\n";
                return 1;
            }
            options.board_x_size = std::stoi(argv[++i]);
        } else if (arg == "-y" || arg == "--board-y") {
            if (i + 1 >= argc) {
                std::cerr << "Error: " << arg << " requires a value\n";
                return 1;
            }
            options.board_y_size = std::stoi(argv[++i]);
        } else if (arg == "--dynamic-batch") {
            if (i + 1 >= argc) {
                std::cerr << "Error: --dynamic-batch requires min,max (e.g. 1,8)\n";
                return 1;
            }
            std::string batch_arg = argv[++i];
            auto comma = batch_arg.find(',');
            if (comma == std::string::npos) {
                std::cerr << "Error: --dynamic-batch format: min,max (e.g. 1,8 or 1,-1)\n";
                return 1;
            }
            options.min_batch_size = std::stoi(batch_arg.substr(0, comma));
            options.max_batch_size = std::stoi(batch_arg.substr(comma + 1));
        } else if (arg == "--author") {
            if (i + 1 >= argc) {
                std::cerr << "Error: " << arg << " requires a value\n";
                return 1;
            }
            options.author = argv[++i];
        } else if (arg == "--license") {
            if (i + 1 >= argc) {
                std::cerr << "Error: " << arg << " requires a value\n";
                return 1;
            }
            options.license = argv[++i];
        } else if (arg[0] == '-') {
            std::cerr << "Error: Unknown option: " << arg << "\n";
            return 1;
        } else if (input_path.empty()) {
            input_path = arg;
        } else if (output_path.empty()) {
            output_path = arg;
        } else {
            std::cerr << "Error: Too many arguments\n";
            return 1;
        }
    }

    if (input_path.empty()) {
        std::cerr << "Error: Input file required\n";
        printUsage(argv[0]);
        return 1;
    }

    try {
        if (info_only) {
            // Show model info
            auto info = katagocoreml::KataGoConverter::getModelInfo(input_path);
            printModelInfo(info);
            return 0;
        }

        if (output_path.empty()) {
            std::cerr << "Error: Output path required\n";
            printUsage(argv[0]);
            return 1;
        }

        if (verbose) {
            std::cout << "Converting " << input_path << " to " << output_path << "\n"
                      << "  Board size: " << options.board_x_size << "x" << options.board_y_size << "\n"
                      << "  Optimize identity mask: " << (options.optimize_identity_mask ? "yes" : "no") << "\n"
                      << "  Compute precision: " << options.compute_precision << "\n"
                      << "  Batch size: " << options.min_batch_size
                      << (options.isDynamicBatch() ? "-" + std::to_string(options.max_batch_size) : "") << "\n";
            if (!options.author.empty()) {
                std::cout << "  Author: " << options.author << "\n";
            }
            if (!options.license.empty()) {
                std::cout << "  License: " << options.license << "\n";
            }
        }

        // First get model info
        auto info = katagocoreml::KataGoConverter::getModelInfo(input_path);
        if (verbose) {
            std::cout << "  Model version: " << info.version << "\n"
                      << "  Residual blocks: " << info.num_blocks << "\n"
                      << "  Trunk channels: " << info.trunk_channels << "\n";
        }

        // Perform conversion
        katagocoreml::KataGoConverter::convert(input_path, output_path, options);

        std::cout << "Successfully converted to " << output_path << "\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
