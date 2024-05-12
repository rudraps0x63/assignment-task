#include "lib/model.h"

#include <vector>
#include <iostream>
#include <cstdint>

static const std::string DEVICE_API = {
#if defined(__x86_64__) || defined(_M_X64)
  "vulkan"
#else
  "UNSUPPORTED_DEVICE"
#endif
};

void verifyCmdlineArguments(int argc, char* argv[])
{
  if (argc < 2 || argc > 3)
    throw std::runtime_error("Incorrect usage, use option: -h|--help for a brief account.");

  if (argc == 2) {
    std::string arg = argv[1];

    if (arg == "-h" || arg == "--help") {
      std::cout << "Usage: Invoke executable with TWO mandatory options: <modelPath> <inputFeatureFile>\n"
                << "Example (assuming inside executable directory): ./assignment-task ../data/whisper-tiny ../data/inputFeatureFile.bin\n";

      exit(0);
    }

    std::cout << "Incorrect usage, single argument supported: -h|--help\n";
    exit(1);
  }
}

int main(int argc, char* argv[])
{
  std::string modelPath;
  std::string inputFeatureFile;
  std::vector<std::string> modelArgs;
  Translator* tr = nullptr;

  verifyCmdlineArguments(argc, argv);

  modelPath = argv[1];
  inputFeatureFile = argv[2];

  modelArgs = {
    modelPath + "/" + DEVICE_API,
    modelPath + "/" + DEVICE_API + "/whisper-tiny.so",
    DEVICE_API
  };

  tr = new Translator(modelArgs);

  std::cout << "Generated transcription for file `" << inputFeatureFile << "` is:\n";
  std::cout << "\"" << tr->generateFromFeatureFile(inputFeatureFile) << "\"\n";

  return EXIT_SUCCESS;
}
