### Assignment.

The task is to fill the incomplete functions with appropriate logic in file `lib/translate.cc`. To identify them, simply search for the string `TASK: Complete this function`. All the functions that need completion have been marked with this comment. All the rest of the functionality has to be used as is.

You may wish to go through the MLC LLM submodule, since this assignment leverages functionality from there as well.
<br><br>
In particular, source files such as `mlc-llm/cpp/llm_chat.cc` might be especially helpful.
<br><br>
Additionally, for most of the incomplete functions you will already have access to equivalent Python implementations. Therefore, going through how the C++ side is structured could help.
<br><br>
The output we are looking for is something similar to:
<br><br>
![output](./data/transcription_output.png)
<br><br>
That is, one should be able to call the executable using the given feature file(s) as argument(s) from the terminal like so (assuming we're in the build directory):
```bash
./assignment-task ../data/whisper-tiny ../data/input_feature_file_0.bin
```
<br>
Build system being used is CMake, which has already been configured for you. You may wish to go through the toplevel <code>CMakeLists.txt</code> once.
<br><br>
Good luck!
