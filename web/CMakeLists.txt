# Vesper Web Server
# Build the web-based inference server for Vesper LLM

add_executable(vesper_server
    server.cpp
)

target_link_libraries(vesper_server PRIVATE vesper)

# Enable threading for httplib
find_package(Threads REQUIRED)
target_link_libraries(vesper_server PRIVATE Threads::Threads)

# Link SentencePiece for tokenization
find_library(SENTENCEPIECE_LIB sentencepiece)
if(SENTENCEPIECE_LIB)
    target_link_libraries(vesper_server PRIVATE ${SENTENCEPIECE_LIB})
    message(STATUS "Found SentencePiece: ${SENTENCEPIECE_LIB}")
else()
    message(FATAL_ERROR "SentencePiece library not found. Install libsentencepiece-dev")
endif()

# C++17 features
target_compile_features(vesper_server PRIVATE cxx_std_17)

# Include the web directory for httplib.h and json.hpp
target_include_directories(vesper_server PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Optional: Install target
install(TARGETS vesper_server
    RUNTIME DESTINATION bin
)
