# Vesper Web Server
# Build the web-based inference server for Vesper LLM

add_executable(vesper_server
    server.cpp
)

target_link_libraries(vesper_server PRIVATE vesper)

# Enable threading for httplib
find_package(Threads REQUIRED)
target_link_libraries(vesper_server PRIVATE Threads::Threads)

# C++17 features
target_compile_features(vesper_server PRIVATE cxx_std_17)

# Include the web directory for httplib.h and json.hpp
target_include_directories(vesper_server PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Optional: Install target
install(TARGETS vesper_server
    RUNTIME DESTINATION bin
)
