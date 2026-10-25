# PulseX Wan - Wan 2.2 TI2V-5B (text or a picture -> a short video) on the Hexagon NPU of a Snapdragon X laptop.
#
# Two programs are built from this folder:
#   wan-stream.exe   the DiT engine (wan_stream.cpp): the 30 transformer blocks streamed through the NPU with ggml's
#                    Hexagon backend. Needs the ggml libraries of the PulseX fork of llama.cpp (ggml-hexagon) - the
#                    packed-tensor entry points it looks up at run time exist only in that fork.
#   pulsex-wan.exe   the window (wan_gui.cpp): Dear ImGui + DX11. It links nothing of ggml; it runs the Python
#                    pipeline (wan_pipeline.py), which in turn starts wan-stream.exe and llama-embedding.exe.
#
# Nothing here points at a folder on a particular machine: every outside input is a CMake variable or a package.
#
# Two ways to build (Windows 11 on ARM64; clang-cl or clang++ --target=arm64-pc-windows-msvc, or MSVC ARM64):
#
#   A) inside the ggml-hexagon fork, like pulsex-image's examples/flux_dit: copy this folder to <fork>/examples/wan22,
#      add "add_subdirectory(wan22)" to <fork>/examples/CMakeLists.txt and build the fork as usual. The fork's own
#      targets ggml / ggml-cpu are used.
#
#   B) on its own, against an INSTALLED fork (cmake --install of the fork, which writes ggml-config.cmake):
#        cmake -S src -B build -G Ninja -DCMAKE_BUILD_TYPE=Release ^
#              -DCMAKE_C_COMPILER=clang-cl -DCMAKE_CXX_COMPILER=clang-cl ^
#              -Dggml_DIR=<fork install>/lib/cmake/ggml ^
#              -DWAN_IMGUI_DIR=<Dear ImGui root>
#        cmake --build build
#      Leave out -Dggml_DIR to build only the window; leave out -DWAN_IMGUI_DIR to build only the engine.
#
# The programs need, next to them at run time: ggml.dll, ggml-base.dll, ggml-cpu.dll, ggml-hexagon.dll and the NPU
# modules libggml-htp-v7x.so + libggml-htp.cat (from the fork), llama-embedding.exe + llama.dll + llama-common.dll
# (the umT5 text encoder), and pulsex-wan.ico (copied by this file). The prebuilt set is in the repository's bin\.
cmake_minimum_required(VERSION 3.21)

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    project(pulsex-wan LANGUAGES C CXX)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endif()

if(POLICY CMP0091)
    cmake_policy(SET CMP0091 NEW)          # MSVC_RUNTIME_LIBRARY below
endif()

# ── the DiT engine ──────────────────────────────────────────────────────────────────────────────────────────────────
set(WAN_GGML "")
if(TARGET ggml)                             # way A: inside the fork
    set(WAN_GGML ggml)
    if(TARGET ggml-cpu)
        list(APPEND WAN_GGML ggml-cpu)      # wan_stream.cpp calls the CPU backend directly (ggml-cpu.h)
    endif()
else()                                      # way B: an installed fork
    find_package(ggml CONFIG QUIET)
    if(ggml_FOUND)
        set(WAN_GGML ggml::ggml)
        if(TARGET ggml::ggml-cpu)
            list(APPEND WAN_GGML ggml::ggml-cpu)
        endif()
    endif()
endif()

if(WAN_GGML)
    find_package(Threads REQUIRED)
    add_executable(wan-stream wan_stream.cpp)
    target_link_libraries(wan-stream PRIVATE ${WAN_GGML} Threads::Threads)
    target_compile_features(wan-stream PRIVATE cxx_std_17)
    target_compile_definitions(wan-stream PRIVATE _CRT_SECURE_NO_WARNINGS)
    if(WIN32)
        # The SHARED C runtime, like ggml's DLLs: wan-stream sets its defaults (FA scale, FA tiling, ...) with _putenv_s,
        # and ggml reads them with getenv - with a static CRT each module has its own environment copy and the defaults
        # never reach ggml (measured: the engine then runs the slow attention path without a word).
        set_property(TARGET wan-stream PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
    endif()
else()
    message(STATUS "pulsex-wan: no ggml found (neither the fork's targets nor -Dggml_DIR) - wan-stream is not built")
endif()

# ── the window ──────────────────────────────────────────────────────────────────────────────────────────────────────
set(WAN_IMGUI_DIR "" CACHE PATH "Dear ImGui root (imgui.cpp, backends/) - needed for the window, pulsex-wan")
if(WIN32 AND EXISTS "${WAN_IMGUI_DIR}/imgui.cpp")
    add_executable(pulsex-wan
        wan_gui.cpp
        ${WAN_IMGUI_DIR}/imgui.cpp
        ${WAN_IMGUI_DIR}/imgui_draw.cpp
        ${WAN_IMGUI_DIR}/imgui_tables.cpp
        ${WAN_IMGUI_DIR}/imgui_widgets.cpp
        ${WAN_IMGUI_DIR}/backends/imgui_impl_win32.cpp
        ${WAN_IMGUI_DIR}/backends/imgui_impl_dx11.cpp
    )
    # flux_image.hpp (picture loading through WIC) and gui.manifest sit beside wan_gui.cpp in this folder
    target_include_directories(pulsex-wan PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${WAN_IMGUI_DIR} ${WAN_IMGUI_DIR}/backends)
    target_compile_features(pulsex-wan PRIVATE cxx_std_17)
    # PULSEX_WAN_PUBLIC: the published window (no developer defaults, no personal texts); see the README's Build section
    target_compile_definitions(pulsex-wan PRIVATE NOMINMAX _CRT_SECURE_NO_WARNINGS PULSEX_WAN_PUBLIC=1)
    target_link_libraries(pulsex-wan PRIVATE d3d11 dxgi d3dcompiler comdlg32 windowscodecs ole32 shell32 dwmapi)
    if(MSVC)
        # UTF-8 code page + per-monitor DPI awareness, merged into the linker's own manifest
        target_link_options(pulsex-wan PRIVATE "/SUBSYSTEM:WINDOWS" "/ENTRY:mainCRTStartup"
                            "/MANIFESTINPUT:${CMAKE_CURRENT_SOURCE_DIR}/gui.manifest")
    else()
        target_link_options(pulsex-wan PRIVATE "LINKER:/SUBSYSTEM:WINDOWS" "LINKER:/ENTRY:mainCRTStartup"
                            "LINKER:/MANIFESTINPUT:${CMAKE_CURRENT_SOURCE_DIR}/gui.manifest" "LINKER:/MANIFEST:EMBED")
    endif()
    # the window loads its icon from the file beside the exe (no resource script)
    add_custom_command(TARGET pulsex-wan POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/pulsex-wan.ico $<TARGET_FILE_DIR:pulsex-wan>)
else()
    message(STATUS "pulsex-wan: WAN_IMGUI_DIR not set (or not Windows) - the window is not built")
endif()
