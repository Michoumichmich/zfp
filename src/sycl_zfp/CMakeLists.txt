###############################################################################
#
#  file: src/sycl_zfp/CMakeLists.txt
#
###############################################################################

include_directories(sycl_intrinsics)

set(syclZFP_sources
        syclZFP.cpp         # main entry point
        decode.hpp
        decode1.hpp
        decode2.hpp
        decode3.hpp
        encode.hpp
        encode1.hpp
        encode2.hpp
        encode3.hpp
        pointers.hpp
        type_info.hpp)

set(syclZFP_headers
        shared.h
        syclZFP.h)
