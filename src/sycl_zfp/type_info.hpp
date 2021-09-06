#pragma once

#include <sycl/sycl.hpp>
#include <cstdint>
#include <limits>

namespace syclZFP {

    template<typename T>
    inline int get_ebias();

    template<>
    inline int get_ebias<double>() { return 1023; }

    template<>
    inline int get_ebias<float>() { return 127; }

    template<>
    inline int get_ebias<int64_t>() { return 0; }

    template<>
    inline int get_ebias<int32_t>() { return 0; }

    template<typename T>
    inline int get_ebits();

    template<>
    inline int get_ebits<double>() { return 11; }

    template<>
    inline int get_ebits<float>() { return 8; }

    template<>
    inline int get_ebits<int32_t>() { return 0; }

    template<>
    inline int get_ebits<int64_t>() { return 0; }

    template<typename T>
    inline int get_precision();

    template<>
    inline int get_precision<double>() { return 64; }

    template<>
    inline int get_precision<int64_t>() { return 64; }

    template<>
    inline int get_precision<float>() { return 32; }

    template<>
    inline int get_precision<int32_t>() { return 32; }

    template<typename T>
    inline int get_min_exp();

    template<>
    inline int get_min_exp<double>() { return -1074; }

    template<>
    inline int get_min_exp<float>() { return -1074; }

    template<>
    inline int get_min_exp<int64_t>() { return 0; }

    template<>
    inline int get_min_exp<int32_t>() { return 0; }

    template<typename T>
    inline int scalar_sizeof();

    template<>
    inline int scalar_sizeof<double>() { return 8; }

    template<>
    inline int scalar_sizeof<int64_t>() { return 8; }

    template<>
    inline int scalar_sizeof<float>() { return 4; }

    template<>
    inline int scalar_sizeof<int32_t>() { return 4; }

    template<typename T>
    struct zfp_traits;

    template<>
    struct zfp_traits<double> {
        typedef uint64_t UInt;
        typedef int64_t Int;
    };

    template<>
    struct zfp_traits<int64_t> {
        typedef uint64_t UInt;
        typedef int64_t Int;
    };

    template<>
    struct zfp_traits<float> {
        typedef uint32_t UInt;
        typedef int32_t Int;
    };

    template<>
    struct zfp_traits<int32_t> {
        typedef uint32_t UInt;
        typedef int32_t Int;
    };

    template<typename T>
    inline bool is_int() {
        return false;
    }

    template<>
    inline bool is_int<int32_t>() {
        return true;
    }

    template<>
    inline bool is_int<int64_t>() {
        return true;
    }

    template<int T>
    struct block_traits;

    template<>
    struct block_traits<1> {
        typedef unsigned char PlaneType;
    };

    template<>
    struct block_traits<2> {
        typedef unsigned short PlaneType;
    };


    template<typename T>
    constexpr inline T get_nbmask();

    template<>
    constexpr inline uint32_t get_nbmask<uint32_t>() { return 0xaaaaaaaau; }

    template<>
    constexpr inline uint64_t get_nbmask<uint64_t>() { return 0xaaaaaaaaaaaaaaaaull; }


}
