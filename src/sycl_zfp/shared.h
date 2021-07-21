#pragma once

#include "syclZFP.h"
#include <sycl/sycl.hpp>
#include "type_info.hpp"
#include "constants.h"
#include <climits>

typedef uint64_t Word;
typedef int64_t ll;


typedef size_t index_t;
typedef int64_t sindex_t;

using const_perm_accessor = sycl::accessor<uchar, 1, sycl::access::mode::read, sycl::access::target::constant_buffer>;

#define Wsize ((uint)(CHAR_BIT * sizeof(Word)))


#define MAX(x, y) (sycl::max((x), (y)))
#define MIN(x, y) (sycl::min((x),(y)))
#define bitsize(x) ((uint)(CHAR_BIT * sizeof(x)))


#ifdef USING_DPCPP
#define LDEXP(x, y) sycl::ldexp((x),(y))
#define FREXP(x, y) sycl::frexp((x),(y))
#define ZFP_ENCODE_ATOMIC_REF_TYPE sycl::ONEAPI::atomic_ref<Word, sycl::ONEAPI::memory_order::relaxed, sycl::ONEAPI::memory_scope::device, sycl::access::address_space::global_device_space>
#else

#ifdef USING_COMPUTECPP
#define LDEXP(x, y) ldexp((x),(y))
#define FREXP(x, y) frexp((x),(y))
#define ZFP_ENCODE_ATOMIC_REF_TYPE auto
#else
#define LDEXP(x, y) ldexp((x),(y))
#define FREXP(x, y) frexp((x),(y))
#define ZFP_ENCODE_ATOMIC_REF_TYPE sycl::atomic_ref<Word, sycl::memory_order::relaxed, sycl::memory_scope::device, address_space::global_space>
#endif

#endif

#define NBMASK 0xaaaaaaaaaaaaaaaaull

#define ZFP_1D_BLOCK_SIZE 4
#define ZFP_2D_BLOCK_SIZE 16
#define ZFP_3D_BLOCK_SIZE 64
#define ZFP_4D_BLOCK_SIZE 256

namespace syclZFP {


    struct int3_t {
        int z{}, y{}, x{};
    };

    struct int2_t {
        int y{}, x{};
    };


    template<typename T>
    void print_bits(sycl::stream &os, const T &bits) {
        const int bit_size = sizeof(T) * 8;

        for (int i = bit_size - 1; i >= 0; --i) {
            T one = 1;
            T mask = one << i;
            T val = (bits & mask) >> i;
            os << (int) val;
        }
        os << '\n';
    }

    size_t calc_device_mem1d(const size_t dim, const int maxbits) {
        const size_t vals_per_block = 4;
        size_t total_blocks = dim / vals_per_block;
        if (dim % vals_per_block != 0) {
            total_blocks++;
        }
        const auto bits_per_block = (size_t) maxbits;
        const size_t bits_per_word = sizeof(Word) * 8;
        const size_t total_bits = bits_per_block * total_blocks;
        size_t alloc_size = total_bits / bits_per_word;
        if (total_bits % bits_per_word != 0) alloc_size++;
        // ensure we have zeros
        return alloc_size * sizeof(Word);
    }

    size_t calc_device_mem2d(const sycl::id<2> dims, const int maxbits) {
        const size_t vals_per_block = 16;
        size_t total_blocks = (dims[0] * dims[1]) / vals_per_block;
        if ((dims[0] * dims[1]) % vals_per_block != 0) total_blocks++;
        const auto bits_per_block = (size_t) maxbits;
        const size_t bits_per_word = sizeof(Word) * 8;
        const size_t total_bits = bits_per_block * total_blocks;
        size_t alloc_size = total_bits / bits_per_word;
        if (total_bits % bits_per_word != 0) alloc_size++;
        return alloc_size * sizeof(Word);
    }

    size_t calc_device_mem3d(const sycl::id<3> encoded_dims, const int bits_per_block) {
        const size_t vals_per_block = 64;
        const size_t size = encoded_dims[0] * encoded_dims[1] * encoded_dims[2];
        size_t total_blocks = size / vals_per_block;
        const size_t bits_per_word = sizeof(Word) * 8;
        const size_t total_bits = (size_t) bits_per_block * total_blocks;
        const size_t alloc_size = total_bits / bits_per_word;
        return alloc_size * sizeof(Word);
    }

    sycl::id<3> get_max_grid_dims(const sycl::queue &q) {
#ifdef SYCL_EXT_ONEAPI_MAX_NUMBER_WORK_GROUPS
        return q.get_device().get_info<sycl::info::device::ext_oneapi_max_number_work_groups>();
#else
#pragma message "Missing SYCL information descriptor to check the max number of work groups allowed"
        return {(1 << 30) - 1, (1 << 30) - 1, (1 << 30) - 1};
#endif
    }

    // size is assumed to have a pad to the nearest preffered block size
    sycl::range<3> calculate_global_work_size(const sycl::queue &q, size_t size, size_t preffered_block_size) {
        size_t grids = size / preffered_block_size; // because of pad this will be exact
        sycl::id<3> max_grid_dims = get_max_grid_dims(q);
        int dims = 1;
        // check to see if we need to add more grids
        if (grids > max_grid_dims[2]) {
            dims = 2;
        }
        if (grids > max_grid_dims[2] * max_grid_dims[1]) {
            dims = 3;
        }

        sycl::range<3> grid_size{1, 1, 1};

        if (dims == 1) {
            grid_size[2] = grids;
        }

        if (dims == 2) {
            double sq_r = std::sqrt(grids);
            double intpart = 0;
            modf(sq_r, &intpart);
            auto base = (size_t) intpart;
            grid_size[2] = base;
            grid_size[1] = base;
            // figure out how many y to add
            size_t rem = (size - base * base);
            size_t y_rows = rem / base;
            if (rem % base != 0) y_rows++;
            grid_size[1] += y_rows;
        }

        if (dims == 3) {
            double cub_r = pow((double) grids, 1. / 3.);
            double intpart = 0;
            modf(cub_r, &intpart);
            auto base = (size_t) intpart;
            grid_size[2] = base;
            grid_size[1] = base;
            grid_size[0] = base;
            // figure out how many z to add
            size_t rem = (size - base * base * base);
            size_t z_rows = rem / (base * base);
            if (rem % (base * base) != 0) z_rows++;
            grid_size[0] += z_rows;
        }
#ifdef SYCL_EXT_ONEAPI_MAX_GLOBAL_NUMBER_WORK_GROUPS
        assert(grid_size[0] * grid_size[1] * grid_size[2] < q.get_device().get_info<sycl::info::device::ext_oneapi_max_global_number_work_groups>());
#else
#pragma message "Missing SYCL information descriptor to check the max number of work groups allowed"
#endif
        grid_size[0] = sycl::max(1ul, grid_size[0]);
        grid_size[1] = sycl::max(1ul, grid_size[1]);
        grid_size[2] = sycl::max(1ul, grid_size[2]);
        //printf("Grid size %lu %lu %lu\n", grid_size[2], grid_size[1], grid_size[0]);
        return grid_size;
    }


// map two's complement signed integer to negabinary unsigned integer
    inline uint64_t int2uint(const int64_t x) {
        return ((uint64_t) x + get_nbmask<uint64_t>() ^ get_nbmask<uint64_t>());
    }

    inline uint32_t int2uint(const int32_t x) {
        return ((uint32_t) x + get_nbmask<uint32_t>() ^ get_nbmask<uint32_t>());
    }


    template<typename Int, typename Scalar>
    Scalar dequantize(Int x, int e);

    template<>
    double dequantize<int64_t, double>(int64_t x, int e) {
        return LDEXP((double) x, e - (int) (CHAR_BIT * scalar_sizeof<double>() - 2));
    }

    template<>
    float dequantize<int32_t, float>(int32_t x, int e) {
        return LDEXP((float) x, e - (int) (CHAR_BIT * scalar_sizeof<float>() - 2));
    }

    template<>
    int32_t dequantize<int32_t, int32_t>(int32_t x, int e) {
        return 1;
    }

    template<>
    int64_t dequantize<int64_t, int64_t>(int64_t x, int e) {
        return 1;
    }

/* inverse lifting transform of 4-vector */
    template<class Int, int s>
    static void inv_lift(Int *p) {
        Int x, y, z, w;
        x = *p;
        p += s;
        y = *p;
        p += s;
        z = *p;
        p += s;
        w = *p;
        p += s;

        /*
        ** non-orthogonal transform
        **       ( 4  6 -4 -1) (x)
        ** 1/4 * ( 4  2  4  5) (y)
        **       ( 4 -2  4 -5) (z)
        **       ( 4 -6 -4  1) (w)
        */
        y += w >> 1;
        w -= y >> 1;
        y += w;
        w <<= 1;
        w -= y;
        z += x;
        x <<= 1;
        x -= z;
        y += z;
        z <<= 1;
        z -= y;
        w += x;
        x <<= 1;
        x -= w;

        p -= s;
        *p = w;
        p -= s;
        *p = z;
        p -= s;
        *p = y;
        p -= s;
        *p = x;
    }

    template<int BlockSize>
    sycl::buffer<uchar, 1> get_perm_buffer();

    template<>
    sycl::buffer<uchar, 1> get_perm_buffer<4>() {
        sycl::buffer<uchar, 1> buf{perm_1, sycl::range<1>(4)};
        buf.set_final_data(nullptr);
        buf.set_write_back(false);
        return buf;
    }

    template<>
    sycl::buffer<uchar, 1> get_perm_buffer<16>() {
        sycl::buffer<uchar, 1> buf{perm_2, sycl::range<1>(16)};
        buf.set_final_data(nullptr);
        buf.set_write_back(false);
        return buf;
    }

    template<>
    sycl::buffer<uchar, 1> get_perm_buffer<64>() {
        sycl::buffer<uchar, 1> buf{perm_3, sycl::range<1>(64)};
        buf.set_final_data(nullptr);
        buf.set_write_back(false);
        return buf;
    }


}

