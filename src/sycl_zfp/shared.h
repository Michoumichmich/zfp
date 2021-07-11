#pragma once

#define SYCL_ZFP_RATE_PRINT 1
typedef unsigned long long Word;
#define Wsize ((uint)(CHAR_BIT * sizeof(Word)))

#include "syclZFP.h"
#include <sycl/sycl.hpp>
#include "type_info.hpp"
#include "constants.h"

#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define bitsize(x) (CHAR_BIT * (uint)sizeof(x))

#define NBMASK 0xaaaaaaaaaaaaaaaaull

#define ZFP_1D_BLOCK_SIZE 4
#define ZFP_2D_BLOCK_SIZE 16
#define ZFP_3D_BLOCK_SIZE 64
#define ZFP_4D_BLOCK_SIZE 256

namespace syclZFP {

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

    size_t calc_device_mem1d(const uint dim, const uint maxbits) {
        const size_t vals_per_block = 4;
        size_t total_blocks = dim / vals_per_block;
        if (dim % vals_per_block != 0) {
            total_blocks++;
        }
        const size_t bits_per_block = maxbits;
        const size_t bits_per_word = sizeof(Word) * 8;
        const size_t total_bits = bits_per_block * total_blocks;
        size_t alloc_size = total_bits / bits_per_word;
        if (total_bits % bits_per_word != 0) alloc_size++;
        // ensure we have zeros
        return alloc_size * sizeof(Word);
    }

    size_t calc_device_mem2d(const sycl::id<2> dims, const uint maxbits) {
        const size_t vals_per_block = 16;
        size_t total_blocks = (dims[sycl::elem::x] * dims[sycl::elem::y]) / vals_per_block;
        if ((dims[sycl::elem::x] * dims[sycl::elem::y]) % vals_per_block != 0) total_blocks++;
        const size_t bits_per_block = maxbits;
        const size_t bits_per_word = sizeof(Word) * 8;
        const size_t total_bits = bits_per_block * total_blocks;
        size_t alloc_size = total_bits / bits_per_word;
        if (total_bits % bits_per_word != 0) alloc_size++;
        return alloc_size * sizeof(Word);
    }

    size_t calc_device_mem3d(const sycl::id<3> encoded_dims, const uint bits_per_block) {
        const size_t vals_per_block = 64;
        const size_t size = encoded_dims[sycl::elem::x] * encoded_dims[sycl::elem::y] * encoded_dims[sycl::elem::z];
        size_t total_blocks = size / vals_per_block;
        const size_t bits_per_word = sizeof(Word) * 8;
        const size_t total_bits = bits_per_block * total_blocks;
        const size_t alloc_size = total_bits / bits_per_word;
        return alloc_size * sizeof(Word);
    }

    sycl::id<3> get_max_grid_dims(const sycl::queue &q) {
#ifdef SYCL_EXT_ONEAPI_MAX_NUMBER_WORK_GROUPS
        return q.get_device().get_info<sycl::info::device::ext_oneapi_max_number_work_groups>();
#else
#pragma message "Missing SYCL information descriptor to check the max number of work groups allowed"
        return {(2**31)-1, (2**31)-1, (2**31)-1 };
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
            uint base = (uint) intpart;
            grid_size[2] = base;
            grid_size[1] = base;
            // figure out how many y to add
            uint rem = (size - base * base);
            uint y_rows = rem / base;
            if (rem % base != 0) y_rows++;
            grid_size[1] += y_rows;
        }

        if (dims == 3) {
            double cub_r = pow((double) grids, 1.f / 3.f);
            double intpart = 0;
            modf(cub_r, &intpart);
            uint base = (uint) intpart;
            grid_size[2] = base;
            grid_size[1] = base;
            grid_size[0] = base;
            // figure out how many z to add
            uint rem = (size - base * base * base);
            uint z_rows = rem / (base * base);
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
        return grid_size;
    }


// map two's complement signed integer to negabinary unsigned integer
    inline unsigned long long int int2uint(const long long int x) {
        return (x + (unsigned long long int) 0xaaaaaaaaaaaaaaaaull) ^ (unsigned long long int) 0xaaaaaaaaaaaaaaaaull;
    }

    inline unsigned int int2uint(const int x) {
        return (x + (unsigned int) 0xaaaaaaaau) ^ (unsigned int) 0xaaaaaaaau;
    }


    template<typename Int, typename Scalar>
    Scalar dequantize(const Int &x, const int &e);

    template<>
    double dequantize<long long int, double>(const long long int &x, const int &e) {
        return sycl::ldexp((double) x, e - (CHAR_BIT * scalar_sizeof<double>() - 2));
    }

    template<>
    float dequantize<int, float>(const int &x, const int &e) {
        return sycl::ldexp((float) x, e - (CHAR_BIT * scalar_sizeof<float>() - 2));
    }

    template<>
    int dequantize<int, int>(const int &x, const int &e) {
        return 1;
    }

    template<>
    long long int dequantize<long long int, long long int>(const long long int &x, const int &e) {
        return 1;
    }

/* inverse lifting transform of 4-vector */
    template<class Int, uint s>
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
    inline const unsigned char *get_perm();

    template<>
    inline const unsigned char *get_perm<64>() {
        return perm_3d;
    }

    template<>
    inline const unsigned char *get_perm<16>() {
        return perm_2;
    }

    template<>
    inline const unsigned char *get_perm<4>() {
        return perm_1;
    }


}

