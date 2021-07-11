#pragma once

#include "shared.h"
#include "encode.hpp"

#include <iostream>

namespace syclZFP {

    template<typename Scalar>
    inline void gather_partial1(Scalar *q, const Scalar *p, int nx, int sx) {
        uint x;
        for (x = 0; x < 4; x++)
            if (x < nx) q[x] = p[x * sx];
        pad_block(q, nx, 1);
    }

    template<typename Scalar>
    inline void gather1(Scalar *q, const Scalar *p, int sx) {
        uint x;
        for (x = 0; x < 4; x++, p += sx)
            *q++ = *p;
    }

    template<class Scalar, bool variable_rate>
    void syclEncode1(
            sycl::nd_item<3> item,
            int minbits,
            const int maxbits,
            const int maxprec,
            const int minexp,
            const Scalar *scalars,
            Word *stream,
            ushort *block_bits,
            const uint dim,
            const int sx,
            const uint padded_dim,
            const uint tot_blocks) {

        typedef unsigned long long int ull;
        typedef long long int ll;
        const ull block_idx = item.get_global_linear_id();

        if (block_idx >= tot_blocks) {
            // we can't launch the exact number of blocks
            // so just exit if this isn't real
            return;
        }

        uint block_dim;
        block_dim = padded_dim >> 2;

        // logical pos in 3d array
        uint block;
        block = (block_idx % block_dim) * 4;

        const ll offset = (ll) block * sx;

        Scalar fblock[ZFP_1D_BLOCK_SIZE];

        bool partial = false;
        if (block + 4 > dim) partial = true;

        if (partial) {
            uint nx = 4 - (padded_dim - dim);
            gather_partial1(fblock, scalars + offset, nx, sx);
        } else {
            gather1(fblock, scalars + offset, sx);
        }

        uint bits = zfp_encode_block<Scalar, ZFP_1D_BLOCK_SIZE>(fblock, minbits, maxbits, maxprec, minexp, block_idx, stream);
        if (variable_rate)
            block_bits[block_idx] = bits;
    }

    //
    // Launch the encode kernel
    //
    template<class Scalar, bool variable_rate>
    size_t encode1launch(
            sycl::queue &q,
            uint dim,
            int sx,
            const Scalar *d_data,
            Word *stream,
            ushort *d_block_bits,
            const int minbits,
            const int maxbits,
            const int maxprec,
            const int minexp) {
        const int preferred_block_size = 128;
        sycl::range<3> block_size(1, 1, preferred_block_size);

        uint zfp_pad(dim);
        if (zfp_pad % 4 != 0) zfp_pad += 4 - dim % 4;

        const uint zfp_blocks = (zfp_pad) / 4;
        //
        // we need to ensure that we launch a multiple of the block size
        //
        long int block_pad = 0;
        if (zfp_blocks % preferred_block_size != 0) {
            block_pad = preferred_block_size - zfp_blocks % preferred_block_size;
        }

        size_t total_blocks = block_pad + zfp_blocks;

        sycl::range<3> grid_size = calculate_global_work_size(q, total_blocks, preferred_block_size);

        size_t stream_bytes = calc_device_mem1d(zfp_pad, maxbits);
        // ensure we have zeros
        sycl::event init_e = q.memset(stream, 0, stream_bytes);

#ifdef SYCL_ZFP_RATE_PRINT
        auto before = std::chrono::steady_clock::now();
#endif
        sycl::nd_range<3> kernel_parameters(grid_size * block_size, block_size);
        q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(init_e);
            cgh.parallel_for(kernel_parameters, [=](sycl::nd_item<3> item) {
                syclEncode1<Scalar, variable_rate>
                        (item, minbits,
                         maxbits,
                         maxprec,
                         minexp,
                         d_data,
                         stream,
                         d_block_bits,
                         dim,
                         sx,
                         zfp_pad,
                         zfp_blocks);
            });
        }).wait();


#ifdef SYCL_ZFP_RATE_PRINT
        auto after = std::chrono::steady_clock::now();
        auto seconds = std::chrono::duration<double>(after - before).count();
        double gb = (float(dim) * float(sizeof(Scalar))) / (1024.f * 1024.f * 1024.f);
        double rate = gb / seconds;
        printf("Encode elapsed time: %.5f (s)\n", seconds);
        printf("# encode1 rate: %.2f (GB / sec) %d\n", rate, maxbits);
#endif
        return stream_bytes;
    }

    //
    // Encode a host vector and output an encoded device vector
    //
    template<class Scalar, bool variable_rate>
    size_t encode1(
            sycl::queue &q,
            int dim,
            int sx,
            Scalar *d_data,
            Word *stream,
            ushort *d_block_bits,
            const int minbits,
            const int maxbits,
            const int maxprec,
            const int minexp) {
        return encode1launch<Scalar, variable_rate>(q, dim, sx, d_data, stream, d_block_bits, minbits, maxbits, maxprec, minexp);
    }

}

