#pragma once

#include "shared.h"
#include "encode.hpp"

#include <iostream>

namespace syclZFP {

    template<class Scalar, bool variable_rate>
    class encode1_kernel;


    template<typename Scalar>
    inline void gather_partial1(Scalar *q, const Scalar *p, int nx, int sx) {
        for (int x = 0; x < nx; x++, p += sx) {
            q[x] = *p;
        }
        pad_block<1>(q, nx);
    }

    template<typename Scalar>
    inline void gather1(Scalar *q, const Scalar *p, int sx) {
#pragma unroll
        for (int x = 0; x < 4; x++, p += sx) {
            *q++ = *p;
        }
    }

    template<class Scalar, bool variable_rate>
    void syclEncode1(
            const size_t &block_idx,
            int minbits,
            const int &maxbits,
            const int &maxprec,
            const int &minexp,
            const Scalar *scalars,
            Word *stream,
            ushort *block_bits,
            const size_t &dim,
            const int &sx,
            const size_t &padded_dim,
            const size_t &tot_blocks) {

        if (block_idx >= tot_blocks) {
            // we can't launch the exact number of blocks
            // so just exit if this isn't real
            return;
        }

        size_t block_dim = padded_dim >> 2;

        // logical pos in 3d array
        size_t block = (block_idx % block_dim) * 4;

        const int64_t offset = (int64_t) block * sx;

        Scalar fblock[ZFP_1D_BLOCK_SIZE];

        bool partial = false;
        if (block + 4 > dim) partial = true;

        if (partial) {
            uint nx = 4 - (padded_dim - dim);
            gather_partial1(fblock, scalars + offset, (int) nx, sx);
        } else {
            gather1(fblock, scalars + offset, sx);
        }

        auto bits = zfp_encode_block<Scalar, ZFP_1D_BLOCK_SIZE>(fblock, minbits, maxbits, maxprec, minexp, block_idx, stream);
        if (variable_rate) {
            block_bits[block_idx] = bits;
        }
    }

    //
    // Launch the encode kernel
    //
    template<class Scalar, bool variable_rate>
    size_t encode1launch(
            sycl::queue &q,
            size_t dim,
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

        size_t zfp_pad(dim);
        if (zfp_pad % 4 != 0) zfp_pad += 4 - dim % 4;

        const size_t zfp_blocks = (zfp_pad) / 4;
        //
        // we need to ensure that we launch a multiple of the block size
        //
        size_t block_pad = 0;
        if (zfp_blocks % preferred_block_size != 0) {
            block_pad = preferred_block_size - zfp_blocks % preferred_block_size;
        }

        size_t total_blocks = block_pad + zfp_blocks;

        sycl::range<3> grid_size = calculate_global_work_size(q, total_blocks, preferred_block_size);

        size_t stream_bytes = calc_device_mem1d(zfp_pad, maxbits);
        // ensure we have zeros
        q.memset(stream, 0, stream_bytes).wait();

        sycl::nd_range<3> kernel_parameters(grid_size * block_size, block_size);
        auto e = q.submit([&](sycl::handler &cgh) {
            cgh.parallel_for<encode1_kernel<Scalar, variable_rate>>(kernel_parameters, [=](sycl::nd_item<3> item) {
                syclEncode1<Scalar, variable_rate>
                        (item.get_global_linear_id(),
                         minbits,
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
        });
        e.wait();

#ifdef SYCL_ZFP_RATE_PRINT
        double ns = e.template get_profiling_info<sycl::info::event_profiling::command_end>()
                    - e.template get_profiling_info<sycl::info::event_profiling::command_start>();
        auto seconds = ns / 1e9;
        double gb = (double(dim) * double(sizeof(Scalar))) / (1024. * 1024. * 1024.);
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
            size_t dim,
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

