#pragma once

#include "shared.h"
#include "encode.hpp"

namespace syclZFP {
    template<class Scalar, bool variable_rate>
    class encode2_kernel;


    template<typename Scalar>
    inline void gather_partial2(Scalar *q, const Scalar *p, int nx, int ny, int sx, int sy) {
        int x, y;
        for (y = 0; y < ny; y++, p += sy - nx * sx) {
            for (x = 0; x < nx; x++, p += sx)
                q[4 * y + x] = *p;
            pad_block(q + 4 * y, nx, 1);
        }
#pragma unroll
        for (x = 0; x < 4; x++)
            pad_block(q + x, ny, 4);
    }

    template<typename Scalar>
    inline void gather2(Scalar *q, const Scalar *p, int sx, int sy) {
        int x, y;
#pragma unroll
        for (y = 0; y < 4; y++, p += sy - 4 * sx)
#pragma unroll
                for (x = 0; x < 4; x++, p += sx)
                    *q++ = *p;
    }

    template<class Scalar, bool variable_rate>
    void syclEncode2(
            const sycl::nd_item<3> &item,
            const const_perm_accessor &acc,
            const int &minbits,
            const int &maxbits,
            const int &maxprec,
            const int &minexp,
            const Scalar *scalars,
            Word *stream,
            ushort *block_bits,
            const sycl::id<2> &dims,
            const int64_2_t &stride,
            const sycl::id<2> &padded_dims,
            const size_t &tot_blocks) {

        const size_t block_idx = item.get_global_linear_id();

        if (block_idx >= tot_blocks) {
            // we can't launch the exact number of blocks
            // so just exit if this isn't real
            return;
        }

        sycl::id<2> block_dims = padded_dims >> 2;

        // logical pos in 3d array
        sycl::id<2> block;
        block[1] = (block_idx % block_dims[1]) * 4; //X
        block[0] = ((block_idx / block_dims[0]) % block_dims[0]) * 4; //Y

        const int64_t offset = (int64_t) block[1] * stride.x + (int64_t) block[0] * stride.y;

        Scalar fblock[ZFP_2D_BLOCK_SIZE];

        bool partial = false;
        if (block[1] + 4 > dims[1]) partial = true;
        if (block[0] + 4 > dims[0]) partial = true;

        if (partial) {
            const uint nx = block[1] + 4 > dims[1] ? dims[1] - block[1] : 4;
            const uint ny = block[0] + 4 > dims[0] ? dims[0] - block[0] : 4;
            gather_partial2(fblock, scalars + offset, (int) nx, (int) ny, stride.x, stride.y);

        } else {
            gather2(fblock, scalars + offset, stride.x, stride.y);
        }

        auto bits = zfp_encode_block<Scalar, ZFP_2D_BLOCK_SIZE>(acc, fblock, minbits, maxbits, maxprec, minexp, block_idx, stream);
        if constexpr (variable_rate) {
            block_bits[block_idx] = bits;
        }

    }

//
// Launch the encode kernel
//
    template<class Scalar, bool variable_rate>
    size_t encode2launch(
            sycl::queue &q,
            sycl::id<2> dims,
            int64_2_t stride,
            const Scalar *d_data,
            Word *stream,
            ushort *d_block_bits,
            const int minbits,
            const int maxbits,
            const int maxprec,
            const int minexp) {
        const int preferred_block_size = 128;
        sycl::range<3> block_size(1, 1, preferred_block_size);

        sycl::id<2> zfp_pad(dims[0], dims[1]);
        if (zfp_pad[1] % 4 != 0) zfp_pad[1] += 4 - dims[1] % 4;
        if (zfp_pad[0] % 4 != 0) zfp_pad[0] += 4 - dims[0] % 4;

        const size_t zfp_blocks = (zfp_pad[1] * zfp_pad[0]) / 16;

        //
        // we need to ensure that we launch a multiple of the block size
        //
        size_t block_pad = 0;
        if (zfp_blocks % preferred_block_size != 0) {
            block_pad = preferred_block_size - zfp_blocks % preferred_block_size;
        }

        size_t total_blocks = block_pad + zfp_blocks;

        sycl::range<3> grid_size = calculate_global_work_size(q, total_blocks, preferred_block_size);

        size_t stream_bytes = calc_device_mem2d(zfp_pad, maxbits);
        // ensure we have zeros
        q.memset(stream, 0, stream_bytes).wait();

        sycl::nd_range<3> kernel_parameters(grid_size * block_size, block_size);
        auto buf = get_perm_buffer<16>();
        auto e = q.submit([&](sycl::handler &cgh) {
            auto acc = buf.get_access<sycl::access::mode::read, sycl::access::target::constant_buffer>(cgh);
            cgh.parallel_for<encode2_kernel<Scalar, variable_rate>>(kernel_parameters, [=](sycl::nd_item<3> item) {
                syclEncode2<Scalar, variable_rate>
                        (item,
                         acc,
                         minbits,
                         maxbits,
                         maxprec,
                         minexp,
                         d_data,
                         stream,
                         d_block_bits,
                         dims,
                         stride,
                         zfp_pad,
                         zfp_blocks);

            });
        });
        e.wait();


#ifdef SYCL_ZFP_RATE_PRINT
        double ns = e.template get_profiling_info<sycl::info::event_profiling::command_end>()
                    - e.template get_profiling_info<sycl::info::event_profiling::command_start>();
        auto seconds = ns / 1e9;
        double mb = (double(dims[1] * dims[0]) * sizeof(Scalar)) / (1024. * 1024. * 1024.);
        double rate = mb / seconds;
        printf("Encode elapsed time: %.5f (s)\n", seconds);
        printf("# encode2 rate: %.2f (GB / sec) %d\n", rate, maxbits);
#endif
        return stream_bytes;
    }

    template<class Scalar, bool variable_rate>
    size_t encode2(
            sycl::queue &q,
            sycl::id<2> dims,
            int64_2_t stride,
            Scalar *d_data,
            Word *stream,
            ushort *d_block_bits,
            const int minbits,
            const int maxbits,
            const int maxprec,
            const int minexp) {
        return encode2launch<Scalar, variable_rate>(q, dims, stride, d_data, stream, d_block_bits, minbits, maxbits, maxprec, minexp);
    }
}

