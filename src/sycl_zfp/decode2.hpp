#pragma once

#include "shared.h"
#include "decode.hpp"

namespace syclZFP {

    template<typename Scalar>
    inline void scatter_partial2(const Scalar *q, Scalar *p, int nx, int ny, int sx, int sy) {
        uint x, y;
        for (y = 0; y < 4; y++)
            if (y < ny) {
                for (x = 0; x < 4; x++)
                    if (x < nx) {
                        *p = q[4 * y + x];
                        p += sx;
                    }
                p += sy - nx * sx;
            }
    }

    template<typename Scalar>
    inline void scatter2(const Scalar *q, Scalar *p, int sx, int sy) {
        uint x, y;
        for (y = 0; y < 4; y++, p += sy - 4 * sx)
            for (x = 0; x < 4; x++, p += sx)
                *p = *q++;
    }


    template<class Scalar, int BlockSize>
    void syclDecode2(
            sycl::nd_item<3> item,
            const Word *blocks,
            Scalar *out,
            const sycl::uint2 dims,
            const sycl::int2 stride,
            const sycl::id<2> padded_dims,
            uint maxbits) {
        typedef unsigned long long int ull;
        typedef long long int ll;
        const ull block_idx = item.get_global_linear_id();

        size_t total_blocks = (padded_dims[1] * padded_dims[0]) / 16;

        if (block_idx >= total_blocks) {
            return;
        }

        BlockReader<BlockSize> reader(blocks, maxbits, block_idx, total_blocks);

        Scalar result[BlockSize];
        memset(result, 0, sizeof(Scalar) * BlockSize);

        zfp_decode(reader, result, maxbits);

        // logical block dims
        sycl::uint2 block_dims;
        block_dims[1] = padded_dims[1] >> 2;
        block_dims[0] = padded_dims[0] >> 2;
        // logical pos in 3d array
        sycl::uint2 block;
        block[1] = (block_idx % block_dims[1]) * 4;
        block[0] = ((block_idx / block_dims[1]) % block_dims[0]) * 4;

        const ll offset = (ll) block[1] * stride[1] + (ll) block[0] * stride[0];

        bool partial = false;
        if (block[1] + 4 > dims[1]) partial = true;
        if (block[0] + 4 > dims[0]) partial = true;
        if (partial) {
            const uint nx = block[1] + 4 > dims[1] ? dims[1] - block[1] : 4;
            const uint ny = block[0] + 4 > dims[0] ? dims[0] - block[0] : 4;
            scatter_partial2(result, out + offset, nx, ny, stride[1], stride[0]);
        } else {
            scatter2(result, out + offset, stride[1], stride[0]);
        }
    }

    template<class Scalar>
    size_t decode2launch(sycl::queue &q, sycl::uint2 dims, sycl::int2 stride, Word *stream, Scalar *d_data, uint maxbits) {
        const int preferred_block_size = 128;
        sycl::range<3> block_size(1, 1, preferred_block_size);

        sycl::id<2> zfp_pad(dims[0], dims[1]);
        // ensure that we have block sizes
        // that are a multiple of 4
        if (zfp_pad[1] % 4 != 0) zfp_pad[1] += 4 - dims[1] % 4;
        if (zfp_pad[0] % 4 != 0) zfp_pad[0] += 4 - dims[0] % 4;

        const size_t zfp_blocks = (zfp_pad[1] * zfp_pad[0]) / 16;

        // we need to ensure that we launch a multiple of the block size
        size_t block_pad = 0;
        if (zfp_blocks % preferred_block_size != 0) {
            block_pad = preferred_block_size - zfp_blocks % preferred_block_size;
        }

        size_t stream_bytes = calc_device_mem2d(zfp_pad, maxbits);
        size_t total_blocks = block_pad + zfp_blocks;
        sycl::range<3> grid_size = calculate_global_work_size(q, total_blocks, preferred_block_size);

#ifdef SYCL_ZFP_RATE_PRINT
        auto before = std::chrono::steady_clock::now();
#endif
        sycl::nd_range<3> kernel_parameters(grid_size * block_size, block_size);
        q.submit([&](sycl::handler &cgh) {
            cgh.parallel_for(kernel_parameters, [=](sycl::nd_item<3> item) {
                syclDecode2<Scalar, 16>
                        (item, stream,
                         d_data,
                         dims,
                         stride,
                         zfp_pad,
                         maxbits);
            });
        }).wait();


#ifdef SYCL_ZFP_RATE_PRINT
        auto after = std::chrono::steady_clock::now();
        auto seconds = std::chrono::duration<double>(after - before).count();
        float rate = (float(dims[1] * dims[0]) * sizeof(Scalar)) / seconds;
        rate /= 1024.f;
        rate /= 1024.f;
        rate /= 1024.f;
        printf("Decode elapsed time: %.5f (s)\n", seconds);
        printf("# decode2 rate: %.2f (GB / sec) %d\n", rate, maxbits);
#endif
        return stream_bytes;
    }

    template<class Scalar>
    size_t decode2(sycl::queue &q, sycl::uint2 dims, sycl::int2 stride, Word *stream, Scalar *d_data, uint maxbits) {
        return decode2launch<Scalar>(q, dims, stride, stream, d_data, maxbits);
    }

} // namespace syclZFP


