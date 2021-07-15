#pragma once

#include "shared.h"
#include "decode.hpp"

namespace syclZFP {

    template<typename Scalar>
    inline void scatter_partial3(const Scalar *q, Scalar *p, int nx, int ny, int nz, int sx, int sy, int sz) {
        int x, y, z;
        for (z = 0; z < nz; z++, p += sz - (ptrdiff_t) ny * sy, q += 4 * (4 - ny))
            for (y = 0; y < ny; y++, p += sy - (ptrdiff_t) nx * sx, q += 1 * (4 - nx))
                for (x = 0; x < nx; x++, p += sx, q++)
                    *p = *q;
    }

    template<typename Scalar>
    inline void scatter3(const Scalar *q, Scalar *p, int sx, int sy, int sz) {
        uint x, y, z;
        for (z = 0; z < 4; z++, p += sz - 4 * sy)
            for (y = 0; y < 4; y++, p += sy - 4 * sx)
                for (x = 0; x < 4; x++, p += sx)
                    *p = *q++;
    }


    template<class Scalar, int BlockSize>
    void syclDecode3(
            sycl::nd_item<3> item,
            const Word *blocks,
            Scalar *out,
            const sycl::uint3 dims,
            const sycl::int3 stride,
            const sycl::id<3> padded_dims,
            uint maxbits) {

        typedef unsigned long long int ull;
        typedef long long int ll;
        const uint block_idx = item.get_global_linear_id();


        const long unsigned int total_blocks = (padded_dims[2] * padded_dims[1] * padded_dims[0]) / 64;

        if (block_idx >= total_blocks) {
            return;
        }

        BlockReader<BlockSize> reader(blocks, maxbits, block_idx, total_blocks);

        Scalar result[BlockSize];
        memset(result, 0, sizeof(Scalar) * BlockSize);

        zfp_decode<Scalar, BlockSize>(reader, result, maxbits);

        // logical block dims
        sycl::id<3> block_dims;
        block_dims[2] = padded_dims[2] >> 2;
        block_dims[1] = padded_dims[1] >> 2;
        block_dims[0] = padded_dims[0] >> 2;
        // logical pos in 3d array
        sycl::id<3> block;
        block[2] = (block_idx % block_dims[2]) * 4;
        block[1] = ((block_idx / block_dims[2]) % block_dims[1]) * 4;
        block[0] = (block_idx / (block_dims[2] * block_dims[1])) * 4;

        // default strides
        const ll offset = (ll) block[2] * stride[2] + (ll) block[1] * stride[1] + (ll) block[0] * stride[0];

        bool partial = false;
        if (block[2] + 4 > dims[2]) partial = true;
        if (block[1] + 4 > dims[1]) partial = true;
        if (block[0] + 4 > dims[0]) partial = true;
        if (partial) {
            const uint nx = block[2] + 4u > dims[2] ? dims[2] - block[2] : 4;
            const uint ny = block[1] + 4u > dims[1] ? dims[1] - block[1] : 4;
            const uint nz = block[0] + 4u > dims[0] ? dims[0] - block[0] : 4;

            scatter_partial3(result, out + offset, nx, ny, nz, stride[2], stride[1], stride[0]);
        } else {
            scatter3(result, out + offset, stride[2], stride[1], stride[0]);
        }
    }

    template<class Scalar>
    size_t decode3launch(sycl::queue &q, sycl::uint3 dims, sycl::int3 stride, Word *stream, Scalar *d_data, uint maxbits) {
        const int preferred_block_size = 128;
        sycl::range<3> block_size(1, 1, preferred_block_size);

        sycl::id<3> zfp_pad(dims[0], dims[1], dims[2]);
        // ensure that we have block sizes
        // that are a multiple of 4
        if (zfp_pad[2] % 4 != 0) zfp_pad[2] += 4 - dims[2] % 4;
        if (zfp_pad[1] % 4 != 0) zfp_pad[1] += 4 - dims[1] % 4;
        if (zfp_pad[0] % 4 != 0) zfp_pad[0] += 4 - dims[0] % 4;

        const size_t zfp_blocks = (zfp_pad[2] * zfp_pad[1] * zfp_pad[0]) / 64;

        // we need to ensure that we launch a multiple of the block size
        size_t block_pad = 0;
        if (zfp_blocks % preferred_block_size != 0) {
            block_pad = preferred_block_size - zfp_blocks % preferred_block_size;
        }

        size_t total_blocks = block_pad + zfp_blocks;
        size_t stream_bytes = calc_device_mem3d(zfp_pad, maxbits);
        sycl::range<3> grid_size = calculate_global_work_size(q, total_blocks, preferred_block_size);

#ifdef SYCL_ZFP_RATE_PRINT
        auto before = std::chrono::steady_clock::now();
#endif

        sycl::nd_range<3> kernel_parameters(grid_size * block_size, block_size);
        q.submit([&](sycl::handler &cgh) {
            cgh.parallel_for(kernel_parameters, [=](sycl::nd_item<3> item) {
                syclDecode3<Scalar, 64>
                        (item,
                         stream,
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
        float rate = (float(dims[2] * dims[1] * dims[0]) * sizeof(Scalar)) / seconds;
        rate /= 1024.f;
        rate /= 1024.f;
        rate /= 1024.f;
        printf("Decode elapsed time: %.5f (s)\n", seconds);
        printf("# decode3 rate: %.2f (GB / sec) %d\n", rate, maxbits);
#endif

        return stream_bytes;
    }

    template<class Scalar>
    size_t decode3(sycl::queue &q, sycl::uint3 dims, sycl::int3 stride, Word *stream, Scalar *d_data, uint maxbits) {
        return decode3launch<Scalar>(q, dims, stride, stream, d_data, maxbits);
    }

} // namespace syclZFP

