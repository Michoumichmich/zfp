#pragma once

#include "shared.h"
#include "decode.hpp"

namespace syclZFP {


    template<typename Scalar>
    inline void scatter_partial1(const Scalar *q, Scalar *p, int nx, int sx) {
        uint x;
        for (x = 0; x < 4; x++)
            if (x < nx) p[x * sx] = q[x];
    }

    template<typename Scalar>


    inline void scatter1(const Scalar *q, Scalar *p, int sx) {
        uint x;
        for (x = 0; x < 4; x++, p += sx)
            *p = *q++;
    }

    template<class Scalar>
    void syclDecode1(
            sycl::nd_item<3> item,
            Word *blocks,
            Scalar *out,
            const uint dim,
            const int stride,
            const uint padded_dim,
            const uint total_blocks,
            uint maxbits) {
        typedef unsigned long long int ull;
        typedef long long int ll;
        typedef typename syclZFP::zfp_traits<Scalar>::UInt UInt;
        typedef typename syclZFP::zfp_traits<Scalar>::Int Int;

        const int intprec = get_precision<Scalar>();

        const ull block_idx = item.get_global_linear_id();

        if (block_idx >= total_blocks) return;

        BlockReader<4> reader(blocks, maxbits, block_idx, total_blocks);
        Scalar result[4] = {0, 0, 0, 0};

        zfp_decode(reader, result, maxbits);

        uint block;
        block = block_idx * 4ull;
        const ll offset = (ll) block * stride;

        bool partial = false;
        if (block + 4 > dim) partial = true;
        if (partial) {
            const uint nx = 4u - (padded_dim - dim);
            scatter_partial1(result, out + offset, nx, stride);
        } else {
            scatter1(result, out + offset, stride);
        }
    }

    template<class Scalar>
    size_t decode1launch(
            sycl::queue &q,
            uint dim,
            int stride,
            Word *stream,
            Scalar *d_data,
            uint maxbits) {
        const int preferred_block_size = 128;

        uint zfp_pad(dim);
        if (zfp_pad % 4 != 0) zfp_pad += 4 - dim % 4;

        uint zfp_blocks = (zfp_pad) / 4;

        if (dim % 4 != 0) zfp_blocks = (dim + (4 - dim % 4)) / 4;

        size_t block_pad = 0;
        if (zfp_blocks % preferred_block_size != 0) {
            block_pad = preferred_block_size - zfp_blocks % preferred_block_size;
        }

        size_t total_blocks = block_pad + zfp_blocks;
        size_t stream_bytes = calc_device_mem1d(zfp_pad, maxbits);

        sycl::range<3> block_size(1, 1, preferred_block_size);
        sycl::range<3> grid_size = calculate_global_work_size(q, total_blocks, preferred_block_size);

#ifdef SYCL_ZFP_RATE_PRINT
        auto before = std::chrono::steady_clock::now();
#endif
        sycl::nd_range<3> kernel_parameters(grid_size * block_size, block_size);
        q.submit([&](sycl::handler &cgh) {
            cgh.parallel_for(kernel_parameters, [=](sycl::nd_item<3> item) {
                syclDecode1<Scalar>
                        (item,
                         stream,
                         d_data,
                         dim,
                         stride,
                         zfp_pad,
                         zfp_blocks, // total blocks to decode
                         maxbits);

            });
        }).wait();

#ifdef SYCL_ZFP_RATE_PRINT
        auto after = std::chrono::steady_clock::now();
        auto seconds = std::chrono::duration<double>(after - before).count();
        float rate = (float(dim) * sizeof(Scalar)) / seconds;
        rate /= 1024.f;
        rate /= 1024.f;
        rate /= 1024.f;
        printf("Decode elapsed time: %.5f (s)\n", seconds);
        printf("# decode1 rate: %.2f (GB / sec) %d\n", rate, maxbits);
#endif
        return stream_bytes;
    }

    template<class Scalar>
    size_t decode1(sycl::queue &q, int dim,
                   int stride,
                   Word *stream,
                   Scalar *d_data,
                   uint maxbits) {
        return decode1launch<Scalar>(q, dim, stride, stream, d_data, maxbits);
    }

} // namespace syclZFP

