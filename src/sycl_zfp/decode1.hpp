#pragma once

#include "shared.h"
#include "decode.hpp"

namespace syclZFP {
    template<class Scalar>
    class decode1_kernel;

    template<typename Scalar>
    inline void scatter_partial1(const Scalar *q, Scalar *p, int nx, int sx) {
        int x;
        for (x = 0; x < nx; x++, p += sx)
            *p = *q++;
    }

    template<typename Scalar>


    inline void scatter1(const Scalar *q, Scalar *p, int sx) {
        int x;
#pragma unroll
        for (x = 0; x < 4; x++, p += sx)
            *p = *q++;
    }

    template<class Scalar>
    void syclDecode1(
            const size_t &block_idx,
            Word *blocks,
            Scalar *out,
            const size_t dim,
            const int stride,
            const size_t padded_dim,
            const size_t total_blocks,
            int maxbits) {

        typedef typename syclZFP::zfp_traits<Scalar>::UInt UInt;
        typedef typename syclZFP::zfp_traits<Scalar>::Int Int;

        const int intprec = get_precision<Scalar>();

        if (block_idx >= total_blocks) return;

        BlockReader<4> reader(blocks, maxbits, block_idx, total_blocks);
        Scalar result[4] = {Scalar(0)};

        zfp_decode(reader, result, maxbits);

        size_t block = block_idx * 4ull;
        const int64_t offset = (int64_t) block * stride;

        bool partial = false;
        if (block + 4 > dim) partial = true;
        if (partial) {
            uint nx = 4 - (padded_dim - dim);
            scatter_partial1(result, out + offset, (int) nx, stride);
        } else {
            scatter1(result, out + offset, stride);
        }
    }

    template<class Scalar>
    size_t decode1launch(
            sycl::queue &q,
            size_t dim,
            int stride,
            Word *stream,
            Scalar *d_data,
            int maxbits) {
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

        sycl::nd_range<3> kernel_parameters(grid_size * block_size, block_size);
        auto e = q.submit([&](sycl::handler &cgh) {
            cgh.parallel_for<decode1_kernel<Scalar>>(kernel_parameters, [=](sycl::nd_item<3> item) {
                syclDecode1<Scalar>
                        (item.get_global_linear_id(),
                         stream,
                         d_data,
                         dim,
                         stride,
                         zfp_pad,
                         zfp_blocks, // total blocks to decode
                         maxbits);
            });
        });
        e.wait();

#ifdef SYCL_ZFP_RATE_PRINT
        double ns = e.template get_profiling_info<sycl::info::event_profiling::command_end>()
                    - e.template get_profiling_info<sycl::info::event_profiling::command_start>();
        auto seconds = ns / 1e9;
        double rate = (double(dim) * sizeof(Scalar)) / seconds;
        rate /= 1024.;
        rate /= 1024.;
        rate /= 1024.;
        printf("Decode elapsed time: %.5f (s)\n", seconds);
        printf("# decode1 rate: %.2f (GB / sec) %d\n", rate, maxbits);
#endif
        return stream_bytes;
    }

    template<class Scalar>
    size_t decode1(sycl::queue &q, size_t dim,
                   int stride,
                   Word *stream,
                   Scalar *d_data,
                   int maxbits) {
        return decode1launch<Scalar>(q, dim, stride, stream, d_data, maxbits);
    }

} // namespace syclZFP

