#pragma once

#include "shared.h"
#include "encode.hpp"

namespace syclZFP {

    template<typename Scalar>
    inline void gather_partial3(Scalar *q, const Scalar *p, int nx, int ny, int nz, int sx, int sy, int sz) {
        int x, y, z;
        for (z = 0; z < nz; z++, p += sz - ny * sy) {
            for (y = 0; y < ny; y++, p += sy - nx * sx) {
                for (x = 0; x < nx; x++, p += sx) {
                    q[16 * z + 4 * y + x] = *p;
                }
                pad_block(q + 16 * z + 4 * y, nx, 1);
            }
            for (x = 0; x < 4; x++) {
                pad_block(q + 16 * z + x, ny, 4);
            }
        }

        for (y = 0; y < 4; y++)
            for (x = 0; x < 4; x++)
                pad_block(q + 4 * y + x, nz, 16);
    }

    template<typename Scalar>
    inline void gather3(Scalar *q, const Scalar *p, int sx, int sy, int sz) {
        int x, y, z;
        for (z = 0; z < 4; z++, p += sz - 4 * sy)
            for (y = 0; y < 4; y++, p += sy - 4 * sx)
                for (x = 0; x < 4; x++, p += sx)
                    *q++ = *p;
    }

    template<class Scalar, bool variable_rate>
    void syclEncode3(
            const size_t block_idx,
            const int minbits,
            const int maxbits,
            const int maxprec,
            const int minexp,
            const Scalar *scalars,
            Word *stream,
            ushort *block_bits,
            const sycl::id<3> dims,
            const int3_t stride,
            const sycl::id<3> padded_dims,
            const size_t tot_blocks
            //sycl::stream os
    ) {

        if (block_idx >= tot_blocks) {
            // we can't launch the exact number of blocks
            // so just exit if this isn't real
            return;
        }

        sycl::id<3> block_dims = padded_dims >> 2;

        // logical pos in 3d array
        sycl::id<3> block;
        block[2] = (block_idx % block_dims[2]) * 4;
        block[1] = ((block_idx / block_dims[2]) % block_dims[1]) * 4;
        block[0] = (block_idx / (block_dims[2] * block_dims[1])) * 4;

        // default strides
        const ll offset = (ll) block[2] * stride.x + (ll) block[1] * stride.y + (ll) block[0] * stride.z;
        Scalar fblock[ZFP_3D_BLOCK_SIZE];

        bool partial = false;
        if (block[2] + 4 > dims[2]) partial = true;
        if (block[1] + 4 > dims[1]) partial = true;
        if (block[0] + 4 > dims[0]) partial = true;

        if (partial) {
            const uint nx = block[2] + 4 > dims[2] ? dims[2] - block[2] : 4;
            const uint ny = block[1] + 4 > dims[1] ? dims[1] - block[1] : 4;
            const uint nz = block[0] + 4 > dims[0] ? dims[0] - block[0] : 4;
            //os << "Partial Block " << block_idx << " offset " << offset << " dims " << dims[0] << " " << dims[1] << " " << dims[2] << '\n' << sycl::flush;
            gather_partial3(fblock, scalars + offset, (int) nx, (int) ny, (int) nz, stride.x, stride.y, stride.z);

        } else {
            //os << "Not partial Block " << block_idx << " offset " << offset << " dims " << dims[0] << " " << dims[1] << " " << dims[2] << '\n'<< sycl::flush;
            gather3(fblock, scalars + offset, stride.x, stride.y, stride.z);
        }

        int bits = zfp_encode_block<Scalar, ZFP_3D_BLOCK_SIZE>(fblock, minbits, maxbits, maxprec, minexp, block_idx, stream);
        if (variable_rate) {
            block_bits[block_idx] = bits;
        }

    }

    //
    // Launch the encode kernel
    //
    template<class Scalar, bool variable_rate>
    size_t encode3launch(
            sycl::queue &q,
            sycl::id<3> dims,
            int3_t stride,
            const Scalar *d_data,
            Word *stream,
            ushort *d_block_bits,
            const int minbits,
            const int maxbits,
            const int maxprec,
            const int minexp) {

        const int preferred_block_size = 128;
        sycl::range<3> block_size(1, 1, preferred_block_size);

        sycl::id<3> zfp_pad(dims[0], dims[1], dims[2]);
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

        sycl::range<3> grid_size = calculate_global_work_size(q, total_blocks, preferred_block_size);

        size_t stream_bytes = calc_device_mem3d(zfp_pad, maxbits);
        //ensure we start with 0s
        sycl::event init_e = q.memset(stream, 0, stream_bytes);

#ifdef SYCL_ZFP_RATE_PRINT
        auto before = std::chrono::steady_clock::now();
#endif
        sycl::nd_range<3> kernel_parameters(grid_size * block_size, block_size);
        q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(init_e);
            //sycl::stream os(10240, 2000, cgh);
            cgh.parallel_for(kernel_parameters, [=](sycl::nd_item<3> item) {
                syclEncode3<Scalar, variable_rate>
                        (item.get_global_linear_id(),
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
                         zfp_blocks
                                //,os
                        );

            });
        }).wait();


#ifdef SYCL_ZFP_RATE_PRINT
        auto after = std::chrono::steady_clock::now();
        auto seconds = std::chrono::duration<double>(after - before).count();
        double rate = (double(dims[2] * dims[1] * dims[0]) * sizeof(Scalar)) / seconds;
        rate /= 1024.;
        rate /= 1024.;
        rate /= 1024.;
        printf("Encode elapsed time: %.5f (s)\n", seconds);
        printf("# encode3 rate: %.2f (GB / sec) \n", rate);
#endif
        return stream_bytes;
    }


    //
    // Just pass the raw pointer to the "real" encode
    //
    template<class Scalar, bool variable_rate>
    size_t encode3(
            sycl::queue &q,
            sycl::id<3> dims,
            int3_t stride,
            Scalar *d_data,
            Word *stream,
            ushort *d_block_bits,
            const int minbits,
            const int maxbits,
            const int maxprec,
            const int minexp) {
        return encode3launch<Scalar, variable_rate>(q, dims, stride, d_data, stream, d_block_bits, minbits, maxbits, maxprec, minexp);
    }
}

