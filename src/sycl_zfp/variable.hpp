#pragma once

#include "sycl_intrinsics/intrinsics.hpp"
#include "sycl_intrinsics/cooperative_groups.hpp"
#include "sycl_intrinsics/parallel_primitives/scan_cooperative.hpp"
#include "shared.h"

namespace syclZFP {

    /**
     * Copy a chunk of 16-bit stream lengths into the 64-bit offsets array
     * to compute prefix sums. The first value in offsets is the "base" of the prefix sum
     */
    void copy_length(size_t index, const ushort *length, size_t *offsets, size_t first_stream, uint nstreams_chunk) {
        if (index >= nstreams_chunk) {
            return;
        }
        offsets[index + 1] = length[first_stream + index];
    }

    class copy_length_launch_kernel;

    void copy_length_launch(sycl::queue &q, ushort *bitlengths, size_t *chunk_offsets, size_t first, uint nstreams_chunk) {
        size_t work_group_count = (nstreams_chunk - 1) / 1024 + 1;
        size_t work_group_size = 1024;
        auto kernel_range = sycl::nd_range<1>(sycl::range<1>(work_group_count * work_group_size), sycl::range<1>(work_group_size));
        q.parallel_for<copy_length_launch_kernel>(kernel_range, [=](sycl::nd_item<1> item) {
            copy_length(item.get_global_linear_id(), bitlengths, chunk_offsets, first, nstreams_chunk);
        }).wait();
    }

    // *******************************************************************************

    // Each tile loads the compressed but uncompacted data to shared memory.
    // Input alignment can be anything (1-bit) as maxbits is not always a multiple of 8,
    // so the data is aligned on the fly (first bit of the bitstream on bit 0 in shared memory)
    template<uint tile_size>
    inline void load_to_shared(
            const sycl::nd_item<2> &item,
            const uint *streams,                     // Input data
            uint *sm,                                // Shared memory
            const size_t &offset_bits,               // Offset in bits for the stream
            const size_t &length_bits,               // Length in bits for this stream
            const size_t &maxpad32)                  // Next multiple of 32 of maxbits
    {
        uint misaligned = offset_bits & 31;
        unsigned long long offset_32 = offset_bits / 32;
        for (size_t i = item.get_local_id(1); i * 32 < length_bits; i += tile_size) {
            // Align even if already aligned
            uint low = streams[offset_32 + i];
            uint high = 0;
            if ((i + 1) * 32 < misaligned + length_bits) {
                high = streams[offset_32 + i + 1];
            }
            sm[item.get_local_id(0) * maxpad32 + i] = sycl::ext::funnelshift_r(low, high, misaligned);
        }
    }

    // Read the input bitstreams from shared memory, align them relative to the
    // final output alignment, compact all the aligned bitstreams in sm_out,
    // then write all the data (coalesced) to global memory, using atomics only
    // for the first and last elements
    template<int tile_size, int num_tiles>
    inline void process(
            const sycl::nd_item<2> &item,
            bool valid_stream,
            size_t &offset0,     // Offset in bits of the first bitstream of the block
            const size_t offset, // Offset in bits for this stream
            const size_t &length_bits,          // length of this stream
            const size_t &add_padding,          // padding at the end of the block, in bits
            const size_t &tid,                  // global thread index inside the thread block
            const uint *sm_in,                  // shared memory containing the compressed input data
            uint *sm_out,                       // shared memory to stage the compacted compressed data
            size_t maxpad32,                    // Leading dimension of the shared memory (padded maxbits)
            uint *sm_length,                    // shared memory to compute a prefix-sum inside the block
            uint *output)                       // output pointer
    {
        // All streams in the block will align themselves on the first stream of the block
        size_t misaligned0 = offset0 & 31;
        size_t misaligned = offset & 31;
        size_t off_smin = item.get_local_id(0) * (size_t) maxpad32;
        int off_smout = ((int) offset - (int) offset0 + (int) misaligned0) / 32;
        offset0 /= 32;

        if (valid_stream) {
            // Loop on the whole bitstream (including misalignment), 32 bits per thread
            for (size_t i = item.get_local_id(1); i * 32 < misaligned + length_bits; i += tile_size) {
                // Merge 2 values to create an aligned value
                uint v0 = i > 0 ? sm_in[off_smin + i - 1] : 0;
                uint v1 = sm_in[off_smin + i];
                v1 = sycl::ext::funnelshift_l(v0, v1, misaligned);

                // Mask out neighbor bitstreams
                uint mask = 0xffffffff;
                if (i == 0) {
                    mask &= 0xffffffff << misaligned;
                }
                if ((i + 1) * 32 > misaligned + length_bits) {
                    mask &= ~(0xffffffff << ((misaligned + length_bits) & 31));
                }
                auto addr = sm_out + off_smout + i;
                sycl::atomic_ref<uint, sycl::memory_order::relaxed, sycl::memory_scope::work_group, sycl::access::address_space::local_space> ref(*addr);
                ref += v1 & mask;
            }
        }

        // First thread working on each bistream writes the length in shared memory
        // Add zero-padding bits if needed (last bitstream of last chunk)
        // The extra bits in shared mempory are already zeroed.
        if (item.get_local_id(1) == 0) {
            sm_length[item.get_local_id(0)] = length_bits + add_padding;
        }

        // This synchthreads protects sm_out and sm_length.
        item.barrier(sycl::access::fence_space::local_space);

        // Compute total length for the threadblock
        uint total_length = 0;
        for (size_t i = tid & 31; i < num_tiles; i += 32) {
            total_length += sm_length[i];
        }
        for (size_t i = 1; i < item.get_sub_group().get_local_range().size(); i *= 2) {
            total_length += sycl::permute_group_by_xor(item.get_sub_group(), total_length, i);
        }

        // Write the shared memory output data to global memory, using all the threads
        for (size_t i = tid; i * 32 < misaligned0 + total_length; i += tile_size * num_tiles) {
            // Mask out the beginning and end of the block if unaligned
            uint mask = 0xffffffff;
            if (i == 0) {
                mask &= 0xffffffff << misaligned0;
            }

            if ((i + 1) * 32 > misaligned0 + total_length) {
                mask &= ~(0xffffffff << ((misaligned0 + total_length) & 31));
            }
            // Reset the shared memory to zero for the next iteration.
            uint value = sm_out[i];
            sm_out[i] = 0;
            // Write to global memory. Use atomicCAS for partially masked values
            // Working in-place, the output buffer has not been memset to zero
            if (mask == 0xffffffff)
                output[offset0 + i] = value;
            else {
                uint assumed, old = output[offset0 + i];
                sycl::atomic_ref<uint,
                        sycl::memory_order::relaxed,
                        sycl::memory_scope::work_group,
#ifdef SYCL_IMPLEMENTATION_ONEAPI
                        sycl::access::address_space::ext_intel_global_device_space
#else
                        sycl::access::address_space::global_device_space
#endif
                > ref(output[offset0 + i]);
                do {
                    assumed = old;
                    old = assumed;
                    ref.compare_exchange_strong(old, (assumed & ~mask) + (value & mask));
                } while (assumed != old);
            }
        }
    }


    // In-place bitstream concatenation: compacting blocks containing different number
    // of bits, with the input blocks stored in bins of the same size
    // Using a 2D tile of threads,
    // threadIdx.y = Index of the stream
    // threadIdx.x = Threads working on the same stream
    // Must launch dim3(tile_size, num_tiles, 1) threads per block.
    // Offset has a length of (nstreams_chunk + 1), offsets[0] is the offset in bits
    // where stream 0 starts, it must be memset to zero before launching the very first chunk,
    // and is updated at the end of this kernel.

    //   __launch_bounds__(tile_size *num_tiles)
    template<int tile_size, int num_tiles>
    void concat_bitstreams_chunk(
            const sycl::nd_item<2> &item,
            nd_range_barrier<2> *grid_barrier,
            uint *__restrict__ streams,
            size_t *__restrict__ offsets,
            size_t first_stream_chunk,
            size_t nstreams_chunk,
            bool last_chunk,
            size_t maxbits,
            size_t maxpad32,
            uint *sm_in,
            uint *sm_length) {

        uint *sm_out = sm_in + num_tiles * maxpad32;
        size_t tid = item.get_local_linear_id();
        size_t grid_stride = item.get_group_range(1) * num_tiles;
        size_t first_bitstream_block = item.get_group(1) * num_tiles;
        size_t my_stream = first_bitstream_block + item.get_local_id(0);

        // Zero the output shared memory. Will be reset again inside process().
        for (size_t i = tid; i < num_tiles * maxpad32 + 2; i += tile_size * num_tiles) {
            sm_out[i] = 0;
        }


        // Loop on all the bitstreams of the current chunk, using the whole resident grid.
        // All threads must enter this loop, as they have to synchronize inside.
        for (size_t i = 0; i < nstreams_chunk; i += grid_stride) {
            bool valid_stream = my_stream + i < nstreams_chunk;
            bool active_thread_block = first_bitstream_block + i < nstreams_chunk;
            size_t offset0;
            size_t offset = 0;
            uint length_bits = 0;
            uint add_padding = 0;
            if (active_thread_block) {
                offset0 = offsets[first_bitstream_block + i];
            }

            if (valid_stream) {
                offset = offsets[my_stream + i];
                size_t offset_bits = (first_stream_chunk + my_stream + i) * maxbits;
                size_t next_offset_bits = offsets[my_stream + i + 1];
                length_bits = (uint)(next_offset_bits - offset);
                load_to_shared<tile_size>(item, streams, sm_in, offset_bits, length_bits, maxpad32);
                if (last_chunk && (my_stream + i == nstreams_chunk - 1)) {
                    uint partial = next_offset_bits & 63;
                    add_padding = (64 - partial) & 63;
                }
            }

            // Check if there is overlap between input and output at the grid level.
            // Grid sync if needed, otherwise just syncthreads to protect the shared memory.
            // All the threads launched must participate in a grid::sync
            size_t last_stream = std::min(nstreams_chunk, i + grid_stride);
            size_t writing_to = (offsets[last_stream] + 31) / 32;
            size_t reading_from = (first_stream_chunk + i) * maxbits;
            if (writing_to >= reading_from) {
                grid_barrier->wait(item);
            } else {
                item.barrier(sycl::access::fence_space::local_space);
            }

            // Compact the shared memory data of the whole thread block and write it to global memory
            if (active_thread_block) {
                process<tile_size, num_tiles>(item, valid_stream, offset0, offset, length_bits, add_padding, tid, sm_in, sm_out, maxpad32, sm_length, streams);
            }

        }

        // Reset the base of the offsets array, for the next chunk's prefix sum
        if (item.get_group(1) == 0 && tid == 0) {
            offsets[0] = offsets[nstreams_chunk];
        }
    }


    template<int tile_size, int num_tiles>
    struct chunk_process_launch_kernel;

    template<int tile_size, int num_tiles>
    void dispatch_chunk_kernel(
            sycl::queue &q,
            uint *streams,
            size_t *chunk_offsets,
            size_t first,
            uint nstream_chunk,
            bool last_chunk,
            uint nbitsmax,
            uint num_sm) {
        using kernel_name = chunk_process_launch_kernel<tile_size, num_tiles>;
        uint maxpad32 = (nbitsmax + 31) / 32;
        uint max_blocks = num_sm;
        size_t shmem_count = (2 * num_tiles * maxpad32 + 2);
        max_blocks = std::min(nstream_chunk, max_blocks);
        sycl::range<2> threads(num_tiles, tile_size);
        sycl::range<2> grid_dim(1, max_blocks);
        sycl::nd_range<2> kernel_parameters(threads * grid_dim, threads);

        auto barrier = nd_range_barrier<2>::make_barrier(q, kernel_parameters);

        q.submit([&](sycl::handler &cgh) {
            sycl::accessor<uint, 1, sycl::access::mode::read_write, sycl::target::local> sm_in(shmem_count, cgh);
            sycl::accessor<uint, 1, sycl::access::mode::read_write, sycl::target::local> sm_length(num_tiles, cgh);
            cgh.parallel_for<kernel_name>(kernel_parameters, [=](sycl::nd_item<2> it) {
                auto sm_in_ptr = sm_in.get_pointer();
                auto sm_length_ptr = sm_length.get_pointer();
                concat_bitstreams_chunk<tile_size, num_tiles>(it, barrier, streams, chunk_offsets, first, nstream_chunk, last_chunk, nbitsmax, maxpad32, sm_in_ptr, sm_length_ptr);
            });
        }).wait();
    }


    void chunk_process_launch(
            sycl::queue &q,
            uint *streams,
            size_t *chunk_offsets,
            size_t first,
            uint nstream_chunk,
            bool last_chunk,
            uint nbitsmax,
            uint num_sm) {

        // Increase the number of threads per ZFP block ("tile") as nbitsmax increases
        // Compromise between coalescing, inactive threads and shared memory size <= 48 KiB
        // Total shared memory used = (2 * num_tiles * maxpad + 2) x 32-bit dynamic shared memory
        // and num_tiles x 32-bit static shared memory.
        // The extra 2 elements of dynamic shared memory are needed to handle unaligned output data
        // and potential zero-padding to the next multiple of 64 bits.
        // Block sizes set so that the shared memory stays < 48 KiB.
        if (nbitsmax <= 352) {
            constexpr
            size_t tile_size = 1;
            constexpr
            size_t num_tiles = 512;
            dispatch_chunk_kernel<tile_size, num_tiles>(q, streams, chunk_offsets, first, nstream_chunk, last_chunk, nbitsmax, num_sm);
        } else if (nbitsmax <= 1504) {
            constexpr
            size_t tile_size = 4;
            constexpr
            size_t num_tiles = 128;
            dispatch_chunk_kernel<tile_size, num_tiles>(q, streams, chunk_offsets, first, nstream_chunk, last_chunk, nbitsmax, num_sm);
        } else if (nbitsmax <= 6112) {
            constexpr
            size_t tile_size = 16;
            constexpr
            size_t num_tiles = 32;
            dispatch_chunk_kernel<tile_size, num_tiles>(q, streams, chunk_offsets, first, nstream_chunk, last_chunk, nbitsmax, num_sm);
        } else {
            constexpr
            size_t tile_size = 64;
            constexpr
            size_t num_tiles = 8;
            dispatch_chunk_kernel<tile_size, num_tiles>(q, streams, chunk_offsets, first, nstream_chunk, last_chunk, nbitsmax, num_sm);
        }
    }
}
