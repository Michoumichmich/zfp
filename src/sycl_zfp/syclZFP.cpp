#include <sycl/sycl.hpp>
#include <cassert>

#include "syclZFP.h"

#include "encode1.hpp"
#include "encode2.hpp"
#include "encode3.hpp"

#include "decode1.hpp"
#include "decode2.hpp"
#include "decode3.hpp"

#include "shared.h"


//#define HAS_VARIABLE
#ifdef HAS_VARIABLE

#include "variable.hpp"

#endif

#include "pointers.hpp"

// we need to know about bitstream, but we don't
// want duplicate symbols.
#ifndef inline_
#define inline_ inline
#endif

#include "../inline/bitstream.c"

namespace internal {


    size_t get_global_range(const sycl::id<3> &dims) {
        size_t res = 1;
        if (dims[0] != 0) res *= dims[0];
        if (dims[1] != 0) res *= dims[1];
        if (dims[2] != 0) res *= dims[2];
        return res;
    }

    int count_used_dims(const sycl::id<3> &dims) {
        int d = 0;
        if (dims[2] != 0) d++;
        if (dims[1] != 0) d++;
        if (dims[0] != 0) d++;
        return d;
    }


    static bool is_contigous3d(const sycl::id<3> dims, const syclZFP::int64_3_t &stride, int64_t &offset) {
        int64_t idims[3];
        idims[0] = (int64_t) dims[0];
        idims[1] = (int64_t) dims[1];
        idims[2] = (int64_t) dims[2];

        int64_t imin = std::min(stride.x, int64_t(0)) * (idims[2] - 1) +
                       std::min(stride.y, int64_t(0)) * (idims[1] - 1) +
                       std::min(stride.z, int64_t(0)) * (idims[0] - 1);

        int64_t imax = std::max(stride.x, int64_t(0)) * (idims[2] - 1) +
                       std::max(stride.y, int64_t(0)) * (idims[1] - 1) +
                       std::max(stride.z, int64_t(0)) * (idims[0] - 1);
        offset = imin;
        int64_t ns = idims[0] * idims[1] * idims[2];

        return (imax - imin + 1 == ns);
    }

    static bool is_contigous2d(const sycl::id<3> dims, const syclZFP::int64_3_t &stride, int64_t &offset) {
        int64_t idims[2];
        idims[0] = (int64_t) dims[1]; //Y
        idims[1] = (int64_t) dims[2]; //X

        int64_t imin = std::min(stride.x, int64_t(0)) * (idims[1] - 1) +
                       std::min(stride.y, int64_t(0)) * (idims[0] - 1);

        int64_t imax = std::max(stride.x, int64_t(0)) * (idims[1] - 1) +
                       std::max(stride.y, int64_t(0)) * (idims[0] - 1);

        offset = imin;
        return (imax - imin + 1) == (idims[0] * idims[1]);
    }

    static bool is_contigous1d(size_t dim, const int64_t &stride, int64_t &offset) {
        offset = 0;
        if (stride < 0) offset = stride * (int(dim) - 1);
        return std::abs(stride) == 1;
    }

    static bool is_contigous(const sycl::id<3> dims, const syclZFP::int64_3_t &stride, int64_t &offset) {
        int d = count_used_dims(dims);
        switch (d) {
            case 1:
                return is_contigous1d(dims[2], stride.x, offset);
            case 2:
                return is_contigous2d(dims, stride, offset);
            case 3:
                return is_contigous3d(dims, stride, offset);
            default:
                assert(false);
        }
        return false;
    }

    //
    // encode expects device pointers
    //
    template<typename T, bool variable_rate>
    size_t encode(sycl::queue &q,
                  sycl::id<3> dims,
                  syclZFP::int64_3_t stride,
                  int minbits,
                  int maxbits,
                  int maxprec,
                  int minexp,
                  T *d_data,
                  Word *d_stream,
                  ushort *d_bitlengths) {

        const int d = count_used_dims(dims);
        const size_t len = get_global_range(dims);

        size_t stream_size = 0;
        if (d == 1) {
            stream_size = syclZFP::encode1<T, variable_rate>(q, dims[2], stride.x, d_data, d_stream, d_bitlengths, minbits, maxbits, maxprec, minexp);
        } else if (d == 2) {
            sycl::id<2> ndims(dims[1], dims[2]);
            syclZFP::int64_2_t s{stride.y, stride.x};
            stream_size = syclZFP::encode2<T, variable_rate>(q, ndims, s, d_data, d_stream, d_bitlengths, minbits, maxbits, maxprec, minexp);
        } else if (d == 3) {
            stream_size = syclZFP::encode3<T, variable_rate>(q, dims, stride, d_data, d_stream, d_bitlengths, minbits, maxbits, maxprec, minexp);
        }
        return stream_size;
    }

    template<typename T>
    size_t decode(sycl::queue &q,
                  sycl::id<3> ndims,
                  syclZFP::int64_3_t stride,
                  int bits_per_block,
                  Word *stream,
                  T *out) {
        const int d = count_used_dims(ndims);
        const size_t out_size = get_global_range(ndims);
        size_t stream_bytes = 0;

        if (d == 3) {
            stream_bytes = syclZFP::decode3<T>(q, ndims, stride, stream, out, bits_per_block);
        } else if (d == 1) {
            size_t dim = ndims[2];
            int64_t sx = stride.x;
            stream_bytes = syclZFP::decode1<T>(q, dim, sx, stream, out, bits_per_block);
        } else if (d == 2) {
            sycl::id<2> dims{ndims[1], ndims[2]};
            syclZFP::int64_2_t s{stride.y, stride.x};
            stream_bytes = syclZFP::decode2<T>(q, dims, s, stream, out, bits_per_block);
        } else std::cerr << " d ==  " << d << " not implemented\n";

        return stream_bytes;
    }

    Word *setup_device_stream_compress(sycl::queue &q, zfp_stream *stream, const zfp_field *field) {
        bool stream_device = syclZFP::queue_can_access_ptr(q, stream->stream->begin);
        assert(sizeof(word) == sizeof(Word)); // "CUDA version currently only supports 64bit words");

        if (stream_device) {
            return (Word *) stream->stream->begin;
        }

        size_t max_size = zfp_stream_maximum_size(stream, field);
        Word *d_stream = (Word *) sycl::malloc_device(max_size, q);
        return d_stream;
    }

    Word *setup_device_stream_decompress(sycl::queue &q, zfp_stream *stream, const zfp_field *field) {
        bool stream_device = syclZFP::queue_can_access_ptr(q, stream->stream->begin);
        assert(sizeof(word) == sizeof(Word)); // "CUDA version currently only supports 64bit words");

        if (stream_device) {
            return (Word *) stream->stream->begin;
        }

        //TODO: change maximum_size to compressed stream size
        size_t size = zfp_stream_maximum_size(stream, field);
        Word *d_stream = (Word *) sycl::malloc_device(size, q);
        q.memcpy(d_stream, stream->stream->begin, size).wait();
        return d_stream;
    }

    static void *offset_void(zfp_type type, void *ptr, int64_t offset) {
        void *offset_ptr = nullptr;
        if (type == zfp_type_float) {
            auto *data = (float *) ptr;
            offset_ptr = (void *) (&data[offset]);
        } else if (type == zfp_type_double) {
            auto *data = (double *) ptr;
            offset_ptr = (void *) (&data[offset]);
        } else if (type == zfp_type_int32) {
            auto *data = (int32_t *) ptr;
            offset_ptr = (void *) (&data[offset]);
        } else if (type == zfp_type_int64) {
            auto *data = (int64_t *) ptr;
            offset_ptr = (void *) (&data[offset]);
        }
        return offset_ptr;
    }

    void *setup_device_field_compress(sycl::queue &q, const zfp_field *field, const syclZFP::int64_3_t &stride, int64_t &offset) {
        bool field_device = syclZFP::queue_can_access_ptr(q, field->data);

        if (field_device) {
            offset = 0;
            return field->data;
        }

        sycl::id<3> dims{field->nz, field->ny, field->nx};
        const size_t type_size = zfp_type_size(field->type);
        const size_t field_size = get_global_range(dims);

        bool contig = internal::is_contigous(dims, stride, offset);

        void *host_ptr = offset_void(field->type, field->data, offset);

        void *d_data = nullptr;
        if (contig) {
            size_t field_bytes = type_size * field_size;
            d_data = (void *) sycl::malloc_device(field_bytes, q);
            q.memcpy(d_data, host_ptr, field_bytes).wait();
        }
        return offset_void(field->type, d_data, -offset);
    }

    void *setup_device_field_decompress(sycl::queue &q, const zfp_field *field, const syclZFP::int64_3_t &stride, int64_t &offset) {
        bool field_device = syclZFP::queue_can_access_ptr(q, field->data);

        if (field_device) {
            offset = 0;
            return field->data;
        }

        sycl::id<3> dims{field->nz, field->ny, field->nx};
        const size_t type_size = zfp_type_size(field->type);
        const size_t field_size = get_global_range(dims);
        bool contig = internal::is_contigous(dims, stride, offset);

        void *d_data = nullptr;
        if (contig) {
            size_t field_bytes = type_size * field_size;
            d_data = (void *) sycl::malloc_device(field_bytes, q);
        }
        return offset_void(field->type, d_data, -offset);
    }

    ushort *setup_device_nbits_compress(sycl::queue &q, zfp_stream *stream, const zfp_field *field, int variable_rate) {
        if (!variable_rate)
            return nullptr;

        bool device_mem = syclZFP::queue_can_access_ptr(q, stream->stream->bitlengths);
        if (device_mem)
            return (ushort *) stream->stream->bitlengths;

        size_t size = zfp_field_num_blocks(field) * sizeof(ushort);
        auto *d_bitlengths = (ushort *) sycl::malloc_device(size, q);
        return d_bitlengths;
    }

    ushort *setup_device_nbits_decompress(sycl::queue &q, zfp_stream *stream, const zfp_field *field, int variable_rate) {
        if (!variable_rate)
            return nullptr;

        if (syclZFP::queue_can_access_ptr(q, stream->stream->bitlengths))
            return stream->stream->bitlengths;

        size_t size = zfp_field_num_blocks(field) * sizeof(ushort);
        auto *d_bitlengths = (ushort *) sycl::malloc_device(size, q);
        q.memcpy(d_bitlengths, stream->stream->bitlengths, size).wait();
        return d_bitlengths;
    }

    void cleanup_device_nbits(sycl::queue &q, zfp_stream *stream, const zfp_field *field, ushort *d_bitlengths, int variable_rate, int copy) {
        if (!variable_rate)
            return;

        if (syclZFP::queue_can_access_ptr(q, stream->stream->bitlengths))
            return;

        size_t size = zfp_field_num_blocks(field) * sizeof(ushort);
        if (copy)
            q.memcpy(stream->stream->bitlengths, d_bitlengths, size).wait();

        sycl::free(d_bitlengths, q);
    }

    static void cleanup_device_ptr(sycl::queue &q, void *orig_ptr, void *d_ptr, size_t bytes, int64_t offset, zfp_type type) {
        bool device = syclZFP::queue_can_access_ptr(q, orig_ptr);
        if (device) {
            return;
        }
        void *d_offset_ptr = offset_void(type, d_ptr, offset);
        void *h_offset_ptr = offset_void(type, orig_ptr, offset);

        if (bytes > 0) {
            q.memcpy(h_offset_ptr, d_offset_ptr, bytes).wait();
        }
        sycl::free(d_offset_ptr, q);
    }

} // namespace internal


size_t sycl_compress(zfp_stream *stream, const zfp_field *field, int variable_rate) {
#ifndef HAS_VARIABLE
    if (variable_rate)
        return 0;
#endif


    if (zfp_stream_compression_mode(stream) == zfp_mode_reversible) {
        // Reversible mode not supported on GPU
        return 0;
    }
    sycl::queue q{sycl::gpu_selector{}};

#ifdef VERBOSE_SYCL
    std::cout << "Compressing on: " << q.get_device().get_info<sycl::info::device::name>() << '\n';
#endif
    assert(q.get_device().has(sycl::aspect::usm_device_allocations));

    sycl::id<3> dims{field->nz, field->ny, field->nx};

    syclZFP::int64_3_t stride;
    stride.x = field->sx ? field->sx : 1;
    stride.y = field->sy ? (int64_t) field->sy : (int64_t) field->nx;
    stride.z = field->sz ? (int64_t) field->sz : (int64_t) (field->nx * field->ny);

    size_t stream_bytes = 0;
    int64_t offset = 0;
    void *d_data = internal::setup_device_field_compress(q, field, stride, offset);

    if (d_data == nullptr) {
        // null means the array is non-contiguous host mem which is not supported
        return 0;
    }

    int num_sm = (int) q.get_device().get_info<sycl::info::device::max_compute_units>();

    Word *d_stream = internal::setup_device_stream_compress(q, stream, field);
    ushort *d_bitlengths = internal::setup_device_nbits_compress(q, stream, field, variable_rate);

    uint buffer_maxbits = MIN (stream->maxbits, zfp_block_maxbits(stream, field));

    if (field->type == zfp_type_float) {
        auto *data = (float *) d_data;
        if (variable_rate)
            stream_bytes = internal::encode<float, true>(q, dims, stride, (int) stream->minbits, (int) buffer_maxbits, (int) stream->maxprec, stream->minexp, data, d_stream, d_bitlengths);
        else
            stream_bytes = internal::encode<float, false>(q, dims, stride, (int) stream->minbits, (int) buffer_maxbits, (int) stream->maxprec, stream->minexp, data, d_stream, d_bitlengths);
    } else if (field->type == zfp_type_double) {
        auto *data = (double *) d_data;
        if (variable_rate)
            stream_bytes = internal::encode<double, true>(q, dims, stride, (int) stream->minbits, (int) buffer_maxbits, (int) stream->maxprec, stream->minexp, data, d_stream, d_bitlengths);
        else
            stream_bytes = internal::encode<double, false>(q, dims, stride, (int) stream->minbits, (int) buffer_maxbits, (int) stream->maxprec, stream->minexp, data, d_stream, d_bitlengths);
    } else if (field->type == zfp_type_int32) {
        auto *data = (int32_t *) d_data;
        if (variable_rate)
            stream_bytes = internal::encode<int32_t, true>(q, dims, stride, (int) stream->minbits, (int) buffer_maxbits, (int) stream->maxprec, stream->minexp, data, d_stream, d_bitlengths);
        else
            stream_bytes = internal::encode<int32_t, false>(q, dims, stride, (int) stream->minbits, (int) buffer_maxbits, (int) stream->maxprec, stream->minexp, data, d_stream, d_bitlengths);
    } else if (field->type == zfp_type_int64) {
        auto *data = (int64_t *) d_data;
        if (variable_rate)
            stream_bytes = internal::encode<int64_t, true>(q, dims, stride, (int) stream->minbits, (int) buffer_maxbits, (int) stream->maxprec, stream->minexp, data, d_stream, d_bitlengths);
        else
            stream_bytes = internal::encode<int64_t, false>(q, dims, stride, (int) stream->minbits, (int) buffer_maxbits, (int) stream->maxprec, stream->minexp, data, d_stream, d_bitlengths);
    }

#ifdef HAS_VARIABLE
    if (variable_rate) {
        int chunk_size = num_sm * 1024;
        auto d_offsets = sycl::malloc_device<uint64_t>(chunk_size + 1, q);
        auto offsets_out = sycl::malloc_device<uint64_t>(chunk_size + 1, q);

        size_t blocks = zfp_field_num_blocks(field);

        for (size_t i = 0; i < blocks; i += chunk_size) {
            using namespace parallel_primitives;
            int cur_blocks = chunk_size;
            bool last_chunk = false;
            if (i + chunk_size > blocks) {
                cur_blocks = (int) (blocks - i);
                last_chunk = true;
            }
            // Copy the 16-bit lengths in the offset array
            syclZFP::copy_length_launch(q, d_bitlengths, d_offsets, i, cur_blocks);

            // Prefix sum to turn length into offsets
            cooperative_scan<scan_type::inclusive, sycl::plus<>>(q, d_offsets, offsets_out, cur_blocks + 1);
            //  cub::DeviceScan::InclusiveSum(d_cubtemp, lcubtemp, d_offsets, d_offsets, cur_blocks + 1);

            // Compact the stream array in-place
            syclZFP::chunk_process_launch(q, (uint *) d_stream, offsets_out, i, cur_blocks, last_chunk, buffer_maxbits, num_sm);
        }
        // The total length in bits is now in the base of the prefix sum.
        q.memcpy(&stream_bytes, offsets_out, sizeof(size_t)).wait();
        stream_bytes = (stream_bytes + 7) / 8;


        sycl::free(offsets_out, q);
        sycl::free(d_offsets, q);
    }
#endif

    internal::cleanup_device_ptr(q, stream->stream->begin, d_stream, stream_bytes, 0, field->type);
    internal::cleanup_device_ptr(q, field->data, d_data, 0, offset, field->type);
    if (variable_rate) {
        if (stream->stream->bitlengths) // Saving the individual block lengths if a pointer exists
        {
            size_t size = zfp_field_num_blocks(field) * sizeof(ushort);
            internal::cleanup_device_ptr(q, stream->stream->bitlengths, d_bitlengths, size, 0, zfp_type_none);
        }
        //internal::cleanup_device_ptr(q, nullptr, d_offsets, 0, 0, zfp_type_none);
        //internal::cleanup_device_ptr(q, nullptr, d_cubtemp, 0, 0, zfp_type_none);
    }

    // zfp wants to flush the stream.
    // set bits to wsize because we already did that.
    //size_t compressed_size = (stream_bytes + sizeof(Word) - 1) / sizeof(Word); //???
    size_t compressed_size = stream_bytes / sizeof(Word); //???
    stream->stream->bits = wsize;
    // set stream pointer to end of stream
    stream->stream->ptr = stream->stream->begin + compressed_size;

    return stream_bytes;
}

void sycl_decompress(zfp_stream *stream, zfp_field *field) {
    sycl::queue q{sycl::gpu_selector{}};
#ifdef VERBOSE_SYCL
    std::cout << "Decompressing on: " << q.get_device().get_info<sycl::info::device::name>() << '\n';
#endif


    assert(q.get_device().has(sycl::aspect::usm_device_allocations));
    sycl::id<3> dims{field->nz, field->ny, field->nx};

    syclZFP::int64_3_t stride;
    stride.x = field->sx ? field->sx : 1;
    stride.y = field->sy ? (int) field->sy : (int) field->nx;
    stride.z = field->sz ? (int) field->sz : (int) (field->nx * field->ny);

    size_t decoded_bytes = 0;
    int64_t offset = 0;
    void *d_data = internal::setup_device_field_decompress(q, field, stride, offset);

    if (d_data == nullptr) {
        // null means the array is non-contiguous host mem which is not supported
        return;
    }

    Word *d_stream = internal::setup_device_stream_decompress(q, stream, field);

    if (field->type == zfp_type_float) {
        auto *data = (float *) d_data;
        decoded_bytes = internal::decode<float>(q, dims, stride, (int) stream->maxbits, d_stream, data);
        d_data = (void *) data;
    } else if (field->type == zfp_type_double) {
        auto *data = (double *) d_data;
        decoded_bytes = internal::decode<double>(q, dims, stride, (int) stream->maxbits, d_stream, data);
        d_data = (void *) data;
    } else if (field->type == zfp_type_int32) {
        auto *data = (int32_t *) d_data;
        decoded_bytes = internal::decode<int32_t>(q, dims, stride, (int) stream->maxbits, d_stream, data);
        d_data = (void *) data;
    } else if (field->type == zfp_type_int64) {
        auto *data = (int64_t *) d_data;
        decoded_bytes = internal::decode<int64_t>(q, dims, stride, (int) stream->maxbits, d_stream, data);
        d_data = (void *) data;
    } else {
        std::cerr << "Cannot decompress: type unknown\n";
    }

    size_t type_size = zfp_type_size(field->type);
    size_t field_size = internal::get_global_range(dims);


    size_t bytes = type_size * field_size;
    internal::cleanup_device_ptr(q, stream->stream->begin, d_stream, 0, 0, field->type);
    internal::cleanup_device_ptr(q, field->data, d_data, bytes, offset, field->type);

    // this is how zfp determines if this was a success
    size_t words_read = decoded_bytes / sizeof(Word);
    stream->stream->bits = wsize;
    // set stream pointer to end of stream
    stream->stream->ptr = stream->stream->begin + words_read;
}
