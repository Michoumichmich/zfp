#pragma once

#include "shared.h"
#include <algorithm>

namespace syclZFP {

    // maximum number of bit planes to encode
    static int precision(int maxexp, int maxprec, int minexp) {
        return MIN(maxprec, MAX(0, maxexp - minexp + 8));
    }

    template<int BlockSize>
    static int precision(int maxexp, int maxprec, int minexp) {
        // Followig logic from precision() in zfp/src/template/codecf.c
        if (BlockSize == ZFP_1D_BLOCK_SIZE)
            return MIN(maxprec, MAX(0, maxexp - minexp + 4));
        else if (BlockSize == ZFP_2D_BLOCK_SIZE)
            return MIN(maxprec, MAX(0, maxexp - minexp + 6));
        else if (BlockSize == ZFP_3D_BLOCK_SIZE)
            return MIN(maxprec, MAX(0, maxexp - minexp + 8));
        else if (BlockSize == ZFP_4D_BLOCK_SIZE)
            return MIN(maxprec, MAX(0, maxexp - minexp + 10));
        else
            return 0;
    }

    template<typename Scalar>
    inline void pad_block(Scalar *p, int n, int s) {
        switch (n) {
            case 0:
                p[0 * s] = 0;
                /* FALLTHROUGH */
            case 1:
                p[1 * s] = p[0 * s];
                /* FALLTHROUGH */
            case 2:
                p[2 * s] = p[1 * s];
                /* FALLTHROUGH */
            case 3:
                p[3 * s] = p[0 * s];
                /* FALLTHROUGH */
            default:
                break;
        }
    }

    template<class Scalar>
    static int exponent(Scalar x) {
        if (x > 0) {
            int e;
            FREXP(x, &e);
            // clamp exponent in case x is denormalized
            return sycl::max(e, 1 - get_ebias<Scalar>());
        }
        return -get_ebias<Scalar>();
    }

    template<class Scalar, int BlockSize>
    static int max_exponent(const Scalar *p) {
        Scalar max_val = 0;
        for (int i = 0; i < BlockSize; ++i) {
            Scalar f = sycl::fabs(p[i]);
            max_val = sycl::max(max_val, f);
        }
        return exponent<Scalar>(max_val);
    }

    // lifting transform of 4-vector
    template<class Int, int s>
    static void fwd_lift(Int *p) {

        Int x = *p;
        p += s;
        Int y = *p;
        p += s;
        Int z = *p;
        p += s;
        Int w = *p;
        p += s;

        // default, non-orthogonal transform (preferred due to speed and quality)
        //        ( 4  4  4  4) (x)
        // 1/16 * ( 5  1 -1 -5) (y)
        //        (-4  4  4 -4) (z)
        //        (-2  6 -6  2) (w)
        x += w;
        x >>= 1;
        w -= x;
        z += y;
        z >>= 1;
        y -= z;
        x += z;
        x >>= 1;
        z -= x;
        w += y;
        w >>= 1;
        y -= w;
        w += y >> 1;
        y -= w >> 1;

        p -= s;
        *p = w;
        p -= s;
        *p = z;
        p -= s;
        *p = y;
        p -= s;
        *p = x;
    }

#if ZFP_ROUNDING_MODE == ZFP_ROUND_FIRST
    // bias values such that truncation is equivalent to round to nearest
    template <typename Int, uint BlockSize>
    static void fwd_round(Int* iblock, uint maxprec)
    {
        // add or subtract 1/6 ulp to unbias errors
        if (maxprec < (uint)(CHAR_BIT * sizeof(Int))) {
            Int bias = (static_cast<typename zfp_traits<Int>::UInt>(NBMASK) >> 2) >> maxprec;
            uint n = BlockSize;
            if (maxprec & 1u)
                do *iblock++ += bias; while (--n);
                else
                    do *iblock++ -= bias; while (--n);
        }
    }
#endif


    template<typename Scalar>
    Scalar inline quantize_factor(const int &exponent, Scalar);

    template<>
    float inline quantize_factor<float>(const int &exponent, float) {
        return LDEXP(1.f, get_precision<float>() - 2 - exponent);
    }

    template<>
    double inline quantize_factor<double>(const int &exponent, double) {
        return LDEXP(1.0, get_precision<double>() - 2 - exponent);
    }

    template<typename Scalar, typename Int, int BlockSize>
    void fwd_cast(Int *iblock, const Scalar *fblock, int emax) {
        Scalar s = quantize_factor(emax, Scalar());
        for (int i = 0; i < BlockSize; ++i) {
            iblock[i] = (Int) (s * fblock[i]);
        }
    }

    template<int BlockSize>
    struct transform;

    template<>
    struct transform<64> {
        template<typename Int>
        void fwd_xform(Int *p) {

            int x, y, z;
            /* transform along x */
            for (z = 0; z < 4; z++)
                for (y = 0; y < 4; y++)
                    fwd_lift<Int, 1>(p + 4 * y + 16 * z);
            /* transform along y */
            for (x = 0; x < 4; x++)
                for (z = 0; z < 4; z++)
                    fwd_lift<Int, 4>(p + 16 * z + 1 * x);
            /* transform along z */
            for (y = 0; y < 4; y++)
                for (x = 0; x < 4; x++)
                    fwd_lift<Int, 16>(p + 1 * x + 4 * y);

        }

    };

    template<>
    struct transform<16> {
        template<typename Int>
        void fwd_xform(Int *p) {

            int x, y;
            /* transform along x */
            for (y = 0; y < 4; y++)
                fwd_lift<Int, 1>(p + 4 * y);
            /* transform along y */
            for (x = 0; x < 4; x++)
                fwd_lift<Int, 4>(p + 1 * x);
        }

    };

    template<>
    struct transform<4> {
        template<typename Int>
        void fwd_xform(Int *p) {
            fwd_lift<Int, 1>(p);
        }

    };

    template<typename Int, typename UInt, int BlockSize>
    void fwd_order(const_perm_accessor acc, UInt *ublock, const Int *iblock) {
        for (uint i = 0; i < BlockSize; ++i) {
            ublock[i] = int2uint(iblock[(uint) acc[i]]);
        }
    }

    template<int block_size>
    struct BlockWriter {
        int m_word_index;
        int m_start_bit;
        int m_current_bit;
        const int m_maxbits;
        Word *m_stream;

        BlockWriter(Word *stream, const int &maxbits, const size_t block_idx)
                : m_current_bit(0),
                  m_maxbits(maxbits),
                  m_stream(stream) {
            m_word_index = int((block_idx * (size_t) maxbits) / (sizeof(Word) * 8));
            m_start_bit = int((block_idx * (size_t) maxbits) % (sizeof(Word) * 8));
        }

        template<typename T>
        void print_bits(T bits, sycl::stream &os) {
            const int bit_size = sizeof(T) * 8;
            for (int i = bit_size - 1; i >= 0; --i) {
                T one = 1;
                T mask = one << i;
                int val = (bits & mask) >> i;
                os << val;
            }
            os << '\n';
        }

        void print(int index, sycl::stream &os) {
            print_bits(m_stream[index], os);
        }


        size_t write_bits(const uint64_t &bits, const int &n_bits) {
            const int wbits = sizeof(Word) * 8;
            int seg_start = (m_start_bit + m_current_bit) % wbits;
            int write_index = m_word_index + ((m_start_bit + m_current_bit) / wbits);
            int seg_end = seg_start + n_bits - 1;
            int shift = seg_start;
            // we may be asked to write less bits than exist in 'bits'
            // so we have to make sure that anything after n is zero.
            // If this does not happen, then we may write into a zfp
            // block not at the specified index
            // uint zero_shift = sizeof(Word) * 8 - n_bits;
            Word left = (bits >> n_bits) << n_bits;

            Word b = bits - left;
            Word add = b << shift;
            ATOMIC_REF_NAMESPACE::atomic_ref<Word, ATOMIC_REF_NAMESPACE::memory_order::relaxed, ATOMIC_REF_NAMESPACE::memory_scope::device, sycl::access::address_space::global_space> ref(m_stream[write_index]);
            ref += add;

            // n_bits straddles the word boundary
            bool straddle = seg_start < (int) sizeof(Word) * 8 && seg_end >= (int) sizeof(Word) * 8;

            if (straddle) {
                Word rem = b >> (sizeof(Word) * 8 - (uint) shift);
                ATOMIC_REF_NAMESPACE::atomic_ref<Word, ATOMIC_REF_NAMESPACE::memory_order::relaxed, ATOMIC_REF_NAMESPACE::memory_scope::device, sycl::access::address_space::global_space> ref_next(
                        m_stream[write_index + 1]);
                ref_next += straddle * rem;
            }
            m_current_bit += n_bits;
            return bits >> (Word) n_bits;
        }

        uint write_bit(const unsigned int &bit) {
            const int wbits = sizeof(Word) * 8;
            int seg_start = (m_start_bit + m_current_bit) % wbits;
            int write_index = m_word_index + ((m_start_bit + m_current_bit) / wbits);
            int shift = seg_start;
            // we may be asked to write less bits than exist in 'bits'
            // so we have to make sure that anything after n is zero.
            // If this does not happen, then we may write into a zfp
            // block not at the specified index
            // uint zero_shift = sizeof(Word) * 8 - n_bits;

            Word add = (Word) bit << shift;
            ATOMIC_REF_NAMESPACE::atomic_ref<Word, ATOMIC_REF_NAMESPACE::memory_order::relaxed, ATOMIC_REF_NAMESPACE::memory_scope::device, sycl::access::address_space::global_space> ref(m_stream[write_index]);
            ref += add;
            m_current_bit += 1;

            return bit;
        }

    };

    template<typename Int, int BlockSize>
    int inline encode_block(const_perm_accessor acc, BlockWriter<BlockSize> &stream, int maxbits, int maxprec, Int *iblock) {
        // perform decorrelating transform
        transform<BlockSize> tform;
        tform.fwd_xform(iblock);

#if ZFP_ROUNDING_MODE == ZFP_ROUND_FIRST
        // bias values to achieve proper rounding
        fwd_round<Int, BlockSize>(iblock, maxprec);
#endif

        // reorder signed coefficients and convert to unsigned integer
        typedef typename zfp_traits<Int>::UInt UInt;
        UInt ublock[BlockSize];
        fwd_order<Int, UInt, BlockSize>(acc, ublock, iblock);

        // encode integer coefficients
        int intprec = (CHAR_BIT * sizeof(UInt));
        int kmin = intprec > maxprec ? intprec - maxprec : 0;
        int bits = maxbits;

        for (int k = intprec, n = 0; bits && k-- > kmin;) {
            /* step 1: extract bit plane #k to x */
            uint64_t x = 0;
            for (int i = 0; i < BlockSize; i++) {
                x += (uint64_t) ((ublock[i] >> k) & 1u) << i;
            }
            /* step 2: encode first n bits of bit plane */
            int m = sycl::min(n, bits);
            bits -= m;
            x = stream.write_bits(x, m);
            /* step 3: unary run-length encode remainder of bit plane */
            for (; n < BlockSize && bits && (bits--, stream.write_bit(!!x)); x >>= 1, n++) {
                for (; n < BlockSize - 1 && bits && (bits--, !stream.write_bit(x & 1u)); x >>= 1, n++) {
                }
            }
        }
        return maxbits - bits;
    }

    template<typename Scalar, int BlockSize>
    int inline zfp_encode_block(
            const_perm_accessor acc,
            Scalar *fblock,
            const int minbits,
            const int maxbits,
            int maxprec,
            const int minexp,
            const size_t block_idx,
            Word *stream) {
        BlockWriter<BlockSize> block_writer(stream, maxbits, block_idx);
        int emax = max_exponent<Scalar, BlockSize>(fblock);
        maxprec = precision<BlockSize>(emax, maxprec, minexp);
        uint e = maxprec ? (uint) (emax + get_ebias<Scalar>()) : 0;
        if (e) {
            const auto ebits = get_ebits<Scalar>() + 1;
            block_writer.write_bits(2 * e + 1u, ebits);
            typedef typename zfp_traits<Scalar>::Int Int;
            Int iblock[BlockSize];
            fwd_cast<Scalar, Int, BlockSize>(iblock, fblock, emax);

            int bits = encode_block<Int, BlockSize>(acc, block_writer, maxbits - ebits, maxprec, iblock);
            return ebits + bits;
        } else
            // Single bit (already memset to zero) to indicate all values are zero
            return sycl::max(minbits, 1);
    }

    template<>
    int inline zfp_encode_block<int32_t, 64>(
            const_perm_accessor acc,
            int32_t *fblock,
            const int minbits,
            const int maxbits,
            int maxprec,
            const int minexp,
            const size_t block_idx,
            Word *stream) {
        BlockWriter<64> block_writer(stream, maxbits, block_idx);
        const int intprec = get_precision<int32_t>();
        int bits = encode_block<int32_t, 64>(acc, block_writer, maxbits, intprec, fblock);
        return sycl::max(bits, minbits);
    }


    template<>
    int inline zfp_encode_block<int64_t, 64>(
            const_perm_accessor acc,
            int64_t *fblock,
            const int minbits,
            const int maxbits,
            int maxprec,
            const int minexp,
            const size_t block_idx,
            Word *stream) {
        BlockWriter<64> block_writer(stream, maxbits, block_idx);
        const int intprec = get_precision<int64_t>();
        int bits = encode_block<int64_t, 64>(acc, block_writer, maxbits, intprec, fblock);
        return sycl::max(bits, minbits);
    }


    template<>
    int inline zfp_encode_block<int32_t, 16>(
            const_perm_accessor acc,
            int32_t *fblock,
            const int minbits,
            const int maxbits,
            int maxprec,
            const int minexp,
            const size_t block_idx,
            Word *stream) {
        BlockWriter<16> block_writer(stream, maxbits, block_idx);
        const int intprec = get_precision<int32_t>();
        int bits = encode_block<int32_t, 16>(acc, block_writer, maxbits, intprec, fblock);
        return sycl::max(bits, minbits);
    }

    template<>
    int inline zfp_encode_block<int64_t, 16>(
            const_perm_accessor acc,
            int64_t *fblock,
            const int minbits,
            const int maxbits,
            int maxprec,
            const int minexp,
            const size_t block_idx,
            Word *stream) {
        BlockWriter<16> block_writer(stream, maxbits, block_idx);
        const int intprec = get_precision<int64_t>();
        int bits = encode_block<int64_t, 16>(acc, block_writer, maxbits, intprec, fblock);
        return sycl::max(bits, minbits);
    }

    template<>
    int inline zfp_encode_block<int32_t, 4>(
            const_perm_accessor acc,
            int32_t *fblock,
            const int minbits,
            const int maxbits,
            int maxprec,
            const int minexp,
            const size_t block_idx,
            Word *stream) {
        BlockWriter<4> block_writer(stream, maxbits, block_idx);
        const int intprec = get_precision<int32_t>();
        int bits = encode_block<int32_t, 4>(acc, block_writer, maxbits, intprec, fblock);
        return sycl::max(bits, minbits);
    }

    template<>
    int inline zfp_encode_block<int64_t, 4>(
            const_perm_accessor acc,
            int64_t *fblock,
            const int minbits,
            const int maxbits,
            int maxprec,
            const int minexp,
            const size_t block_idx,
            Word *stream) {
        BlockWriter<4> block_writer(stream, maxbits, block_idx);
        const int intprec = get_precision<int64_t>();
        int bits = encode_block<int64_t, 4>(acc, block_writer, maxbits, intprec, fblock);
        return sycl::max(bits, minbits);
    }

}
