/**
 * Inspired from git@github.com:mattdean1/cuda.git
 */


#pragma once

#include <sycl/sycl.hpp>

#define SHARED_MEMORY_BANKS 32
#define LOG_MEM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_MEM_BANKS)


constexpr size_t THREADS_PER_BLOCK = 512;
constexpr size_t ELEMENTS_PER_BLOCK = THREADS_PER_BLOCK * 2;

namespace internal {
    // from https://stackoverflow.com/a/3638454
    static inline bool isPowerOfTwo(size_t x) {
        return x && !(x & (x - 1));
    }

    // from https://stackoverflow.com/a/12506181
    static inline size_t nextPowerOfTwo(size_t x) {
        size_t power = 1;
        while (power < x) {
            power *= 2;
        }
        return power;
    }


    template<typename T>
    static void prescan_large(T *output, const T *input, size_t n, T *sums, sycl::nd_item<1> item_ct1, T *local) {
        size_t blockID = item_ct1.get_group(0);
        size_t threadID = item_ct1.get_local_id(0);
        size_t blockOffset = blockID * n;

        size_t ai = threadID;
        size_t bi = threadID + (n / 2);
        size_t bankOffsetA = CONFLICT_FREE_OFFSET(ai);
        size_t bankOffsetB = CONFLICT_FREE_OFFSET(bi);
        local[ai + bankOffsetA] = input[blockOffset + ai];
        local[bi + bankOffsetB] = input[blockOffset + bi];

        int64_t offset = 1;
        for (size_t d = n >> 1; d > 0; d >>= 1) // build sum in place up the tree
        {
            /*
            DPCT1065:8: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space)
            for better performance, if there is no access to global memory.
            */
            item_ct1.barrier();
            if (threadID < d) {
                int64_t ai = offset * (2 * (int64_t) threadID + 1) - 1;
                int64_t bi = offset * (2 * (int64_t) threadID + 2) - 1;
                ai += CONFLICT_FREE_OFFSET(ai);
                bi += CONFLICT_FREE_OFFSET(bi);

                local[bi] += local[ai];
            }
            offset *= 2;
        }
        /*
        DPCT1065:6: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance, if there is no access to global memory.
        */
        item_ct1.barrier();

        if (threadID == 0) {
            sums[blockID] = local[n - 1 + CONFLICT_FREE_OFFSET(n - 1)];
            local[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0;
        }

        for (size_t d = 1; d < n; d *= 2) // traverse down tree & build scan
        {
            offset >>= 1;
            /*
            DPCT1065:9: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space)
            for better performance, if there is no access to global memory.
            */
            item_ct1.barrier();
            if (threadID < d) {
                int64_t ai = offset * (2 * (int64_t) threadID + 1) - 1;
                int64_t bi = offset * (2 * (int64_t) threadID + 2) - 1;
                ai += CONFLICT_FREE_OFFSET(ai);
                bi += CONFLICT_FREE_OFFSET(bi);

                T t = local[ai];
                local[ai] = local[bi];
                local[bi] += t;
            }
        }
        /*
        DPCT1065:7: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance, if there is no access to global memory.
        */
        item_ct1.barrier();

        output[blockOffset + ai] = local[ai + bankOffsetA];
        output[blockOffset + bi] = local[bi + bankOffsetB];
    }


    template<typename T>
    static void add(T *output, size_t length, const T *n, sycl::nd_item<1> item_ct1) {
        size_t blockID = item_ct1.get_group(0);
        size_t threadID = item_ct1.get_local_id(0);
        size_t blockOffset = blockID * length;

        output[blockOffset + threadID] += n[blockID];
    }

    template<typename T>
    static void add(T *output, size_t length, const T *n1, const T *n2, sycl::nd_item<1> item_ct1) {
        size_t blockID = item_ct1.get_group(0);
        size_t threadID = item_ct1.get_local_id(0);
        size_t blockOffset = blockID * length;

        output[blockOffset + threadID] += n1[blockID] + n2[blockID];
    }


    template<typename T>
    void scanLargeDeviceArray(sycl::queue &q, T *d_out, const T *d_in, size_t length);

    template<typename T>
    void scanSmallDeviceArray(sycl::queue &q, T *d_out, const T *d_in, size_t length) {

        /*
        DPCT1049:30: The workgroup size passed to the SYCL kernel may
        exceed the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the workgroup size if
        needed.
        */
        q.submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                    sycl::nd_range<1>(sycl::range<1>(length), sycl::range<1>(length)),
                    [=](sycl::nd_item<1> item) {
                        sycl::joint_exclusive_scan(item.get_sub_group(), d_in, d_in + length, d_out, sycl::plus<>());
                    });
        }).wait();
    }


    template<typename T>
    void scanLargeEvenDeviceArray(sycl::queue &q, T *d_out, const T *d_in, size_t length) {
        const size_t blocks = length / ELEMENTS_PER_BLOCK;
        const size_t sharedMemArrayCount = ELEMENTS_PER_BLOCK;
        auto d_sums = sycl::malloc_device<T>(blocks, q);
        auto d_incr = sycl::malloc_device<T>(blocks, q);

        /*
        DPCT1049:32: The workgroup size passed to the SYCL kernel may
        exceed the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the workgroup size if
        needed.
        */
        q.submit([&](sycl::handler &cgh) {
            sycl::accessor<T, 1, sycl::access_mode::read_write, sycl::access::target::local> local_acc(sycl::range<1>(2 * sharedMemArrayCount), cgh);
            cgh.parallel_for(
                    sycl::nd_range<1>(sycl::range<1>(blocks * THREADS_PER_BLOCK), sycl::range<1>(THREADS_PER_BLOCK)),
                    [=](sycl::nd_item<1> item_ct1) {
                        prescan_large(d_out, d_in, ELEMENTS_PER_BLOCK, d_sums, item_ct1, (T *) local_acc.get_pointer());
                    });
        });


        const size_t sumsArrThreadsNeeded = (blocks + 1) / 2;
        if (sumsArrThreadsNeeded > THREADS_PER_BLOCK) {
            // perform a large scan on the sums arr
            scanLargeDeviceArray(q, d_incr, d_sums, blocks);
        } else {
            // only need one block to scan sums arr so can use small scan
            scanSmallDeviceArray(q, d_incr, d_sums, blocks);
        }

        /*
        DPCT1049:34: The workgroup size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the workgroup size if needed.
        */
        q.submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                    sycl::nd_range<1>(sycl::range<1>(blocks * ELEMENTS_PER_BLOCK), sycl::range<1>(ELEMENTS_PER_BLOCK)),
                    [=](sycl::nd_item<1> item_ct1) {
                        add(d_out, ELEMENTS_PER_BLOCK, d_incr, item_ct1);
                    });
        }).wait();

        sycl::free(d_sums, q);
        sycl::free(d_incr, q);
    }

    template<typename T>
    void scanLargeDeviceArray(sycl::queue &q, T *d_out, const T *d_in, size_t length) {
        size_t remainder = length % (ELEMENTS_PER_BLOCK);
        if (remainder == 0) {
            scanLargeEvenDeviceArray(q, d_out, d_in, length);
        } else {
            // perform a large scan on a compatible multiple of elements
            size_t lengthMultiple = length - remainder;
            scanLargeEvenDeviceArray(q, d_out, d_in, lengthMultiple);

            // scan the remaining elements and add the (inclusive) last element of the large scan to this
            T *startOfOutputArray = &(d_out[lengthMultiple]);
            scanSmallDeviceArray(q, startOfOutputArray, &(d_in[lengthMultiple]), remainder);

            /*
            DPCT1049:29: The workgroup size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the workgroup size if
            needed.
            */
            q.submit([&](sycl::handler &cgh) {
                const T *d_in_lengthMultiple = &(d_in[lengthMultiple - 1]);
                T *d_out_lengthMultiple = &(d_out[lengthMultiple - 1]);

                cgh.parallel_for(
                        sycl::nd_range<1>(sycl::range<1>(remainder), sycl::range<1>(remainder)),
                        [=](sycl::nd_item<1> item_ct1) {
                            add(startOfOutputArray, remainder, d_in_lengthMultiple, d_out_lengthMultiple, item_ct1);
                        });
            });
        }
    }
}

template<typename T>
void blockscan(sycl::queue &q, T *output, const T *input, size_t length) {
    auto d_out = sycl::malloc_device<T>(length, q);
    auto d_in = sycl::malloc_device<T>(length, q);
    q.memcpy(d_out, output, length * sizeof(T)).wait();
    q.memcpy(d_in, input, length * sizeof(T)).wait();

    size_t powerOfTwo = internal::nextPowerOfTwo(length);
    /*
    DPCT1049:19: The workgroup size passed to the SYCL kernel may
    exceed the limit. To get the device limit, query
    info::device::max_work_group_size. Adjust the workgroup size if
    needed.
    */
    q.submit([&](sycl::handler &cgh) {
        sycl::accessor<T, 1, sycl::access_mode::read_write, sycl::access::target::local> local_acc(sycl::range<1>(2 * powerOfTwo), cgh);
        cgh.parallel_for(
                sycl::nd_range<1>(sycl::range<1>((length + 1) / 2), sycl::range<1>((length + 1) / 2)),
                [=](sycl::nd_item<1> item) {
                    prescan_arbitrary(d_out, d_in, length, powerOfTwo, item, local_acc.get_pointer());
                });
    }).wait();

    q.memcpy(output, d_out, length * sizeof(T)).wait();
    sycl::free(d_out, q);
    sycl::free(d_in, q);
}


enum class scan_type {
    inclusive,
    exclusive
};

template<scan_type type, typename T>
void scan(sycl::queue &q, T *output, T *input, size_t length) {
    size_t alloc_length = length;
    size_t offset = 0;

    if constexpr (type == scan_type::inclusive) {
        alloc_length = length + 1;
        offset = 1;
    }

    auto d_out = sycl::malloc_device<T>(alloc_length, q);
    auto d_in = sycl::malloc_device<T>(alloc_length, q);
    q.memcpy(d_out, output, length * sizeof(T)).wait();
    q.memcpy(d_in, input, length * sizeof(T)).wait();


    //sycl::event start, stop;
    //std::chrono::time_point<std::chrono::steady_clock> start_ct1;
    //std::chrono::time_point<std::chrono::steady_clock> stop_ct1;
    //start_ct1 = std::chrono::steady_clock::now();

    if (length > ELEMENTS_PER_BLOCK) {
        internal::scanLargeDeviceArray(q, d_out, d_in, length);
    } else {
        internal::scanSmallDeviceArray(q, d_out, d_in, length);
    }

    //stop_ct1 = std::chrono::steady_clock::now();
    //float elapsedTime = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();

    //printf("Time %f \n", elapsedTime);

    q.memcpy(output, d_out + offset, length * sizeof(T)).wait();

    if (type == scan_type::inclusive && length > 1) {
        output[length - 1] = output[length - 2] + input[length - 1];
    }

    if (type == scan_type::inclusive && length == 1) {
        output[0] = input[0];
    }


    sycl::free(d_out, q);
    sycl::free(d_in, q);
}
