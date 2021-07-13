#pragma once

#include <sycl/sycl.hpp>
#include "syclZFP.h"
#include <iostream>

namespace syclZFP {
#ifdef IMPLICIT_MEMORY_COPY

#include <sys/mman.h>
#include <unistd.h>

    /**
     * Checks whether a pointer was allocated on the host device
     * @see http://si-head.nl/articles/msync
     * @tparam T
     * @param p
     * @return
     */
    template<typename T>
    inline bool valid_pointer(T *p) {
        // Get page size and calculate page mask
        size_t pagesz = sysconf(_SC_PAGESIZE);
        size_t pagemask = ~(pagesz - 1);
        // Calculate base address
        void *base = (void *) (((size_t) p) & pagemask);
        return msync(base, sizeof(char), MS_ASYNC) == 0;
    }

#endif

    /**
     * https://github.com/Michoumichmich/SYCL-Hashing-Algorithms/blob/main/include/tools/sycl_queue_helpers.hpp
     * WARNING: If the queue is associated to the host device, some implementation systematically returns true, for any pointer.
     * @tparam debug whether to print the memory location
     * @param q
     * @param ptr
     * @return
     */
    template<typename T, bool debug=false>
    inline bool queue_can_access_ptr(sycl::queue &q, const T *ptr) {
#ifndef IMPLICIT_MEMORY_COPY
        (T *) ptr;
        return false; // If we're not doing implicit memory copies, this test should always fail
#else
        if (q.get_device().is_host()) {
            return valid_pointer(ptr);
        }

        try {
            sycl::get_pointer_device(ptr, q.get_context());
            sycl::usm::alloc alloc_type = sycl::get_pointer_type(ptr, q.get_context());
            if constexpr(debug) {
                std::cerr << "Allocated on:" << q.get_device().get_info<sycl::info::device::name>() << " USM type: ";
                switch (alloc_type) {
                    case sycl::usm::alloc::host:
                        std::cerr << "alloc::host" << '\n';
                        break;
                    case sycl::usm::alloc::device:
                        std::cerr << "alloc::device" << '\n';
                        break;
                    case sycl::usm::alloc::shared:
                        std::cerr << "alloc::shared" << '\n';
                        break;
                    case sycl::usm::alloc::unknown:
                        std::cerr << "alloc::unknown" << '\n';
                        break;
                }
            }
            return alloc_type == sycl::usm::alloc::shared // Shared memory is ok
                   || alloc_type == sycl::usm::alloc::device // Device memory is ok
                   || (alloc_type == sycl::usm::alloc::host && q.get_device().is_cpu()) // We discard host allocated memory because of poor performance unless on the CPU
                    ;
        } catch (...) {
            if constexpr (debug) {
                std::cerr << "Not allocated on:" << q.get_device().get_info<sycl::info::device::name>() << '\n';
            }
            return false;
        }
#endif
    }
} // namespace syclZFP

