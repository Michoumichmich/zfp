#pragma once

#include <sycl/sycl.hpp>
#include <iostream>

namespace syclZFP {
    /**
     * https://github.com/Michoumichmich/SYCL-Hashing-Algorithms/blob/main/include/tools/sycl_queue_helpers.hpp
     * WARNING: If the queue is associated to the host device, some implementation systematically returns true, for any pointer.
     * @tparam debug whether to print the memory location
     * @param q
     * @param ptr
     * @return
     */
    template<bool debug = false>
    bool queue_can_access_ptr(sycl::queue &q, const void *ptr) {
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
                   || (q.get_device().is_host() && alloc_type != sycl::usm::alloc::unknown) // If we're on the host, anything known is OK.
                   || (q.get_device().is_cpu() && alloc_type != sycl::usm::alloc::unknown) // ???? is accessing host mem from CPU backend fine ?
                    ;
        } catch (...) {
            if constexpr (debug) {
                std::cerr << "Not allocated on:" << q.get_device().get_info<sycl::info::device::name>() << '\n';
            }
            return false;
        }
    }


} // namespace syclZFP

