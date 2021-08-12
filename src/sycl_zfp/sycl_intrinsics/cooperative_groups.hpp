#pragma once

#include <sycl/sycl.hpp>

#ifndef ATOMIC_REF_NAMESPACE
#ifdef USING_DPCPP
#define ATOMIC_REF_NAMESPACE sycl::ext::oneapi
#else
#define ATOMIC_REF_NAMESPACE sycl
#endif
#endif

template<int dim>
class nd_range_barrier {
private:
    using mask_t = uint64_t;
    mask_t groups_waiting_ = 0;
    const mask_t barrier_mask_ = 0;
    mask_t reached_ = 0;

    static mask_t compute_barrier_mask(size_t group_count, const std::initializer_list<size_t> &cooperating_groups) {
        mask_t out = 0;
        if (cooperating_groups.size() == 0) {
            for (size_t gr = 0; gr < group_count; ++gr) {
                out |= mask_t(1) << gr;
            }
        } else {
            for (auto e: cooperating_groups) {
                if (e >= group_count) throw std::out_of_range("Making barrier on out of range group");
                out |= mask_t(1) << e;
            }
        }
        return out;
    }

    static mask_t compute_item_mask(const sycl::nd_item<dim> &this_item) {
        return mask_t(1) << this_item.get_group_linear_id();
    }

    nd_range_barrier(
            sycl::queue &q,
            sycl::nd_range<dim> kernel_range,
            std::initializer_list<size_t> cooperating_groups) :
            barrier_mask_(compute_barrier_mask(kernel_range.get_group_range().size(), cooperating_groups)) {
        if (kernel_range.get_group_range().size() > sizeof(mask_t) * 8) {
            throw std::runtime_error("Not implemented.");
        }

        if (kernel_range.get_group_range().size() > q.get_device().get_info<sycl::info::device::max_compute_units>()) {
            throw std::runtime_error("Too much groups requested on cooperative barrier. Forward progress not guaranteed.");
        }

        if (kernel_range.get_local_range().size() > q.get_device().get_info<sycl::info::device::max_work_group_size>()) {
            throw std::runtime_error("Too much items per group. Forward progress not guaranteed.");
        }
    }

public:

    static nd_range_barrier<dim> *make_barrier(
            sycl::queue &q,
            sycl::nd_range<dim> kernel_range,
            std::initializer_list<size_t> cooperating_groups = {}) {
        auto barrier = sycl::malloc_shared<nd_range_barrier<dim>>(1, q);
        return new(barrier) nd_range_barrier<dim>(q, kernel_range, cooperating_groups);
    }


    void wait(sycl::nd_item<dim> this_item) {
        const mask_t this_group_mask = compute_item_mask(this_item);

        if ((this_group_mask & barrier_mask_) == 0) return;

        this_item.barrier(sycl::access::fence_space::local_space);
        /* Choosing one work item to perform the work */
        if (this_item.get_local_linear_id() == 0) {
            using atomic_ref_t = ATOMIC_REF_NAMESPACE::atomic_ref<
                    mask_t,
                    ATOMIC_REF_NAMESPACE::memory_order::acq_rel,
                    ATOMIC_REF_NAMESPACE::memory_scope::device,
                    sycl::access::address_space::global_space
            >;
            atomic_ref_t groups_waiting_ref(groups_waiting_);
            atomic_ref_t barrier_reached_ref(reached_);

            /* Waiting before entering the barrier */
            while (barrier_reached_ref.load() != 0) {}

            /* Registring this group at the barrier. */
            groups_waiting_ref.fetch_or(this_group_mask);

            if (groups_waiting_ref.load() == barrier_mask_) {
                barrier_reached_ref.store(1);
            } else {
                while (barrier_reached_ref.load() != 1) {}
            }

            /* This group leaves the barrier. */
            groups_waiting_ref.fetch_and(~this_group_mask);

            if (groups_waiting_ref.load() == 0) {
                barrier_reached_ref.store(0);
            } else {
                while (barrier_reached_ref.load() != 0) {}
            }

        }

        this_item.barrier(sycl::access::fence_space::local_space);
    }
};

