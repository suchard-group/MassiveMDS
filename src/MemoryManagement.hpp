#ifndef _MEMORYMANAGEMENT_HPP
#define _MEMORYMANAGEMENT_HPP

#include <vector>
#include <algorithm>
#include <iostream> // TODO Remove
#include "aligned_allocator.hpp"

namespace mds {
namespace mm {

// enum class Alignment : size_t
// {
//     Normal = sizeof(void*),
//     SSE    = 16,
//     AVX    = 32,
// };
//
// namespace detail {
//     void* allocate_aligned_memory(size_t align, size_t size);
//     void deallocate_aligned_memory(void* ptr) noexcept;
// }
//
// template <typename T, Alignment Align = Alignment::AVX>
// class AlignedAllocator;
//
// template <typename T>
// using MemoryManager = std::vector<T>;

template <typename T>
using MemoryManager = std::vector<T, util::aligned_allocator<T, 16> >;


// Copy functionality

template <typename RealVectorPtr, typename Buffer>
void bufferedCopy(double *begin, double *end, RealVectorPtr destination, Buffer& buffer);

template <typename Buffer>
void bufferedCopy(double *begin, double *end,
		mm::MemoryManager<double>::iterator destination, Buffer&) {
	std::copy(begin, end, destination);
}

template <typename Buffer>
void bufferedCopy(double *begin, double *end,
		mm::MemoryManager<float>::iterator destination, Buffer& buffer) {
	std::copy(begin, end, destination);
}

template <typename RealVectorPtr, typename Buffer>
void bufferedCopy(RealVectorPtr begin, RealVectorPtr end, double* destination, Buffer& buffer);

template <typename Buffer>
void bufferedCopy(mm::MemoryManager<double>::iterator begin, 
		mm::MemoryManager<double>::iterator end, 
        double* destination, Buffer& buffer) {
	std::copy(begin, end, destination);                  
}

template <typename Buffer>
void bufferedCopy(mm::MemoryManager<float>::iterator begin, 
		mm::MemoryManager<float>::iterator end, 
        double* destination, Buffer& buffer) {
	std::copy(begin, end, destination);                  
}
        // Padded copy functionality

        template <typename SourceType, typename DestinationType, typename Buffer>
        void paddedBufferedCopy(SourceType source, int sourceStride, int length,
                                DestinationType destination, int destinationStride,
                                int count,
                                Buffer& buffer) {

            for (int i = 0; i < count; ++i) {
                mm::bufferedCopy(source, source + length, destination, buffer);
                source += sourceStride;
                destination += destinationStride;
            }
        }

    } // namespace mm
} // namespace mds

#endif // _MEMORYMANAGEMENT_HPP
