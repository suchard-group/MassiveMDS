#ifndef _MEMORYMANAGEMENT_HPP
#define _MEMORYMANAGEMENT_HPP

#include <vector>
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

} // namespace mm
} // namespace mds

#endif // _MEMORYMANAGEMENT_HPP
