#ifndef _OPECLMEMORYMANAGEMENT_HPP
#define _OPECLMEMORYMANAGEMENT_HPP

#include <boost/compute.hpp>

namespace mds {
namespace mm {

template <typename T>
using GPUMemoryManager = std::vector<T, util::aligned_allocator<T, 16> >;

} // namespace mm
} // namespace mds

#endif // _OPECLMEMORYMANAGEMENT_HPP
