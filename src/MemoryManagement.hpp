#ifndef _MEMORYMANAGEMENT_HPP
#define _MEMORYMANAGEMENT_HPP

#include <vector>

namespace mds {
namespace mm {

template <typename T>
using MemoryManager = std::vector<T>;


} // namespace mm
} // namespace mds

#endif // _MEMORYMANAGEMENT_HPP