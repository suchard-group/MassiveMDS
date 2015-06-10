#ifndef _OPECLMEMORYMANAGEMENT_HPP
#define _OPECLMEMORYMANAGEMENT_HPP

#include <boost/compute.hpp>

namespace mds {
namespace mm {

template <typename T>
using GPUMemoryManager = boost::compute::vector<T>;

} // namespace mm
} // namespace mds

#endif // _OPECLMEMORYMANAGEMENT_HPP
