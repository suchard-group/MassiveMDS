#include <iostream>

#include <boost/compute/core.hpp>
#include <boost/compute/detail/vendor.hpp>

namespace compute = boost::compute;

int main()
{
    // get the default device
    compute::device device = compute::system::default_device();

    // print the device's vendor
    std::cout << device.vendor() << std::endl;
    std::cout << device.platform().vendor() << std::endl;

    std::cout << compute::detail::is_nvidia_device(device) << std::endl;

    return 0;
}
