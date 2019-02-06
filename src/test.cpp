
#include <chrono>
#include <random>
#include <iostream>
#include <fstream>


#define XSIMD_ENABLE_FALLBACK

#include "xsimd/xsimd.hpp"
#include "AbstractMultiDimensionalScaling.hpp"

int main(int argc, char* argv[]) {

        xsimd::batch<double, 2> x{1.0, 1.0};
        x = mds::math::phi_new(x);
        std::cout << x << " : " << decltype(x)::size << std::endl;

        xsimd::batch<double, 1> y{1.0};
        y = mds::math::phi_new(y);
        std::cout << y << " : " << decltype(y)::size << std::endl;

        double z{1.0};
        z = mds::math::phi_new(z);
        std::cout << z << std::endl;

        const int length = 11;

        // How to non-mod-0 loops
        const int simd_size = xsimd::batch<double, 4>::size;
        const int vec_size = length - length % simd_size;

        for (int i = 0; i < vec_size; i += simd_size) {
			// Do stuff mod 2
        }

        for (int i = vec_size; i < length; ++i) {
        	std::cout << "Remainder" << std::endl;
        }

        xsimd::batch_bool<double, 2> flag{true, false};
        using st = xsimd::batch<double, 2>;

        const auto result = xsimd::select(flag, st(1.0,1.0), st(0.0,0.0));

        std::cout << result << std::endl;

}
