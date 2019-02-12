
#include <chrono>
#include <random>
#include <iostream>
#include <fstream>


#define XSIMD_ENABLE_FALLBACK

#ifdef USE_SIMD
#include "xsimd/xsimd.hpp"
#endif
#include "AbstractMultiDimensionalScaling.hpp"

int main(int argc, char* argv[]) {

#ifdef USE_SIMD
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

//         const auto result = xsimd::select(flag, st(1.0,1.0), st(0.0,0.0));

        const auto result = st(flag()) & st(2.0, 1.0);

        std::cout << result << std::endl;

        xsimd::batch<double, 4> w(double(0));

#endif

#ifdef USE_AVX512

        using D8 = xsimd::batch<double, 8>;
        using D8Bool = xsimd::batch_bool<double, 8>;

        const auto a = D8(1.0);
        const auto b = D8Bool(true, false, true, false,true, false, true, false);

        const auto c = a;
        c ^= b.;

        const auto kernel = xsimd::bool_cast(b);

        std::cout << a <<  " " << xsimd::select(b, a, D8(0.0)) << " " << (xsimd::bool_cast(b) & a) << std::endl;

#endif // USE_AVX512
}
