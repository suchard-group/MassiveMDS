#ifndef _DISTANCE_HPP
#define _DISTANCE_HPP

#include <numeric>
#include <vector>

//#define XSIMD_ENABLE_FALLBACK

#ifdef USE_SIMD
  #if defined(__ARM64_ARCH_8__)
    #include "sse2neon.h"
    #undef USE_AVX
    #undef USE_AVX512
  #endif
  #include "xsimd/xsimd.hpp"
#endif

#include "MemoryManagement.hpp"

namespace mds {

struct Generic {};
struct NonGeneric {};

#ifdef USE_SIMD

#ifdef USE_AVX
    using D4 = xsimd::batch<double, 4>;
    using D4Bool = xsimd::batch_bool<double, 4>;
#endif

#ifdef USE_SSE
using D2 = xsimd::batch<double, 2>;
using D2Bool = xsimd::batch_bool<double, 2>;

//using D1 = xsimd::batch<double, 1>;
//using D1Bool = xsimd::batch_bool<double, 1>;

using S4 = xsimd::batch<float, 4>;
using S4Bool = xsimd::batch_bool<float, 4>;
#endif

#endif

#ifdef USE_AVX512
using D8 = xsimd::batch<double, 8>;
using D8Bool = xsimd::batch_bool<double, 8>;
#endif // USE_AVX512

template <typename SimdType, typename RealType, typename Algorithm>
class DistanceDispatch {
private:

  const mm::MemoryManager<RealType>& locations;
  const int i;
  const int embeddingDimension;
  const int rowOffset;
  const int columnOffset;

  const decltype(std::begin(locations)) iterator;
  const decltype(std::begin(locations)) start;

public:

	using IteratorType = typename mm::MemoryManager<RealType>::iterator;

	DistanceDispatch(const mm::MemoryManager<RealType>& locations, const int i, const int embeddingDimension, const int rowOffset, const int columnOffset) :
		locations(locations), i(i), embeddingDimension(embeddingDimension),
		rowOffset(rowOffset),
		columnOffset(columnOffset),
		iterator(locations.begin() + columnOffset * embeddingDimension), start(locations.begin() + (i + rowOffset) * embeddingDimension) { }

	inline SimdType calculate(int j) const;

	inline int getRowLocationOffset() const {
	  return rowOffset * embeddingDimension;
	}

	inline int getRowOffset() const {
	  return rowOffset;
	}

  inline int getColumnLocationOffset() const {
    return columnOffset * embeddingDimension;
  }

	inline int getColumnOffset() const {
	  return columnOffset;
	}
};

 namespace impl {

	template <typename Iterator>
	typename Iterator::value_type calculateDistanceGeneric2(Iterator iX, Iterator iY, int length) {

		typename Iterator::value_type sum{0};

		for (int i = 0; i < length; ++i, ++iX, ++iY) {
			const auto diff = *iX - *iY;
			sum += diff * diff;
		}

		return std::sqrt(sum);
	}

#ifdef USE_SIMD
    template <typename Iterator>
    float calculateDistance2(Iterator iX, Iterator iY, int length, float) {

        //using AlignedValueType = typename HostVectorType::allocator_type::aligned_value_type;

        typedef const float aligned_float __attribute__((aligned(16)));
        typedef aligned_float* SSE_PTR;

        SSE_PTR __restrict__ x = &*iX;
        SSE_PTR __restrict__ y = &*iY;

        auto a = _mm_loadu_ps(x);
        auto b = _mm_loadu_ps(y); // TODO second call is not aligned without padding
        auto c = a - b;

       // const int mask = 0x31;
        __m128 d = _mm_dp_ps(c, c, 0x31);
        return  _mm_cvtss_f32(_mm_sqrt_ps(d));
    }

    template <typename Iterator>
    double calculateDistance2(Iterator iX, Iterator iY, int length, double) {

        using AlignedValueType = const typename mm::MemoryManager<double>::allocator_type::aligned_value_type;

        AlignedValueType* x = &*iX;
        AlignedValueType* y = &*iY;

        __m128d xv = _mm_load_pd(x);
        __m128d yv = _mm_load_pd(y);

       __m128d diff = _mm_sub_pd(xv, yv);

	   //const int mask = 0x31;
	   __m128d d = _mm_dp_pd(diff, diff, 0x31);
	   return  std::sqrt(_mm_cvtsd_f64(d));
    }
#endif // USE_SIMD

    template <typename Iterator>
    typename Iterator::value_type calculateDistanceScalar(Iterator x, Iterator y, int) {

        typename Iterator::value_type sum{0};

        for (int i = 0; i < 2; ++i, ++x, ++y) {
            const auto difference = *x - *y;
            sum += difference * difference;
        }
        return std::sqrt(sum);
    }

	template <typename Iterator>
	typename Iterator::value_type calculateDistanceGenericScalar(Iterator x, Iterator y, int length) {
		typename Iterator::value_type sum{0};

		for (int i = 0; i < length; ++i, ++x, ++y) {
			const auto difference = *x - *y;
			sum += difference * difference;
		}
		return std::sqrt(sum);
	}


} // namespace impl

#ifdef USE_SIMD

#ifdef USE_AVX
    template <>
    inline D4 DistanceDispatch<D4, D4::value_type, Generic>::calculate(int j) const {

        const auto distance = D4(
                impl::calculateDistanceGeneric2(
                        start,
                        iterator + j * embeddingDimension,
                        embeddingDimension),
                impl::calculateDistanceGeneric2(
                        start,
                        iterator + (j + 1) * embeddingDimension,
                        embeddingDimension),
                impl::calculateDistanceGeneric2(
                        start,
                        iterator + (j + 2) * embeddingDimension,
                        embeddingDimension),
                impl::calculateDistanceGeneric2(
                        start,
                        iterator + (j + 3) * embeddingDimension,
                        embeddingDimension)
        );

        return distance;
    }
#endif

#ifdef USE_AVX
    template <>
    inline D4 DistanceDispatch<D4, D4::value_type, NonGeneric>::calculate(int j) const {

        const auto distance = D4(
                impl::calculateDistance2(
                        start,
                        iterator + j * embeddingDimension,
                        embeddingDimension, D4::value_type()),
                impl::calculateDistance2(
                        start,
                        iterator + (j + 1) * embeddingDimension,
                        embeddingDimension, D4::value_type()),
                impl::calculateDistance2(
                        start,
                        iterator + (j + 2) * embeddingDimension,
                        embeddingDimension, D4::value_type()),
                impl::calculateDistance2(
                        start,
                        iterator + (j + 3) * embeddingDimension,
                        embeddingDimension, D4::value_type())
        );

        return distance;
    }
#endif

#ifdef USE_SSE
template <>
inline D2 DistanceDispatch<D2, D2::value_type, Generic>::calculate(int j) const {

	const auto distance = D2(
						impl::calculateDistanceGeneric2(
								start,
								iterator + j * embeddingDimension,
								embeddingDimension),
						impl::calculateDistanceGeneric2(
								start,
								iterator + (j + 1) * embeddingDimension,
								embeddingDimension)
				);

	return distance;
}

template <>
inline D2 DistanceDispatch<D2, D2::value_type, NonGeneric>::calculate(int j) const {

	const auto distance = D2(
						impl::calculateDistance2(
								start,
								iterator + j * embeddingDimension,
								embeddingDimension, D2::value_type()),
						impl::calculateDistance2(
								start,
								iterator + (j + 1) * embeddingDimension,
								embeddingDimension, D2::value_type())
				);

	return distance;
}

#endif //USE_SSE

//template <>
//inline D1 DistanceDispatch<D1, D1::value_type, Generic>::calculate(int j) const {
//
//	const auto distance = D1(
//						impl::calculateDistanceGeneric2(
//								start,
//								iterator + j * embeddingDimension,
//								embeddingDimension)
//				);
//
//	return distance;
//}

//template <>
//inline D1 DistanceDispatch<D1, D1::value_type, NonGeneric>::calculate(int j) const {
//
//	const auto distance = D1(
//						impl::calculateDistance2(
//								start,
//								iterator + j * embeddingDimension,
//								embeddingDimension, D1::value_type())
//				);
//
//	return distance;
//}
#endif // USE_SIMD

#ifdef USE_AVX512
    template <>
    inline D8 DistanceDispatch<D8, D8::value_type, Generic>::calculate(int j) const {

        const auto distance = D8(
                impl::calculateDistanceGeneric2(
                        start,
                        iterator + j * embeddingDimension,
                        embeddingDimension),
                impl::calculateDistanceGeneric2(
                        start,
                        iterator + (j + 1) * embeddingDimension,
                        embeddingDimension),
                impl::calculateDistanceGeneric2(
                        start,
                        iterator + (j + 2) * embeddingDimension,
                        embeddingDimension),
                impl::calculateDistanceGeneric2(
                        start,
                        iterator + (j + 3) * embeddingDimension,
                        embeddingDimension),
                impl::calculateDistanceGeneric2(
                        start,
                        iterator + (j + 4) * embeddingDimension,
                        embeddingDimension),
                impl::calculateDistanceGeneric2(
                        start,
                        iterator + (j + 5) * embeddingDimension,
                        embeddingDimension),
                impl::calculateDistanceGeneric2(
                        start,
                        iterator + (j + 6) * embeddingDimension,
                        embeddingDimension),
                impl::calculateDistanceGeneric2(
                        start,
                        iterator + (j + 7) * embeddingDimension,
                        embeddingDimension)
        );

        return distance;
    }

    template <>
    inline D8 DistanceDispatch<D8, D8::value_type, NonGeneric>::calculate(int j) const {

        const auto distance = D8(
                impl::calculateDistance2(
                        start,
                        iterator + j * embeddingDimension,
                        embeddingDimension, D8::value_type()),
                impl::calculateDistance2(
                        start,
                        iterator + (j + 1) * embeddingDimension,
                        embeddingDimension, D8::value_type()),
                impl::calculateDistance2(
                        start,
                        iterator + (j + 2) * embeddingDimension,
                        embeddingDimension, D8::value_type()),
                impl::calculateDistance2(
                        start,
                        iterator + (j + 3) * embeddingDimension,
                        embeddingDimension, D8::value_type()),
                impl::calculateDistance2(
                        start,
                        iterator + (j + 4) * embeddingDimension,
                        embeddingDimension, D8::value_type()),
                impl::calculateDistance2(
                        start,
                        iterator + (j + 5) * embeddingDimension,
                        embeddingDimension, D8::value_type()),
                impl::calculateDistance2(
                        start,
                        iterator + (j + 6) * embeddingDimension,
                        embeddingDimension, D8::value_type()),
                impl::calculateDistance2(
                        start,
                        iterator + (j + 7) * embeddingDimension,
                        embeddingDimension, D8::value_type())
        );

        return distance;
    }

#endif // USE_AVX512

template <>
inline double DistanceDispatch<double, double, Generic>::calculate(int j) const {
	return
						impl::calculateDistanceGeneric2(
								start,
								iterator + j * embeddingDimension,
								embeddingDimension);

}

template <>
inline double DistanceDispatch<double, double, NonGeneric>::calculate(int j) const {
#if USE_SIMD
    return impl::calculateDistance2(
            start,
            iterator + j * embeddingDimension,
            embeddingDimension, double());
#else
    return impl::calculateDistanceScalar(
            start,
            iterator + j * embeddingDimension,
            embeddingDimension);
#endif
}

#ifdef USE_SIMD
#ifdef USE_SSE
	template <>
	inline S4 DistanceDispatch<S4, S4::value_type, Generic>::calculate(int j) const {

		const auto distance = S4(
				impl::calculateDistanceGeneric2(
						start,
						iterator + j * embeddingDimension,
						embeddingDimension),
				impl::calculateDistanceGeneric2(
						start,
						iterator + (j + 1) * embeddingDimension,
						embeddingDimension),
				impl::calculateDistanceGeneric2(
						start,
						iterator + (j + 2) * embeddingDimension,
						embeddingDimension),
				impl::calculateDistanceGeneric2(
						start,
						iterator + (j + 3) * embeddingDimension,
						embeddingDimension)
		);

		return distance;
	}

	template <>
	inline S4 DistanceDispatch<S4, S4::value_type, NonGeneric>::calculate(int j) const {

		const auto distance = S4(
				impl::calculateDistance2(
						start,
						iterator + j * embeddingDimension,
						embeddingDimension, S4::value_type()),
				impl::calculateDistance2(
						start,
						iterator + (j + 1) * embeddingDimension,
						embeddingDimension, S4::value_type()),
				impl::calculateDistance2(
						start,
						iterator + (j + 2) * embeddingDimension,
						embeddingDimension, S4::value_type()),
				impl::calculateDistance2(
						start,
						iterator + (j + 3) * embeddingDimension,
						embeddingDimension, S4::value_type())
		);

		return distance;
	}

#endif //USE_SSE
#endif //USE_SIMD

	template <>
	inline float DistanceDispatch<float, float, Generic>::calculate(int j) const {

		return
				impl::calculateDistanceGeneric2(
						start,
						iterator + j * embeddingDimension,
						embeddingDimension);

	}

	template <>
	inline float DistanceDispatch<float, float, NonGeneric>::calculate(int j) const {
#ifdef USE_SIMD
        return impl::calculateDistance2(
                start,
                iterator + j * embeddingDimension,
                embeddingDimension, float());
#else
        return impl::calculateDistanceScalar(
                start,
                iterator + j * embeddingDimension,
                embeddingDimension);
#endif
	}

template <typename SimdType, typename RealType>
class SimdHelper {
public:

// 	using SimdBool = xsimd::batch_bool<typename SimdType::value_type, SimdType::size>;

	static inline SimdType get(const RealType* iterator);
	static inline void put(SimdType x, RealType* iterator);
// 	static inline SimdBool missing(int i, int j, SimdType x);
// 	static inline SimdType mask(SimdBool x);
// 	static inline bool any(SimdBool x);

};

#ifdef USE_SIMD
#ifdef USE_AVX
    template <>
    inline D4 SimdHelper<D4, D4::value_type>::get(const double* iterator) {
        return D4(iterator, xsimd::unaligned_mode());
    }
#endif

#ifdef USE_SSE
    template <>
    inline D2 SimdHelper<D2, D2::value_type>::get(const double* iterator) {
        return {iterator, xsimd::unaligned_mode()};
    }

	template <>
	inline S4 SimdHelper<S4, S4::value_type>::get(const float* iterator) {
		return {iterator, xsimd::unaligned_mode()};
	}
#endif
#ifdef USE_AVX
    template <>
    inline void SimdHelper<D4, D4::value_type>::put(D4 x, double* iterator) {
        x.store_unaligned(iterator);
    }
#endif
#ifdef USE_SSE
    template <>
    inline void SimdHelper<D2, D2::value_type>::put(D2 x, double* iterator) {
        x.store_unaligned(iterator);
    }


//template <>
//inline D1 SimdHelper<D1, D1::value_type>::get(const double* iterator) {
//		return D1(iterator, xsimd::unaligned_mode());
//}
//
//template <>
//inline void SimdHelper<D1, D1::value_type>::put(D1 x, double* iterator) {
//	x.store_unaligned(iterator);
//}

	template <>
	inline void SimdHelper<S4, S4::value_type>::put(S4 x, float* iterator) {
		x.store_unaligned(iterator);
	}

#endif //USE_SSE
#endif // USE_SIMD

#ifdef USE_AVX512
    template <>
    inline D8 SimdHelper<D8, D8::value_type>::get(const double* iterator) {
        return D8(iterator, xsimd::unaligned_mode());
    }

    template <>
    inline void SimdHelper<D8, D8::value_type>::put(D8 x, double* iterator) {
        x.store_unaligned(iterator);
    }
#endif // USE_AVX512

    template <>
    inline double SimdHelper<double, double>::get(const double* iterator) {
        return *iterator;
    }

    template <>
    inline float SimdHelper<float, float>::get(const float* iterator) {
        return *iterator;
    }

    template <>
    inline void SimdHelper<double, double>::put(double x, double* iterator) {
        *iterator = x;
    }

    template <>
    inline void SimdHelper<float, float>::put(float x, float* iterator) {
        *iterator = x;
    }
} // namespace mds

#endif // _DISTANCE_HPP
