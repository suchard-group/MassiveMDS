#ifndef _DISTANCE_HPP
#define _DISTANCE_HPP

#include <numeric>
#include <vector>

#define XSIMD_ENABLE_FALLBACK

#include "xsimd/xsimd.hpp"
#include "MemoryManagement.hpp"

namespace mds {

struct Generic {};
struct NonGeneric {};

using D2 = xsimd::batch<double, 2>;
using D2Bool = xsimd::batch_bool<double, 2>;

using D1 = xsimd::batch<double, 1>;
using D1Bool = xsimd::batch_bool<double, 1>;

template <typename SimdType, typename RealType, typename Algorithm>
class DistanceDispatch {

public:

	using IteratorType = typename mm::MemoryManager<RealType>::iterator;

	DistanceDispatch(const mm::MemoryManager<RealType>& locations, const int i, const int embeddingDimension) :
		locations(locations), i(i), embeddingDimension(embeddingDimension),
		iterator(locations.begin()), start(iterator + i * embeddingDimension) { }


	inline SimdType calculate(int j) const;


private:

	const mm::MemoryManager<RealType>& locations;
	const int i;
	const int embeddingDimension;

	const decltype(std::begin(locations)) iterator;
	const decltype(std::begin(locations)) start;
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

    template <typename Iterator>
    float calculateDistance2(Iterator iX, Iterator iY, int length, float) {

        //using AlignedValueType = typename HostVectorType::allocator_type::aligned_value_type;

        typedef float aligned_float __attribute__((aligned(16)));
        typedef aligned_float* SSE_PTR;

        SSE_PTR __restrict__ x = &*iX;
        SSE_PTR __restrict__ y = &*iY;

        auto a = _mm_loadu_ps(x);
        auto b = _mm_loadu_ps(y); // TODO second call is not aligned without padding
        auto c = a - b;

        const int mask = 0x31;
        __m128 d = _mm_dp_ps(c, c, mask);
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

	   const int mask = 0x31;
	   __m128d d = _mm_dp_pd(diff, diff, mask);
	   return  std::sqrt(_mm_cvtsd_f64(d));
    }

    template <typename Iterator>
    typename Iterator::value_type calculateDistanceScalar(Iterator x, Iterator y, int length) {

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

template <>
inline D1 DistanceDispatch<D1, D1::value_type, Generic>::calculate(int j) const {

	const auto distance = D1(
						impl::calculateDistanceGeneric2(
								start,
								iterator + j * embeddingDimension,
								embeddingDimension)
				);

	return distance;
}

template <>
inline D1 DistanceDispatch<D1, D1::value_type, NonGeneric>::calculate(int j) const {

	const auto distance = D1(
						impl::calculateDistance2(
								start,
								iterator + j * embeddingDimension,
								embeddingDimension, D1::value_type())
				);

	return distance;
}

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

	return
						impl::calculateDistance2(
								start,
								iterator + j * embeddingDimension,
								embeddingDimension, double());
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

template <>
inline D2 SimdHelper<D2, D2::value_type>::get(const double* iterator) {
		return D2(iterator, xsimd::unaligned_mode());
}

template <>
inline void SimdHelper<D2, D2::value_type>::put(D2 x, double* iterator) {
	x.store_unaligned(iterator);
}

template <>
inline D1 SimdHelper<D1, D1::value_type>::get(const double* iterator) {
		return D1(iterator, xsimd::unaligned_mode());
}

template <>
inline void SimdHelper<D1, D1::value_type>::put(D1 x, double* iterator) {
	x.store_unaligned(iterator);
}

template <>
inline double SimdHelper<double, double>::get(const double* iterator) {
		return *iterator;
}

template <>
inline void SimdHelper<double, double>::put(double x, double* iterator) {
	*iterator = x;
}

// template <>
// inline D2Bool SimdHelper<D2>::missing(int i, int j, D2 x) {
// 	return D2Bool(i == j, i == j + 1) || xsimd::isnan(x);
// }
//
// template <>
// inline D2 SimdHelper<D2>::mask(D2Bool x) {
// 	return xsimd::select(x, D2(0.0, 0.0), D2(1.0, 1.0));
// }
//
// template <>
// inline bool SimdHelper<D2>::any(D2Bool x) {
// 	return xsimd::any(x);
// }


} // namespace mds

#endif // _DISTANCE_HPP
