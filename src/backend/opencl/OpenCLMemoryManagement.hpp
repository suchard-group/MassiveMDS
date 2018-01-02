#ifndef _OPECLMEMORYMANAGEMENT_HPP
#define _OPECLMEMORYMANAGEMENT_HPP

// #include <boost/compute.hpp>

#include <boost/compute/types.hpp>

namespace mds {

    template <int VectorDimension>
    struct OpenCLFloat {
	    typedef float BaseType;
	    typedef boost::compute::float2_ VectorType;
        static const int dim = 2;
    };

    template <>
    struct OpenCLFloat<4> {
        typedef float BaseType;
        typedef boost::compute::float4_ VectorType;
        static const int dim = 4;
    };

    template <>
    struct OpenCLFloat<8> {
        typedef float BaseType;
        typedef boost::compute::float8_ VectorType;
        static const int dim = 8;
    };

    template <int VectorDimension>
    struct OpenCLDouble {
	    typedef double BaseType;
	    typedef boost::compute::double2_ VectorType;
        static const int dim = 2;
    };

    template <>
    struct OpenCLDouble<4> {
        typedef double BaseType;
        typedef boost::compute::double4_ VectorType;
        static const int dim = 4;
    };

    template <>
    struct OpenCLDouble<8> {
        typedef double BaseType;
        typedef boost::compute::double8_ VectorType;
        static const int dim = 8;
    };

namespace mm {

template <typename T>
using GPUMemoryManager = boost::compute::vector<T>;

	template <typename RealVectorPtr, typename Buffer, typename Queue>
	void bufferedCopyFromDevice(RealVectorPtr begin,
								RealVectorPtr end,
								double *destination,
								Buffer& buffer, Queue& queue);

	template <typename Buffer, typename Queue>
	void bufferedCopyFromDevice(mm::GPUMemoryManager<double>::iterator begin,
								mm::GPUMemoryManager<double>::iterator end,
								double *destination,
							    Buffer&, Queue& queue) {

		boost::compute::copy(begin, end, destination, queue);
	}

	template <typename Buffer, typename Queue>
	void bufferedCopyFromDevice(mm::GPUMemoryManager<boost::compute::double2_>::iterator begin,
								mm::GPUMemoryManager<boost::compute::double2_>::iterator end,
								double *destination,
								Buffer&, Queue& queue) {
		using namespace boost::compute;

		copy(begin, end, reinterpret_cast<double2_ *>(destination), queue);
	}

    template <typename Buffer, typename Queue>
    void bufferedCopyFromDevice(mm::GPUMemoryManager<boost::compute::double4_>::iterator begin,
                                mm::GPUMemoryManager<boost::compute::double4_>::iterator end,
                                double *destination,
                                Buffer&, Queue& queue) {
        using namespace boost::compute;

        copy(begin, end, reinterpret_cast<double4_ *>(destination), queue);
    }

    template <typename Buffer, typename Queue>
    void bufferedCopyFromDevice(mm::GPUMemoryManager<boost::compute::double8_>::iterator begin,
                                mm::GPUMemoryManager<boost::compute::double8_>::iterator end,
                                double *destination,
                                Buffer&, Queue& queue) {
        using namespace boost::compute;

        copy(begin, end, reinterpret_cast<double8_ *>(destination), queue);
    }

	template <typename Buffer, typename Queue>
	void bufferedCopyFromDevice(mm::GPUMemoryManager<float>::iterator begin,
								mm::GPUMemoryManager<float>::iterator end,
								double *destination,
								Buffer& buffer, Queue& queue) {

		const auto length = std::distance(begin, end);
		if (buffer.size() < length) {
			buffer.resize(length);
		}

		boost::compute::copy(begin, end, std::begin(buffer), queue);
		std::copy(std::begin(buffer), std::begin(buffer) + length, destination);
	}

	template <typename Buffer, typename Queue>
	void bufferedCopyFromDevice(mm::GPUMemoryManager<boost::compute::float2_>::iterator begin,
								mm::GPUMemoryManager<boost::compute::float2_>::iterator end,
								double* destination,
								Buffer& buffer, Queue& queue) {
		using namespace boost::compute;

		const auto length = std::distance(begin, end) * 2;
		if (buffer.size() < length) {
			buffer.resize(length);
		}

		boost::compute::copy(begin, end, reinterpret_cast<float2_ *>(&buffer[0]), queue);
		std::copy(std::begin(buffer), std::begin(buffer) + length, destination);
	}

    template <typename Buffer, typename Queue>
    void bufferedCopyFromDevice(mm::GPUMemoryManager<boost::compute::float4_>::iterator begin,
                                mm::GPUMemoryManager<boost::compute::float4_>::iterator end,
                                double* destination,
                                Buffer& buffer, Queue& queue) {
        using namespace boost::compute;

        const auto length = std::distance(begin, end) * 4;
        if (buffer.size() < length) {
            buffer.resize(length);
        }

        boost::compute::copy(begin, end, reinterpret_cast<float4_ *>(&buffer[0]), queue);
        std::copy(std::begin(buffer), std::begin(buffer) + length, destination);
    }

    template <typename Buffer, typename Queue>
    void bufferedCopyFromDevice(mm::GPUMemoryManager<boost::compute::float8_>::iterator begin,
                                mm::GPUMemoryManager<boost::compute::float8_>::iterator end,
                                double* destination,
                                Buffer& buffer, Queue& queue) {
        using namespace boost::compute;

        const auto length = std::distance(begin, end) * 8;
        if (buffer.size() < length) {
            buffer.resize(length);
        }

        boost::compute::copy(begin, end, reinterpret_cast<float8_ *>(&buffer[0]), queue);
        std::copy(std::begin(buffer), std::begin(buffer) + length, destination);
    }

template <typename RealVectorPtr, typename Buffer, typename Queue>
void bufferedCopyToDevice(double *begin, double *end, RealVectorPtr destination,
		Buffer& buffer, Queue& queue);

template <typename Buffer, typename Queue>
void bufferedCopyToDevice(double *begin, double *end,
		mm::GPUMemoryManager<double>::iterator destination, Buffer&, Queue& queue) {

	boost::compute::copy(begin, end, destination, queue);
}

template <typename Buffer, typename Queue>
void bufferedCopyToDevice(double *begin, double *end,
		mm::GPUMemoryManager<boost::compute::double2_>::iterator destination, Buffer&, Queue& queue) {
	using namespace boost::compute;

	copy(
		reinterpret_cast<double2_ *>(begin),
		reinterpret_cast<double2_ *>(begin) + std::distance(begin, end) / 2,
		destination, queue);
}

    template <typename Buffer, typename Queue>
    void bufferedCopyToDevice(double *begin, double *end,
                              mm::GPUMemoryManager<boost::compute::double4_>::iterator destination, Buffer&, Queue& queue) {
        using namespace boost::compute;

        copy(
                reinterpret_cast<double4_ *>(begin),
                reinterpret_cast<double4_ *>(begin) + std::distance(begin, end) / 4,
                destination, queue);
    }

template <typename Buffer, typename Queue>
void bufferedCopyToDevice(double *begin, double *end,
		mm::GPUMemoryManager<float>::iterator destination, Buffer& buffer, Queue& queue) {

	const auto length = std::distance(begin, end);
	if (buffer.size() < length) {
		buffer.resize(length);
	}
	std::copy(begin, end, std::begin(buffer));

	boost::compute::copy(std::begin(buffer), std::begin(buffer) + length,
		destination, queue);
}

template <typename Buffer, typename Queue>
void bufferedCopyToDevice(double *begin, double *end,
		mm::GPUMemoryManager<boost::compute::float2_>::iterator destination, Buffer& buffer, Queue& queue) {
	using namespace boost::compute;

	const auto length = std::distance(begin, end);
	if (buffer.size() < length) {
		buffer.resize(length);
	}
	std::copy(begin, end, std::begin(buffer));

	copy(
		reinterpret_cast<float2_ *>(&buffer[0]),
		reinterpret_cast<float2_ *>(&buffer[0]) + length / 2,
		destination, queue);
}

    template <typename Buffer, typename Queue>
    void bufferedCopyToDevice(double *begin, double *end,
                              mm::GPUMemoryManager<boost::compute::float4_>::iterator destination, Buffer& buffer, Queue& queue) {
        using namespace boost::compute;

        const auto length = std::distance(begin, end);
        if (buffer.size() < length) {
            buffer.resize(length);
        }
        std::copy(begin, end, std::begin(buffer));

        copy(reinterpret_cast<float4_ *>(&buffer[0]),
             reinterpret_cast<float4_ *>(&buffer[0]) + length / 4,
             destination, queue);
    }

    template <typename BufferPtr, typename Queue>
    void copyToDevice(BufferPtr begin, BufferPtr end,
                      mm::GPUMemoryManager<boost::compute::float2_>::iterator destination, Queue& queue) {
        using namespace boost::compute;

        const auto length = std::distance(begin, end);
        copy(reinterpret_cast<float2_ *>(&(*begin)),
             reinterpret_cast<float2_ *>(&(*begin)) + length / 2,
             destination, queue);
    };

    template <typename BufferPtr, typename Queue>
    void copyToDevice(BufferPtr begin, BufferPtr end,
                      mm::GPUMemoryManager<boost::compute::float4_>::iterator destination, Queue& queue) {
        using namespace boost::compute;

        const auto length = std::distance(begin, end);
        copy(reinterpret_cast<float4_ *>(&(*begin)),
             reinterpret_cast<float4_ *>(&(*begin)) + length / 4,
             destination, queue);
    };

    template <typename BufferPtr, typename Queue>
    void copyToDevice(BufferPtr begin, BufferPtr end,
                      mm::GPUMemoryManager<boost::compute::float8_>::iterator destination, Queue& queue) {
        using namespace boost::compute;

        const auto length = std::distance(begin, end);
        copy(reinterpret_cast<float8_ *>(&(*begin)),
             reinterpret_cast<float8_ *>(&(*begin)) + length / 8,
             destination, queue);
    };

    template <typename BufferPtr, typename Queue>
    void copyToDevice(BufferPtr begin, BufferPtr end,
                      mm::GPUMemoryManager<boost::compute::double2_>::iterator destination, Queue& queue) {
        using namespace boost::compute;

        const auto length = std::distance(begin, end);
        copy(reinterpret_cast<double2_ *>(&*begin),
             reinterpret_cast<double2_ *>(&*begin) + length / 2,
             destination, queue);
    };

    template <typename BufferPtr, typename Queue>
    void copyToDevice(BufferPtr begin, BufferPtr end,
                      mm::GPUMemoryManager<boost::compute::double4_>::iterator destination, Queue& queue) {
        using namespace boost::compute;

        const auto length = std::distance(begin, end);
        copy(reinterpret_cast<double4_ *>(&*begin),
             reinterpret_cast<double4_ *>(&*begin) + length / 4,
             destination, queue);
    };

    template <typename BufferPtr, typename Queue>
    void copyToDevice(BufferPtr begin, BufferPtr end,
                      mm::GPUMemoryManager<boost::compute::double8_>::iterator destination, Queue& queue) {
        using namespace boost::compute;

        const auto length = std::distance(begin, end);
        copy(reinterpret_cast<double8_ *>(&*begin),
             reinterpret_cast<double8_ *>(&*begin) + length / 8,
             destination, queue);
    };

} // namespace mm
} // namespace mds

#endif // _OPECLMEMORYMANAGEMENT_HPP
