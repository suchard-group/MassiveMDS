#include "AbstractMultiDimensionalScaling.hpp"

// forward reference
namespace mds {
	SharedPtr constructMultiDimensionalScalingDouble(int, int, long);
	SharedPtr constructNewMultiDimensionalScalingDoubleNoParallel(int, int, long, int);
	SharedPtr constructNewMultiDimensionalScalingDoubleTbb(int, int, long, int);
	SharedPtr constructOpenCLMultiDimensionalScalingDouble(int, int, long, int);

	SharedPtr constructMultiDimensionalScalingFloat(int, int, long);
	SharedPtr constructNewMultiDimensionalScalingFloatNoParallel(int, int, long, int);
	SharedPtr constructNewMultiDimensionalScalingFloatTbb(int, int, long, int);
	SharedPtr constructOpenCLMultiDimensionalScalingFloat(int, int, long, int);


SharedPtr factory(int dim1, int dim2, long flags, int device, int threads) {
	bool useFloat = flags & mds::Flags::FLOAT;
	bool useOpenCL = flags & mds::Flags::OPENCL;
	bool useTbb = flags & mds::Flags::TBB;

	if (useFloat) {
		if (useOpenCL) {
			return constructOpenCLMultiDimensionalScalingFloat(dim1, dim2, flags, device);
		} else {
			if (useTbb) {
				return constructNewMultiDimensionalScalingFloatTbb(dim1, dim2, flags, threads);
			} else {
				return constructNewMultiDimensionalScalingFloatNoParallel(dim1, dim2, flags, threads);
			}
		}
	} else {
		if (useOpenCL) {
			return constructOpenCLMultiDimensionalScalingDouble(dim1, dim2, flags, device);
		} else {
			if (useTbb) {
				return constructNewMultiDimensionalScalingDoubleTbb(dim1, dim2, flags, threads);
			} else {
				return constructNewMultiDimensionalScalingDoubleNoParallel(dim1, dim2, flags, threads);
			}
		}
	}
}

} // namespace mds