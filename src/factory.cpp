#include "AbstractMultiDimensionalScaling.hpp"

// forward reference
namespace mds {
	SharedPtr constructMultiDimensionalScalingDouble(int, int, long);
	SharedPtr constructNewMultiDimensionalScalingDoubleNoParallel(int, int, long);
	SharedPtr constructNewMultiDimensionalScalingDoubleTbb(int, int, long);
	SharedPtr constructOpenCLMultiDimensionalScalingDouble(int, int, long);

	SharedPtr constructMultiDimensionalScalingFloat(int, int, long);
	SharedPtr constructNewMultiDimensionalScalingFloatNoParallel(int, int, long);
	SharedPtr constructNewMultiDimensionalScalingFloatTbb(int, int, long);
	SharedPtr constructOpenCLMultiDimensionalScalingFloat(int, int, long);


SharedPtr factory(int dim1, int dim2, long flags) {
	bool useFloat = flags & mds::Flags::FLOAT;
	bool useOpenCL = flags & mds::Flags::OPENCL;
	bool useTbb = flags & mds::Flags::TBB;

	if (useFloat) {
		if (useOpenCL) {
			return constructOpenCLMultiDimensionalScalingFloat(dim1, dim2, flags);
		} else {
			if (useTbb) {
				return constructNewMultiDimensionalScalingFloatTbb(dim1, dim2, flags);
			} else {
				return constructNewMultiDimensionalScalingFloatNoParallel(dim1, dim2, flags);
			}
		}
	} else {
		if (useOpenCL) {
			return constructOpenCLMultiDimensionalScalingDouble(dim1, dim2, flags);
		} else {
			if (useTbb) {
				return constructNewMultiDimensionalScalingDoubleTbb(dim1, dim2, flags);
			} else {
				return constructNewMultiDimensionalScalingDoubleNoParallel(dim1, dim2, flags);
			}
		}
	}
}

} // namespace mds