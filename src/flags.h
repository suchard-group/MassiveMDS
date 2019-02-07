#ifndef _FLAGS_H
#define _FLAGS_H

namespace mds {

enum Flags {
	FLOAT = 1 << 2,
	TBB = 1 << 3,
	OPENCL = 1 << 4,
    LEFT_TRUNCATION = 1 << 5,
    EGPU = 1 << 6,
    SIMD = 1 << 7
};

} // namespace mds

#endif // _FLAGS_H
