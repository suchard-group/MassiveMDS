#ifndef _FLAGS_H
#define _FLAGS_H

namespace mds {

enum Flags {
	FLOAT = 1 << 2,
	TBB = 1 << 3,
	OPENCL = 1 << 4,
    LEFT_TRUNCATION = 1 << 5,
    EGPU = 1 << 6,
    TBB2 = 1 << 7,
    TBB3 = 1 << 8,
	TBB4 = 1 << 9,
	TBB5 = 1 << 10,
	TBB6 = 1 << 11,
	TBB7 = 1 << 12,
	TBB8 = 1 << 13,
	TBB9 = 1 << 14,
	TBB10 = 1 << 15,
};

} // namespace mds

#endif // _FLAGS_H
