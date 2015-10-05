#ifndef _FLAGS_H
#define _FLAGS_H

namespace mds {

enum Flags {
	FLOAT = 1 << 1,
	OPENCL = 1 << 2,
    LEFT_TRUNCATION = 1 << 5,
    TBB = 1 << 10,
};

} // namespace mds

#endif // _FLAGS_H
