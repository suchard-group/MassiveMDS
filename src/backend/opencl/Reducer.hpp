#ifndef _REDUCER_HPP
#define _REDUCER_HPP

namespace mds {
    namespace reduce {

        template<typename T, bool isNvidiaDevice>
        struct ReduceBody1 {
            static std::string body() {
                std::stringstream k;
                // local reduction for non-NVIDIA device
                k <<
                  "   for(int k = 1; k < TPB; k <<= 1) {        \n" <<
                  "       barrier(CLK_LOCAL_MEM_FENCE);         \n" <<
                  "       uint mask = (k << 1) - 1;             \n" <<
                  "       if ((lid & mask) == 0) {              \n" <<
                  "           scratch[lid] += scratch[lid + k]; \n" <<
                  "       }                                     \n" <<
                  "   }                                         \n";
                return k.str();
            }
        };

        template<typename T, bool isNvidiaDevice>
        struct ReduceBody2 {
            static std::string body() {
                std::stringstream k;
                // local reduction for non-NVIDIA device
                k <<
                  "   for(int k = 1; k < TPB; k <<= 1) {          \n" <<
                  "       barrier(CLK_LOCAL_MEM_FENCE);           \n" <<
                  "       uint mask = (k << 1) - 1;               \n" <<
                  "       if ((lid & mask) == 0) {                \n" <<
                  "           scratch[0][lid] += scratch[0][lid + k]; \n" <<
                  "           scratch[1][lid] += scratch[1][lid + k]; \n" <<
                  "       }                                       \n" <<
                  "   }                                           \n";
                return k.str();
            }
        };

        template<typename T>
        struct ReduceBody1<T, true> {
            static std::string body() {
                std::stringstream k;
                k <<
                  "   barrier(CLK_LOCAL_MEM_FENCE);\n" <<
                  "   if (TPB >= 1024) { if (lid < 512) { sum += scratch[lid + 512]; scratch[lid] = sum; } barrier(CLK_LOCAL_MEM_FENCE); } \n"
                  <<
                  "   if (TPB >=  512) { if (lid < 256) { sum += scratch[lid + 256]; scratch[lid] = sum; } barrier(CLK_LOCAL_MEM_FENCE); } \n"
                  <<
                  "   if (TPB >=  256) { if (lid < 128) { sum += scratch[lid + 128]; scratch[lid] = sum; } barrier(CLK_LOCAL_MEM_FENCE); } \n"
                  <<
                  "   if (TPB >=  128) { if (lid <  64) { sum += scratch[lid +  64]; scratch[lid] = sum; } barrier(CLK_LOCAL_MEM_FENCE); } \n"
                  <<
                  // warp reduction
                  "   if (lid < 32) { \n" <<
                  // volatile this way we don't need any barrier
                  "       volatile __local TMP_REAL *lmem = scratch;                 \n" <<
                  "       if (TPB >= 64) { lmem[lid] = sum = sum + lmem[lid + 32]; } \n" <<
                  "       if (TPB >= 32) { lmem[lid] = sum = sum + lmem[lid + 16]; } \n" <<
                  "       if (TPB >= 16) { lmem[lid] = sum = sum + lmem[lid +  8]; } \n" <<
                  "       if (TPB >=  8) { lmem[lid] = sum = sum + lmem[lid +  4]; } \n" <<
                  "       if (TPB >=  4) { lmem[lid] = sum = sum + lmem[lid +  2]; } \n" <<
                  "       if (TPB >=  2) { lmem[lid] = sum = sum + lmem[lid +  1]; } \n" <<
                  "   }                                                            \n";
                return k.str();
            }
        };

        template<typename T>
        struct ReduceBody2<T, true> {
            static std::string body() {
                std::stringstream k;
                k <<
                  "   barrier(CLK_LOCAL_MEM_FENCE); \n" <<
                  "   if (TPB >= 1024) { if (lid < 512) { sum0 += scratch[0][lid + 512]; scratch[0][lid] = sum0; sum1 += scratch[1][lid + 512]; scratch[1][lid] = sum1; } barrier(CLK_LOCAL_MEM_FENCE); } \n"
                  <<
                  "   if (TPB >=  512) { if (lid < 256) { sum0 += scratch[0][lid + 256]; scratch[0][lid] = sum0; sum1 += scratch[1][lid + 256]; scratch[1][lid] = sum1; } barrier(CLK_LOCAL_MEM_FENCE); } \n"
                  <<
                  "   if (TPB >=  256) { if (lid < 128) { sum0 += scratch[0][lid + 128]; scratch[0][lid] = sum0; sum1 += scratch[1][lid + 128]; scratch[1][lid] = sum1; } barrier(CLK_LOCAL_MEM_FENCE); } \n"
                  <<
                  "   if (TPB >=  128) { if (lid <  64) { sum0 += scratch[0][lid +  64]; scratch[0][lid] = sum0; sum1 += scratch[1][lid +  64]; scratch[1][lid] = sum1; } barrier(CLK_LOCAL_MEM_FENCE); } \n"
                  <<
                  // warp reduction
                  "   if (lid < 32) { \n" <<
                  // volatile this way we don't need any barrier
                  "       volatile __local TMP_REAL *lmem = scratch; \n" <<
                  //"       volatile __local TMP_REAL *lmem1 = scratch1; \n" <<
                  "       if (TPB >= 64) { lmem[0][lid] = sum0 = sum0 + lmem[0][lid+32]; lmem[1][lid] = sum1 = sum1 + lmem[1][lid+32]; } \n"
                  <<
                  "       if (TPB >= 32) { lmem[0][lid] = sum0 = sum0 + lmem[0][lid+16]; lmem[1][lid] = sum1 = sum1 + lmem[1][lid+16]; } \n"
                  <<
                  "       if (TPB >= 16) { lmem[0][lid] = sum0 = sum0 + lmem[0][lid+ 8]; lmem[1][lid] = sum1 = sum1 + lmem[1][lid+ 8]; } \n"
                  <<
                  "       if (TPB >=  8) { lmem[0][lid] = sum0 = sum0 + lmem[0][lid+ 4]; lmem[1][lid] = sum1 = sum1 + lmem[1][lid+ 4]; } \n"
                  <<
                  "       if (TPB >=  4) { lmem[0][lid] = sum0 = sum0 + lmem[0][lid+ 2]; lmem[1][lid] = sum1 = sum1 + lmem[1][lid+ 2]; } \n"
                  <<
                  "       if (TPB >=  2) { lmem[0][lid] = sum0 = sum0 + lmem[0][lid+ 1]; lmem[1][lid] = sum1 = sum1 + lmem[1][lid+ 1]; } \n"
                  <<
                  "   }                                            \n";
                return k.str();
            }
        };

    } // namespace reduce
} // namespace mds

#endif // _REDUCER_HPP
