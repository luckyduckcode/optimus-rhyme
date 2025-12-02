#include "q4_kernel.h"

void q4_pack_scalar(const uint8_t* src, uint8_t* dst, size_t n) {
    for (size_t i = 0; i < n; i += 2) {
        dst[i/2] = (src[i] & 0x0F) | ((src[i+1] & 0x0F) << 4);
    }
}

void q4_unpack_scalar(const uint8_t* src, uint8_t* dst, size_t n) {
    for (size_t i = 0; i < n; i += 2) {
        uint8_t val = src[i/2];
        dst[i] = val & 0x0F;
        dst[i+1] = (val >> 4) & 0x0F;
    }
}

#ifndef __AVX2__
void q4_pack(const uint8_t* src, uint8_t* dst, size_t n) {
    q4_pack_scalar(src, dst, n);
}

void q4_unpack(const uint8_t* src, uint8_t* dst, size_t n) {
    q4_unpack_scalar(src, dst, n);
}
#endif
