#pragma once
#include <cstdint>
#include <vector>

// Pack 8-bit integers (values 0-15) into 4-bit packed format
// src: array of N uint8_t values (each < 16)
// dst: array of N/2 uint8_t values
// n: number of elements in src (must be even)
void q4_pack(const uint8_t* src, uint8_t* dst, size_t n);

// Unpack 4-bit packed format into 8-bit integers
// src: array of N/2 uint8_t values
// dst: array of N uint8_t values
// n: number of elements in dst (must be even)
void q4_unpack(const uint8_t* src, uint8_t* dst, size_t n);
