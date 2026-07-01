#pragma once

#include <cstdint>
#include <cstring>
#include <type_traits>

namespace vesper {

/**
 * @brief IEEE 754 half-precision floating-point (FP16)
 * 
 * Format: 1 sign bit, 5 exponent bits, 10 mantissa bits
 * Range: ±65504, precision: ~0.1% relative error
 * 
 * FP16 is useful when:
 * - Memory bandwidth is the bottleneck
 * - Values stay within ±65504 range
 * - Loss scaling is used to prevent gradient underflow
 */
struct Float16 {
    uint16_t data;
    
    Float16() = default;
    
    explicit Float16(float f) {
        // IEEE 754 FP32 to FP16 conversion
        uint32_t bits;
        std::memcpy(&bits, &f, sizeof(float));
        
        uint32_t sign = (bits >> 16) & 0x8000;
        int32_t exponent = ((bits >> 23) & 0xFF) - 127 + 15;
        uint32_t mantissa = bits & 0x7FFFFF;
        
        if (exponent <= 0) {
            // Subnormal or zero
            if (exponent < -10) {
                data = static_cast<uint16_t>(sign);  // Underflow to zero
            } else {
                // Subnormal FP16
                mantissa |= 0x800000;  // Add implicit leading 1
                int shift = 1 - exponent + 13;
                mantissa >>= shift;
                data = static_cast<uint16_t>(sign | mantissa);
            }
        } else if (exponent >= 31) {
            // Overflow to infinity or NaN
            if (exponent == 31 && mantissa != 0) {
                // NaN
                data = static_cast<uint16_t>(sign | 0x7C00 | (mantissa >> 13));
            } else {
                // Infinity
                data = static_cast<uint16_t>(sign | 0x7C00);
            }
        } else {
            // Normal number: round mantissa to nearest, ties to even.
            // Combine the exponent and rounded mantissa with integer addition
            // (not OR) so a carry out of the 10-bit mantissa increments the
            // exponent. At the largest normal this overflows exponent 30 -> 31,
            // correctly producing infinity.
            uint32_t lsb = (mantissa >> 13) & 1u;
            uint32_t rounding_bias = 0xFFFu + lsb;  // (0x1000 - 1) + result LSB
            uint32_t rounded = mantissa + rounding_bias;
            uint16_t out = static_cast<uint16_t>(
                (static_cast<uint32_t>(exponent) << 10) + (rounded >> 13));
            data = static_cast<uint16_t>(sign | out);
        }
    }
    
    explicit operator float() const {
        uint32_t sign = (data & 0x8000) << 16;
        uint32_t exponent = (data >> 10) & 0x1F;
        uint32_t mantissa = data & 0x3FF;
        
        uint32_t bits;
        if (exponent == 0) {
            if (mantissa == 0) {
                // Zero
                bits = sign;
            } else {
                // Subnormal: normalize
                exponent = 1;
                while ((mantissa & 0x400) == 0) {
                    mantissa <<= 1;
                    exponent--;
                }
                mantissa &= 0x3FF;
                bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
            }
        } else if (exponent == 31) {
            // Inf or NaN
            bits = sign | 0x7F800000 | (mantissa << 13);
        } else {
            // Normal
            bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
        }
        
        float result;
        std::memcpy(&result, &bits, sizeof(float));
        return result;
    }
    
    static Float16 from_bits(uint16_t bits) {
        Float16 h;
        h.data = bits;
        return h;
    }
    
    uint16_t to_bits() const { return data; }
    
    // Arithmetic operators (promote to float, compute, convert back)
    Float16 operator+(Float16 other) const {
        return Float16(float(*this) + float(other));
    }
    
    Float16 operator-(Float16 other) const {
        return Float16(float(*this) - float(other));
    }
    
    Float16 operator*(Float16 other) const {
        return Float16(float(*this) * float(other));
    }
    
    Float16 operator/(Float16 other) const {
        return Float16(float(*this) / float(other));
    }
    
    Float16& operator+=(Float16 other) {
        *this = *this + other;
        return *this;
    }
    
    Float16& operator-=(Float16 other) {
        *this = *this - other;
        return *this;
    }
    
    Float16& operator*=(Float16 other) {
        *this = *this * other;
        return *this;
    }
    
    Float16& operator/=(Float16 other) {
        *this = *this / other;
        return *this;
    }
    
    bool operator==(Float16 other) const { return data == other.data; }
    bool operator!=(Float16 other) const { return data != other.data; }
    bool operator<(Float16 other) const { return float(*this) < float(other); }
    bool operator>(Float16 other) const { return float(*this) > float(other); }
    bool operator<=(Float16 other) const { return float(*this) <= float(other); }
    bool operator>=(Float16 other) const { return float(*this) >= float(other); }
};

/**
 * @brief Brain floating-point 16-bit (BF16)
 * 
 * Format: 1 sign bit, 8 exponent bits, 7 mantissa bits
 * Range: Same as FP32 (±3.4e38), precision: ~1% relative error
 * 
 * BF16 is preferred over FP16 when:
 * - Same dynamic range as FP32 is needed (no overflow/underflow issues)
 * - Hardware supports it (MI200+, A100+)
 * - Loss scaling is not desired
 */
struct BFloat16 {
    uint16_t data;
    
    BFloat16() = default;
    
    explicit BFloat16(float f) {
        // Simple truncation from FP32 - just take upper 16 bits
        // This preserves the sign and exponent exactly, truncates mantissa
        uint32_t bits;
        std::memcpy(&bits, &f, sizeof(float));
        
        // Round to nearest even (optional but improves accuracy)
        uint32_t rounding_bias = 0x7FFF + ((bits >> 16) & 1);
        bits += rounding_bias;
        
        data = static_cast<uint16_t>(bits >> 16);
    }
    
    explicit operator float() const {
        // Expand back to FP32 - just shift up and zero-fill mantissa
        uint32_t bits = static_cast<uint32_t>(data) << 16;
        float result;
        std::memcpy(&result, &bits, sizeof(float));
        return result;
    }
    
    static BFloat16 from_bits(uint16_t bits) {
        BFloat16 h;
        h.data = bits;
        return h;
    }
    
    uint16_t to_bits() const { return data; }
    
    // Arithmetic operators
    BFloat16 operator+(BFloat16 other) const {
        return BFloat16(float(*this) + float(other));
    }
    
    BFloat16 operator-(BFloat16 other) const {
        return BFloat16(float(*this) - float(other));
    }
    
    BFloat16 operator*(BFloat16 other) const {
        return BFloat16(float(*this) * float(other));
    }
    
    BFloat16 operator/(BFloat16 other) const {
        return BFloat16(float(*this) / float(other));
    }
    
    BFloat16& operator+=(BFloat16 other) {
        *this = *this + other;
        return *this;
    }
    
    BFloat16& operator-=(BFloat16 other) {
        *this = *this - other;
        return *this;
    }
    
    BFloat16& operator*=(BFloat16 other) {
        *this = *this * other;
        return *this;
    }
    
    BFloat16& operator/=(BFloat16 other) {
        *this = *this / other;
        return *this;
    }
    
    bool operator==(BFloat16 other) const { return data == other.data; }
    bool operator!=(BFloat16 other) const { return data != other.data; }
    bool operator<(BFloat16 other) const { return float(*this) < float(other); }
    bool operator>(BFloat16 other) const { return float(*this) > float(other); }
    bool operator<=(BFloat16 other) const { return float(*this) <= float(other); }
    bool operator>=(BFloat16 other) const { return float(*this) >= float(other); }
};

// Type traits for half types
template<typename T> struct is_half_type : std::false_type {};
template<> struct is_half_type<Float16> : std::true_type {};
template<> struct is_half_type<BFloat16> : std::true_type {};

template<typename T>
inline constexpr bool is_half_type_v = is_half_type<T>::value;

} // namespace vesper
