#include <vesper/core/dtype.h>
#include <iostream>
#include <sstream>
#include <cassert>

void test_dtype_sizes() {
    std::cout << "Testing DType sizes..." << std::endl;
    assert(vesper::GetDTypeSize(vesper::DType::Float32) == 4);
    assert(vesper::GetDTypeSize(vesper::DType::Float64) == 8);
    assert(vesper::GetDTypeSize(vesper::DType::Float16) == 2);
    assert(vesper::GetDTypeSize(vesper::DType::BFloat16) == 2);
    assert(vesper::GetDTypeSize(vesper::DType::Int32) == 4);
    assert(vesper::GetDTypeSize(vesper::DType::Int64) == 8);
}

void test_dtype_printing() {
    std::cout << "Testing DType printing..." << std::endl;
    {
        std::stringstream ss;
        ss << vesper::DType::Float32;
        assert(ss.str() == "Float32");
    }
    {
        std::stringstream ss;
        ss << vesper::DType::Float64;
        assert(ss.str() == "Float64");
    }
    {
        std::stringstream ss;
        ss << vesper::DType::Float16;
        assert(ss.str() == "Float16");
    }
    {
        std::stringstream ss;
        ss << vesper::DType::BFloat16;
        assert(ss.str() == "BFloat16");
    }
    {
        std::stringstream ss;
        ss << vesper::DType::Int32;
        assert(ss.str() == "Int32");
    }
    {
        std::stringstream ss;
        ss << vesper::DType::Int64;
        assert(ss.str() == "Int64");
    }
}

int main() {
    test_dtype_sizes();
    test_dtype_printing();
    std::cout << "DType granular tests passed!" << std::endl;
    return 0;
}
