#include "QuantumComputation.hpp"

#include <iostream>

using namespace qc;

int main(){ 
    // auto qcc = qc::QuantumComputation();
    qc::QuantumComputation qcc = QuantumComputation();
    // const std::size_t nqubits = 2;
    // qc::QuantumComputation qcc{};
    std::cout << "hello! " << qc::QuantumComputation::test_flag << "\n";
    // std::cout << qc::QuantumComputation::test_flag << "\n";
    // std::cout << &Format::OpenQASM << "\n";
}
