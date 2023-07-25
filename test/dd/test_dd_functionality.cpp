#include "CircuitOptimizer.hpp"
#include "QuantumComputation.hpp"
#include "dd/FunctionalityConstruction.hpp"
#include "dd/Simulation.hpp"

#include "gtest/gtest.h"
#include <random>
using namespace std::chrono;

using namespace qc;

class DDFunctionality : public testing::TestWithParam<qc::OpType> {
protected:
  void TearDown() override {
    if (!e.isTerminal()) {
      dd->decRef(e);
    }
    dd->garbageCollect(true);

    // number of complex table entries after clean-up should equal initial
    // number of entries
    EXPECT_EQ(dd->cn.complexTable.getCount(), initialComplexCount);
    // number of available cache entries after clean-up should equal initial
    // number of entries
    EXPECT_EQ(dd->cn.complexCache.getCount(), initialCacheCount);
  }

  void SetUp() override {
    // dd
    dd = std::make_unique<dd::Package<>>(nqubits);
    initialCacheCount = dd->cn.complexCache.getCount();
    initialComplexCount = dd->cn.complexTable.getCount();

    // initial state preparation
    e = ident = dd->makeIdent(nqubits);
    dd->incRef(ident);

    std::array<std::mt19937_64::result_type, std::mt19937_64::state_size>
        randomData{};
    std::random_device rd;
    std::generate(begin(randomData), end(randomData), [&]() { return rd(); });
    std::seed_seq seeds(begin(randomData), end(randomData));
    mt.seed(seeds);
    dist = std::uniform_real_distribution<dd::fp>(0.0, 2. * dd::PI);
  }

  dd::QubitCount nqubits = 4U;
  std::size_t initialCacheCount = 0U;
  std::size_t initialComplexCount = 0U;
  qc::MatrixDD e{}, ident{};
  std::unique_ptr<dd::Package<>> dd;
  std::mt19937_64 mt;
  std::uniform_real_distribution<dd::fp> dist;
};

INSTANTIATE_TEST_SUITE_P(
    Parameters, DDFunctionality,
    testing::Values(qc::GPhase, qc::I, qc::H, qc::X, qc::Y, qc::Z, qc::S,
                    qc::Sdag, qc::T, qc::Tdag, qc::SX, qc::SXdag, qc::V,
                    qc::Vdag, qc::U3, qc::U2, qc::Phase, qc::RX, qc::RY, qc::RZ,
                    qc::Peres, qc::Peresdag, qc::SWAP, qc::iSWAP, qc::DCX,
                    qc::ECR, qc::RXX, qc::RYY, qc::RZZ, qc::RZX, qc::XXminusYY,
                    qc::XXplusYY),
    [](const testing::TestParamInfo<DDFunctionality::ParamType>& inf) {
      const auto gate = inf.param;
      return toString(gate);
    });

TEST_P(DDFunctionality, standardOpBuildInverseBuild) {
  using namespace qc::literals;
  auto gate = static_cast<qc::OpType>(GetParam());

  qc::StandardOperation op;
  switch (gate) {
  case qc::GPhase:
    op = qc::StandardOperation(nqubits, Controls{}, Targets{}, gate,
                               std::vector{dist(mt)});
    break;
  case qc::U3:
    op = qc::StandardOperation(nqubits, 0, gate,
                               std::vector{dist(mt), dist(mt), dist(mt)});
    break;
  case qc::U2:
    op = qc::StandardOperation(nqubits, 0, gate,
                               std::vector{dist(mt), dist(mt)});
    break;
  case qc::RX:
  case qc::RY:
  case qc::RZ:
  case qc::Phase:
    op = qc::StandardOperation(nqubits, 0, gate, std::vector{dist(mt)});
    break;

  case qc::SWAP:
  case qc::iSWAP:
  case qc::DCX:
  case qc::ECR:
    op = qc::StandardOperation(nqubits, Controls{}, 0, 1, gate);
    break;
  case qc::Peres:
  case qc::Peresdag:
    op = qc::StandardOperation(nqubits, {0_pc}, 1, 2, gate);
    break;
  case qc::RXX:
  case qc::RYY:
  case qc::RZZ:
  case qc::RZX:
    op = qc::StandardOperation(nqubits, Controls{}, 0, 1, gate,
                               std::vector{dist(mt)});
    break;
  case qc::XXminusYY:
  case qc::XXplusYY:
    op = qc::StandardOperation(nqubits, Controls{}, 0, 1, gate,
                               std::vector{dist(mt), dist(mt)});
    break;
  default:
    op = qc::StandardOperation(nqubits, 0, gate);
  }

  ASSERT_NO_THROW({ e = dd->multiply(getDD(&op, dd), getInverseDD(&op, dd)); });
  dd->incRef(e);

  EXPECT_EQ(ident, e);
}

TEST_F(DDFunctionality, buildCircuit) {
  qc::QuantumComputation qc(nqubits);

  qc.x(0);
  qc.swap(0, 1);
  qc.h(0);
  qc.s(3);
  qc.sdag(2);
  qc.v(0);
  qc.t(1);
  qc.x(1, 0_pc);
  qc.x(2, 3_pc);
  qc.x(0, {2_pc, 3_pc});
  qc.dcx(0, 1);
  qc.dcx(0, 1, 2_pc);
  qc.ecr(0, 1);
  qc.ecr(0, 1, 2_pc);
  const auto theta = dist(mt);
  qc.rxx(0, 1, theta);
  qc.rxx(0, 1, 2_pc, theta);
  qc.ryy(0, 1, theta);
  qc.ryy(0, 1, 2_pc, theta);
  qc.rzz(0, 1, theta);
  qc.rzz(0, 1, 2_pc, theta);
  qc.rzx(0, 1, theta);
  qc.rzx(0, 1, 2_pc, theta);
  const auto beta = dist(mt);
  qc.xx_minus_yy(0, 1, theta, beta);
  qc.xx_minus_yy(0, 1, 2_pc, theta, beta);
  qc.xx_plus_yy(0, 1, theta, beta);
  qc.xx_plus_yy(0, 1, 2_pc, theta, beta);

  // invert the circuit above
  qc.xx_plus_yy(0, 1, 2_pc, -theta, beta);
  qc.xx_plus_yy(0, 1, -theta, beta);
  qc.xx_minus_yy(0, 1, 2_pc, -theta, beta);
  qc.xx_minus_yy(0, 1, -theta, beta);
  qc.rzx(0, 1, 2_pc, -theta);
  qc.rzx(0, 1, -theta);
  qc.rzz(0, 1, 2_pc, -theta);
  qc.rzz(0, 1, -theta);
  qc.ryy(0, 1, 2_pc, -theta);
  qc.ryy(0, 1, -theta);
  qc.rxx(0, 1, 2_pc, -theta);
  qc.rxx(0, 1, -theta);
  qc.ecr(0, 1, 2_pc);
  qc.ecr(0, 1);
  qc.dcx(1, 0, 2_pc);
  qc.dcx(1, 0);
  qc.x(0, {2_pc, 3_pc});
  qc.x(2, 3_pc);
  qc.x(1, 0_pc);
  qc.tdag(1);
  qc.vdag(0);
  qc.s(2);
  qc.sdag(3);
  qc.h(0);
  qc.swap(0, 1);
  qc.x(0);

  e = buildFunctionality(&qc, dd);
  EXPECT_EQ(ident, e);

  qc.x(0);
  e = buildFunctionality(&qc, dd);
  dd->incRef(e);
  EXPECT_NE(ident, e);
}

TEST_F(DDFunctionality, nonUnitary) {
  const qc::QuantumComputation qc{};
  auto dummyMap = Permutation{};
  auto op = qc::NonUnitaryOperation(nqubits, {0, 1, 2, 3}, {0, 1, 2, 3});
  EXPECT_FALSE(op.isUnitary());
  EXPECT_THROW(getDD(&op, dd), qc::QFRException);
  EXPECT_THROW(getInverseDD(&op, dd), qc::QFRException);
  EXPECT_THROW(getDD(&op, dd, dummyMap), qc::QFRException);
  EXPECT_THROW(getInverseDD(&op, dd, dummyMap), qc::QFRException);
  for (Qubit i = 0; i < nqubits; ++i) {
    EXPECT_TRUE(op.actsOn(i));
  }

  for (Qubit i = 0; i < nqubits; ++i) {
    dummyMap[i] = i;
  }
  auto barrier =
      qc::NonUnitaryOperation(nqubits, {0, 1, 2, 3}, qc::OpType::Barrier);
  EXPECT_EQ(getDD(&barrier, dd), dd->makeIdent(nqubits));
  EXPECT_EQ(getInverseDD(&barrier, dd), dd->makeIdent(nqubits));
  EXPECT_EQ(getDD(&barrier, dd, dummyMap), dd->makeIdent(nqubits));
  EXPECT_EQ(getInverseDD(&barrier, dd, dummyMap), dd->makeIdent(nqubits));
  for (Qubit i = 0; i < nqubits; ++i) {
    EXPECT_FALSE(barrier.actsOn(i));
  }
}

TEST_F(DDFunctionality, CircuitEquivalence) {
  // verify that the IBM decomposition of the H gate into RZ-SX-RZ works as
  // expected (i.e., realizes H up to a global phase)
  qc::QuantumComputation qc1(1);
  qc1.h(0);

  qc::QuantumComputation qc2(1);
  qc2.rz(0, PI_2);
  qc2.sx(0);
  qc2.rz(0, PI_2);

  const qc::MatrixDD dd1 = buildFunctionality(&qc1, dd);
  const qc::MatrixDD dd2 = buildFunctionality(&qc2, dd);

  EXPECT_EQ(dd1.p, dd2.p);
}

TEST_F(DDFunctionality, changePermutation) {
  qc::QuantumComputation qc{};
  std::stringstream ss{};
  ss << "// o 1 0\n"
     << "OPENQASM 2.0;"
     << "include \"qelib1.inc\";"
     << "qreg q[2];"
     << "x q[0];" << std::endl;
  qc.import(ss, qc::Format::OpenQASM);
  auto sim = simulate(
      &qc, dd->makeZeroState(static_cast<dd::QubitCount>(qc.getNqubits())), dd);
  EXPECT_TRUE(sim.p->e[0].isZeroTerminal());
  EXPECT_TRUE(sim.p->e[1].w.approximatelyOne());
  EXPECT_TRUE(sim.p->e[1].p->e[1].isZeroTerminal());
  EXPECT_TRUE(sim.p->e[1].p->e[0].w.approximatelyOne());
  auto func = buildFunctionality(&qc, dd);
  EXPECT_FALSE(func.p->e[0].isZeroTerminal());
  EXPECT_FALSE(func.p->e[1].isZeroTerminal());
  EXPECT_FALSE(func.p->e[2].isZeroTerminal());
  EXPECT_FALSE(func.p->e[3].isZeroTerminal());
  EXPECT_TRUE(func.p->e[0].p->e[1].w.approximatelyOne());
  EXPECT_TRUE(func.p->e[1].p->e[3].w.approximatelyOne());
  EXPECT_TRUE(func.p->e[2].p->e[0].w.approximatelyOne());
  EXPECT_TRUE(func.p->e[3].p->e[2].w.approximatelyOne());
}

TEST_F(DDFunctionality, basicTensorDumpTest) {
  QuantumComputation qc(2);
  qc.h(1);
  qc.x(0, 1_pc);

  std::stringstream ss{};
  dd::dumpTensorNetwork(ss, qc);

  const std::string reference =
      "{\"tensors\": [\n"
      "[[\"h   \", \"Q1\", \"GATE0\"], [\"q1_0\", \"q1_1\"], [2, 2], "
      "[[0.70710678118654757, 0], [0.70710678118654757, 0], "
      "[0.70710678118654757, 0], [-0.70710678118654757, 0]]],\n"
      "[[\"x   \", \"Q1\", \"Q0\", \"GATE1\"], [\"q1_1\", \"q0_0\", \"q1_2\", "
      "\"q0_1\"], [2, 2, 2, 2], [[1, 0], [0, 0], [0, 0], [0, 0], [0, 0], [1, "
      "0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [0, 0], [0, 0], [1, "
      "0], [0, 0]]]\n"
      "]}\n";
  EXPECT_EQ(ss.str(), reference);
}

TEST_F(DDFunctionality, compoundTensorDumpTest) {
  QuantumComputation qc(2);
  QuantumComputation comp(2);
  comp.h(1);
  comp.x(0, 1_pc);
  qc.emplace_back(comp.asOperation());

  std::stringstream ss{};
  dd::dumpTensorNetwork(ss, qc);

  const std::string reference =
      "{\"tensors\": [\n"
      "[[\"h   \", \"Q1\", \"GATE0\"], [\"q1_0\", \"q1_1\"], [2, 2], "
      "[[0.70710678118654757, 0], [0.70710678118654757, 0], "
      "[0.70710678118654757, 0], [-0.70710678118654757, 0]]],\n"
      "[[\"x   \", \"Q1\", \"Q0\", \"GATE1\"], [\"q1_1\", \"q0_0\", \"q1_2\", "
      "\"q0_1\"], [2, 2, 2, 2], [[1, 0], [0, 0], [0, 0], [0, 0], [0, 0], [1, "
      "0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [0, 0], [0, 0], [1, "
      "0], [0, 0]]]\n"
      "]}\n";
  EXPECT_EQ(ss.str(), reference);
}

TEST_F(DDFunctionality, errorTensorDumpTest) {
  QuantumComputation qc(2);
  qc.classicControlled(qc::X, 0, {0, 1U}, 1U);

  std::stringstream ss{};
  EXPECT_THROW(dd::dumpTensorNetwork(ss, qc), qc::QFRException);

  ss.str("");
  qc.erase(qc.begin());
  qc.barrier(0);
  qc.measure(0, 0);
  EXPECT_NO_THROW(dd::dumpTensorNetwork(ss, qc));

  ss.str("");
  qc.reset(0);
  EXPECT_THROW(dd::dumpTensorNetwork(ss, qc), qc::QFRException);
}

TEST_F(DDFunctionality, FuseTwoSingleQubitGates) {
  nqubits = 1;
  QuantumComputation qc(nqubits);
  qc.x(0);
  qc.h(0);

  qc.print(std::cout);
  e = buildFunctionality(&qc, dd);
  CircuitOptimizer::singleQubitGateFusion(qc);
  const auto f = buildFunctionality(&qc, dd);
  std::cout << "-----------------------------" << std::endl;
  qc.print(std::cout);
  EXPECT_EQ(qc.getNops(), 1);
  EXPECT_EQ(e, f);
}

TEST_F(DDFunctionality, FuseThreeSingleQubitGates) {
  nqubits = 1;
  QuantumComputation qc(nqubits);
  qc.x(0);
  qc.h(0);
  qc.y(0);

  e = buildFunctionality(&qc, dd);
  std::cout << "-----------------------------" << std::endl;
  qc.print(std::cout);
  CircuitOptimizer::singleQubitGateFusion(qc);
  const auto f = buildFunctionality(&qc, dd);
  std::cout << "-----------------------------" << std::endl;
  qc.print(std::cout);
  EXPECT_EQ(qc.getNops(), 1);
  EXPECT_EQ(e, f);
}

TEST_F(DDFunctionality, FuseNoSingleQubitGates) {
  nqubits = 2;
  QuantumComputation qc(nqubits);
  qc.h(0);
  qc.x(1, 0_pc);
  qc.y(0);
  e = buildFunctionality(&qc, dd);
  std::cout << "-----------------------------" << std::endl;
  qc.print(std::cout);
  CircuitOptimizer::singleQubitGateFusion(qc);
  const auto f = buildFunctionality(&qc, dd);
  std::cout << "-----------------------------" << std::endl;
  qc.print(std::cout);
  EXPECT_EQ(qc.getNops(), 3);
  EXPECT_EQ(e, f);
}

TEST_F(DDFunctionality, FuseSingleQubitGatesAcrossOtherGates) {
  nqubits = 2;
  QuantumComputation qc(nqubits);
  qc.h(0);
  qc.z(1);
  qc.y(0);
  e = buildFunctionality(&qc, dd);
  std::cout << "-----------------------------" << std::endl;
  qc.print(std::cout);
  CircuitOptimizer::singleQubitGateFusion(qc);
  const auto f = buildFunctionality(&qc, dd);
  std::cout << "-----------------------------" << std::endl;
  qc.print(std::cout);
  EXPECT_EQ(qc.getNops(), 2);
  EXPECT_EQ(e, f);
}

TEST_F(DDFunctionality, constructCircuitDD) {
  QuantumComputation qc{};
  const std::string qasm = "// i 0 1\n"
                           "// o 0 1\n"
                           "OPENQASM 2.0;\n"
                           "include \"qelib1.inc\";\n"
                           "qreg q[2];\n"
                           "rz(1/8) q[0];\n"
                           "p(1/8) q[1];\n"
                           "crz(1/8) q[0],q[1];\n"
                           "cp(1/8) q[0],q[1];\n";
  std::stringstream ss;
  ss << qasm;
  /*
  // Show stringstream interpretation
  std::string aux; 
  for (int i = 0; i < 10; i++) {
    ss >> aux;
    std::cout << aux << "\n";
  }
  // End
  */
  qc.import(ss, qc::Format::OpenQASM);
  std::unique_ptr<dd::Package<>> dd_temp;
  dd_temp = std::make_unique<dd::Package<>>(2);
  const qc::MatrixDD ddf = buildFunctionality(&qc, dd_temp);

  EXPECT_TRUE(true);
}

TEST_F(DDFunctionality, constructCircuitDDFromFileBasic) {
  QuantumComputation qc{};
  std::stringstream ss;
  std::string qasm;
  std::unique_ptr<dd::Package<>> dd_temp;
  std::ifstream myfile; 
  myfile.open("test.qasm");
  EXPECT_TRUE(myfile.is_open());
  std::string myline;
  if ( myfile.is_open() ) {
    qasm = "";
    while ( myfile ) {
      std::getline (myfile, myline);
      qasm = qasm + myline + "\n";
    }
  }
  ss << qasm;
  qc.import(ss, qc::Format::OpenQASM);
  dd_temp = std::make_unique<dd::Package<>>(2);
  const qc::MatrixDD ddf = buildFunctionality(&qc, dd_temp);
}

TEST_F(DDFunctionality, constructCircuitDDFromFile) {
  QuantumComputation qc{};
  std::stringstream ss;
  // First Circuit
  std::string qasm = "// i 0 1\n"
                      "// o 0 1\n"
                      "OPENQASM 2.0;\n"
                      "qreg q[2];\n"
                      "rz(1/8) q[0];\n"
                      "p(1/8) q[1];\n"
                      "crz(1/8) q[0],q[1];\n"
                      "cp(1/8) q[0],q[1];\n";
  ss << qasm;
  qc.import(ss, qc::Format::OpenQASM);
  std::unique_ptr<dd::Package<>> dd_temp;
  dd_temp = std::make_unique<dd::Package<>>(2);
  const qc::MatrixDD ddf = buildFunctionality(&qc, dd_temp);
  // Second Circuit
  std::ifstream myfile; 
  myfile.open("test.qasm");
  EXPECT_TRUE(myfile.is_open());
  std::string myline;
  if ( myfile.is_open() ) {
    qasm = "";
    while ( myfile ) {
      std::getline (myfile, myline);
      qasm = qasm + myline + "\n";
    }
  }
  ss << qasm;
  qc.import(ss, qc::Format::OpenQASM);
  dd_temp = std::make_unique<dd::Package<>>(2);
  const qc::MatrixDD ddf2 = buildFunctionality(&qc, dd_temp);
  // Comprobations
  EXPECT_TRUE(ddf == ddf2);
}


void testCircuitRecursive(std::string file, int n) {
  //Setup();

  dd::QubitCount nqubits = n;
  std::size_t initialCacheCount = 0U;
  std::size_t initialComplexCount = 0U;
  qc::MatrixDD e{}, ident{};
  std::unique_ptr<dd::Package<>> dd;
  std::mt19937_64 mt;
  std::uniform_real_distribution<dd::fp> dist;
  // dd
  dd = std::make_unique<dd::Package<>>(nqubits);
  initialCacheCount = dd->cn.complexCache.getCount();
  initialComplexCount = dd->cn.complexTable.getCount();

  // initial state preparation
  e = ident = dd->makeIdent(nqubits);
  dd->incRef(ident);

  std::array<std::mt19937_64::result_type, std::mt19937_64::state_size>
      randomData{};
  std::random_device rd;
  std::generate(begin(randomData), end(randomData), [&]() { return rd(); });
  std::seed_seq seeds(begin(randomData), end(randomData));
  mt.seed(seeds);



  QuantumComputation qc{};
  std::stringstream ss;
  std::string qasm ;
  std::unique_ptr<dd::Package<>> dd_temp;
  std::ifstream myfile; 
  myfile.open(file);
  EXPECT_TRUE(myfile.is_open());
  std::string myline;
  if ( myfile.is_open() ) {
    qasm = "";
    while ( myfile ) {
      std::getline (myfile, myline);
      qasm = qasm + myline + "\n";
    }
  }
  ss << qasm;

  qc.import(ss, qc::Format::OpenQASM);
  //dd = std::make_unique<dd::Package<>>(n);
  auto start = high_resolution_clock::now();
  const qc::MatrixDD ddf2 = dd::buildFunctionality(&qc, dd);
  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<milliseconds>(stop - start);
  std::cout << "Time: " << duration.count() / 1000.0 << std::endl;
}

TEST_F(DDFunctionality, constructCircuitDDSycamore552) {
  int n = 12;
  testCircuitRecursive("Circuits/sycamore_" + std::__cxx11::to_string(552) + ".qasm", n);
}

TEST_F(DDFunctionality, constructCircuitDDQFT10) {
  nqubits = 10U;
  testCircuitRecursive("Circuits/qftentangled_indep_qiskit_" + std::__cxx11::to_string(nqubits) + ".qasm", nqubits);
}

TEST_F(DDFunctionality, constructCircuitDDQFTFast) {
  for (int n = 7; n <= 11; n++) {
    std::cout << "Testing with " << n << " qubits: \n";
    testCircuitRecursive("Circuits/qft_" + std::__cxx11::to_string(n) + ".qasm", n);
  }
}

TEST_F(DDFunctionality, constructCircuitDDQFTEntangled7) {
  int n = 7;
  testCircuitRecursive("Circuits/qftentangled_indep_qiskit_" + std::__cxx11::to_string(n) + ".qasm", n);
}

TEST_F(DDFunctionality, constructCircuitDDQFTEntangled8) {
  int n = 8;
  testCircuitRecursive("Circuits/qftentangled_indep_qiskit_" + std::__cxx11::to_string(n) + ".qasm", n);
}

TEST_F(DDFunctionality, constructCircuitDDQFTEntangled9) {
  int n = 9;
  testCircuitRecursive("Circuits/qftentangled_indep_qiskit_" + std::__cxx11::to_string(n) + ".qasm", n);
}

TEST_F(DDFunctionality, constructCircuitDDQFTEntangled10) {
  int n = 10;
  testCircuitRecursive("Circuits/qftentangled_indep_qiskit_" + std::__cxx11::to_string(n) + ".qasm", n);
}

TEST_F(DDFunctionality, constructCircuitDDQFTEntangled11) {
  int n = 11;
  testCircuitRecursive("Circuits/qftentangled_indep_qiskit_" + std::__cxx11::to_string(n) + ".qasm", n);
}

TEST_F(DDFunctionality, constructCircuitDDQFTIndep7) {
  int n = 7;
  testCircuitRecursive("Circuits/qft_indep_qiskit_" + std::__cxx11::to_string(n) + ".qasm", n);
}

TEST_F(DDFunctionality, constructCircuitDDQFTIndep8) {
  int n = 8;
  testCircuitRecursive("Circuits/qft_indep_qiskit_" + std::__cxx11::to_string(n) + ".qasm", n);
}

TEST_F(DDFunctionality, constructCircuitDDQFTIndep9) {
  int n = 9;
  testCircuitRecursive("Circuits/qft_indep_qiskit_" + std::__cxx11::to_string(n) + ".qasm", n);
}

TEST_F(DDFunctionality, constructCircuitDDQFTIndep10) {
  int n = 10;
  testCircuitRecursive("Circuits/qft_indep_qiskit_" + std::__cxx11::to_string(n) + ".qasm", n);
}

TEST_F(DDFunctionality, constructCircuitDDQFTIndep11) {
  int n = 11;
  testCircuitRecursive("Circuits/qft_indep_qiskit_" + std::__cxx11::to_string(n) + ".qasm", n);
}

TEST_F(DDFunctionality, constructCircuitDDQFTIBM0Q7) {
  int n = 7;
  testCircuitRecursive("Circuits/qft_nativegates_ibm_qiskit_opt0_" + std::__cxx11::to_string(n) + ".qasm", n);
}

TEST_F(DDFunctionality, constructCircuitDDQFTIBM0Q8) {
  int n = 8;
  testCircuitRecursive("Circuits/qft_nativegates_ibm_qiskit_opt0_" + std::__cxx11::to_string(n) + ".qasm", n);
}

TEST_F(DDFunctionality, constructCircuitDDQFTIBM0Q9) {
  int n = 9;
  testCircuitRecursive("Circuits/qft_nativegates_ibm_qiskit_opt0_" + std::__cxx11::to_string(n) + ".qasm", n);
}

TEST_F(DDFunctionality, constructCircuitDDQFTIBM0Q10) {
  int n = 10;
  testCircuitRecursive("Circuits/qft_nativegates_ibm_qiskit_opt0_" + std::__cxx11::to_string(n) + ".qasm", n);
}

TEST_F(DDFunctionality, constructCircuitDDQFTIBM0Q11) {
  int n = 11;
  testCircuitRecursive("Circuits/qft_nativegates_ibm_qiskit_opt0_" + std::__cxx11::to_string(n) + ".qasm", n);
}

TEST_F(DDFunctionality, constructCircuitDDQFTIBM3Q7) {
  int n = 7;
  testCircuitRecursive("Circuits/qft_nativegates_ibm_qiskit_opt3_" + std::__cxx11::to_string(n) + ".qasm", n);
}

TEST_F(DDFunctionality, constructCircuitDDQFTIBM3Q8) {
  int n = 8;
  testCircuitRecursive("Circuits/qft_nativegates_ibm_qiskit_opt3_" + std::__cxx11::to_string(n) + ".qasm", n);
}

TEST_F(DDFunctionality, constructCircuitDDQFTIBM3Q9) {
  int n = 9;
  testCircuitRecursive("Circuits/qft_nativegates_ibm_qiskit_opt3_" + std::__cxx11::to_string(n) + ".qasm", n);
}

TEST_F(DDFunctionality, constructCircuitDDQFTIBM3Q10) {
  int n = 10;
  testCircuitRecursive("Circuits/qft_nativegates_ibm_qiskit_opt3_" + std::__cxx11::to_string(n) + ".qasm", n);
}

TEST_F(DDFunctionality, constructCircuitDDQFTIBM3Q11) {
  int n = 11;
  testCircuitRecursive("Circuits/qft_nativegates_ibm_qiskit_opt3_" + std::__cxx11::to_string(n) + ".qasm", n);
}

TEST_F(DDFunctionality, constructCircuitDDQFTIonQ0Q7) {
  int n = 7;
  testCircuitRecursive("Circuits/qft_nativegates_ionq_qiskit_opt0_" + std::__cxx11::to_string(n) + ".qasm", n);
}

TEST_F(DDFunctionality, constructCircuitDDQFTIonQ0Q8) {
  int n = 8;
  testCircuitRecursive("Circuits/qft_nativegates_ionq_qiskit_opt0_" + std::__cxx11::to_string(n) + ".qasm", n);
}

TEST_F(DDFunctionality, constructCircuitDDQFTIonQ0Q9) {
  int n = 9;
  testCircuitRecursive("Circuits/qft_nativegates_ionq_qiskit_opt0_" + std::__cxx11::to_string(n) + ".qasm", n);
}

TEST_F(DDFunctionality, constructCircuitDDQFTIonQ0Q10) {
  int n = 10;
  testCircuitRecursive("Circuits/qft_nativegates_ionq_qiskit_opt0_" + std::__cxx11::to_string(n) + ".qasm", n);
}

TEST_F(DDFunctionality, constructCircuitDDQFTIonQ0Q11) {
  int n = 11;
  testCircuitRecursive("Circuits/qft_nativegates_ionq_qiskit_opt0_" + std::__cxx11::to_string(n) + ".qasm", n);
}

TEST_F(DDFunctionality, constructCircuitDDQFTIonQ3Q7) {
  int n = 7;
  testCircuitRecursive("Circuits/qft_nativegates_ionq_qiskit_opt3_" + std::__cxx11::to_string(n) + ".qasm", n);
}

TEST_F(DDFunctionality, constructCircuitDDQFTIonQ3Q8) {
  int n = 8;
  testCircuitRecursive("Circuits/qft_nativegates_ionq_qiskit_opt3_" + std::__cxx11::to_string(n) + ".qasm", n);
}

TEST_F(DDFunctionality, constructCircuitDDQFTIonQ3Q9) {
  int n = 9;
  testCircuitRecursive("Circuits/qft_nativegates_ionq_qiskit_opt3_" + std::__cxx11::to_string(n) + ".qasm", n);
}

TEST_F(DDFunctionality, constructCircuitDDQFTIonQ3Q10) {
  int n = 10;
  testCircuitRecursive("Circuits/qft_nativegates_ionq_qiskit_opt3_" + std::__cxx11::to_string(n) + ".qasm", n);
}

TEST_F(DDFunctionality, constructCircuitDDQFTIonQ3Q11) {
  int n = 11;
  testCircuitRecursive("Circuits/qft_nativegates_ionq_qiskit_opt3_" + std::__cxx11::to_string(n) + ".qasm", n);
}

TEST_F(DDFunctionality, constructCircuitDDQFToqc0Q) {
  for (int n = 7; n <= 11; n++) {
    std::cout << "Testing with " << n << " qubits: \n";
    testCircuitRecursive("Circuits/qft_nativegates_oqc_qiskit_opt0_" + std::__cxx11::to_string(n) + ".qasm", n);
  }
}

TEST_F(DDFunctionality, constructCircuitDDQFToqc3Q) {
  for (int n = 7; n <= 11; n++) {
    std::cout << "Testing with " << n << " qubits: \n";
    testCircuitRecursive("Circuits/qft_nativegates_oqc_qiskit_opt3_" + std::__cxx11::to_string(n) + ".qasm", n);
  }
}

TEST_F(DDFunctionality, constructCircuitDDQFTQuantinum0Q) {
  for (int n = 7; n <= 11; n++) {
    std::cout << "Testing with " << n << " qubits: \n";
    testCircuitRecursive("Circuits/qft_nativegates_quantinuum_qiskit_opt0_" + std::__cxx11::to_string(n) + ".qasm", n);
  }
}

TEST_F(DDFunctionality, constructCircuitDDQFTQuantinum3Q) {
  for (int n = 7; n <= 11; n++) {
    std::cout << "Testing with " << n << " qubits: \n";
    testCircuitRecursive("Circuits/qft_nativegates_quantinuum_qiskit_opt3_" + std::__cxx11::to_string(n) + ".qasm", n);
  }
}

TEST_F(DDFunctionality, constructCircuitDDQFTRiguetti0Q) {
  for (int n = 7; n <= 11; n++) {
    std::cout << "Testing with " << n << " qubits: \n";
    testCircuitRecursive("Circuits/qft_nativegates_rigetti_qiskit_opt0_" + std::__cxx11::to_string(n) + ".qasm", n);
  }
}

TEST_F(DDFunctionality, constructCircuitDDQFTRiguetti3Q) {
  for (int n = 7; n <= 11; n++) {
    std::cout << "Testing with " << n << " qubits: \n";
    testCircuitRecursive("Circuits/qft_nativegates_rigetti_qiskit_opt3_" + std::__cxx11::to_string(n) + ".qasm", n);
  }
}
