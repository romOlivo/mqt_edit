// i 0 1
// o 0 1
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
rz(1/8) q[0];
p(1/8) q[1];
crz(1/8) q[0],q[1];
cp(1/8) q[0],q[1];

