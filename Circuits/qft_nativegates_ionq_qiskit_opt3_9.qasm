// Benchmark was created by MQT Bench on 2023-06-29
// For more information about MQT Bench, please visit https://www.cda.cit.tum.de/mqtbench/
// MQT Bench version: v1.0.0
// Qiskit version: {'qiskit-terra': '0.24.1', 'qiskit-aer': '0.12.0', 'qiskit-ignis': None, 'qiskit-ibmq-provider': '0.20.2', 'qiskit': '0.43.1', 'qiskit-nature': '0.6.2', 'qiskit-finance': '0.3.4', 'qiskit-optimization': '0.5.0', 'qiskit-machine-learning': '0.6.1'}
// Used Gate Set: ['rxx', 'rz', 'ry', 'rx', 'measure']

OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
creg c[9];
creg meas[9];
rz(-pi/4) q[7];
ry(pi/2) q[7];
rz(2.593601761179481) q[8];
ry(0.9762597372169783) q[8];
rz(2.3990735165900112) q[8];
rxx(pi/2) q[7],q[8];
rz(-pi) q[7];
rx(-pi/2) q[7];
rz(-2.85666852696773) q[8];
ry(2.5935642459694797) q[8];
rz(-0.28492412662206235) q[8];
rxx(pi/2) q[7],q[8];
rz(-pi/4) q[7];
ry(-pi/2) q[7];
rz(0.8883376665185629) q[8];
ry(0.9108465872773697) q[8];
rz(-2.6793301358014503) q[8];
rxx(pi/2) q[8],q[6];
rx(-pi/2) q[6];
rz(-pi/8) q[6];
rx(-21.604585416581372) q[8];
rxx(pi/2) q[8],q[6];
rx(-pi/2) q[6];
rz(pi/8) q[6];
rxx(pi/2) q[7],q[6];
rx(-pi/2) q[6];
rz(-pi/4) q[6];
rx(-21.21802225803419) q[7];
rxx(pi/2) q[7],q[6];
ry(-pi/4) q[6];
rx(-15.732506960555137) q[6];
rxx(pi/2) q[8],q[5];
rx(-pi/2) q[5];
rz(-pi/16) q[5];
rxx(pi/2) q[8],q[5];
rx(-pi/2) q[5];
rz(pi/16) q[5];
rxx(pi/2) q[7],q[5];
rx(-pi/2) q[5];
rz(-pi/8) q[5];
rxx(pi/2) q[7],q[5];
rx(-pi/2) q[5];
rz(pi/8) q[5];
rxx(pi/2) q[6],q[5];
rx(-pi/2) q[5];
rz(-pi/4) q[5];
rxx(pi/2) q[6],q[5];
ry(-pi/4) q[5];
rx(-12.615457999571515) q[5];
rxx(pi/2) q[8],q[4];
rx(-pi/2) q[4];
rz(-pi/32) q[4];
rxx(pi/2) q[8],q[4];
rx(-pi/2) q[4];
rz(pi/32) q[4];
rxx(pi/2) q[7],q[4];
rx(-pi/2) q[4];
rz(-pi/16) q[4];
rxx(pi/2) q[7],q[4];
rx(-pi/2) q[4];
rz(pi/16) q[4];
rxx(pi/2) q[6],q[4];
rx(-pi/2) q[4];
rz(-pi/8) q[4];
rxx(pi/2) q[6],q[4];
rx(-pi/2) q[4];
rz(pi/8) q[4];
rxx(pi/2) q[5],q[4];
rx(-pi/2) q[4];
rz(-pi/4) q[4];
rxx(pi/2) q[5],q[4];
ry(-pi/4) q[4];
rx(-9.522952731194062) q[4];
rxx(pi/2) q[8],q[3];
rx(-pi/2) q[3];
rz(-pi/64) q[3];
rxx(pi/2) q[8],q[3];
rx(-pi/2) q[3];
rz(pi/64) q[3];
rxx(pi/2) q[7],q[3];
rx(-pi/2) q[3];
rz(-pi/32) q[3];
rxx(pi/2) q[7],q[3];
rx(-pi/2) q[3];
rz(pi/32) q[3];
rxx(pi/2) q[6],q[3];
rx(-pi/2) q[3];
rz(-pi/16) q[3];
rxx(pi/2) q[6],q[3];
rx(-pi/2) q[3];
rz(pi/16) q[3];
rxx(pi/2) q[5],q[3];
rx(-pi/2) q[3];
rz(-pi/8) q[3];
rxx(pi/2) q[5],q[3];
rx(-pi/2) q[3];
rz(pi/8) q[3];
rxx(pi/2) q[4],q[3];
rx(-pi/2) q[3];
rz(-pi/4) q[3];
rxx(pi/2) q[4],q[3];
ry(-pi/4) q[3];
rx(-8.050331174823846) q[3];
rxx(pi/2) q[8],q[2];
rx(-pi/2) q[2];
rz(-pi/128) q[2];
rxx(pi/2) q[8],q[2];
rx(-pi/2) q[2];
rz(pi/128) q[2];
rxx(pi/2) q[7],q[2];
rx(-pi/2) q[2];
rz(-pi/64) q[2];
rxx(pi/2) q[7],q[2];
rx(-pi/2) q[2];
rz(pi/64) q[2];
rxx(pi/2) q[6],q[2];
rx(-pi/2) q[2];
rz(-pi/32) q[2];
rxx(pi/2) q[6],q[2];
rx(-pi/2) q[2];
rz(pi/32) q[2];
rxx(pi/2) q[5],q[2];
rx(-pi/2) q[2];
rz(-pi/16) q[2];
rxx(pi/2) q[5],q[2];
rx(-pi/2) q[2];
rz(pi/16) q[2];
rxx(pi/2) q[4],q[2];
rx(-pi/2) q[2];
rz(-pi/8) q[2];
rxx(pi/2) q[4],q[2];
rx(-pi/2) q[2];
rz(pi/8) q[2];
rxx(pi/2) q[3],q[2];
rx(-pi/2) q[2];
rz(-pi/4) q[2];
rxx(pi/2) q[3],q[2];
ry(-pi/4) q[2];
rx(-13*pi/8) q[2];
rxx(pi/2) q[8],q[1];
rx(-pi/2) q[1];
rz(-pi/256) q[1];
rxx(pi/2) q[8],q[1];
rx(-pi/2) q[1];
rz(pi/256) q[1];
rxx(pi/2) q[7],q[1];
rx(-pi/2) q[1];
rz(-pi/128) q[1];
rxx(pi/2) q[7],q[1];
rx(-pi/2) q[1];
rz(pi/128) q[1];
rxx(pi/2) q[6],q[1];
rx(-pi/2) q[1];
rz(-pi/64) q[1];
rxx(pi/2) q[6],q[1];
rx(-pi/2) q[1];
rz(pi/64) q[1];
rxx(pi/2) q[5],q[1];
rx(-pi/2) q[1];
rz(-pi/32) q[1];
rxx(pi/2) q[5],q[1];
rx(-pi/2) q[1];
rz(pi/32) q[1];
rxx(pi/2) q[4],q[1];
rx(-pi/2) q[1];
rz(-pi/16) q[1];
rxx(pi/2) q[4],q[1];
rx(-pi/2) q[1];
rz(pi/16) q[1];
rxx(pi/2) q[3],q[1];
rx(-pi/2) q[1];
rz(-pi/8) q[1];
rxx(pi/2) q[3],q[1];
rx(-pi/2) q[1];
rz(pi/8) q[1];
rxx(pi/2) q[2],q[1];
rx(-pi/2) q[1];
rz(-pi/4) q[1];
rxx(pi/2) q[2],q[1];
ry(-pi/4) q[1];
rx(-3*pi/4) q[1];
rxx(pi/2) q[8],q[0];
rx(-pi/2) q[0];
rz(-pi/512) q[0];
rxx(pi/2) q[8],q[0];
rx(-pi/2) q[0];
rz(pi/512) q[0];
rxx(pi/2) q[7],q[0];
rx(-pi/2) q[0];
rz(-pi/256) q[0];
rxx(pi/2) q[7],q[0];
rx(-pi/2) q[0];
rz(pi/256) q[0];
rxx(pi/2) q[6],q[0];
rx(-pi/2) q[0];
rz(-pi/128) q[0];
rxx(pi/2) q[6],q[0];
rx(-pi/2) q[0];
rz(pi/128) q[0];
rxx(pi/2) q[5],q[0];
rx(-pi/2) q[0];
rz(-pi/64) q[0];
rxx(pi/2) q[5],q[0];
rx(-pi/2) q[0];
rz(pi/64) q[0];
rxx(pi/2) q[4],q[0];
rx(-pi/2) q[0];
rz(-pi/32) q[0];
rxx(pi/2) q[4],q[0];
rx(-pi/2) q[0];
rz(pi/32) q[0];
rxx(pi/2) q[3],q[0];
rx(-pi/2) q[0];
rz(-pi/16) q[0];
rxx(pi/2) q[3],q[0];
rx(-pi/2) q[0];
rz(pi/16) q[0];
rxx(pi/2) q[2],q[0];
rx(-pi/2) q[0];
rz(-pi/8) q[0];
rxx(pi/2) q[2],q[0];
rx(-pi/2) q[0];
rz(pi/8) q[0];
rxx(pi/2) q[1],q[0];
rx(-pi/2) q[0];
rz(-pi/4) q[0];
rxx(pi/2) q[1],q[0];
rx(pi/2) q[0];
rz(-pi/4) q[0];
ry(-pi/2) q[4];
ry(-pi/2) q[5];
rxx(pi/2) q[3],q[5];
ry(-pi/2) q[3];
rx(-pi/2) q[5];
ry(pi/2) q[5];
rxx(pi/2) q[5],q[3];
rx(-pi/2) q[3];
ry(pi/2) q[3];
rx(-pi/2) q[5];
ry(-pi/2) q[5];
rxx(pi/2) q[3],q[5];
rx(-pi/2) q[3];
ry(-pi/2) q[3];
rx(-pi/2) q[5];
ry(-pi/2) q[6];
rxx(pi/2) q[2],q[6];
ry(-pi/2) q[2];
rx(-pi/2) q[6];
ry(pi/2) q[6];
rxx(pi/2) q[6],q[2];
rx(-pi/2) q[2];
ry(pi/2) q[2];
rx(-pi/2) q[6];
ry(-pi/2) q[6];
rxx(pi/2) q[2],q[6];
rx(-pi/2) q[2];
ry(-pi/2) q[2];
rx(-pi/2) q[6];
ry(-pi/2) q[7];
rxx(pi/2) q[1],q[7];
ry(-pi/2) q[1];
rx(-pi/2) q[7];
ry(pi/2) q[7];
rxx(pi/2) q[7],q[1];
rx(-pi/2) q[1];
ry(pi/2) q[1];
rx(-pi/2) q[7];
ry(-pi/2) q[7];
rxx(pi/2) q[1],q[7];
rx(-pi/2) q[1];
ry(-pi/2) q[1];
rx(-pi/2) q[7];
ry(-pi/2) q[8];
rxx(pi/2) q[0],q[8];
rx(-pi/2) q[0];
ry(-pi/2) q[0];
rx(-pi/2) q[8];
ry(pi/2) q[8];
rxx(pi/2) q[8],q[0];
rx(-pi/2) q[0];
ry(pi/2) q[0];
rx(-pi/2) q[8];
ry(-pi/2) q[8];
rxx(pi/2) q[0],q[8];
rx(-pi/2) q[0];
ry(-pi/2) q[0];
rx(-pi/2) q[8];
