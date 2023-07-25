// Benchmark was created by MQT Bench on 2023-06-29
// For more information about MQT Bench, please visit https://www.cda.cit.tum.de/mqtbench/
// MQT Bench version: v1.0.0
// Qiskit version: {'qiskit-terra': '0.24.1', 'qiskit-aer': '0.12.0', 'qiskit-ignis': None, 'qiskit-ibmq-provider': '0.20.2', 'qiskit': '0.43.1', 'qiskit-nature': '0.6.2', 'qiskit-finance': '0.3.4', 'qiskit-optimization': '0.5.0', 'qiskit-machine-learning': '0.6.1'}
// Used Gate Set: ['rx', 'rz', 'cz', 'measure']

OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
creg c[8];
creg meas[8];
rz(pi) q[0];
rz(-pi/2) q[1];
rx(-pi/2) q[1];
rz(-pi/2) q[1];
rz(-pi/2) q[2];
rx(-pi/2) q[2];
rz(-pi/2) q[2];
rz(-pi/2) q[3];
rx(-pi/2) q[3];
rz(-pi/2) q[3];
rz(-pi/2) q[4];
rx(-pi/2) q[4];
rz(-pi/2) q[4];
rz(-pi/2) q[5];
rx(-pi/2) q[5];
rz(-pi/2) q[5];
rz(pi) q[6];
rx(-2.644621760451785) q[7];
rz(-pi) q[7];
cz q[6],q[7];
rx(pi) q[6];
rx(pi/4) q[7];
rz(pi/2) q[7];
cz q[6],q[7];
rz(-3*pi/4) q[6];
rx(pi/2) q[6];
rz(-1.595340019401064) q[6];
rx(pi/2) q[7];
rz(-0.5092427394410929) q[7];
cz q[7],q[5];
rx(-pi/8) q[5];
cz q[7],q[5];
rx(pi/8) q[5];
cz q[6],q[5];
rx(-pi/4) q[5];
cz q[6],q[5];
rx(pi/4) q[5];
rz(-pi/64) q[5];
cz q[7],q[4];
rx(-pi/16) q[4];
cz q[7],q[4];
rx(pi/16) q[4];
cz q[6],q[4];
rx(-pi/8) q[4];
cz q[6],q[4];
rx(pi/8) q[4];
cz q[5],q[4];
rx(-pi/4) q[4];
cz q[5],q[4];
rx(pi/4) q[4];
rz(-pi/32) q[4];
cz q[7],q[3];
rx(-pi/32) q[3];
cz q[7],q[3];
rx(pi/32) q[3];
cz q[6],q[3];
rx(-pi/16) q[3];
cz q[6],q[3];
rx(pi/16) q[3];
cz q[5],q[3];
rx(-pi/8) q[3];
cz q[5],q[3];
rx(pi/8) q[3];
cz q[4],q[3];
rx(-pi/4) q[3];
cz q[4],q[3];
rx(pi/4) q[3];
rz(-pi/16) q[3];
cz q[7],q[2];
rx(-pi/64) q[2];
cz q[7],q[2];
rx(pi/64) q[2];
cz q[6],q[2];
rx(-pi/32) q[2];
cz q[6],q[2];
rx(pi/32) q[2];
cz q[5],q[2];
rx(-pi/16) q[2];
cz q[5],q[2];
rx(pi/16) q[2];
cz q[4],q[2];
rx(-pi/8) q[2];
cz q[4],q[2];
rx(pi/8) q[2];
cz q[3],q[2];
rx(-pi/4) q[2];
cz q[3],q[2];
rx(pi/4) q[2];
rz(-pi/8) q[2];
cz q[7],q[1];
rx(-0.02454369260617001) q[1];
cz q[7],q[1];
rx(0.02454369260617062) q[1];
cz q[6],q[1];
rx(-pi/64) q[1];
cz q[6],q[1];
rx(pi/64) q[1];
cz q[5],q[1];
rx(-pi/32) q[1];
cz q[5],q[1];
rx(pi/32) q[1];
cz q[4],q[1];
rx(-pi/16) q[1];
cz q[4],q[1];
rx(pi/16) q[1];
cz q[3],q[1];
rx(-pi/8) q[1];
cz q[3],q[1];
rx(pi/8) q[1];
cz q[2],q[1];
rx(-pi/4) q[1];
cz q[2],q[1];
rx(pi/4) q[1];
rz(-pi/4) q[1];
rx(pi/2) q[7];
rz(-pi/2) q[7];
cz q[0],q[7];
rx(pi) q[0];
rx(0.01227184630308494) q[7];
cz q[0],q[7];
rz(-1.5830681730979796) q[0];
rx(pi/2) q[0];
rz(-pi/2) q[0];
cz q[6],q[0];
rx(-0.02454369260617001) q[0];
cz q[6],q[0];
rx(0.02454369260617062) q[0];
cz q[5],q[0];
rx(-pi/64) q[0];
cz q[5],q[0];
rx(pi/64) q[0];
cz q[4],q[0];
rx(-pi/32) q[0];
cz q[4],q[0];
rx(pi/32) q[0];
cz q[3],q[0];
rx(-pi/16) q[0];
cz q[3],q[0];
rx(pi/16) q[0];
cz q[2],q[0];
rx(-pi/8) q[0];
cz q[2],q[0];
rx(pi/8) q[0];
cz q[1],q[0];
rx(-pi/4) q[0];
cz q[1],q[0];
rx(pi/4) q[0];
rx(-pi/2) q[4];
rz(-pi) q[4];
cz q[3],q[4];
rx(-pi/2) q[3];
rz(-pi) q[3];
rx(-pi/2) q[4];
rz(-pi) q[4];
cz q[4],q[3];
rx(-pi/2) q[3];
rz(-pi/2) q[3];
rx(-pi/2) q[4];
rz(-pi) q[4];
cz q[3],q[4];
rx(-pi/2) q[4];
rz(-pi/2) q[4];
rx(-pi/2) q[5];
rz(-pi) q[5];
cz q[2],q[5];
rx(-pi/2) q[2];
rz(-pi) q[2];
rx(-pi/2) q[5];
rz(-pi) q[5];
cz q[5],q[2];
rx(-pi/2) q[2];
rz(-pi/2) q[2];
rx(-pi/2) q[5];
rz(-pi) q[5];
cz q[2],q[5];
rx(-pi/2) q[5];
rz(-pi/2) q[5];
rx(-pi/2) q[6];
rz(-pi) q[6];
cz q[1],q[6];
rx(-pi/2) q[1];
rz(-pi) q[1];
rx(-pi/2) q[6];
rz(-pi) q[6];
cz q[6],q[1];
rx(-pi/2) q[1];
rz(-pi/2) q[1];
rx(-pi/2) q[6];
rz(-pi) q[6];
cz q[1],q[6];
rx(-pi/2) q[6];
rz(-pi/2) q[6];
rx(pi/2) q[7];
cz q[0],q[7];
rz(pi/2) q[0];
rx(pi/2) q[0];
rz(pi/2) q[7];
rx(pi/2) q[7];
cz q[7],q[0];
rx(-pi/2) q[0];
rz(-pi/2) q[0];
rx(-pi/2) q[7];
rz(-pi) q[7];
cz q[0],q[7];
rx(-pi/2) q[7];
rz(-pi/2) q[7];
