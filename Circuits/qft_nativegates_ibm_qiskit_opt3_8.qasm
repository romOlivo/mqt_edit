// Benchmark was created by MQT Bench on 2023-06-29
// For more information about MQT Bench, please visit https://www.cda.cit.tum.de/mqtbench/
// MQT Bench version: v1.0.0
// Qiskit version: {'qiskit-terra': '0.24.1', 'qiskit-aer': '0.12.0', 'qiskit-ignis': None, 'qiskit-ibmq-provider': '0.20.2', 'qiskit': '0.43.1', 'qiskit-nature': '0.6.2', 'qiskit-finance': '0.3.4', 'qiskit-optimization': '0.5.0', 'qiskit-machine-learning': '0.6.1'}
// Used Gate Set: ['rz', 'sx', 'x', 'cx', 'measure']

OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
creg c[8];
creg meas[8];
rz(pi/2) q[7];
sx q[7];
rz(3.1293208072867076) q[7];
cx q[7],q[6];
rz(-pi/4) q[6];
cx q[7],q[6];
rz(3*pi/4) q[6];
sx q[6];
rz(3.1170489609836225) q[6];
cx q[7],q[5];
rz(-pi/8) q[5];
cx q[7],q[5];
rz(pi/8) q[5];
cx q[6],q[5];
rz(-pi/4) q[5];
cx q[6],q[5];
rz(3*pi/4) q[5];
sx q[5];
rz(3.0925052683774523) q[5];
cx q[7],q[4];
rz(-pi/16) q[4];
cx q[7],q[4];
rz(pi/16) q[4];
cx q[6],q[4];
rz(-pi/8) q[4];
cx q[6],q[4];
rz(pi/8) q[4];
cx q[5],q[4];
rz(-pi/4) q[4];
cx q[5],q[4];
rz(3*pi/4) q[4];
sx q[4];
rz(3.043417883165112) q[4];
cx q[7],q[3];
rz(-pi/32) q[3];
cx q[7],q[3];
rz(pi/32) q[3];
cx q[6],q[3];
rz(-pi/16) q[3];
cx q[6],q[3];
rz(pi/16) q[3];
cx q[5],q[3];
rz(-pi/8) q[3];
cx q[5],q[3];
rz(pi/8) q[3];
cx q[4],q[3];
rz(-pi/4) q[3];
cx q[4],q[3];
rz(3*pi/4) q[3];
sx q[3];
rz(15*pi/16) q[3];
cx q[7],q[2];
rz(-pi/64) q[2];
cx q[7],q[2];
rz(pi/64) q[2];
cx q[6],q[2];
rz(-pi/32) q[2];
cx q[6],q[2];
rz(pi/32) q[2];
cx q[5],q[2];
rz(-pi/16) q[2];
cx q[5],q[2];
rz(pi/16) q[2];
cx q[4],q[2];
rz(-pi/8) q[2];
cx q[4],q[2];
rz(pi/8) q[2];
cx q[3],q[2];
rz(-pi/4) q[2];
cx q[3],q[2];
rz(3*pi/4) q[2];
sx q[2];
rz(7*pi/8) q[2];
cx q[7],q[1];
rz(-pi/128) q[1];
cx q[7],q[1];
rz(pi/128) q[1];
cx q[6],q[1];
rz(-pi/64) q[1];
cx q[6],q[1];
rz(pi/64) q[1];
cx q[5],q[1];
rz(-pi/32) q[1];
cx q[5],q[1];
rz(pi/32) q[1];
cx q[4],q[1];
rz(-pi/16) q[1];
cx q[4],q[1];
rz(pi/16) q[1];
cx q[3],q[1];
rz(-pi/8) q[1];
cx q[3],q[1];
rz(pi/8) q[1];
cx q[2],q[1];
rz(-pi/4) q[1];
cx q[2],q[1];
rz(3*pi/4) q[1];
sx q[1];
rz(3*pi/4) q[1];
cx q[7],q[0];
rz(-pi/256) q[0];
cx q[7],q[0];
rz(pi/256) q[0];
cx q[6],q[0];
rz(-pi/128) q[0];
cx q[6],q[0];
rz(pi/128) q[0];
cx q[5],q[0];
rz(-pi/64) q[0];
cx q[5],q[0];
rz(pi/64) q[0];
cx q[4],q[0];
rz(-pi/32) q[0];
cx q[4],q[0];
rz(pi/32) q[0];
cx q[3],q[0];
rz(-pi/16) q[0];
cx q[3],q[0];
rz(pi/16) q[0];
cx q[2],q[0];
rz(-pi/8) q[0];
cx q[2],q[0];
rz(pi/8) q[0];
cx q[1],q[0];
rz(-pi/4) q[0];
cx q[1],q[0];
rz(3*pi/4) q[0];
sx q[0];
rz(pi/2) q[0];
cx q[0],q[7];
cx q[1],q[6];
cx q[2],q[5];
cx q[3],q[4];
cx q[4],q[3];
cx q[3],q[4];
cx q[5],q[2];
cx q[2],q[5];
cx q[6],q[1];
cx q[1],q[6];
cx q[7],q[0];
cx q[0],q[7];
