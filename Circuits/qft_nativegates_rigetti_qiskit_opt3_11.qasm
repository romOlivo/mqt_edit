// Benchmark was created by MQT Bench on 2023-06-29
// For more information about MQT Bench, please visit https://www.cda.cit.tum.de/mqtbench/
// MQT Bench version: v1.0.0
// Qiskit version: {'qiskit-terra': '0.24.1', 'qiskit-aer': '0.12.0', 'qiskit-ignis': None, 'qiskit-ibmq-provider': '0.20.2', 'qiskit': '0.43.1', 'qiskit-nature': '0.6.2', 'qiskit-finance': '0.3.4', 'qiskit-optimization': '0.5.0', 'qiskit-machine-learning': '0.6.1'}
// Used Gate Set: ['rx', 'rz', 'cz', 'measure']

OPENQASM 2.0;
include "qelib1.inc";
qreg q[11];
creg c[11];
creg meas[11];
rz(-pi/2) q[0];
rx(pi) q[0];
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
rz(-pi/2) q[6];
rx(-pi/2) q[6];
rz(-pi/2) q[6];
rz(-pi/2) q[7];
rx(-pi/2) q[7];
rz(-pi/2) q[7];
rz(-pi/2) q[8];
rx(-pi/2) q[8];
rz(-pi/2) q[8];
rz(pi) q[9];
rx(-2.644621760451785) q[10];
rz(-pi) q[10];
cz q[9],q[10];
rx(pi/4) q[10];
rz(pi/2) q[10];
rx(pi) q[9];
cz q[9],q[10];
rx(pi/2) q[10];
rz(-0.49850487392589343) q[10];
cz q[10],q[8];
rx(-pi/8) q[8];
cz q[10],q[8];
cz q[10],q[7];
rx(-pi/16) q[7];
cz q[10],q[7];
cz q[10],q[6];
rx(-pi/32) q[6];
cz q[10],q[6];
cz q[10],q[5];
rx(-pi/64) q[5];
cz q[10],q[5];
cz q[10],q[4];
rx(-0.02454369260617001) q[4];
cz q[10],q[4];
cz q[10],q[3];
rx(-0.012271846303084928) q[3];
cz q[10],q[3];
cz q[10],q[2];
rx(-0.0061359231515423504) q[2];
cz q[10],q[2];
cz q[10],q[1];
rx(-0.003067961575771) q[1];
cz q[10],q[1];
rx(0.0030679615757715595) q[1];
rx(-pi/2) q[10];
rz(-pi/2) q[10];
cz q[0],q[10];
rz(pi/2) q[0];
rx(pi) q[0];
rx(0.0015339807878852084) q[10];
rz(pi) q[10];
cz q[0],q[10];
rz(-1.5692623460070116) q[0];
rx(pi/2) q[0];
rz(pi/2) q[0];
rx(-pi/2) q[10];
rx(0.006135923151542877) q[2];
rx(0.012271846303085377) q[3];
rx(0.02454369260617062) q[4];
rx(pi/64) q[5];
rx(pi/32) q[6];
rx(pi/16) q[7];
rx(pi/8) q[8];
rz(-3*pi/4) q[9];
rx(pi/2) q[9];
rz(-1.5738642883706653) q[9];
cz q[9],q[8];
rx(-pi/4) q[8];
cz q[9],q[8];
rx(pi/4) q[8];
rz(-0.006135923151542766) q[8];
cz q[9],q[7];
rx(-pi/8) q[7];
cz q[9],q[7];
rx(pi/8) q[7];
cz q[8],q[7];
rx(-pi/4) q[7];
cz q[8],q[7];
rx(pi/4) q[7];
rz(-0.01227184630308531) q[7];
cz q[9],q[6];
rx(-pi/16) q[6];
cz q[9],q[6];
rx(pi/16) q[6];
cz q[8],q[6];
rx(-pi/8) q[6];
cz q[8],q[6];
rx(pi/8) q[6];
cz q[7],q[6];
rx(-pi/4) q[6];
cz q[7],q[6];
rx(pi/4) q[6];
rz(-pi/128) q[6];
cz q[9],q[5];
rx(-pi/32) q[5];
cz q[9],q[5];
rx(pi/32) q[5];
cz q[8],q[5];
rx(-pi/16) q[5];
cz q[8],q[5];
rx(pi/16) q[5];
cz q[7],q[5];
rx(-pi/8) q[5];
cz q[7],q[5];
rx(pi/8) q[5];
cz q[6],q[5];
rx(-pi/4) q[5];
cz q[6],q[5];
rx(pi/4) q[5];
rz(1.521708941582556) q[5];
cz q[9],q[4];
rx(-pi/64) q[4];
cz q[9],q[4];
rx(pi/64) q[4];
cz q[8],q[4];
rx(-pi/32) q[4];
cz q[8],q[4];
rx(pi/32) q[4];
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
cz q[9],q[3];
rx(-0.02454369260617001) q[3];
cz q[9],q[3];
rx(0.02454369260617062) q[3];
cz q[8],q[3];
rx(-pi/64) q[3];
cz q[8],q[3];
rx(pi/64) q[3];
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
cz q[9],q[2];
rx(-0.012271846303084928) q[2];
cz q[9],q[2];
rx(0.012271846303085377) q[2];
cz q[8],q[2];
rx(-0.02454369260617001) q[2];
cz q[8],q[2];
rx(0.02454369260617062) q[2];
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
cz q[9],q[1];
rx(-0.0061359231515423504) q[1];
cz q[9],q[1];
rx(0.006135923151542877) q[1];
cz q[8],q[1];
rx(-0.012271846303084928) q[1];
cz q[8],q[1];
rx(0.012271846303085377) q[1];
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
cz q[9],q[0];
rx(-0.003067961575771) q[0];
cz q[9],q[0];
rx(0.0030679615757715595) q[0];
cz q[8],q[0];
rx(-0.0061359231515423504) q[0];
cz q[8],q[0];
rx(0.006135923151542877) q[0];
cz q[7],q[0];
rx(-0.012271846303084928) q[0];
cz q[7],q[0];
rx(0.012271846303085377) q[0];
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
cz q[0],q[10];
rz(pi/2) q[0];
rx(pi/2) q[0];
rz(pi/2) q[10];
rx(pi/2) q[10];
cz q[10],q[0];
rx(-pi/2) q[0];
rz(-pi/2) q[0];
rx(-pi/2) q[10];
rz(-pi) q[10];
cz q[0],q[10];
rx(-pi/2) q[10];
rz(-pi/2) q[10];
rx(-pi/2) q[6];
rz(-pi) q[6];
cz q[4],q[6];
rx(-pi/2) q[4];
rz(-pi) q[4];
rx(-pi/2) q[6];
rz(-pi) q[6];
cz q[6],q[4];
rx(-pi/2) q[4];
rz(-pi/2) q[4];
rx(-pi/2) q[6];
rz(-pi) q[6];
cz q[4],q[6];
rx(-pi/2) q[6];
rz(-pi/2) q[6];
rx(-pi/2) q[7];
rz(-pi) q[7];
cz q[3],q[7];
rx(-pi/2) q[3];
rz(-pi) q[3];
rx(-pi/2) q[7];
rz(-pi) q[7];
cz q[7],q[3];
rx(-pi/2) q[3];
rz(-pi/2) q[3];
rx(-pi/2) q[7];
rz(-pi) q[7];
cz q[3],q[7];
rx(-pi/2) q[7];
rz(-pi/2) q[7];
rx(-pi/2) q[8];
rz(-pi) q[8];
cz q[2],q[8];
rx(-pi/2) q[2];
rz(-pi) q[2];
rx(-pi/2) q[8];
rz(-pi) q[8];
cz q[8],q[2];
rx(-pi/2) q[2];
rz(-pi/2) q[2];
rx(-pi/2) q[8];
rz(-pi) q[8];
cz q[2],q[8];
rx(-pi/2) q[8];
rz(-pi/2) q[8];
rx(-pi/2) q[9];
rz(-pi) q[9];
cz q[1],q[9];
rx(-pi/2) q[1];
rz(-pi) q[1];
rx(-pi/2) q[9];
rz(-pi) q[9];
cz q[9],q[1];
rx(-pi/2) q[1];
rz(-pi/2) q[1];
rx(-pi/2) q[9];
rz(-pi) q[9];
cz q[1],q[9];
rx(-pi/2) q[9];
rz(-pi/2) q[9];