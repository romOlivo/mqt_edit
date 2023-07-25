// Benchmark was created by MQT Bench on 2023-06-29
// For more information about MQT Bench, please visit https://www.cda.cit.tum.de/mqtbench/
// MQT Bench version: v1.0.0
// Qiskit version: {'qiskit-terra': '0.24.1', 'qiskit-aer': '0.12.0', 'qiskit-ignis': None, 'qiskit-ibmq-provider': '0.20.2', 'qiskit': '0.43.1', 'qiskit-nature': '0.6.2', 'qiskit-finance': '0.3.4', 'qiskit-optimization': '0.5.0', 'qiskit-machine-learning': '0.6.1'}
// Used Gate Set: ['rz', 'sx', 'x', 'ecr', 'measure']

OPENQASM 2.0;
include "qelib1.inc";
opaque ecr q0,q1;
qreg q[7];
creg c[7];
creg meas[7];
rz(-pi/2) q[0];
rz(-pi/2) q[1];
rz(-pi/2) q[2];
x q[3];
rz(-pi/2) q[3];
rz(-pi/2) q[4];
rz(-pi/2) q[5];
rz(-pi/2) q[6];
sx q[6];
rz(-0.17496904566568894) q[6];
ecr q[5],q[6];
sx q[6];
rz(3*pi/4) q[6];
sx q[6];
rz(-pi) q[6];
ecr q[5],q[6];
rz(-3*pi/4) q[5];
sx q[5];
rz(pi/4) q[5];
rz(-2.5739245262253787) q[6];
ecr q[4],q[6];
sx q[6];
rz(7*pi/8) q[6];
sx q[6];
rz(-pi) q[6];
ecr q[4],q[6];
rz(-3*pi/8) q[4];
sx q[4];
rz(-pi) q[4];
ecr q[5],q[4];
rz(3*pi/4) q[4];
sx q[4];
rz(-pi) q[4];
x q[5];
rz(-pi/2) q[5];
ecr q[5],q[4];
rz(3*pi/4) q[4];
sx q[4];
rz(pi/4) q[4];
x q[5];
rz(-3*pi/8) q[5];
x q[6];
rz(-15*pi/16) q[6];
ecr q[3],q[6];
sx q[6];
rz(15*pi/16) q[6];
sx q[6];
rz(-pi) q[6];
ecr q[3],q[6];
rz(7*pi/16) q[3];
sx q[3];
ecr q[5],q[3];
rz(7*pi/8) q[3];
sx q[3];
rz(-pi) q[3];
x q[5];
rz(-pi/2) q[5];
ecr q[5],q[3];
rz(-7*pi/8) q[3];
sx q[3];
rz(-pi) q[3];
ecr q[4],q[3];
rz(3*pi/4) q[3];
sx q[3];
rz(-pi) q[3];
x q[4];
rz(-pi/2) q[4];
ecr q[4],q[3];
rz(3*pi/4) q[3];
sx q[3];
rz(pi/4) q[3];
x q[4];
rz(-3*pi/8) q[4];
x q[5];
rz(-7*pi/16) q[5];
x q[6];
rz(3.0434178831651115) q[6];
ecr q[2],q[6];
sx q[6];
rz(3.0434178831651124) q[6];
sx q[6];
rz(-pi) q[6];
ecr q[2],q[6];
rz(-1.4726215563702145) q[2];
sx q[2];
rz(-pi) q[2];
ecr q[5],q[2];
rz(15*pi/16) q[2];
sx q[2];
rz(-pi) q[2];
x q[5];
rz(-pi/2) q[5];
ecr q[5],q[2];
rz(-15*pi/16) q[2];
sx q[2];
rz(-pi) q[2];
ecr q[4],q[2];
rz(7*pi/8) q[2];
sx q[2];
rz(-pi) q[2];
x q[4];
rz(-pi/2) q[4];
ecr q[4],q[2];
rz(-7*pi/8) q[2];
sx q[2];
rz(-pi) q[2];
ecr q[3],q[2];
rz(3*pi/4) q[2];
sx q[2];
rz(-pi) q[2];
x q[3];
rz(-pi/2) q[3];
ecr q[3],q[2];
rz(3*pi/4) q[2];
sx q[2];
rz(pi/4) q[2];
x q[3];
rz(-3*pi/8) q[3];
x q[4];
rz(-7*pi/16) q[4];
x q[5];
rz(-1.4726215563702159) q[5];
rz(-0.049087385212343015) q[6];
ecr q[1],q[6];
sx q[6];
rz(3.0925052683774528) q[6];
sx q[6];
rz(-pi) q[6];
ecr q[1],q[6];
rz(-1.521708941582557) q[1];
sx q[1];
rz(-pi) q[1];
ecr q[5],q[1];
rz(3.0434178831651115) q[1];
sx q[1];
rz(-pi) q[1];
x q[5];
rz(-pi/2) q[5];
ecr q[5],q[1];
rz(-3.0434178831651124) q[1];
sx q[1];
rz(-pi) q[1];
ecr q[4],q[1];
rz(15*pi/16) q[1];
sx q[1];
rz(-pi) q[1];
x q[4];
rz(-pi/2) q[4];
ecr q[4],q[1];
rz(-15*pi/16) q[1];
sx q[1];
rz(-pi) q[1];
ecr q[3],q[1];
rz(7*pi/8) q[1];
sx q[1];
rz(-pi) q[1];
x q[3];
rz(-pi/2) q[3];
ecr q[3],q[1];
rz(-7*pi/8) q[1];
sx q[1];
rz(-pi) q[1];
ecr q[2],q[1];
rz(3*pi/4) q[1];
sx q[1];
rz(-pi) q[1];
x q[2];
rz(-pi/2) q[2];
ecr q[2],q[1];
rz(3*pi/4) q[1];
sx q[1];
rz(pi/4) q[1];
x q[2];
rz(-3*pi/8) q[2];
x q[3];
rz(-7*pi/16) q[3];
rz(3.0434178831651124) q[4];
x q[5];
rz(pi/64) q[5];
rz(-pi/128) q[6];
ecr q[0],q[6];
sx q[6];
rz(3.117048960983622) q[6];
sx q[6];
rz(-pi) q[6];
ecr q[0],q[6];
x q[0];
rz(3.117048960983622) q[0];
ecr q[0],q[5];
sx q[5];
rz(3.0925052683774528) q[5];
sx q[5];
rz(-pi) q[5];
ecr q[0],q[5];
x q[0];
rz(-3.0925052683774545) q[0];
ecr q[0],q[4];
sx q[4];
rz(3.0434178831651124) q[4];
sx q[4];
rz(-pi) q[4];
ecr q[0],q[4];
rz(-1.4726215563702159) q[0];
sx q[0];
rz(-pi) q[0];
ecr q[3],q[0];
rz(15*pi/16) q[0];
sx q[0];
rz(-pi) q[0];
x q[3];
rz(-pi/2) q[3];
ecr q[3],q[0];
rz(-15*pi/16) q[0];
sx q[0];
rz(-pi) q[0];
ecr q[2],q[0];
rz(7*pi/8) q[0];
sx q[0];
rz(-pi) q[0];
x q[2];
rz(-pi/2) q[2];
ecr q[2],q[0];
rz(-7*pi/8) q[0];
sx q[0];
rz(-pi) q[0];
ecr q[1],q[0];
rz(3*pi/4) q[0];
sx q[0];
rz(-pi) q[0];
x q[1];
rz(-pi/2) q[1];
ecr q[1],q[0];
rz(3*pi/4) q[0];
sx q[0];
x q[1];
rz(-pi/2) q[1];
x q[2];
rz(-pi/2) q[2];
x q[3];
rz(-pi) q[4];
sx q[4];
ecr q[2],q[4];
sx q[2];
rz(-pi/2) q[4];
ecr q[4],q[2];
rz(-pi/2) q[2];
sx q[4];
ecr q[2],q[4];
x q[2];
rz(-pi) q[5];
sx q[5];
rz(-pi) q[5];
ecr q[1],q[5];
sx q[1];
rz(-pi/2) q[5];
ecr q[5],q[1];
rz(-pi/2) q[1];
sx q[5];
ecr q[1],q[5];
x q[1];
rz(-pi) q[6];
sx q[6];
ecr q[0],q[6];
sx q[0];
rz(-pi/2) q[6];
ecr q[6],q[0];
rz(-pi/2) q[0];
sx q[6];
ecr q[0],q[6];
x q[0];
