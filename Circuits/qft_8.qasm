OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
creg c[8];
h q[0];
cu1(pi/2) q[0],q[1];
cu1(pi/4) q[0],q[2];
cu1(pi/8) q[0],q[3];
cu1(pi/16) q[0],q[4];
cu1(pi/32) q[0],q[5];
cu1(pi/64) q[0],q[6];
cu1(pi/128) q[0],q[7];
h q[1];
cu1(pi/2) q[1],q[2];
cu1(pi/4) q[1],q[3];
cu1(pi/8) q[1],q[4];
cu1(pi/16) q[1],q[5];
cu1(pi/32) q[1],q[6];
cu1(pi/64) q[1],q[7];
h q[2];
cu1(pi/2) q[2],q[3];
cu1(pi/4) q[2],q[4];
cu1(pi/8) q[2],q[5];
cu1(pi/16) q[2],q[6];
cu1(pi/32) q[2],q[7];
h q[3];
cu1(pi/2) q[3],q[4];
cu1(pi/4) q[3],q[5];
cu1(pi/8) q[3],q[6];
cu1(pi/16) q[3],q[7];
h q[4];
cu1(pi/2) q[4],q[5];
cu1(pi/4) q[4],q[6];
cu1(pi/8) q[4],q[7];
h q[5];
cu1(pi/2) q[5],q[6];
cu1(pi/4) q[5],q[7];
h q[6];
cu1(pi/2) q[6],q[7];
h q[7];
