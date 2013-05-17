#!/usr/bin/env python2

from pylab import *

times0_1 = array([8.297250,1.017617,0.392766,0.224927,0.143942,0.137313])
times1 = array([214.799995,16.630342,4.268002,1.379326,0.602422,0.388496])
times10 = array([2500.1,1146.261074,369.145711,128.303774,47.854218,25.066478])
timesBig  = array([566.56,98.426960,42.935925,23.341612,13.792518,9.511550])

# times0_1 = times0_1[::-1]
# times1 = times1[::-1]
# times10 = times10[::-1]
# timesBig = timesBig[::-1]

p = array([1, 4, 9, 16, 25, 36])
pBig = array([1, 4, 9, 16, 25, 36])

speedup0_1 = times0_1[0] / (times0_1)
speedup1 = times1[0] / (times1)
speedup10 = times10[0] / (times10)
speedupBig = timesBig[0] / timesBig

effic0_1 = times0_1[0] / (p * times0_1)
effic1 = times1[0] / (p * times1)
effic10 = times10[0] / (p * times10)
efficBig = timesBig[0] / (pBig * timesBig)

f1 = figure ()
plot (p, p, '--', label='ideal')
hold ('on')
plot (p, speedup0_1, label='0.10% nonzero')
plot (p, speedup1, label='1.18% nonzero')
plot (p, speedup10, label='11.63% nonzero')
# axis ('equal')
grid ('on')
ylabel ('Speedup')
xlabel ('Number of Processing Elements')
legend ()

f1.show ()
f1.savefig ('speedup_K_1e4.pdf')

f2 = figure ()
plot (p, p, '--', label='ideal')
hold ('on')
plot (pBig, speedupBig, label='0.05% nonzero')
# axis ('equal')
grid ('on')
ylabel ('Speedup')
xlabel ('Number of Processing Elements')
legend ()

f2.show ()
f2.savefig ('speedup_K_1e5.pdf')

f3 = figure ()
plot (p, ones (p.shape), '--', label='ideal')
hold ('on')
plot (p, effic0_1, label='0.10% nonzero')
plot (p, effic1, label='1.18% nonzero')
plot (p, effic10, label='11.63% nonzero')
grid ('on')
xlabel ('Number of Processing Elements')
ylabel ('Efficiency')
legend (loc='upper left')

f3.show ()
f3.savefig ('effic_K_1e4.pdf')

f4 = figure ()
plot (p, ones (p.shape), '--', label='ideal')
hold ('on')
plot (pBig, efficBig, label='0.05% nonzero')
grid ('on')
xlabel ('Number of Processing Elements')
ylabel ('Efficiency')
legend ()

f4.show ()
f4.savefig ('effic_K_1e5.pdf')
