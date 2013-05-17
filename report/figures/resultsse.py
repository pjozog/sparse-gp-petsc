#!/bin/env/python2

from pylab import *

times0_1 = array([187.150625,108.330420,74.426996,57.390306,29.875077,32.799355,30.877238,40.771062])
times1 = array([2739.799414,1436.301511,1055.523877,840.427201,493.569283,384.757745,314.533461,279.590304])
times10 = array([22353.0, 12010.3, 7584.1, 5601.3, 2817.093872,1505.675602,1050.419739,775.594670])

timesBig = array([26000, 13883.32, 9190.707582,6686.790996,4058.741773,1877.304765,1111.624388,887.110239])

p = array([1, 2, 3, 4, 9, 16, 25, 36])

speedup0_1 = times0_1[0] / (times0_1)
speedup1 = times1[0] / (times1)
speedup10 = times10[0] / (times10)
speedupBig = timesBig[0] / timesBig

effic0_1 = times0_1[0] / (p * times0_1)
effic1 = times1[0] / (p * times1)
effic10 = times10[0] / (p * times10)
efficBig = timesBig[0] / (p * timesBig)

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
f1.savefig ('speedup_grad_1e4.pdf')

f2 = figure ()
plot (p, p, '--', label='ideal')
hold ('on')
plot (p, speedupBig, label='0.05% nonzero')
# axis ('equal')
grid ('on')
ylabel ('Speedup')
xlabel ('Number of Processing Elements')
legend ()

f2.show ()
f2.savefig ('speedup_grad_1e5.pdf')

f3 = figure ()
plot (p, ones (p.shape), '--', label='ideal')
hold ('on')
plot (p, effic0_1, label='0.10% nonzero')
plot (p, effic1, label='1.18% nonzero')
plot (p, effic10, label='11.63% nonzero')
grid ('on')
xlabel ('Number of Processing Elements')
ylabel ('Efficiency')
legend (loc='lower left')

f3.show ()
f3.savefig ('effic_grad_1e4.pdf')

f4 = figure ()
plot (p, ones (p.shape), '--', label='ideal')
hold ('on')
plot (p, efficBig, label='0.05% nonzero')
grid ('on')
xlabel ('Number of Processing Elements')
ylabel ('Efficiency')
legend ()

f4.show ()
f4.savefig ('effic_grad_1e5.pdf')
