import chimera
from chimera import runCommand

for i in range(1, 35):
    # runCommand("spec 20 #4." + str(i) + ":1-2.A@ca ")
    # runCommand('shape tube #4.' + str(i) + ':1-2.A@ca radius 1.7 color #22c74e')
    runCommand('shape tube #2.' + str(i) + ':1-1000.A@ca radius 1.5 color #0d521f')
    # runCommand('shape tube #4.' + str(i) + ':11-17.A@ca radius 1.5 color orange')
runCommand("color green #2:1.A@ca")
runCommand("vdwdefine 8 #2:1.A@ca")
runCommand("vdw #2:1.A@ca")
runCommand("~ribbon")
