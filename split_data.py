import random

with open("issuelist", "r") as inp:
    with open("datalist", "w") as outp:
        with open("testlist", "w") as testp:
            while True:
                line = inp.readline()
                if not line:
                    break
                if random.random() < 0.1:
                    testp.write(line)
                else:
                    outp.write(line)
