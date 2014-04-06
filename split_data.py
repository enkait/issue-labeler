import random

with open("issue_list", "r") as inp:
    with open("data_list", "w") as outp:
        with open("cv_list", "w") as cvp:
            with open("test_list", "w") as testp:
                while True:
                    line = inp.readline()
                    if not line:
                        break
                    if random.random() < 0.05:
                        testp.write(line)
                    elif random.random() < 0.05:
                        cvp.write(line)
                    else:
                        outp.write(line)
