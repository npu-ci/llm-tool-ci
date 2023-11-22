"""
Check whether the loss is less than or equal to 0
"""
import sys
if __name__ == "__main__":
    args = sys.argv[1:]
    
    with open(args[0], mode='r') as f:
        line = f.readline()
        while line:
            if "'loss': 0.0," in line or "'loss': -":
                raise ValueError("Got loss <=0.0, some errors caused by precision have occurred.")
            line = f.readline()

