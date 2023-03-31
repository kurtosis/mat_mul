# from argparse import ArgumentParser
# from pathlib import Path
# from time import time

# from model import *
# from data_generation import *
# from strassen_training import *
from synthetic_training import *
# from utils import *


def main():
    # StrassenTrainingApp().main()
    SyntheticDemoTrainingApp().main()
    # if args.task == "strassen":
    #     StrassenTrainingApp().main()
    # elif args.task == "synthetic":
    #     SyntheticDemoTrainingApp().main()


if __name__ == "__main__":
    main()
