import sys
import yy_final_exercise as yy


def main(argv):
    if len(argv) != 2:
        print("Usage: python main.py path/to/file.csv")
        exit(1)

    csv_name = argv[1]

    yy.demo(csv_name)

if __name__ == '__main__':
    main(sys.argv)
