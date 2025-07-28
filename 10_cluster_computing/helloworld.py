import argparse
import socket
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('id', type=int, help='a number to print')
    args = parser.parse_args()

    time.sleep(10)
    print(f'Hello World! I am task number {args.id} on host {socket.gethostname()}.')
