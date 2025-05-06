import os
from colorama import Fore, Style

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_info_list(info_list):
    for info in info_list:
        for key, value in info.items():
            if key == 'success':
                print(f'{Style.BRIGHT}{Fore.GREEN}[success] {Fore.BLUE}{value}')
            elif key == 'error':
                print(f'{Style.BRIGHT}{Fore.RED}[error] {Fore.BLUE}{value}')

    print('')