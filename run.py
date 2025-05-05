import sys
sys.dont_write_bytecode = True


if __name__ == '__main__':
    from src.cli import ner_cli
    ner_cli.run()