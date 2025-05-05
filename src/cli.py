import inquirer
from pathlib import Path
from pyfiglet import Figlet
from riposte import Riposte
from colorama import Fore, Style


from .utils.os_utils import clear_screen, print_info_list

info_list = []

class CLI(Riposte):
    BANNER = (
        f'{Fore.GREEN}'
        f'{Style.BRIGHT}'
        f'{Figlet(font="slant").renderText("NER CLI")}'
        f'{Fore.YELLOW}'
        f'Wellcome to NER CLI !!!\n'
        f'{Fore.CYAN}'
        f'Start with:\n'
        f'{Fore.MAGENTA}[0] {Fore.BLUE}exit (This command is used to exit the program)\n'
        f'{Fore.MAGENTA}[1] {Fore.BLUE}commands (This command is used to view all available commands)\n'
        f'{Style.RESET_ALL}'
    )
    def __init__(self):
        self._archive_path = Path(__file__).parent.parent / 'home'
        self._history_file = self._archive_path / 'history'
        super().__init__(
            '',
            CLI.BANNER,
            self._history_file,
            100)
        
        self._cur_path = Path(__file__).parent.parent.parent / 'home'
        
    @property
    def prompt(self):
        cur_path = str(self._cur_path).replace(str(self._archive_path), '/⌂')
        return f'{Fore.YELLOW}{Style.BRIGHT}NER-cli:[ {cur_path}]$ {Style.RESET_ALL}'

clear_screen()
ner_cli = CLI()


@ner_cli.command('exit')
def exit():
    clear_screen()
    quit()


@ner_cli.command('commands')
def commands():
    clear_screen()
    print((
        f'{Fore.GREEN}'
        f'{Style.BRIGHT}'
        f'{Figlet(font="slant").renderText("NER CLI")}'
        f'{Fore.YELLOW}'
        f'All commands:\n'
        f'{Fore.CYAN}'
        f'{Fore.MAGENTA}[0] {Fore.BLUE}build_database (This command is used to call the build_database function)\n'
        f'{Fore.MAGENTA}[0] {Fore.BLUE}train (This command is used to call the training data generation function)\n'
        f'{Fore.MAGENTA}[1] {Fore.BLUE}test (This command is used to call the testing data generation function)\n'
        f'{Fore.MAGENTA}[2] {Fore.BLUE}evaluate (This command is used to call the evaluate function)\n'
        f'{Style.RESET_ALL}'
    ))

@ner_cli.command('build_database')
def build_database():
    questions = [inquirer.Text('chunk_size', message='What\'s your chunk_size'),
                 inquirer.Text('batch_size', message='What\'s your batch_size')]
    answers = inquirer.prompt(questions)
    try:
        chunk_size = int(answers['chunk_size'])
        batch_size = int(answers['batch_size'])
    except Exception as e:
        info_list.append({
            'error': f'{e}'
            })
        clear_screen()
        print(CLI.BANNER + '\n')
        print_info_list(info_list)
        return
    print('waiting...\n')
    from .utils.database import DataBase
    database = DataBase(chunk_size=chunk_size, batch_size=batch_size)
    info_list.append({
            'success': f'documents per domain: {database.count()}.'
        })
    clear_screen()
    print(CLI.BANNER + '\n')
    print_info_list(info_list)


@ner_cli.command('train')
def train():
    clear_screen()
    print((
        f'{Fore.GREEN}'
        f'{Style.BRIGHT}'
        f'{Figlet(font="slant").renderText("NER CLI")}'
    ))
    questions = [
        inquirer.List('action',
                        message='Select the action you want to perform',
                        choices=['Get the augmented iepile dataset.',
                                 'Get the extension correction iepile dataset.'],
                    )]
    answers = inquirer.prompt(questions)
    if answers['action'] == 'Get the augmented iepile dataset.':
        print('waiting...\n')
        from .train.iepile_augmentation_build import Builder as IEPILEAugBuilder
        builder = IEPILEAugBuilder()
        sizes = builder.build()
        info_list.append({
            'success': f'iepile training set size: {sizes["train"]}; iepile dev set size: {sizes["dev"]}.'
        })
    if answers['action'] == 'Get the extension correction iepile dataset.':
        questions = [inquirer.Text('model_name', message='What\'s your model_name'),
                     inquirer.List('with_dev',message='Whether the dev dataset needs to be processed',choices=['Yes', 'No'],
            )]
        answers = inquirer.prompt(questions)
        print('waiting...\n')
        from .train.iepile_ec_build import Builder as IEPILEECBuilder
        builder = IEPILEECBuilder(answers['model_name'])
        if answers['with_dev'] == 'Yes':
            with_dev = True
        else:
            with_dev = False
        sizes = builder.build(with_dev=with_dev)
        info_list.append({
            'success': f'iepile ec training set size: {sizes["train"]}; iepile ec dev set size: {sizes["dev"]}.'
        })
    clear_screen()
    print(CLI.BANNER + '\n')
    print_info_list(info_list)


@ner_cli.command('test')
def test():
    clear_screen()
    print((
        f'{Fore.GREEN}'
        f'{Style.BRIGHT}'
        f'{Figlet(font="slant").renderText("NER CLI")}'
    ))
    questions = [
        inquirer.List('action',
                        message='Select the action you want to perform',
                        choices=['Get the base crossner dataset.',
                                 'Get the extension correction crossner dataset.',
                                 'Get the documents-based correction crossner dataset.'],
                    )]
    answers = inquirer.prompt(questions)
    if answers['action'] == 'Get the base crossner dataset.':
        try:
            questions = [inquirer.Text('num_schema', message='What\'s your num_schema')]
            answers = inquirer.prompt(questions)
            num_schema = int(answers['num_schema'])
        except Exception as e:
            info_list.append({
            'error': f'{e}'
            })
            clear_screen()
            print(CLI.BANNER + '\n')
            print_info_list(info_list)
            return
        print('waiting...\n')
        from .test.crossner_build import Builder as BaseBuilder
        builder = BaseBuilder()
        data_size = builder.build(num_schema=num_schema)
        info_list.append({
            'success': f'crossner set size: {data_size}.'
        })
    elif answers['action'] == 'Get the extension correction crossner dataset.':
        questions = [inquirer.Text('model_name', message='What\'s your model_name')]
        answers = inquirer.prompt(questions)
        print('waiting...\n')
        from .test.crossner_ec_build import Builder as CrossNERECBuilder
        builder = CrossNERECBuilder(answers['model_name'])
        data_size = builder.build()
        info_list.append({
            'success': f'({answers["model_name"]}) extension correction crossner size: {data_size}.'
        })
    elif answers['action'] == 'Get the documents-based correction crossner dataset.':
        questions = [inquirer.Text('model_name', message='What\'s your model_name'),
                     inquirer.Text('chunk_size', message='What\'s your chunk_size'),
                     inquirer.Text('top_n', message='How many documents are found by similarity (n)'),
                     inquirer.Text('top_k', message='How many documents should you choose last (k)')]
        answers = inquirer.prompt(questions)
        try:
            model_name = answers['model_name']
            chunk_size = int(answers['chunk_size'])
            top_n = int(answers['top_n'])
            top_k = int(answers['top_k'])
        except Exception as e:
            info_list.append({
            'error': f'{e}'
            })
            clear_screen()
            print(CLI.BANNER + '\n')
            print_info_list(info_list)
            return
        print('waiting...\n')
        if top_k != 0:
            from .test.crossner_retrieval_docs_build import Builder as CrossNERRetrievalDocsBuilder
            builder = CrossNERRetrievalDocsBuilder(model_name=model_name,chunk_size=chunk_size,top_n=top_n,top_k=top_k)
            data_size = builder.build()
            info_list.append({
                'success': f'({answers["model_name"]}) documents-based[{top_k}] correction crossner size: {data_size}.'
            })
    clear_screen()
    print(CLI.BANNER + '\n')
    print_info_list(info_list)


@ner_cli.command('evaluate')
def evaluate():
    clear_screen()
    print((
        f'{Fore.GREEN}'
        f'{Style.BRIGHT}'
        f'{Figlet(font="slant").renderText("NER CLI")}'
    ))
    questions = [inquirer.List('action',
                    message='Select the action you want to perform',
                    choices=['Evaluate the results on crossner.'],
                    )]
    answers = inquirer.prompt(questions)
    if answers['action'] == 'Evaluate the results on crossner.':
        questions = [inquirer.Text('model_name', message='What\'s your model_name')]
        answers = inquirer.prompt(questions)
        model_name = answers['model_name']
        print('waiting...\n')
        from .evaluate.crossner_evaluator import Evaluator as CrossnerEvaluator
        evaluator = CrossnerEvaluator()
        save_path = evaluator.evaluate(model_name=model_name)
        info_list.append({
                'success': f'The result has been written to file [{save_path}].'
            })
    clear_screen()
    print(CLI.BANNER + '\n')
    print_info_list(info_list)