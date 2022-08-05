class Color:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    RED = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_decision_and_migrations(decision, migrations):
    print('Decision: [', end='')
    for i, d in enumerate(decision):
        if d not in migrations:
            print(Color.FAIL, end='')
        print(d, end='')
        if d not in migrations:
            print(Color.ENDC, end='')
        print(',', end='') if i != len(decision) - 1 else print(']')
    print()
