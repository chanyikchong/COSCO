import os
from .ColorUtils import color


def generate_decision_migration_string(decisions, migrations):
    sub_string_list = list()
    for i, d in enumerate(decisions):
        if d not in migrations:
            sub_string = "%s%s%s" % (color.FAIL, str(d), color.ENDC)
        else:
            sub_string = str(d)
        sub_string_list.append(sub_string)
    sub_string = ', '.join(sub_string_list)
    return 'Decision: [%s]' % sub_string


def print_decision_and_migrations(decision, migrations):
    string = generate_decision_migration_string(decision, migrations)
    print(string)


def unixify(paths):
    for path in paths:
        for file in os.listdir(path):
            if '.py' in file or '.sh' in file:
                _ = os.system("bash -c \"dos2unix " + path + file + " 2&> /dev/null\"")
