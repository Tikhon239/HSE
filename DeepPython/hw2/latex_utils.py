from functools import reduce


def convert_tags(tags, begin=True, initializer=''):
    if begin:
        return reduce(lambda a, b: a + f'\\begin' + '{' + b + '}\n', tags, initializer)
    else:
        return reduce(lambda a, b: a + f'\\end' + '{' + b + '}\n', tags[::-1], initializer)


def get_table_raw(table_row):
    return reduce(lambda a, b: a + ' & ' + b, table_row) + '\\\\'


def get_table_center(list_table):
     header = '{ |' + ' c |' * len(list_table[0]) + ' }'
     table_center = map(get_table_raw, list_table)
     table_center = reduce(lambda a, b: a + '\n\hline \n' + b, table_center, '')
     return header + table_center + '\n\hline \n'


def generate_latex_table(list_table):
    table_tags = ['table', 'center', 'tabular']
    table_begin = convert_tags(table_tags)
    table_end = convert_tags(table_tags, begin=False)

    table_center = get_table_center(list_table)

    return table_begin + table_center + table_end


def get_image_center(image_path):
    return '\\includegraphics[width = 15cm]{' + image_path + '}\n'


def generate_latex_image(image_path):
    image_tags = ['center']
    image_begin = convert_tags(image_tags)
    image_end = convert_tags(image_tags, begin=False)

    image_center = get_image_center(image_path)

    return image_begin + image_center + image_end
