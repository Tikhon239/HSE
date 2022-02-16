from latex_utils import convert_tags, generate_latex_table


def compile_latex_document_with_table(list_table):
    document_tags = ['document']
    document_begin = convert_tags(document_tags, initializer='\\documentclass{article}\n')
    document_end = convert_tags(document_tags, begin=False)

    latex_table = generate_latex_table(list_table)

    return document_begin + latex_table + document_end


if __name__ == '__main__':
    list_table = [
        ['', 'col name 1', 'col name 2'],
        ['row name 1', 'value 1', 'value 2'],
        ['row name 2', 'value 3', 'value 4']
    ]

    latex_document = compile_latex_document_with_table(list_table)

    with open("artifacts/latex_table.tex", "w+") as file:
        file.write(latex_document)

