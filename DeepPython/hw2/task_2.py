import os

from hw1.ast_visualization import ast_visualization
from hw1.fibonacci import Fibonacci
from pdflatex import PDFLaTeX

from latex_utils import convert_tags, generate_latex_table, generate_latex_image


def compile_latex_document_with_table_and_image(list_table, image_path):
    document_tags = ['document']
    header = '\\documentclass{article}\n\\usepackage{graphicx}\n'
    document_begin = convert_tags(document_tags, initializer=header)
    document_end = convert_tags(document_tags, begin=False)

    latex_table = generate_latex_table(list_table)
    latex_image = generate_latex_image(image_path)

    return document_begin + latex_table + latex_image + document_end


if __name__ == '__main__':
    list_table = [
        ['', 'col name 1', 'col name 2'],
        ['row name 1', 'value 1', 'value 2'],
        ['row name 2', 'value 3', 'value 4']
    ]

    f = Fibonacci()
    ast_visualization(f.__call__, 'artifacts')

    latex_document = compile_latex_document_with_table_and_image(list_table, os.path.join('artifacts', 'ast.png'))

    latex_file_name = os.path.join('artifacts', 'latex_table_and_image.tex')
    with open(latex_file_name, "w+") as file:
        file.write(latex_document)

    pdf_latex = PDFLaTeX.from_texfile(latex_file_name)
    pdf, log, completed_process = pdf_latex.create_pdf(keep_pdf_file=True, keep_log_file=False)
