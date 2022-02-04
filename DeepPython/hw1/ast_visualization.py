import ast
import inspect
import os
import textwrap
from typing import Callable

import networkx as nx

from fibonacci import Fibonacci
from graph_editor import ASTGraphEditor


def ast_visualization(function: Callable, save_path: str) -> None:
    function_text = textwrap.dedent(inspect.getsource(function))
    ast_tree = ast.parse(function_text)

    ast_graph_editor = ASTGraphEditor()
    graph = ast_graph_editor.build_graph(ast_tree)
    graph = ast_graph_editor.delete_nodes(graph, [ast.Store, ast.Module, ast.Load])

    p = nx.drawing.nx_pydot.to_pydot(graph)
    p.write_png(os.path.join(save_path, "ast.png"))


if __name__ == "__main__":
    f = Fibonacci()
    ast_visualization(f.__call__, "artifacts")
