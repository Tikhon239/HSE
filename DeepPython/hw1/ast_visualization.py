import ast
import inspect
import os
import textwrap
from typing import List, Callable

import networkx as nx

from fibonacci import Fibonacci


class ASTGraphEditor:
    @staticmethod
    def get_node_description(node):
        label = node.__class__.__name__
        color = "gray"
        shape = "rectangle"
        style = "filled"

        if isinstance(node, ast.If):
            color = "yellow"
            shape = "triangle"

        if isinstance(node, ast.Return):
            color = "red"
            shape = "circle"

        if isinstance(node, ast.BinOp):
            label = f'{node.__class__.__name__}: {node.op.__class__.__name__}'

        return {"label": label, "color": color, "shape": shape, "style": style}

    def build_graph(self, node):
        graph = nx.DiGraph()
        return self.add_node(node, graph)

    def add_node(self, node, graph):
        graph.add_node(node, **self.get_node_description(node))
        for _, value in ast.iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        graph.add_edge(node, item)
                        self.add_node(item, graph)

            elif isinstance(value, ast.AST):
                graph.add_edge(node, value)
                self.add_node(value, graph)
        return graph

    @staticmethod
    def delete_nodes(graph, deleted_nodes: List[str]):
        candidates = []
        for node in graph.nodes:
            if node.__class__.__name__ in deleted_nodes:
                candidates.append(node)
        # плохо что inplace
        graph.remove_nodes_from(candidates)
        return graph


def ast_visualization(function: Callable, save_path: str):
    function_text = textwrap.dedent(inspect.getsource(function))
    ast_tree = ast.parse(function_text)

    ast_graph_editor = ASTGraphEditor()
    graph = ast_graph_editor.build_graph(ast_tree)
    graph = ast_graph_editor.delete_nodes(graph, ["Store", "Module", "Load"])

    p = nx.drawing.nx_pydot.to_pydot(graph)
    p.write_png(os.path.join(save_path, "ast.png"))


if __name__ == "__main__":
    f = Fibonacci()
    ast_visualization(f.__call__, "artifacts")
