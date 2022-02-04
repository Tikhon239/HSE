import ast
import random
from collections import defaultdict
from typing import List

import networkx as nx


class ASTGraphEditor:
    def __init__(self, colors_seed: int = 239):
        class_names = [
            ast.FunctionDef, ast.If, ast.Return, ast.BinOp,
            ast.arguments, ast.arg, ast.Compare, ast.Load,
            ast.Name, ast.Attribute, ast.Assign, ast.Module,
            ast.Constant, ast.Expr, ast.Call, ast.Subscript,
            ast.Sub, ast.Lt, ast.Add, ast.Assign, ast.Store,
        ]
        self.class_colors = defaultdict(lambda: "gray")
        random.seed(colors_seed)
        for class_name in class_names:
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            self.class_colors[class_name] = '#%02x%02x%02x' % (r, g, b)

    def get_node_description(self, node: ast.AST):
        label = node.__class__.__name__
        color = self.class_colors[node.__class__]
        shape = "rectangle"
        style = "filled"

        if isinstance(node, ast.If):
            shape = "triangle"

        if isinstance(node, ast.Return):
            shape = "circle"

        if isinstance(node, ast.BinOp):
            label = f'{node.__class__.__name__}: {node.op.__class__.__name__}'

        return {"label": label, "color": color, "shape": shape, "style": style}

    def build_graph(self, node: ast.AST) -> nx.Graph:
        graph = nx.DiGraph()
        return self.add_node(node, graph)

    def add_node(self, node: ast.AST, graph: nx.Graph) -> nx.Graph:
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
    def delete_nodes(graph: nx.Graph, deleted_nodes: List[ast.AST]) -> nx.Graph:
        candidates = []
        for node in graph.nodes:
            if node.__class__ in deleted_nodes:
                candidates.append(node)
        # плохо что inplace
        graph.remove_nodes_from(candidates)
        return graph
