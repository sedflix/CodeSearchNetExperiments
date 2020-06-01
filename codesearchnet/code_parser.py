"""
This file is responsible for taking in a code block and producing a networkx graph out of it!
"""

from queue import Queue
from typing import Dict, Any, Tuple, List

import networkx as nx
from matplotlib import pyplot as plt
from tree_sitter import Language, Parser, Node

SO_PATH: str = "../resources/csnet_parse_build.so"


class TreeSitterNode(object):

    def __init__(self, node: Node, program: str = None):
        """
        :param node: The tree_sitter node
        :param program: the str of the program
        """

        self.type = node.type
        self.start_byte = node.start_byte
        self.end_byte = node.end_byte
        self.name = self.get_name(node, program)

    def get_name(self, node: Node, program: str = None) -> str:
        if 'identifier' in node.type and node.is_named and not program is None:
            return program[self.start_byte:self.end_byte]
        return node.type

    def __eq__(self, obj):
        return self.type == obj.type and self.start_byte == obj.start_byte and self.end_byte == obj.end_byte

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f'{self.name} @ [{self.start_byte}, {self.end_byte}]'

    def __hash__(self):
        return hash(self.__str__())


PARSERS = {}


def get_parser(lang: str, so_path: str = None) -> Parser:
    if so_path is None:
        so_path = SO_PATH

    # global PARSERS
    # if lang in PARSERS:
    #     return PARSERS[lang]

    LANG = Language(so_path, lang)

    parser = Parser()
    parser.set_language(LANG)

    # PARSERS[lang] = parser

    return parser


def parse_program(program: str, lang: str = None, parser: Parser = None) -> nx.DiGraph:
    if parser is None:
        if lang is None:
            raise Exception("either lang should be giver or parser should be given")
        parser: Parser = get_parser(lang)

    tree = parser.parse(bytes(program, "utf8"))

    g: nx.DiGraph = nx.DiGraph()

    queue: Queue = Queue()
    queue.put(tree.root_node)

    while not queue.empty():

        node = queue.get()

        if not hasattr(node, 'children'):
            continue

        for child in node.children:
            g.add_edge(TreeSitterNode(node, program), TreeSitterNode(child, program))
            queue.put(child)

    return g


def plot_graph(g: nx.DiGraph, labels: str = 'name', figsize: Tuple[float] = (20.0, 10.0)):
    """

    :param figsize:
    :param g:
    :param labels:
    :return:
    """
    from networkx.drawing.nx_agraph import graphviz_layout
    labels_map: Dict[TreeSitterNode, Any] = None
    if labels is 'name':
        labels_map = {node: node.name for node in g.nodes}
    elif labels is 'number':
        g = nx.convert_node_labels_to_integers(g)

    pos = graphviz_layout(g, prog='dot')
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    nx.draw(g, pos, ax=ax,
            with_labels=True,
            labels=labels_map,
            arrows=True,
            font_size=min(figsize[0], figsize[1]),
            node_color="#C2C2C7")


def get_tokens_from_graph(g: nx.DiGraph, kind_of_tokens="name") -> List[str]:
    tokens: List[str] = []
    for node in nx.dfs_preorder_nodes(g):
        token = ""
        if kind_of_tokens == "name":
            token = node.name
        elif kind_of_tokens == 'type':
            token = node.type
        tokens.append(token)
    return tokens


if __name__ == '__main__':
    program_str_1 = """
    def add(a, b):
        return a + b
    """

    parser = get_parser("python", SO_PATH)
    g1 = parse_program(program_str_1, parser=parser)
    print(get_tokens_from_graph(g1, kind_of_tokens="name"))

    # plot_graph(g1)
    # plt.show()
