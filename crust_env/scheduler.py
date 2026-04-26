"""
CRust Dependency Scheduler — scheduler.py

Builds a Directed Acyclic Graph (DAG) from a legacy C codebase by parsing
local files into AST nodes (Semantic Slicing) using tree-sitter, then produces
a bottom-up topological migration schedule.
"""

import os
import re
from collections import deque, defaultdict
from typing import Dict, List, Set, Optional, Tuple

try:
    import tree_sitter_c
    from tree_sitter import Language, Parser
except ImportError:
    pass # Will be installed via requirements.txt


class CDependencyGraph:
    """
    Parses a C project directory tree and constructs an AST-level dependency DAG.

    Nodes  = functions, structs, and declarations
    Edges  = directed from caller/user → callee/definition
    """

    def __init__(self, c_code_dir: str):
        self.c_code_dir = c_code_dir
        self.graph: Dict[str, List[str]] = defaultdict(list)
        self.reverse_graph: Dict[str, List[str]] = defaultdict(list)
        self.nodes: Set[str] = set()
        self._node_code: Dict[str, str] = {}
        self._node_file: Dict[str, str] = {}
        
        if hasattr(tree_sitter_c, 'LANGUAGE'):
            self.LANGUAGE = tree_sitter_c.LANGUAGE
        else:
            self.LANGUAGE = Language(tree_sitter_c.language())
            
        self.parser = Parser()
        if hasattr(self.parser, 'language'):
            self.parser.language = self.LANGUAGE
        else:
            self.parser.set_language(self.LANGUAGE)

    def _find_files(self) -> List[str]:
        files: List[str] = []
        if not os.path.isdir(self.c_code_dir):
            return files
        for root, _, filenames in os.walk(self.c_code_dir):
            for filename in filenames:
                if filename.endswith((".c", ".h")):
                    files.append(os.path.join(root, filename))
        return files

    def build_graph(self) -> None:
        files = self._find_files()
        
        # Parse all files and extract nodes (functions/structs)
        for f in files:
            with open(f, 'rb') as file:
                source_code = file.read()
                
            tree = self.parser.parse(source_code)
            root_node = tree.root_node
            
            for node in root_node.children:
                if node.type in ['function_definition', 'struct_specifier', 'declaration']:
                    name = None
                    if node.type == 'function_definition':
                        declarator = node.child_by_field_name('declarator')
                        while declarator and declarator.type != 'identifier':
                            declarator = declarator.child_by_field_name('declarator')
                        if declarator:
                            name = source_code[declarator.start_byte:declarator.end_byte].decode('utf8')
                    elif node.type == 'struct_specifier':
                        name_node = node.child_by_field_name('name')
                        if name_node:
                            name = "struct_" + source_code[name_node.start_byte:name_node.end_byte].decode('utf8')
                    elif node.type == 'declaration':
                        type_node = node.child_by_field_name('type')
                        if type_node and type_node.type == 'struct_specifier':
                            name_node = type_node.child_by_field_name('name')
                            if name_node:
                                name = "struct_" + source_code[name_node.start_byte:name_node.end_byte].decode('utf8')
                    
                    if name:
                        self.nodes.add(name)
                        self._node_code[name] = source_code[node.start_byte:node.end_byte].decode('utf8')
                        self._node_file[name] = os.path.basename(f)
                        
                        # Find dependencies
                        def find_deps(n):
                            if n.type == 'call_expression':
                                func_node = n.child_by_field_name('function')
                                if func_node and func_node.type == 'identifier':
                                    dep_name = source_code[func_node.start_byte:func_node.end_byte].decode('utf8')
                                    if dep_name != name:
                                        self.graph[name].append(dep_name)
                                        self.reverse_graph[dep_name].append(name)
                            elif n.type == 'type_identifier':
                                type_name = source_code[n.start_byte:n.end_byte].decode('utf8')
                                dep_name = "struct_" + type_name
                                if dep_name != name:
                                    self.graph[name].append(dep_name)
                                    self.reverse_graph[dep_name].append(name)
                            for child in n.children:
                                find_deps(child)
                                
                        find_deps(node)

    def get_topological_schedule(self) -> List[Dict]:
        """
        Kahn's algorithm for topological sort — returns leaf-first order of AST nodes.
        Returns: List of dicts with node info.
        """
        self.build_graph()

        if not self.nodes:
            return []

        in_degree: Dict[str, int] = {node: 0 for node in self.nodes}
        for node, deps in self.graph.items():
            in_degree[node] = len([d for d in deps if d in self.nodes])

        queue: deque = deque(sorted(n for n in self.nodes if in_degree[n] == 0))
        schedule_names: List[str] = []

        while queue:
            node = queue.popleft()
            schedule_names.append(node)

            for dependent in sorted(self.reverse_graph.get(node, [])):
                if dependent in in_degree:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)

        if len(schedule_names) != len(self.nodes):
            scheduled_set = set(schedule_names)
            for node in sorted(self.nodes):
                if node not in scheduled_set:
                    schedule_names.append(node)

        schedule = []
        for name in schedule_names:
            schedule.append({
                "name": name,
                "file": self._node_file[name],
                "code": self._node_code[name]
            })

        return schedule

    def get_dependency_info(self) -> Dict[str, Dict]:
        self.build_graph()
        info: Dict[str, Dict] = {}
        for node in self.nodes:
            info[node] = {
                "depends_on": list(set(self.graph.get(node, []))),
                "depended_by": list(set(self.reverse_graph.get(node, []))),
                "file": self._node_file.get(node),
            }
        return info
