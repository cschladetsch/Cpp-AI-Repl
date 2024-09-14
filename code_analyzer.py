import clang.cindex
from functools import lru_cache
import networkx as nx
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from colorama import Fore, Style

class CodeAnalyzer:
    def __init__(self):
        self.index = clang.cindex.Index.create()
        self.cursor_kind = clang.cindex.CursorKind

    @lru_cache(maxsize=100)
    def parse_cpp_file(self, file_path):
        try:
            return self.index.parse(file_path)
        except Exception as e:
            print(f"{Fore.RED}Error parsing C++ file: {e}{Style.RESET_ALL}")
            return None

    def build_ast_graph(self, cursor):
        G = nx.DiGraph()

        def add_node_and_edges(node, parent=None):
            node_id = f"{node.kind}:{node.spelling}:{node.location.line}:{node.location.column}"
            G.add_node(node_id, kind=node.kind.name, spelling=node.spelling)
            if parent:
                G.add_edge(parent, node_id)
            for child in node.get_children():
                add_node_and_edges(child, node_id)

        add_node_and_edges(cursor)
        return G

    def extract_code_features(self, G):
        if G is None:
            return [0, 0, 0, 0, 0, 0]
        features = {
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'avg_degree': sum(dict(G.degree()).values()) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0,
            'num_functions': sum(1 for _, data in G.nodes(data=True) if data['kind'] == 'FUNCTION_DECL'),
            'num_classes': sum(1 for _, data in G.nodes(data=True) if data['kind'] == 'CLASS_DECL'),
            'max_depth': max(nx.shortest_path_length(G, source=list(G.nodes())[0]).values()) if G.number_of_nodes() > 0 else 0,
        }
        return list(features.values())

    def detect_code_anomalies(self, code_info):
        features = self.extract_code_features(code_info.get('ast_graph'))
        features = StandardScaler().fit_transform([features])

        clf = IsolationForest(random_state=0).fit(features)
        labels = clf.predict(features)

        if -1 in labels:
            return f"{Fore.RED}Potential code anomaly detected.{Style.RESET_ALL}"
        return f"{Fore.GREEN}No significant code anomalies detected.{Style.RESET_ALL}"

    def analyze_file(self, file_path):
        translation_unit = self.parse_cpp_file(file_path)
        if not translation_unit:
            return None

        cursor = translation_unit.cursor
        code_info = {
            'file_path': file_path,
            'includes': [],
            'namespaces': [],
            'classes': [],
            'structs': [],
            'enums': [],
            'global_variables': [],
            'functions': [],
            'templates': [],
            'typedefs': [],
            'macros': [],
            'comments': [],
            'local_variables': [],
            'member_variables': [],
            'constructors': [],
            'destructors': [],
            'operator_overloads': [],
            'friend_functions': [],
            'virtual_functions': [],
            'pure_virtual_functions': [],
            'lambda_expressions': [],
            'exception_handlers': [],
            'memory_allocations': [],
            'static_assertions': [],
            'ast_graph': None
        }

        try:
            code_info['ast_graph'] = self.build_ast_graph(cursor)
        except Exception as e:
            print(f"{Fore.RED}Error building AST graph: {e}{Style.RESET_ALL}")

        def visit_node(node):
            location = f"{node.location.file.name}:{node.location.line}" if node.location.file else "Unknown"
            
            if node.kind == self.cursor_kind.INCLUSION_DIRECTIVE:
                code_info['includes'].append((node.displayname, location))
            elif node.kind == self.cursor_kind.NAMESPACE:
                code_info['namespaces'].append((node.displayname, location))
            elif node.kind == self.cursor_kind.CLASS_DECL:
                code_info['classes'].append((node.displayname, location))
            elif node.kind == self.cursor_kind.STRUCT_DECL:
                code_info['structs'].append((node.displayname, location))
            elif node.kind == self.cursor_kind.ENUM_DECL:
                code_info['enums'].append((node.displayname, location))
            elif node.kind == self.cursor_kind.VAR_DECL and node.semantic_parent.kind == self.cursor_kind.TRANSLATION_UNIT:
                code_info['global_variables'].append((node.displayname, node.type.spelling, location))
            elif node.kind == self.cursor_kind.FUNCTION_DECL:
                code_info['functions'].append((node.displayname, node.result_type.spelling, location))
            elif hasattr(self.cursor_kind, 'TEMPLATE_DECL') and node.kind == self.cursor_kind.TEMPLATE_DECL:
                code_info['templates'].append((node.displayname, location))
            elif node.kind == self.cursor_kind.TYPEDEF_DECL:
                code_info['typedefs'].append((node.displayname, node.underlying_typedef_type.spelling, location))
            elif node.kind == self.cursor_kind.MACRO_DEFINITION:
                code_info['macros'].append((node.displayname, location))
            elif node.kind == self.cursor_kind.VAR_DECL and node.semantic_parent.kind != self.cursor_kind.TRANSLATION_UNIT:
                code_info['local_variables'].append((node.displayname, node.type.spelling, location))
            elif node.kind == self.cursor_kind.FIELD_DECL:
                code_info['member_variables'].append((node.displayname, node.type.spelling, location))
            elif node.kind == self.cursor_kind.CONSTRUCTOR:
                code_info['constructors'].append((node.displayname, location))
            elif node.kind == self.cursor_kind.DESTRUCTOR:
                code_info['destructors'].append((node.displayname, location))
            elif node.kind == self.cursor_kind.CXX_METHOD and node.spelling.startswith('operator'):
                code_info['operator_overloads'].append((node.spelling, location))
            elif node.kind == self.cursor_kind.FRIEND_DECL:
                code_info['friend_functions'].append((node.displayname, location))
            elif node.kind == self.cursor_kind.CXX_METHOD and node.is_virtual_method():
                if node.is_pure_virtual_method():
                    code_info['pure_virtual_functions'].append((node.displayname, location))
                else:
                    code_info['virtual_functions'].append((node.displayname, location))
            elif node.kind == self.cursor_kind.LAMBDA_EXPR:
                code_info['lambda_expressions'].append((node.displayname or "Anonymous Lambda", location))
            elif node.kind == self.cursor_kind.CXX_CATCH_STMT:
                code_info['exception_handlers'].append((node.displayname or "Catch Block", location))
            elif node.kind == self.cursor_kind.CALL_EXPR and node.spelling in ['new', 'malloc', 'calloc', 'realloc']:
                code_info['memory_allocations'].append((node.spelling, location))
            elif node.kind == self.cursor_kind.STATIC_ASSERT:
                code_info['static_assertions'].append((node.displayname or "Static Assertion", location))

            for child in node.get_children():
                visit_node(child)

        visit_node(cursor)

        # Process comments separately as they are not part of the AST
        for token in cursor.get_tokens():
            if token.kind == clang.cindex.TokenKind.COMMENT:
                code_info['comments'].append((token.spelling, f"{token.location.file.name}:{token.location.line}"))

        code_info['anomalies'] = self.detect_code_anomalies(code_info)
        return code_info
