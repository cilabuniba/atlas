import sys
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PyQt6.QtWidgets import QApplication
app = QApplication(sys.argv)

from utils.code_importer import GraphImporter
from views.graph_scene import GraphScene

scene = GraphScene()

def test_code(name, code):
    print(f"\n--- Testing: {name} ---")
    try:
        GraphImporter.import_from_code(code, scene)
        print("✅ SUCCESS (Code executed without security errors)")
    except Exception as e:
        print(f"❌ BLOCKED: {e}")

# Case 1: Normal graph generation (Should PASS)
code_normal = """
import networkx as nx
G = nx.erdos_renyi_graph(10, 0.5)
pos = nx.spring_layout(G)
for n in G.nodes():
    G.nodes[n]['pos'] = pos[n].tolist()
"""
test_code("Normal NetworkX Graph", code_normal)

# Case 2: Attempting to import 'os' (Should BLOCK)
code_import_os = """
import networkx as nx
import os
G = nx.Graph()
"""
test_code("Importing 'os' module", code_import_os)

# Case 3: Attempting to use open() (Should BLOCK)
code_open = """
import networkx as nx
f = open('test.txt', 'w')
f.write('hello')
G = nx.Graph()
"""
test_code("Using open() function", code_open)

# Case 4: Attempting to access __class__ (Should BLOCK)
code_dunder = """
import networkx as nx
x = [].__class__
G = nx.Graph()
"""
test_code("Accessing __class__ attribute", code_dunder)

