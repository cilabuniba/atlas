<h1 align="center">ATLAS: A Software for Interactive Modeling and Analysis of Homogeneous and Heterogeneous Graphs</h1>
<p align="center">
    <img src="docs/imgs/overview.png" width="500">
</p>

Despite the increasing adoption of graph-based learning methods, the integration between interactive graph modelling and programmatic graph analytics remains limited. We present ATLAS, a general-purpose software that bridges interactive graph modelling with modern graph analytics and graph-learning workflows. The software is implemented in Python and supports the interactive construction of both homogeneous and heterogeneous graphs through a graphical interface. It enables the definition and customisation of nodes and edges, the real-time computation of structural metrics, and the automatic generation of executable Python code compatible with major graph libraries, including NetworkX, igraph, PyVis, DGL, and PyTorch Geometric. In addition, existing code can be imported to reconstruct graph structures, supporting bidirectional integration between visual modelling and programmatic workflows. The software also facilitates the visual inspection of complex graph structures and graph-based explanatory outputs. Two real-world application scenarios demonstrate the applicability of ATLAS in practical contexts.

## Environment setup

ATLAS targets **Python 3.12** and uses [uv](https://github.com/astral-sh/uv). for dependency management. After cloning the repository, install all required packages with:

```bash
uv sync
```

The project dependencies can be grouped according to their purpose.

### GUI and graph visualization

The following packages are required to run the ATLAS graphical interface and support graph visualization, configuration management, and data manipulation:

| Package       | Purpose                                                    |
| ------------- | ---------------------------------------------------------- |
| `click`       | Command-line interface utilities.                          |
| `jupyter`     | Interactive notebooks for development and experimentation. |
| `networkx`    | Graph representation and visualization.                    |
| `pandas`      | Tabular data manipulation.                                 |
| `pyqt6`       | Desktop graphical user interface.                          |
| `ruamel-yaml` | Reading and writing YAML configuration files.              |
| `scipy`       | Scientific computing utilities.                            |

### GNN training and explanation

The experimental pipeline used to train Graph Neural Networks and generate explanations relies on the following machine learning libraries:

| Package           | Purpose                                                                                         |
| ----------------- | ----------------------------------------------------------------------------------------------- |
| `torch`           | Deep learning framework used to train GNN models.                                               |
| `torch-geometric` | Graph learning library providing datasets, models, and explanation algorithms.                  |
| `torchmetrics`    | Evaluation metrics for model training and testing.                                              |
| `transformers`    | Transformer-based models used by the multimodal and language-based components of the framework. |

All dependencies are installed automatically through `uv sync`. Users interested only in the visualization interface can ignore the machine learning libraries, whereas reproducing the experiments described in the paper requires the complete environment.

Once all dependencies have been installed, you can launch the GUI by running

``` bash
uv run main.py
```

## Software Description
<p align="center">
    <img src="docs/imgs/graph_choice.png" width="200">
</p>

The first interactive window you see after the command line app launch is the "*Select Mode*", with which you can choose either to manage homogeneous or heterogeneous graphs.

### Homogeneous Graphs

<p align="center">
    <img src="docs/imgs/homogeneous_interface.png" width="900">
</p>

ATLAS enables the interactive construction, customization, and analysis of homogeneous graphs through a set of coordinated graphical interface components.

The workflow starts from the graph editor toolbar, which provides the main interaction modes, including node and edge creation, node movement, and canvas navigation.

- **Node Creation** — Nodes can be introduced into the canvas through the *Add Node* function. Each node is initialized with default visual properties and can be repositioned via drag-and-drop. The system supports the modification of node attributes, including label, shape, color, and size, enabling flexible visual representation.

- **Edge Creation** — Edges can be created interactively between pairs of nodes through the *Add Edge* function. During this process, additional attributes, such as edge type and description, can be specified. Existing edges can also be modified or removed directly from the interface.

- **Visual Customization** — The software provides global and local customization options for nodes, edges, and labels through integrated configuration panels. Visual properties such as colors, shapes, edge thickness, and node size can be adjusted dynamically with immediate visual feedback.

- **Backend Library Selection** — ATLAS supports compatibility with multiple graph-processing ecosystems. Users can select the target backend library directly from the interface, including:
  - NetworkX
  - igraph
  - PyVis
  - DGL
  - PyTorch Geometric

- **Code Preview and Export** — The *Code Preview* panel maintains synchronization between the graphical editor and the programmatic representation by automatically generating executable Python code corresponding to the current graph state. The generated code can be copied or exported to external files for integration into external workflows.

- **Real-Time Metrics Computation** — The integrated *Metrics* panel continuously computes and updates structural graph properties, including:
  - Number of nodes
  - Number of edges
  - Graph density
  - Average degree
  - Additional graph indicators

  This functionality enables immediate inspection and analysis of graph structures during interactive exploration.

### Heterogeneous Graphs

<p align="center">
    <img src="docs/imgs/heterogeneous_interface.png" width="900">
</p>

Selecting the heterogeneous graph modality configures ATLAS to support multiple node and edge types, enabling the construction of semantically rich graph structures. This modality affects the available tools, the behavior of the interface during graph creation, and the export options, which become compatible with heterogeneous graph libraries such as DGL and PyTorch Geometric.

The interface is organized into multiple coordinated components supporting the heterogeneous graph construction workflow.

- **Node Creation** — Nodes can be placed directly onto the canvas through the *Add Node* function. Unlike homogeneous graphs, each node is associated with a predefined node type that determines both its semantic meaning and visual appearance.

- **Node Type Definition** — Node types can be defined through a dedicated configuration panel where users can specify:
  - Type name
  - Node shape
  - Node color
  - Optional custom image/icon

  When a custom image is provided, it replaces the default geometric representation, allowing domain-specific visual abstractions. This feature improves readability and facilitates the interpretation of complex heterogeneous graph structures.

- **Semantic Node Representation** — Each node maintains:
  - A unique identifier
  - A textual label or description
  - Internal metadata

  These attributes are preserved during export to ensure compatibility with graph-learning and graph-processing frameworks.

- **Edge Creation** — Edges are created interactively between nodes through the *Add Edge* function. In heterogeneous mode, edges represent semantic relationships between node types and can include textual labels describing the relation type.

- **Backend Library Selection** — ATLAS supports exporting heterogeneous graphs to multiple frameworks, including:
  - DGL
  - PyTorch Geometric

  The generated code automatically adapts to the selected framework:
  - **DGL** exports relations as triplets:
    `(source_type, relation_type, destination_type)`
  - **PyTorch Geometric** exports graphs using the `HeteroData` structure.

- **Code Preview and Import** — The *Code Preview/Import* panel continuously synchronizes the graphical representation with the generated executable Python code. Any modification applied to the graph is immediately reflected in the exported implementation.

- **Legend Panel** — The integrated *Legend* component provides a complete overview of all defined node types, including:
  - Shape
  - Color
  - Associated image/icon

  This functionality facilitates navigation and interpretation of complex heterogeneous graph structures.

- **Real-Time Metrics Computation** — The *Metrics* panel continuously computes structural graph properties and updates them in real time during graph editing and exploration, extending the analytical capabilities available for homogeneous graphs.

## Case Study — Cora Citation Network

This case study demonstrates the modelling of a real-world citation network using the homogeneous graph functionality provided by ATLAS.

The Cora dataset is one of the most widely adopted benchmarks in graph machine learning research. It represents a citation network in which:
- Nodes correspond to scientific publications
- Directed edges represent citation relationships
- Each paper belongs to one of seven research topics

Because all nodes and edges belong to a single semantic category, Cora naturally fits the homogeneous graph setting.

The construction workflow begins by selecting the **Homogeneous** modality during the initialization phase. A representative subgraph can then be assembled interactively by placing paper nodes directly onto the canvas through the graphical editor. Citation relationships are created using directed edges labeled `cites`.

<p align="center">
    <img src="docs/imgs/cora_dataset_graph.png" width="900">
</p>

The resulting graph captures the topology of the citation network while maintaining a compact and interpretable visual structure.

Once the graph is assembled, ATLAS automatically generates executable Python code corresponding to the current graph state. By selecting **PyTorch Geometric (PyG)** as the target backend library, the software produces code compatible with graph deep learning pipelines.

The generated script:
- Instantiates the graph structure
- Defines node feature placeholders
- Encodes connectivity through the `edge_index` tensor representation
- Maps nodes to integer identifiers
- Produces executable PyG-compatible code in real time

The generated code can be:
- Copied directly to the clipboard
- Exported as a standalone Python file
- Integrated into external machine learning workflows

In addition, the integrated metrics module continuously computes structural properties of the graph during the modelling process, enabling immediate analytical feedback.

This example demonstrates how ATLAS bridges interactive graph modelling and executable graph-learning representations with minimal manual effort.

ATLAS can also be employed to visualize raw dataset. To this end, you need to first generate the `python` code to do that, executing
```
uv run script_exe.py dataset_code --parameters configs/dataset_download/planetoid.yaml 
``` 
Once the code is generated (see [dataset_code/planetoid/graph_export.py](dataset_code/planetoid/graph_export.py)), you can copy-paste it into the GUI import section, obtaining the following graph:
<p align="center">
    <img src="docs/imgs/code/cora_dataset.png" width="900">
</p>

Additionally, we provide explanation visualization. To do this, we firstly train a GAT to solve, in this case, a node classification task; then we leverage GNNExplainer to compute the explanation for a target node, and lastly we extract the code to draw the explanatin graph.

To train the GNN, run:
```
uv run script_exe.py training --parameters configs/training/planetoid.yaml --cls PlanetoidRun
```

To extract explanations, run:
```
uv run script_exe.py explain --parameters configs/explain/planetoid.yaml --cls PlanetoidExplainerRun
```

And to extract the code to draw the explanation, execute:
```
uv run script_exe.py dataset_code --parameters configs/explanation_code/planetoid.yaml --explanation
```

Once the code has been generated (see [dataset_code/planetoid/explanation_export.py](dataset_code/planetoid/explanation_export.py)), it can be directly imported in the GUI to visualize the obtained explanation. 
<p align="center">
    <img src="docs/imgs/code/cora_explanation.png" width="900">
</p>


---

## Case Study — MovieLens Recommendation System

This case study demonstrates the modelling of a heterogeneous recommendation graph using ATLAS.

The MovieLens dataset is a standard benchmark for recommender system research and naturally represents a multi-relational heterogeneous graph. The dataset includes:
- Users
- Movies
- Directors
- Genres

connected through semantic relationships such as:
- `rates`
- `directed_by`
- `has_genre`

Because the graph contains multiple node and edge types, it is particularly suitable for heterogeneous graph modelling.

The workflow begins by selecting the **Heterogeneous** modality within ATLAS. Node categories are then defined through the dedicated type-management interface, where each type can be associated with:
- A custom name
- A specific color
- A geometric shape
- An optional custom icon

This visual abstraction improves readability and semantic interpretation of complex graph structures.

For this example:
- **Director** nodes use a camera icon
- **User** nodes use a person icon
- **Movie** nodes use a clapperboard icon
- **Genre** nodes use a theatre-mask icon

Individual nodes are then placed on the interactive canvas and connected through typed semantic relations.

<p align="center">
    <img src="docs/imgs/movielens_graph.png" width="900">
</p>

The resulting graph contains:
- Multiple node categories
- Typed semantic relations
- A fully heterogeneous graph structure suitable for recommendation tasks

The integrated legend panel provides a complete overview of all node types and their associated visual representations, improving navigation and interpretability.

For programmatic export, **PyTorch Geometric (PyG)** is selected as the target backend. ATLAS automatically converts the visual graph into executable heterogeneous graph code using the `HeteroData` representation.

The generated code:
- Organizes nodes by semantic type
- Creates relation-specific edge indices
- Preserves typed graph semantics
- Defines placeholder feature tensors for each node category
- Generates executable PyG-compatible heterogeneous graph code

All generated code is updated dynamically in real time as the graph changes.

The final script can be:
- Exported as a standalone Python file
- Copied into external projects
- Integrated into graph-learning and recommendation-system pipelines

Unlike homogeneous graphs, the heterogeneous representation explicitly preserves semantic relationships between node categories, enabling downstream tasks such as:
- Recommendation systems
- Link prediction
- Heterogeneous node classification
- Knowledge graph modelling

This example highlights how ATLAS simplifies the transition from conceptual heterogeneous graph design to executable graph-learning implementations.

As explained for the Cora dataset, ATLAS features the drawing of the entire dataset and the explanation subgraph for heterogeneous graph too.

To gather the code for the raw dataset visualization, please run:
```
uv run script_exe.py dataset_code --parameters configs/dataset_download/movielens.yaml --hetero
```
The obtained graph (see [dataset_code/movielens/graph_export.py](dataset_code/movielens/graph_export.py)) can be directly imported in ATLAS, obtaining:
<p align="center">
    <img src="docs/imgs/code/movielens_dataset.png" width="900">
</p>

After that, we simulated a general purpose link prediction task, training a GNN (in this case a GAT) and an explainer (GNNExplainer) to gather graph based explanations.
To do this, run:
```
uv run script_exe.py training --parameters configs/training/movielens.yaml --cls MovielensRun
uv run script_exe.py explain --parameters configs/explain/movielens.yaml --cls MovielensExplainerRun
```

Finally, generate the explanation graph code by executing:
```
uv run script_exe.py dataset_code --parameters configs/explanation_code/movielens.yaml --explanation --hetero
```

The network, visualized inside ATLAS, is as follows:
<p align="center">
    <img src="docs/imgs/code/movielens_explanation.png" width="900">
</p>
