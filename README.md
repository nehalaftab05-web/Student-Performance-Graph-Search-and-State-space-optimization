# Student-Performance-Graph-Search-and-State-space-optimization
Artificial-Intelligence
Student Performance Graph Search & State-Space Optimization
This repository implements an AI Agent Environment built over the UCI Student Performance Dataset (Track C: Education). The project constructs a state-space graph connecting student behavioral metrics (study time) to final academic results (
G
3
 grades) and systematically evaluates Uninformed, Informed, Local, and Adversarial (Game-Playing) search algorithms.

📊 Dataset & State-Space Graph Architecture
Domain Context
The underlying dataset consists of student profiles from two Portuguese schools. It details demographic factors, social characteristics, and academic tracking markers distributed across 33 properties.

Graph Construction
A network graph is systematically constructed by mapping features to distinct node environments:

Source Nodes: Operationalized behaviors, explicitly targeting study frequencies (e.g., StudyTime_2).
Target Nodes: Final scholastic evaluations graded out of 20 points (e.g., Grade_15).
Edges: Represent an unweighted, undirected connection (
cost
=
1
) dynamically added when a data record satisfies both the specific habit profile and the grading tier.
[StudyTime_1] -------- (Student Records) -------- [Grade_8]
[StudyTime_2] <=======> [Grade_15] (Target Goal)
[StudyTime_3] -------- (Student Records) -------- [Grade_10]

Environment Characteristics
Total Identified Nodes: 22
Total Extracted Edges: 59
Target State Variable: Final Assessment Score (
G
3
)
🤖 AI Agent Architecture
The environmental agent abstracts perception and routing capabilities across the graph via the standard AIAgent interface:

perceive(state): Probes the environment topology to return accessible neighbors.
act(action): Transitions the internal state machine to the targeted node.
goal_test(state): Verifies if the agent has reached the target grade node (e.g., Grade_15).
get_cost(state1, state2): Enforces an isotropic uniform movement step cost of 
1
.
🔍 Evaluated Search Frameworks
1. Uninformed Search (Blind Routings)
Implements baseline traversal logic through structured memory spaces without structural optimization heuristics:

Breadth-First Search (BFS): FIFO implementation exploring frontier nodes uniformly. Finds the optimal path with shallowest-node evaluation constraints.
Depth-First Search (DFS): LIFO stack deployment maximizing branch traversal before backtracking. Minimizes memory footprints under favorable path selections.
Depth-Limited Search (DLS): DFS variation bounded by a hard depth maximum to mitigate structural trapping inside infinite loop spaces.
Iterative Deepening Search (IDS): Iteratively expands DLS boundary thresholds. Combines the completeness of BFS with the space efficiency of DFS.
Uniform Cost Search (UCS): Dijkstra-variant priority queue routing that expands along paths minimizing cumulative operational costs 
g
(
n
)
.
2. Informed Search (Heuristic Routings)
Utilizes an admissible evaluation function 
h
(
n
)
 to estimate remaining path costs to the target:

Admissible 
h
(
n
)
=
{
0
if 
n
=
Goal
 
1
if 
n
∈
StudyTime
 
2
if 
n
∈
Alternative Grade

Greedy Best-First Search: Evaluates paths strictly using the structural heuristic estimation 
h
(
n
)
, minimizing immediate expansion trajectories.
A Search:* Minimizes total estimated path function costs 
f
(
n
)
=
g
(
n
)
+
h
(
n
)
, guaranteeing an optimal path trajectory.
3. Local Search & Optimization (Heuristics Optimization)
Formulates traversal around maximizing state-objective performance scores rather than preserving complete historical paths:

Hill Climbing: Greedily moves towards neighboring states that maximize structural performance value. It can get trapped at local optima depending on initial graph configurations.
Simulated Annealing: Employs a physical thermodynamic cooling function (
T
0
=
100
,
α
=
0.95
). It systematically permits down-slope exploratory transitions to avoid tracking traps:
P
(
accept
)
=
e
Δ
E
T

Local Beam Search: Evaluates 
k
 parallel paths concurrently (
k
∈
3
,
5
), consolidating data tracks around top-performing nodes.
4. Adversarial Game Search
Models academic prediction tracking as an alternating two-player game tree to evaluate state values under adversarial conditions:

Minimax Search: Recursively computes optimal policy values by modeling alternating maximizing adjustments (representing optimal study trajectories) against minimizing adjustments (representing resource constraints or behavioral regressions).
📈 Performance Benchmarks
Uninformed vs. Informed Efficiency
The experimental run logs demonstrate a clear decrease in node exploration overhead when utilizing informed search methods:

Algorithm Standard	Path Length	Explored Node Cardinality	Optimization Target
BFS	1	18	Shortest Path
DFS	1	2	Deep Branch
UCS	1	8	Cheapest Evaluation
Greedy Best-First	1	2	Heuristic Guided
A*	1	2	Optimal 
g
(
n
)
+
h
(
n
)
Local Search Optimization Results
Hill Climbing: 100% success rate across 10 random initialization passes due to the interconnected nature of the study-grade graph.
Simulated Annealing: Reached the global maximum on all evaluation iterations.
Local Beam Search: Successfully found the global target state across both test parameters (
k
=
3
 and 
k
=
5
).
🛠️ Project Structure & Setup
Environment Dependencies
The analytical execution workspace relies on standard scientific Python libraries:

pip install numpy pandas matplotlib scikit-learn
Core Pipeline Execution
Ingestion: Reads student profiles from track_c_education/student_data.csv.
Profiling: Validates dataset shapes, column properties, and handles any missing values using pd.DataFrame utilities.
Graph Generation: Extracts feature cross-connections and maps them into an adjacency list dictionary structure.
Simulation Execution: Iterates through the search algorithms to benchmark graph traversals, route metrics, and exploration costs.
