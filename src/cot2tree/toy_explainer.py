#%% 
import networkx as nx
from torch_geometric.explain import Explainer, GNNExplainer, GraphMaskExplainer, PGExplainer, AttentionExplainer
import torch
from gatv2 import *
from fast_gspan import FastgSpan
import matplotlib.pyplot as plt
def get_explanations(explainer:Explainer, loader:DataLoader, dataset_name:str, model_name:str, parent_dir:str):
    print(len([data for data in loader]))
    weighted_graphs = []
    for data in loader:
        print(f"Data.X: {data.x}")
        print(f"Edge index: {data.edge_index}")
        explanation = explainer(data.x, data.edge_index, batch=data.batch)
        print(explanation)
        print(explanation.edge_mask)
        print(explanation.get_explanation_subgraph())
        print(explanation.get_complement_subgraph())
        #explanation.visualize_graph()
        print(f"The index index of our data: {data.edge_index}")
        print(f"The edge index of the explanation: {explanation.edge_index}")
        
        edge_weight = explanation.edge_mask
        print(f"The weights are: {edge_weight} before filtering.")
        edge_weight = edge_weight - edge_weight.min()
        edge_weight = edge_weight / edge_weight.max()
        ind = 1.5
        edge_weight_t = edge_weight ** ind
        while len([1 for t in edge_weight_t if t > 0.66]) > len(edge_weight)/5:
            ind += 0.2
            edge_weight_t = edge_weight ** ind
            if ind > 5:
                break
        edge_weight = edge_weight_t
        edge_weight = edge_weight * 0.9 + 0.1
        print(f"The weights are: {edge_weight} after filtering.")
        gr = nx.DiGraph()
        edge_index = data.edge_index
        print(f"edge_index.view(-1): {edge_index.view(-1)}")

        for node in edge_index.view(-1).unique().tolist():  # gets the list of nodes
            gr.add_node(node)
        for (src, dst), w in zip(edge_index.t().tolist(), edge_weight.tolist()):
            gr.add_edge(src, dst, alpha=w)
        weighted_graphs.append(gr)
        print(gr)
    fastgs = FastgSpan()
    df = fastgs.run_from_graphs(weighted_graphs)
    print(df)
    #print("support")
    #print(df["support"])
    patterns = []
    subgraphs = []
    for i, (v,e) in enumerate(zip(df["vertices"], df["edges"])):
        print(v)
        print(type(v))
        print(e)
        print(type(e))
        pattern = {'vertices':v, 'edges':e}
        patterns.append(pattern)
        subg = fastgs.pattern_to_graph(pattern)
        subgraphs.append(subg)
        print(subg)
        fig = plt.figure()
        nx.draw(subg, ax=fig.add_subplot(), arrows=True, arrowstyle="-|>")
        local_path = f".local/patterns/graph_{dataset_name}_{model_name}{i}.png"
        save_path = os.join(parent_dir, local_path)
        fig.savefig(save_path)
    return subgraphs


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device', device)
# create fake data and edge_index
g = nx.DiGraph()
g.add_node(1)
g.add_node(2)
g.add_node(3)
g.add_node(4)
g.add_edge(1,2)
g.add_edge(1,3)
g.add_edge(3,4)
h = nx.DiGraph()
h.add_node(1)
h.add_node(2)
h.add_node(3)
h.add_edge(1,2)
h.add_edge(2,3)
adj = nx.to_numpy_array(g)
#print(adj.shape)
#print(adj)
coo = coo_matrix(adj)
#print(coo)
#print(coo.shape)
#print(coo.col)
#print(coo.row)
coo_h = coo_matrix(nx.to_numpy_array(h))
#print(np.array([coo.row, coo.col]))
# let's use fake features for now
x = torch.tensor([[3,8,9,1],[1,5,3,7],[33,22,11,0],[0,0,0,8]],dtype=torch.float)
y = torch.tensor([[42,33,7,9],[1,6,1,2],[107,0,5,10],[3,18,9,5]],dtype=torch.float)
#print(x)
data = Data(x=x, edge_index=torch.tensor(np.array([coo.row, coo.col])), y=0)
d1 = Data(x=y, edge_index=torch.tensor(np.array([coo_h.row, coo_h.col])), y=1)
loader = DataLoader([data, d1], batch_size=1)
model = GAT(in_channels=4, out_channels=2, hidden=64).to(device)

explainer = Explainer(
        model=model, 
        algorithm=GNNExplainer(epochs=200), 
        explanation_type='model', 
        edge_mask_type='object', 
        model_config=dict(mode="multiclass_classification", task_level="graph", return_type="log_probs")
    )

get_explanations(explainer, loader)



# %%
