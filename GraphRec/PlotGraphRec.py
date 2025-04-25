import torch
import torch.nn as nn
import pickle
import numpy as np
import pandas as pd

from UV_Encoders import UV_Encoder
from UV_Aggregators import UV_Aggregator
from Social_Encoders import Social_Encoder
from Social_Aggregators import Social_Aggregator
import torch.nn.functional as F
import torch.utils.data

import argparse
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "8"

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import seaborn as sns
from collections import defaultdict, Counter


class GraphRec(nn.Module):

    def __init__(self, enc_u, enc_v_history, r2e):
        super(GraphRec, self).__init__()
        self.enc_u = enc_u
        self.enc_v_history = enc_v_history
        self.embed_dim = enc_u.embed_dim

        self.w_ur1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_ur2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_vr1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_vr2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_uv1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.w_uv2 = nn.Linear(self.embed_dim, 16)
        self.w_uv3 = nn.Linear(16, 1)
        self.r2e = r2e
        self.bn1 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn2 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn3 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn4 = nn.BatchNorm1d(16, momentum=0.5)
        self.criterion = nn.MSELoss()

    def forward(self, nodes_u, nodes_v):
        embeds_u = self.enc_u(nodes_u)
        embeds_v = self.enc_v_history(nodes_v)

        x_u = F.relu(self.bn1(self.w_ur1(embeds_u)))
        x_u = F.dropout(x_u, training=self.training)
        x_u = self.w_ur2(x_u)
        x_v = F.relu(self.bn2(self.w_vr1(embeds_v)))
        x_v = F.dropout(x_v, training=self.training)
        x_v = self.w_vr2(x_v)

        x_uv = torch.cat((x_u, x_v), 1)
        x = F.relu(self.bn3(self.w_uv1(x_uv)))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.bn4(self.w_uv2(x)))
        x = F.dropout(x, training=self.training)
        scores = self.w_uv3(x)
        return scores.squeeze()

    def loss(self, nodes_u, nodes_v, labels_list):
        scores = self.forward(nodes_u, nodes_v)
        return self.criterion(scores, labels_list)


def plot_embeddings(embeddings, labels=None, title="Embeddings Visualization", sample_size=1000):
    indices = torch.randperm(embeddings.shape[0])[:sample_size]
    sampled_embeds = embeddings[indices].detach().cpu().numpy()
    
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000)
    embeds_2d = tsne.fit_transform(sampled_embeds)

    plt.figure(figsize=(8, 8))
    plt.scatter(embeds_2d[:, 0], embeds_2d[:, 1], c='blue', alpha=0.6, label="Embedding points")
    if labels is not None:
        for i, label in enumerate(labels[indices].cpu().numpy()):
            plt.annotate(str(label), (embeds_2d[i, 0], embeds_2d[i, 1]))
    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend()
    plt.savefig(f"../Plots/{title}.png")
    plt.show()


def plot_embeddings_3d(embeddings, title="Embeddings (3D)", sample_size=1000):
    indices = torch.randperm(embeddings.shape[0])[:sample_size]
    sampled_embeds = embeddings[indices].detach().cpu().numpy()

    tsne = TSNE(n_components=3, perplexity=30, max_iter=1000, init='pca', random_state=42)
    embeds_3d = tsne.fit_transform(sampled_embeds)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(embeds_3d[:, 0], embeds_3d[:, 1], embeds_3d[:, 2], c='blue', alpha=0.6, s=20)
    ax.set_title(title)
    ax.set_xlabel("t-SNE Dim 1")
    ax.set_ylabel("t-SNE Dim 2")
    ax.set_zlabel("t-SNE Dim 3")
    plt.savefig(f"../Plots/{title}.png")
    plt.show()


def plot_embeddings_3d_colored(embeddings, labels, title="Embeddings (3D Colored)", sample_size=1000):

    assert len(labels) == embeddings.shape[0], "Le nombre de labels doit correspondre au nombre d'embeddings."

    # Random sampling
    indices = torch.randperm(embeddings.shape[0])[:sample_size]
    sampled_embeds = embeddings[indices].detach().cpu().numpy()
    sampled_labels = np.array(labels)[indices]

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(sampled_labels)

    # t-SNE 3D
    tsne = TSNE(n_components=3, perplexity=30, max_iter=1000, init='pca', random_state=42)
    embeds_3d = tsne.fit_transform(sampled_embeds)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(embeds_3d[:, 0], embeds_3d[:, 1], embeds_3d[:, 2],
                         c=encoded_labels, cmap='tab20', s=30, alpha=0.75)

    unique_categories = label_encoder.classes_
    cbar = plt.colorbar(scatter, pad=0.1)
    cbar.set_ticks(np.arange(len(unique_categories)))
    cbar.set_ticklabels(unique_categories)
    cbar.ax.tick_params(labelsize=8)

    ax.set_title(title)
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.set_zlabel("t-SNE Dimension 3")
    plt.tight_layout()
    plt.savefig(f"../Plots/{title}.png")
    plt.show()


def plot_embeddings_3d_kmeans(embeddings, title="Embeddings (3D with KMeans)", sample_size=1000, n_clusters=10):

    # Random sampling
    indices = torch.randperm(embeddings.shape[0])[:sample_size]
    sampled_embeds = embeddings[indices].detach().cpu().numpy()

    tsne = TSNE(n_components=3, perplexity=30, init='pca', random_state=42)
    embeds_3d = tsne.fit_transform(sampled_embeds)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeds_3d)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(embeds_3d[:, 0], embeds_3d[:, 1], embeds_3d[:, 2],
                         c=cluster_labels, cmap='tab10', s=30, alpha=0.75)

    cbar = plt.colorbar(scatter, pad=0.1)
    cbar.set_label("KMeans Cluster", fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    ax.set_title(title)
    ax.set_xlabel("t-SNE Dim 1")
    ax.set_ylabel("t-SNE Dim 2")
    ax.set_zlabel("t-SNE Dim 3")
    plt.tight_layout()
    plt.savefig(f"../Plots/{title.replace(' ', '_').lower()}.png")
    plt.show()


def plot_similarity_heatmap(embeddings, title="Similarity Heatmap", sample_size=400, n_clusters=10):
    indices = np.random.choice(embeddings.shape[0], sample_size, replace=False)
    sampled_embeds = embeddings[indices].detach().cpu().numpy()

    norm_embeds = sampled_embeds / np.linalg.norm(sampled_embeds, axis=1, keepdims=True)
    similarity_matrix = np.dot(norm_embeds, norm_embeds.T)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(norm_embeds)

    sorted_idx = np.argsort(labels)
    sorted_matrix = similarity_matrix[sorted_idx][:, sorted_idx]

    plt.figure(figsize=(10, 8))
    sns.heatmap(sorted_matrix, cmap="viridis")
    plt.title(f"{title} (sorted by KMeans clusters)")
    plt.xlabel("Sampled Items (cluster-sorted)")
    plt.ylabel("Sampled Items (cluster-sorted)")
    plt.savefig(f"../Plots/{title.replace(' ', '_').lower()}_kmeans.png")
    plt.show()


def plot_cluster_category_distribution(item_embeddings, idx2prod, ratings_df,
                                       n_clusters=8, top_k=5, figsize=(12,6)):

    if 'ProductID' in ratings_df.columns:
        ratings_unique = ratings_df.drop_duplicates(subset='ProductID')
        category_map = ratings_unique.set_index('ProductID')['Category'].to_dict()
    else:
        ratings_unique = ratings_df.reset_index().drop_duplicates(subset='ProductID')
        category_map = ratings_unique.set_index('ProductID')['Category'].to_dict()
        
    # Clustering KMeans
    embeds_np = item_embeddings.detach().cpu().numpy()
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeds_np)
    
    clusters = defaultdict(list)
    for idx, cid in enumerate(labels):
        prod = idx2prod.get(idx, None)
        if prod is not None:
            clusters[cid].append(prod)
    
    dist_data = {}
    for cid in range(n_clusters):
        prods = clusters[cid]
        cats = [category_map.get(p, 'Unknown') for p in prods]
        cnt = Counter(cats)
        
        top = cnt.most_common(top_k)
        total = sum(cnt.values())
        
        pct = {cat: count/total*100 for cat, count in top}
        others_pct = 100 - sum(pct.values())
        if others_pct > 0:
            pct['Other'] = others_pct
        dist_data[f'Cluster {cid}'] = pct
    
    df_pct = pd.DataFrame(dist_data).T.fillna(0)
    
    ax = df_pct.plot(kind='bar', 
                     stacked=True, 
                     figsize=figsize, 
                     colormap='tab20')
    ax.set_ylabel("Percentage of Top Categories (%)")
    ax.set_xlabel("Clusters")
    ax.set_title("Cluster-wise Top-Category Distribution")
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.savefig("../Plots/cluster_category_distribution.png")
    plt.show()


def plot_prediction_error_histogram(model, device, test_loader, bins=50):
    model.eval()
    errors = []

    with torch.no_grad():
        for test_u, test_v, tmp_target in test_loader:
            test_u, test_v, tmp_target = test_u.to(device), test_v.to(device), tmp_target.to(device)
            predictions = model.forward(test_u, test_v)
            batch_errors = predictions - tmp_target
            errors.extend(batch_errors.cpu().numpy())

    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=bins, edgecolor='black', color='skyblue')
    plt.title("Distribution of Prediction Errors")
    plt.xlabel("Prediction Error (Predicted - Actual)")
    plt.ylabel("Number of Samples")
    plt.tight_layout()
    plt.savefig("../Plots/prediction_error_histogram.png")
    plt.show()



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Social Recommendation: GraphRec model')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size for training')
    parser.add_argument('--embed_dim', type=int, default=64, metavar='N', help='embedding size')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
    parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N', help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    embed_dim = args.embed_dim

    with open('../Data/Ciao/dataset/processed_data.pkl', "rb") as f:
        ciao_graph_data = pickle.load(f)

    history_u_lists = ciao_graph_data["history_u_lists"]
    history_ur_lists = ciao_graph_data["history_ur_lists"]
    history_v_lists = ciao_graph_data["history_v_lists"]
    history_vr_lists = ciao_graph_data["history_vr_lists"]
    social_adj_lists= ciao_graph_data["social_adj_lists"]
    ratings_list    = ciao_graph_data["ratings_list"]

    num_users = max(history_u_lists) + 1
    num_items = max(history_v_lists) + 1
    num_ratings = ratings_list.__len__()

    u2e = nn.Embedding(num_users, embed_dim).to(device)
    v2e = nn.Embedding(num_items, embed_dim).to(device)
    r2e = nn.Embedding(num_ratings, embed_dim).to(device)

    # user feature
    # features: item * rating
    agg_u_history = UV_Aggregator(v2e, r2e, u2e, embed_dim, cuda=device, uv=True)
    enc_u_history = UV_Encoder(u2e, embed_dim, history_u_lists, history_ur_lists, agg_u_history, cuda=device, uv=True)
    # neighobrs
    agg_u_social = Social_Aggregator(lambda nodes: enc_u_history(nodes).t(), u2e, embed_dim, cuda=device)
    enc_u = Social_Encoder(lambda nodes: enc_u_history(nodes).t(), embed_dim, social_adj_lists, agg_u_social,
                           base_model=enc_u_history, cuda=device)

    # item feature: user * rating
    agg_v_history = UV_Aggregator(v2e, r2e, u2e, embed_dim, cuda=device, uv=False)
    enc_v_history = UV_Encoder(v2e, embed_dim, history_v_lists, history_vr_lists, agg_v_history, cuda=device, uv=False)


    # Load model
    graphrec = GraphRec(enc_u, enc_v_history, r2e).to(device)
    graphrec.load_state_dict(torch.load("../Data/GraphRecComputation/graphrec_model.pth", map_location=device))
    graphrec.eval()


    # ----------------------------------- Plots ---------------------------------- #

    user_embeddings = graphrec.enc_u.base_model.features.weight 
    #plot_embeddings(user_embeddings, title="User Embeddings (t-SNE)")
    plot_embeddings_3d(user_embeddings, title="User Embeddings (t-SNE 3D)")

    item_embeddings = graphrec.enc_v_history.features.weight

    # Import
    with open("../Data/GraphRecComputation/categories_by_index.pkl", "rb") as f:
        categories_by_index = pickle.load(f)

    #plot_embeddings(item_embeddings, title="Item Embeddings (t-SNE)")
    #plot_embeddings_3d(item_embeddings, title="Item Embeddings (t-SNE 3D)")
    plot_embeddings_3d_colored(item_embeddings, labels=categories_by_index, title="Item Embeddings by Category (t-SNE 3D)", sample_size=500)
    plot_embeddings_3d_kmeans(item_embeddings, title="Item Embeddings by Category (t-SNE 3D)", sample_size=500)

    plot_similarity_heatmap(user_embeddings, title="User Embeddings Similarity", sample_size=400, n_clusters=10)
    plot_similarity_heatmap(item_embeddings, title="Item Embeddings Similarity", sample_size=400, n_clusters=10)


    # Import
    with open("../Data/GraphRecComputation/prod2idx.pkl", "rb") as f:
        prod2idx = pickle.load(f)

    ratings_df = pd.read_csv("../Data/GraphRecComputation/ratings_df.csv", sep=";", low_memory=False)
    idx2prod = {v: k for k, v in prod2idx.items()}

    plot_cluster_category_distribution(item_embeddings, idx2prod, ratings_df, n_clusters=10)

    # Construct train and test histories using direct arrays
    test_u          = ciao_graph_data["test_u"]
    test_v          = ciao_graph_data["test_v"]
    test_r          = ciao_graph_data["test_r"]
    testset = torch.utils.data.TensorDataset(torch.LongTensor(test_u), torch.LongTensor(test_v), torch.FloatTensor(test_r))
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True)

    plot_prediction_error_histogram(graphrec, device, test_loader, bins=25)


if __name__ == "__main__":
    main()

