import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
from torch import amp
import pickle
import numpy as np
import time
import random
from collections import defaultdict
from UV_Encoders import UV_Encoder
from UV_Aggregators import UV_Aggregator
from Social_Encoders import Social_Encoder
from Social_Aggregators import Social_Aggregator
import torch.nn.functional as F
import torch.utils.data
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
import datetime
import argparse
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

os.environ["CUDA_LAUNCH_BLOCKING"] = "1" # Debugging error
torch.backends.cudnn.benchmark = True # Dynamic optimizations

"""
GraphRec: Graph Neural Networks for Social Recommendation. 
Wenqi Fan, Yao Ma, Qing Li, Yuan He, Eric Zhao, Jiliang Tang, and Dawei Yin. 
In Proceedings of the 28th International Conference on World Wide Web (WWW), 2019. Preprint[https://arxiv.org/abs/1902.07243]
"""


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


def train(model, device, train_loader, optimizer, best_rmse, best_mae, scaler):
    model.train()
    losses = []
    running_loss = 0.0

    progress_bar = tqdm(enumerate(train_loader, 0), desc="Training", total=len(train_loader), ncols=100)

    for i, data in progress_bar:
        batch_nodes_u, batch_nodes_v, labels_list = data

        optimizer.zero_grad()

        with amp.autocast(device_type=device.type): # (fp16)
            loss = model.loss(batch_nodes_u.to(device), batch_nodes_v.to(device), labels_list.to(device))
        
        scaler.scale(loss).backward(retain_graph=True)
        scaler.step(optimizer)
        scaler.update()
        losses.append(loss.item())
        running_loss += loss.item()
        if i % 100 == 0:
            avg_loss = running_loss / 100
            progress_bar.set_postfix(loss=avg_loss, rmse=best_rmse, mae=best_mae)
            running_loss = 0.0

    return np.mean(losses)


def test(model, device, test_loader):
    model.eval()
    tmp_pred = []
    target = []
    losses = []
    expected_rmse = 99999.0
    with torch.no_grad():
        for test_u, test_v, tmp_target in test_loader:
            test_u, test_v, tmp_target = test_u.to(device), test_v.to(device), tmp_target.to(device)
            val_output = model.forward(test_u, test_v)
            loss = model.criterion(val_output, tmp_target)
            losses.append(loss.item())
            tmp_pred.append(list(val_output.data.cpu().numpy()))
            target.append(list(tmp_target.data.cpu().numpy()))
    tmp_pred = np.array(sum(tmp_pred, []))
    target = np.array(sum(target, []))
    try:
        expected_rmse = sqrt(mean_squared_error(tmp_pred, target))
    except ValueError:
        pass
    mae = mean_absolute_error(tmp_pred, target)
    return expected_rmse, mae, np.mean(losses)


def plot_loss(train_loss, test_loss):
    """
    Function to plot the evolution of train and test loss over epochs.
    """
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_loss, 'b-', marker='o', label='Train Loss')
    plt.plot(epochs, test_loss, 'r-', marker='o', label='Test Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Evolution of Loss (Train vs Test)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("../Plots/losses_evolution.png")
    plt.show()


def plot_rmse_mae(rmse_list, mae_list):
    """
    Function to plot the evolution of RMSE and MAE over epochs,
    with one Y-axis for RMSE and another for MAE.
    """
    epochs = range(1, len(rmse_list) + 1)

    fig, ax1 = plt.subplots(figsize=(8, 6))

    # Plot RMSE
    color = 'tab:blue'
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("RMSE", color=color)
    ax1.plot(epochs, rmse_list, color=color, marker='o', label='RMSE')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True)

    # Create a second axis for MAE
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel("MAE", color=color)
    ax2.plot(epochs, mae_list, color=color, marker='s', label='MAE')
    ax2.tick_params(axis='y', labelcolor=color)

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.title("Evolution of RMSE and MAE per Epoch")
    fig.tight_layout()  # Adjust layout
    plt.savefig("../Plots/metrics_evolution.png")
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
    print(f"Device choosen for computation : {device}")

    embed_dim = args.embed_dim

    with open('../Data/Ciao/dataset/processed_data.pkl', "rb") as f:
        ciao_graph_data = pickle.load(f)

    history_u_lists = ciao_graph_data["history_u_lists"]
    history_ur_lists = ciao_graph_data["history_ur_lists"]
    history_v_lists = ciao_graph_data["history_v_lists"]
    history_vr_lists = ciao_graph_data["history_vr_lists"]
    train_u         = ciao_graph_data["train_u"]
    train_v         = ciao_graph_data["train_v"]
    train_r         = ciao_graph_data["train_r"]
    test_u          = ciao_graph_data["test_u"]
    test_v          = ciao_graph_data["test_v"]
    test_r          = ciao_graph_data["test_r"]
    social_adj_lists= ciao_graph_data["social_adj_lists"]
    ratings_list    = ciao_graph_data["ratings_list"]
    """
    history_u_lists, history_ur_lists:  user's purchased history (item set in training set), and his/her rating score (dict)
    history_v_lists, history_vr_lists:  user set (in training set) who have interacted with the item, and rating score (dict)
    
    train_u, train_v, train_r: training_set (user, item, rating)
    test_u, test_v, test_r: testing set (user, item, rating)
    
    social_adj_lists: user's connected neighborhoods
    ratings_list: rating value from 0.5 to 4.0 (8 opinion embeddings)
    """

    trainset = torch.utils.data.TensorDataset(torch.LongTensor(train_u), torch.LongTensor(train_v),
                                              torch.FloatTensor(train_r))
    testset = torch.utils.data.TensorDataset(torch.LongTensor(test_u), torch.LongTensor(test_v),
                                             torch.FloatTensor(test_r))
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True)
    num_users = max(history_u_lists) + 1
    num_items = max(history_v_lists) + 1
    num_ratings = ratings_list.__len__()

    print(f"Number of points : {len(train_u)}\n")

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

    # model
    graphrec = GraphRec(enc_u, enc_v_history, r2e).to(device)
    optimizer = torch.optim.RMSprop(graphrec.parameters(), lr=args.lr, alpha=0.9)
    scaler = amp.GradScaler(device.type)

    best_rmse = 9999.0
    best_mae = 9999.0
    endure_count = 0

    # Store for plotting
    train_loss = []
    test_loss = []
    rmse_list = []
    mae_list = []

    for epoch in range(1, args.epochs + 1):

        mean_loss_train = train(graphrec, device, train_loader, optimizer, best_rmse, best_mae, scaler)
        train_loss.append(mean_loss_train)

        expected_rmse, mae, mean_loss_test = test(graphrec, device, test_loader)

        rmse_list.append(expected_rmse)
        mae_list.append(mae)
        test_loss.append(mean_loss_test)

        # early stopping 
        if best_rmse > expected_rmse:
            best_rmse = expected_rmse
            best_mae = mae
            endure_count = 0
        else:
            endure_count += 1
        print("rmse: %.4f, mae:%.4f " % (expected_rmse, mae))

        if endure_count > 5:
            break
    
    # Plot les graphiques après l'entraînement
    plot_loss(train_loss, test_loss)
    plot_rmse_mae(rmse_list, mae_list)

    # Save model
    torch.save(graphrec.state_dict(), "../Data/GraphRecComputation/graphrec_model.pth")
    print("Model saved successfully !")

if __name__ == "__main__":
    main()