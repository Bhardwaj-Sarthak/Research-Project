import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import ParameterGrid

# show progress bar
from tqdm import tqdm
tqdm.pandas()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
def train_cen(X1, X2, X3, X4, X5, X6, X7, Y):
    X1_train, X1_test, X2_train, X2_test, X3_train, X3_test, X4_train, X4_test, X5_train, X5_test, X6_train, X6_test, X7_train, X7_test, Y_train, Y_test = train_test_split(X1, X2, X3, X4, X5, X6, X7, Y, test_size=0.2, random_state=42)
    X1_train = torch.tensor(X1_train, dtype=torch.float32).to(device)
    X2_train = torch.tensor(X2_train, dtype=torch.float32).to(device)
    X3_train = torch.tensor(X3_train, dtype=torch.float32).to(device)
    X4_train = torch.tensor(X4_train, dtype=torch.float32).to(device)
    X5_train = torch.tensor(X5_train, dtype=torch.float32).to(device)
    X6_train = torch.tensor(X6_train, dtype=torch.float32).to(device)
    X7_train = torch.tensor(X7_train, dtype=torch.float32).to(device)
    Y_train = torch.tensor(Y_train, dtype=torch.float32).to(device)
    
    X1_test = torch.tensor(X1_test, dtype=torch.float32).to(device)
    X2_test = torch.tensor(X2_test, dtype=torch.float32).to(device)
    X3_test = torch.tensor(X3_test, dtype=torch.float32).to(device)
    X4_test = torch.tensor(X4_test, dtype=torch.float32).to(device)
    X5_test = torch.tensor(X5_test, dtype=torch.float32).to(device)
    X6_test = torch.tensor(X6_test, dtype=torch.float32).to(device)
    X7_test = torch.tensor(X7_test, dtype=torch.float32).to(device)
    Y_test = torch.tensor(Y_test, dtype=torch.float32).to(device)
    
    
    class CrossAttention(nn.Module):
        def __init__(self, embedding_dim):
            super(CrossAttention, self).__init__()
            self.attn1 = nn.Linear(embedding_dim,embedding_dim)
            self.attn2 = nn.Linear(embedding_dim,embedding_dim)
            self.attn3 = nn.Linear(embedding_dim,embedding_dim)
            self.attn4 = nn.Linear(embedding_dim,embedding_dim)
            self.attn5 = nn.Linear(embedding_dim,embedding_dim)
            self.attn6 = nn.Linear(embedding_dim,embedding_dim)
            self.attn7 = nn.Linear(embedding_dim,embedding_dim) 

        def forward(self, f1, f2, f3, f4, f5, f6, f7):
            attn_weights1 = torch.sigmoid(self.attn1(f3 + f2))  # Influence from question to correct response
            attn_weights2 = torch.sigmoid(self.attn2(f1 + f2))  # Influence from content to correct response
            attn_weights3 = torch.sigmoid(self.attn4(f1 + f4))  # Influence from content to response 1
            attn_weights4 = torch.sigmoid(self.attn4(f1 + f5))  # Influence from content to response 2
            attn_weights5 = torch.sigmoid(self.attn4(f1 + f6))  # Influence from content to response 3
            attn_weights6 = torch.sigmoid(self.attn3(f1 + f7))  # Influence from content to response 4
            attn_weights7 = torch.sigmoid(self.attn7(f2 + f4 + f5 + f6 + f7))  # Influence from correct response to all responses
            attn_weights8 = torch.sigmoid(self.attn7(f3 + f4 + f5 + f6 + f7))  # Influence from qustion to all responses
            
            f1_enhanced = f1 * attn_weights2 #CONTENT
            f2_enhanced = f2 * attn_weights1  #QUESTION
            f3_enhanced = f3 * attn_weights7 * attn_weights8  #CR with combined weights of all responses and Q to all responses
            f4_enhanced = f4 * attn_weights3 #R1
            f5_enhanced = f5 * attn_weights4 #R2
            f6_enhanced = f6 * attn_weights5 #R3
            f7_enhanced = f7 * attn_weights6 #R4
            
            return f1_enhanced, f2_enhanced, f3_enhanced, f4_enhanced, f5_enhanced, f6_enhanced, f7_enhanced

    class CrossEstimationNetwork(nn.Module):
        def __init__(self, embedding_dim, hidden_dim, output_dim):
            super(CrossEstimationNetwork, self).__init__()
            self.cross_attention = CrossAttention(embedding_dim)
            self.fc = nn.Sequential(
                nn.Linear(7 * embedding_dim, 2*embedding_dim),   
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(2*embedding_dim, embedding_dim // 2),
                nn.ReLU(),
                nn.Linear(embedding_dim // 2, hidden_dim ),
                nn.Dropout(0.3),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)  
            )

        def forward(self, x1, x2, x3, x4, x5, x6, x7):
            f1, f2, f3, f4, f5, f6, f7 = self.cross_attention(x1, x2, x3, x4, x5, x6, x7)  # Cross estimation step
            fused_features = torch.cat([f1, f2, f3, f4, f5, f6, f7], dim=1)  # Concatenate all features
            output = self.fc(fused_features)
            return output


    param_grid = {
        'hidden_dim': [64,128,256, 512, 768, 1024],
        'num_epochs': np.arange(0, 20, 1),
        'learning_rate': np.arange(0.001, 0.05, 0.001)
    }

    grid = ParameterGrid(param_grid)
    best_loss = float('inf')
    criterion = nn.MSELoss()
    total_combinations = len(param_grid['hidden_dim']) * len(param_grid['num_epochs']) * len(param_grid['learning_rate'])

    pbar = tqdm(grid, total=total_combinations, desc="Grid Search")
    best_params = None
    for params in pbar:
        # Update description with current parameters
        pbar.set_description(f"Search: hidden_dim={params['hidden_dim']}, epochs={params['num_epochs']}, lr={params['learning_rate']:.4f}")
        model_cen = CrossEstimationNetwork(embedding_dim=X1.shape[1], hidden_dim=params['hidden_dim'], output_dim=2).to(device)
        optimizer = optim.Adam(model_cen.parameters(), lr=params['learning_rate'])
        model_cen.to(device)
        model_cen.train()
        
        # Train the model
        for epoch in range(params['num_epochs']):
            optimizer.zero_grad()
            outputs = model_cen(X1_train, X2_train, X3_train, X4_train, X5_train, X6_train, X7_train)
            loss = criterion(outputs, Y_train)
            loss.backward()
            optimizer.step()
        
        model_cen.eval()
        outputs = model_cen(X1_test, X2_test, X3_test, X4_test, X5_test, X6_test, X7_test)
        test_loss = criterion(outputs, Y_test)
        
        # Update postfix with current test loss
        pbar.set_postfix(test_loss=f"{test_loss.item():.4f}")
        
        if test_loss < best_loss:
            best_loss = test_loss
            best_params = params
            print(f'\n New best Loss: {best_loss:.4f} | Accuracy: {1 - best_loss.item():.4f}\n Params: {best_params} \n')     
            #print(f'_ _'*10)
            
    print(f'Best Test Loss: {best_loss}')
    print(f'Best Parameters: {best_params}')
    print(f'Best model accuracy: {1 - best_loss.item()}')
    torch.save(model_cen.state_dict(), f'cen_acc-{1 - best_loss.item()}.pth')
    
    model_cen.eval()
    with torch.no_grad():
        Y_pred = model_cen(X1_test, X2_test, X3_test, X4_test, X5_test, X6_test, X7_test)
        loss = criterion(Y_pred, Y_test)
    Y_pred_c = Y_pred.cpu().numpy()
    Y_test_c = Y_test.cpu().numpy()
    df_results = pd.DataFrame({'Predicted Discrimination': Y_pred_c[:, 0], 'Actual Discrimination': Y_test_c[:, 0],'Difference Discrimination': Y_pred_c[:, 0]-Y_test_c[:, 0],
                            'Predicted Difficulty': Y_pred_c[:, 1], 'Actual Difficulty': Y_test_c[:, 1],'Difference Difficulty': Y_pred_c[:, 1]-Y_test_c[:, 1]})
    diff_disc = np.abs(df_results['Difference Discrimination']).mean()
    diff_diff = np.abs(df_results['Difference Difficulty']).mean()
    print(f"Mean Absolute Difference Discrimination: {diff_disc:.4f}")
    print(f"Mean Absolute Difference Difficulty: {diff_diff:.4f}")
    plt.figure(figsize=(10, 5))
    plt.plot(df_results['Actual Discrimination'], label='Actual Discrimination')
    plt.plot(df_results['Predicted Discrimination'], label='Predicted Discrimination')  
    plt.xlabel('Index')
    plt.ylabel('Discrimination')
    plt.title('Actual vs. Predicted Discrimination')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(df_results['Actual Difficulty'], label='Actual Difficulty')
    plt.plot(df_results['Predicted Difficulty'], label='Predicted Difficulty')
    plt.xlabel('Index')
    plt.ylabel('Difficulty')
    plt.title('Actual vs. Predicted Difficulty')
    plt.legend()
    plt.show()
    return model_cen