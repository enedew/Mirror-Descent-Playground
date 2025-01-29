from MirrorDescent import MirrorDescent
import torch 
import numpy as np 
import matplotlib.pyplot as plt 

# constant seed so results are reproducible 
torch.manual_seed(0)

# creating input data
X = np.linspace(-5, 5, 500).astype(np.float32)
Y = X**2 + 4*X
# convert to torchtensors 
X = torch.tensor(X).unsqueeze(1)
Y = torch.tensor(Y).unsqueeze(1)

# defining a simple linear model 
# 2 layer MLP 

model = torch.nn.Sequential(
    torch.nn.Linear(1,10),
    torch.nn.ReLU(),
    torch.nn.Linear(10,1)
)

# initiate md optimiser
optimiser = MirrorDescent(model.parameters(), lr=0.01, bregman='EUCLID')
criterion = torch.nn.MSELoss()

# defining the training loop 
losses = []
for epoch in range(5000): 
    pred = model(X) 
    loss = criterion(pred, Y)

    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
    losses.append(loss.item())
    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}") 

with torch.no_grad():

    final_preds = model(X) 

plt.plot(losses, label="loss over epochs")
plt.xlabel("Epochs")
plt.ylabel("MSE loss")
plt.title("Training loss") 
plt.legend()
plt.show()

plt.plot(X.numpy(), Y.numpy(), label = "True values", color="blue")
plt.plot(X.numpy(), final_preds.numpy(), label="Predictions", color="red")
plt.xlabel('x')
plt.ylabel('y')
plt.title('True vs Predicted Values')
plt.legend()
plt.show()