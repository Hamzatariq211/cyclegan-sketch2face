import torch
from generator import Generator  # make sure this file contains your Generator class

checkpoint = torch.load("checkpoint_latest.pth", map_location="cpu")

G_AB = Generator()
G_AB.load_state_dict(checkpoint['G_AB'])
torch.save(G_AB.state_dict(), "G_AB_trained.pth")

print("✅ G_AB generator weights saved as G_AB_trained.pth")
