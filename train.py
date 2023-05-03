import time
import torch
import torch
import pickle
from tqdm.auto import tqdm
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss

from src.constants import *
from src.models.unet import UNet
from src.data.load_data import load_data


def main() :
    # Load data
    dsets, _, _ = load_data()
    
    # Set up dataset loaders
    dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=BATCH_SIZE,
                                               shuffle=True, num_workers=0)
                    for x in ['train', 'test']}
    train_loader = dset_loaders['train']
    test_loader = dset_loaders['test']

    # Initialize u-net
    unet = UNet(3, 1)
    lossFunc = BCEWithLogitsLoss()
    opt = Adam(unet.parameters(), lr=BASE_LR)
    trainSteps = len(dsets['train']) // BATCH_SIZE
    testSteps = len(dsets['test']) // BATCH_SIZE
    H = {"train_loss": [], "test_loss": []}

    # Train 
    startTime = time.time()
    for e in range(NUM_EPOCHS):
        unet.train()
        totalTrainLoss = 0
        totalTestLoss = 0
        loop = tqdm(train_loader)
        
        for (i, (x, y)) in enumerate(loop):
            pred = unet(x)
            loss = lossFunc(pred, y.float())
            opt.zero_grad()
            loss.backward()
            opt.step()
            totalTrainLoss += loss
            loop.set_description(f"Epoch [{e + 1}/{NUM_EPOCHS}]")
            loop.set_postfix(loss=loss)
        
        with torch.no_grad():
            unet.eval()
            for (x, y) in test_loader:
                pred = unet(x)
                totalTestLoss += lossFunc(pred, y.float())
            
        avgTrainLoss = totalTrainLoss / trainSteps
        avgTestLoss = totalTestLoss / testSteps
        H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        H["test_loss"].append(avgTestLoss.cpu().detach().numpy())

        print("[INFO] EPOCH: {}/{}".format(e + 1, NUM_EPOCHS))
        print("Train loss: {:.6f}, Test loss: {:.4f}".format(
            avgTrainLoss, avgTestLoss))

        torch.save(unet.state_dict(), f"../models/unet_base_new_model_epoch_{e}.pt") # Change to appropriate name

    with open('/content/drive/MyDrive/School/Harvard/NEUROBIO240/Project/Losses/losses_base.pkl', 'wb') as fp:
        pickle.dump(H, fp)
        print('dictionary saved successfully to file')

    endTime = time.time()
    print("[INFO] total time taken to train the model: {:.2f}s".format(
        endTime - startTime))

if __name__ == "__main__":
    main()