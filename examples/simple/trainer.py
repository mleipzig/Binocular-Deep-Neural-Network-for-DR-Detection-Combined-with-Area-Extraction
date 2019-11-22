import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class Trainer():
    def __init__(self, model, args):
        self.learing_rate = args.lr
        self.model = model
        self.criteria = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learing_rate)

    def train_a_batch_binary(self, batch, labels):
        self.model.train()
        batch = batch.to(device)
        labels = labels.to(device)
        print("begin infer")
        outputs = self.model(batch)
        print("begin optimize")
        loss = self.criteria(outputs[:, 5:], labels[1])
        self.optimizer.zero_grad()
        loss.backward()
        print("end optimize")
        self.optimizer.step()
        return loss.item()

    def train_a_batch_five(self, batch, labels):
        self.model.train()
        batch = batch.to(device)
        labels = labels.to(device)
        outputs = self.model(batch)
        loss = self.criteria(outputs[:, 0:5], labels[0])
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluate_binary(self, batch, labels):
        self.model.eval()
        batch = batch.to(device)
        labels = labels.to(device)
        batch_size = batch.shape[0]
        accuracy = 0
        with torch.no_grad():
            for i in range(batch_size):
                outputs = self.model(batch[i].view((1,) + batch[i].shape))
                output_2 = outputs[0][5:]
                healthy = torch.argmax(output_2)
                print('-----')
                print(" label:", labels[1][i].item(), " predict:", healthy.item(), " prob:",
                      torch.max(torch.softmax(output_2, dim=0)).item())
                if healthy == labels[1][i]:
                    accuracy += 1
        return accuracy / batch_size