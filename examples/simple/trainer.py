import torch

device = torch.device('cpu')


class Trainer():
    def __init__(self, model, args):
        self.learning_rate = args.lr
        self.model = model
        self.criteria = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.model_type = args.model_type
    def train(self, batch, labels):
        if self.model_type == "binary":
            return self.train_a_batch_binary(batch, labels)
        elif self.model_type == "five":
            return self.train_a_batch_five(batch, labels)
        else:
            print("no such model")

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

    def train_a_batch_binary(self, batch, labels):
        self.model.train()
        batch = batch.to(device)
        labels = labels.to(device)
        outputs = self.model(batch)
        loss = self.criteria(outputs[:, 5:], labels[1])
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    def evaluate(self, batch, labels):
        if self.model_type == "binary":
            return self.evaluate_binary(batch, labels)
        elif self.model_type == "five":
            return self.evaluate_five(batch, labels)
        else:
            print("no such model")
    def evaluate_five(self, batch, labels):
        self.model.eval()
        batch = batch.to(device)
        labels = labels.to(device)
        batch_size = batch.shape[0]
        accuracy = 0
        with torch.no_grad():
            outputs = self.model(batch)
            loss = self.criteria(outputs[:, 0:5], labels[0])
            for i in range(batch_size):
                output = outputs[i][0:5]
                healthy = torch.argmax(output)
                print(" label:", labels[0][i].item(), " predict:", healthy.item(), " prob:",
                      torch.max(torch.softmax(output, dim=0)).item())
                if healthy == labels[0][i]:
                    accuracy += 1
        return accuracy / batch_size, loss.item()

    def evaluate_binary(self, batch, labels):
        self.model.eval()
        batch = batch.to(device)
        labels = labels.to(device)
        batch_size = batch.shape[0]
        accuracy = 0
        with torch.no_grad():
            outputs = self.model(batch)
            loss = self.criteria(outputs[:, 5:], labels[1])
            for i in range(batch_size):
                output = outputs[i][5:]
                healthy = torch.argmax(output)
                print(" label:", labels[1][i].item(), " predict:", healthy.item(), " prob:",
                      torch.max(torch.softmax(output, dim=0)).item())
                if healthy == labels[1][i]:
                    accuracy += 1
        return accuracy / batch_size, loss.item()
