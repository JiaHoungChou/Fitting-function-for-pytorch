# Fitting-function-for-pytorch
This function can use for training in Pytorch directly.

class Training_model:
    def __init__(self, model, max_epoch, batch_size, learning_rate, Input_shape, loss_function):
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.channal_size = Input_shape[1]
        self.net_model = model
        self.optimizer = torch.optim.Adam(self.net_model.parameters(), lr=self.learning_rate)
        self.loss_function_name = loss_function

        if self.loss_function_name == "mean_square_error_loss":
            self.loss_function = nn.MSELoss()

        elif self.loss_function_name == "cross_entorpy_loss":
            self.loss_function = nn.CrossEntropyLoss()

    def training(self, x, y):
        print("---------------------- model training ------------------------")
        self.net = self.net_model.to(device)
        x = variable(torch.FloatTensor(x)).to(device)

        if self.loss_function_name == "cross_entorpy_loss":
            y = variable(torch.LongTensor(y)).to(device)

        else:
            y = variable(torch.FloatTensor(y)).to(device)

        Tensorloader = data.TensorDataset(x, y)
        Tensorloader = data.DataLoader(dataset=Tensorloader, batch_size=self.batch_size, shuffle=True)

        Training_Loss = [];
        Training_Acc = []
        for epoch in range(1, self.max_epoch + 1):
            self.net.train()

            Training_Batch_Loss = [];
            Training_Batch_Acc = []
            for i, data_loader in enumerate(Tensorloader):
                self.optimizer.zero_grad()
                In_data, T_data = data_loader

                output = self.net(In_data)
                loss = self.loss_function(output, T_data)

                loss.backward()
                self.optimizer.step()

                Training_Batch_Loss.append(loss.item())

                if self.loss_function_name == "cross_entorpy_loss":
                    accuracy_training = accuracy_score(T_data.cpu().data.numpy().ravel(),
                                                       np.argmax(output.cpu().data.numpy(), axis=1))
                    Training_Batch_Acc.append(accuracy_training)

            Training_Batch_Loss = np.mean(Training_Batch_Loss)
            Training_Loss.append(Training_Batch_Loss)

            if self.loss_function_name == "cross_entorpy_loss":
                Training_Batch_Acc = np.mean(Training_Batch_Acc)
                Training_Acc.append(Training_Batch_Acc)

            if epoch == 1:
                if self.loss_function_name == "mean_square_error_loss":
                    print("Training -->  epoch %3d, mean square error: %4.4f" % (epoch, Training_Batch_Loss))
                elif self.loss_function_name == "cross_entorpy_loss":
                    print("Training -->  epoch %3d, cross entropy loss: %4.4f, accuracy: %4.4f" % (
                    epoch, Training_Batch_Loss, Training_Batch_Acc))

            elif epoch % 50 == 0:
                if self.loss_function_name == "mean_square_error_loss":
                    print("         -->  epoch %3d, mean square error: %4.4f" % (epoch, Training_Batch_Loss))
                elif self.loss_function_name == "cross_entorpy_loss":
                    print("         -->  epoch %3d, cross entropy loss: %4.4f, accuracy: %4.4f" % (
                    epoch, Training_Batch_Loss, Training_Batch_Acc))

        Training_Loss = np.array(Training_Loss)
        Training_Acc = np.array(Training_Acc)

        plt.figure(figsize=(8, 4))
        plt.title("Training Loss Curve", fontsize=15)
        plt.plot(np.arange(1, len(Training_Loss) + 1), Training_Loss, color="darkblue", label="Training Loss")
        plt.xlabel("epoch", fontsize=10)
        plt.ylabel("loss", fontsize=10)
        plt.grid(True)
        plt.legend(loc="best", fontsize=12)

        if self.loss_function_name == "cross_entorpy_loss":
            plt.figure(figsize=(8, 4))
            plt.title("Training Accuracy Curve", fontsize=15)
            plt.plot(np.arange(1, len(Training_Loss) + 1), Training_Acc, color="red", label="Training Accuracy")
            plt.xlabel("epoch", fontsize=10)
            plt.ylabel("loss", fontsize=10)
            plt.grid(True)
            plt.legend(loc="best", fontsize=12)

        plt.show()
        return self
