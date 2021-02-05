class Training_model:
    ### only classiication model need to put test_x, test_y for validation ###
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

    def training(self, Train_x, Train_y, Test_x, Test_y):
        print("---------------------- model training ------------------------")
        self.net = self.net_model.to(device)
        Train_x = variable(torch.FloatTensor(Train_x)).to(device)

        if self.loss_function_name == "cross_entorpy_loss":
            Train_y = variable(torch.LongTensor(Train_y)).to(device)

            Test_x = variable(torch.FloatTensor(Test_x)).to(device)
            Test_y = variable(torch.LongTensor(Test_y)).to(device)

        else:
            Train_y = variable(torch.FloatTensor(Train_y)).to(device)

        Tensorloader = data.TensorDataset(Train_x, Train_y)
        Tensorloader = data.DataLoader(dataset=Tensorloader, batch_size=self.batch_size, shuffle=True)

        Training_Loss = [];
        Training_Acc = [];
        Validation_Acc = []
        for epoch in range(1, self.max_epoch + 1):
            self.net.train()

            Training_Batch_Loss = [];
            Training_Batch_Acc = [];
            Validation_Batch_Acc = []
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

                    self.net.eval()
                    accuracy_test = accuracy_score(Test_y.cpu().data.numpy(),
                                                   np.argmax(self.net(Test_x).cpu().data.numpy(), axis=1))
                    Validation_Batch_Acc.append(accuracy_test)

                    self.net.train()

            Training_Batch_Loss = np.mean(Training_Batch_Loss)
            Training_Loss.append(Training_Batch_Loss)

            if self.loss_function_name == "cross_entorpy_loss":
                Training_Batch_Acc = np.mean(Training_Batch_Acc)
                Training_Acc.append(Training_Batch_Acc)

                Validation_Batch_Acc = np.mean(Validation_Batch_Acc)
                Validation_Acc.append(Validation_Batch_Acc)

            if epoch == 1:
                if self.loss_function_name == "mean_square_error_loss":
                    print("Training -->  epoch %4d, mean square error: %5.4f" % (epoch, Training_Batch_Loss))
                elif self.loss_function_name == "cross_entorpy_loss":
                    print("Training -->  epoch %4d, cross entropy loss: %5.4f, accuracy: %5.4f, Val accuracy: %4.4f" % (
                    epoch, Training_Batch_Loss, Training_Batch_Acc, Validation_Batch_Acc))

            elif epoch % 50 == 0:
                if self.loss_function_name == "mean_square_error_loss":
                    print("         -->  epoch %4d, mean square error: %5.4f" % (epoch, Training_Batch_Loss))
                elif self.loss_function_name == "cross_entorpy_loss":
                    print("         -->  epoch %4d, cross entropy loss: %5.4f, accuracy: %5.4f, Val accuracy: %4.4f" % (
                    epoch, Training_Batch_Loss, Training_Batch_Acc, Validation_Batch_Acc))

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
            plt.plot(np.arange(1, len(Validation_Acc) + 1), Validation_Acc, color="blue", label="Validation Accuracy")
            plt.xlabel("epoch", fontsize=10)
            plt.xlabel("epoch", fontsize=10)
            plt.ylabel("Accuracy", fontsize=10)
            plt.grid(True)
            plt.legend(loc="best", fontsize=12)

        plt.show()
        return self
    
########################################### e.g. ###########################################
Stacked_LSTM= net()
Stacked_LSTM_Training= Training_model(model= Stacked_LSTM, max_epoch= 100, batch_size= 64, learning_rate=0.01, Input_shape= x_train.shape, loss_function= "mean_square_error_loss")
Stacked_LSTM_Training= Stacked_LSTM_Training.training(x_train, y_train, x_train, y_train)

Cov2d_Auto= net()
Cov2d_Auto_Training= Training_model(model= Cov2d_Auto, max_epoch= 100, batch_size= 64, learning_rate=0.01, Input_shape= x_train.shape, loss_function= "cross_entorpy_loss")
Cov2d_Auto_Training= Stacked_LSTM_Training.training(x_train, y_train, x_validation, x_validation)
