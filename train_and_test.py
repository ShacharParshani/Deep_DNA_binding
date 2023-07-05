import torch


# This is the train function, returns a trained model
def train(model, train_loader, criterion, optimizer, number_of_epochs, batch_size):
    # set the model into train mode
    model.train()

    for epoch in range(number_of_epochs):
        train_loss = 0
        correct = 0
        total = 0
        for x_seq, y_label in train_loader:
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward step
            y_pred = model(x_seq)
            # calculate the loss
            loss = criterion(y_pred, y_label)
            # backward pass: compute gradient of the loss
            loss.backward()
            # update the weights
            optimizer.step()
            # update running training loss
            train_loss += loss.item()

            print("y_pred shape: ", y_pred.shape)
            print("y_pred:\n", y_pred)
            # for each row of y_pred, find the index of the max value and save it in predicted vector
            predicted = torch.argmax(y_pred, dim=1)
            print("predicted shape: ", predicted.shape)
            print("predicted:\n", predicted)
            print("y_label shape: ", y_label.shape)
            print("y_label:\n", y_label)

            # Convert predicted and y_label to one-hot encoding
            num_classes = y_pred.size(1)
            predicted_one_hot = torch.zeros(batch_size, num_classes)
            predicted_one_hot.scatter_(1, predicted.unsqueeze(1), 1)
            print("predicted_one_hot:\n", predicted_one_hot)

            total += y_label.size(0)
            correct += (predicted_one_hot == y_label).sum().item()

        # calculate average loss over an epoch
        train_loss = train_loss / len(train_loader.dataset)
        accuracy = correct / total
        print('Epoch: {} \tTraining Loss: {:.6f} \tAccuracy: {:.6f}'.format(epoch + 1, train_loss, 100*accuracy))
    return model


# This is the test function, returns the accuracy of the model on the test set
def test(model, test_loader, criterion, batch_size):
    test_loss = 0
    # set the model into evaluation mode
    model.eval()
    correct = 0

    for x_seq, y_label in test_loader:
        # forward step
        y_pred = model(x_seq)
        # calculate the loss
        loss = criterion(y_pred, y_label)
        # update test loss
        test_loss += loss.item()
        # count the number of correct predictions
        y_pred = y_pred.max(1, keepdim=True)[1]
        for index in range(batch_size):
            if y_label[index] == y_pred[index]:
                correct += 1

    # print the test loss and accuracy
    test_loss = test_loss/len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)
