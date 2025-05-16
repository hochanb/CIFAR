import torch
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
import time
import torch.optim as optim
import torchvision
import random


cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
   
def train(net, train_loader, epochs = 2):
    lr = 0.003
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    
    current_time = time.time()
    log_interval = 1000
    loss_history = []  # 손실 기록용 리스트
    
    for epoch in range(epochs):   # 데이터셋을 수차례 반복합니다.

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # [inputs, labels]의 목록인 data로부터 입력을 받은 후;
            inputs, labels = data
            if cuda:
                inputs, labels = inputs.to(device), labels.to(device)    

            # 변화도(Gradient) 매개변수를 0으로 만들고
            optimizer.zero_grad()

            # 순전파 + 역전파 + 최적화를 한 후
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 통계를 출력합니다.
            running_loss += loss.item()
            if i % log_interval == log_interval-1:    # print every n mini-batches
                avg_loss = running_loss / log_interval
                loss_history.append(avg_loss)
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {avg_loss:.3f}')
                running_loss = 0.0
                
        lr = max(lr * 0.95, 0.0003)
        optimizer.param_groups[0]['lr'] = lr  # learning rate decay

    print(f'Finished Training, {time.time() - current_time:.2f} seconds')
    # graph the loss
    plt.figure(figsize=(8, 5))
    plt.ylim(0,2.5)
    plt.plot(range(1, len(loss_history) + 1), loss_history)
    plt.title('Training Loss every 1000 iterations')
    plt.xlabel('Log step (x1000 iters)')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()
    
    return loss_history


def test_accuracy(net, test_loader, classes):
    correct = 0
    total = 0
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            if cuda:
                images, labels = images.to(device), labels.to(device)
            # 신경망에 이미지를 통과시켜 출력을 계산합니다
            outputs = net(images)
            # 가장 높은 값(energy)를 갖는 분류(class)를 정답으로 선택하겠습니다
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            for label, prediction in zip(labels, predicted):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1


    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

    # 클래스별 정확도 계산
    class_accuracies = []
    for classname in classes:
        accuracy = 100 * float(correct_pred[classname]) / total_pred[classname]
        class_accuracies.append(f'{accuracy:5.1f}%')

    print("Accuracy for each class:")
    print('  '.join(f'{name:7s}' for name in classes))       # 클래스 이름들 정렬 출력
    print('  '.join(class_accuracies))                       # 정확도들 정렬 출력
    
    return correct // total
     
def show_prediction(net, testset, classes):
    net.eval()
    n_samples = 10
    # 무작위 인덱스 n개 선택
    indices = random.sample(range(len(testset)), n_samples)
    images = torch.stack([testset[i][0] for i in indices])
    labels = torch.tensor([testset[i][1] for i in indices])

    images = images.to(device)
    labels = labels.to(device)

    # 이미지 출력
    imshow(torchvision.utils.make_grid(images.cpu(), nrow=10))
    print('True labels:    ', ' '.join(f'{classes[labels[j]]:7s}' for j in range(n_samples)))

    # 예측 출력
    with torch.no_grad():
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)

    print('Predicted:      ', ' '.join(f'{classes[predicted[j]]:7s}' for j in range(n_samples)))
    
    

def train_and_test(net, trainloader, testloader, testset, classes, epochs=2):
    loss = train(net, trainloader, epochs=epochs)
    accuracy = test_accuracy(net, testloader,classes)
    show_prediction(net, testset, classes)
    return loss, accuracy

def draw_loss_graph_1000_iters(title, loss_list, label_list):
    # graph the all loss
    plt.figure(figsize=(8, 5))
    for loss, label in zip(loss_list, label_list):
        plt.plot(range(1, len(loss) + 1), loss, label=label)

    plt.title(title)
    plt.xlabel('Log step (x1000 iters)')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.ylim(0,2.5)
    plt.show()