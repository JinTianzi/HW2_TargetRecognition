import matplotlib.pyplot as plt

for model in ['baseline','cutmix','cutout','mixup']:
    train_acc = []
    test_acc = [0]
    with open(model+'_train_accuracy.txt', 'r') as file:
        for line in file:
            number = float(line.strip())
            train_acc.append(number)
    with open(model+'_test_accuracy.txt', 'r') as file:
        for line in file:
            number = float(line.strip())
            test_acc.append(number)

    iters = range(len(train_acc))

    plt.figure()
    plt.plot(iters, train_acc, linewidth = 2, label='train accuracy')
    plt.plot([0,10,20,30,40,50,60,70], test_acc, linewidth = 2, label='val accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.ylim(0, 1)
    plt.title('CNN Acurracy - '+model)
    plt.legend()
    plt.savefig(( model+"_acc.png"))