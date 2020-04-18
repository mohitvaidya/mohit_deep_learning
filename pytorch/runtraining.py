from Network import *
from Runbuilder import *
from RunManager import *
from collections import OrderedDict


def start_train():
    train_set = torchvision.datasets.FashionMNIST(root='./data/FashionMNIST/', train= False,
                                              transform= transforms.Compose([transforms.ToTensor()]))
    parameters = OrderedDict(lr =[0.01], batch_size = [100,1000])

    m= RunManager()

    for run in Runbuilder.get_runs(parameters):

        network = Network()
        loader = torch.utils.data.DataLoader(train_set,batch_size=run.batch_size)
        optimizer = optim.Adam(network.parameters(), lr = run.lr)

        m.begin_run(run, network, loader)

        for epoch in range(2):
            m.begin_epoch()
            for batch in loader:
                images , labels = batch
                preds = network(images)
                loss = F.cross_entropy(preds, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                m.track_loss(loss)
                m.track_num_correct(preds, labels)
            m.end_epoch()

        m.end_run()
    m.save('results')
if __name__=='__main__':
    print('training started')
    start_train()
    print('training ended')