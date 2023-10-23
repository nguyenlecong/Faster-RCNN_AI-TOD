class EarlyStopping:
    def __init__(self, patience=100):
        self.best_fitness = float('inf')
        self.best_epoch = 0
        self.patience = patience or float('inf')  # epochs to wait after fitness stops improving to stop

    @staticmethod
    def mean(x):
        return sum(x) / len(x)
    
    @staticmethod
    def log(path, line):
        logger = open(path, "a")
        logger.write(line + '\n')
        logger.close()

    def __call__(self, epoch, fitness):
        if fitness <= self.best_fitness:
            self.best_epoch = epoch
            self.best_fitness = fitness
        delta = epoch - self.best_epoch  # epochs without improvement
        stop = delta >= self.patience  # stop training if patience exceeded
        if stop:
            print(f'Stopping training early as no improvement observed in last {self.patience} epochs. '
                  f'Best results observed at epoch {self.best_epoch}')
        return stop
    
if __name__ == '__main__':
    patience = 10
    early_stopper = EarlyStopping(patience)

    path = 'training/3/val_log.txt'
    file = open(path, 'r')
    lines = file.readlines()
    loss = [float(str(i).split('  ')[0].split(' ')[2][1:-1]) for i in lines]

    for epoch, loss in enumerate(loss):
        stop = early_stopper(epoch, loss)
        if stop:
            print(epoch)