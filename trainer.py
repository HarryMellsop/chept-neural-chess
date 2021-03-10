from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader


class TrainerConfig:

    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 1e-4
    grad_norm_clip = 1.0

    # checkpoint settings
    num_workers = 0

    def __init__(self, func, state_dict, args_dict):
        self.func = func
        self.state_dict = state_dict
        self.__dict__.update(args_dict)


class Trainer:

    def __init__(self, model, train_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.config = config
        self.func = self.config.func
        self.load_params = self.config.state_dict

        if self.func == 'finetune' and self.load_params:
            print('\nLoading pretrain params in...\n')
            self.model.load_state_dict(self.load_params)

        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def save_checkpoint(self, path):
        ckpt_model = self.model.module if hasattr(self.model, "module") else self.model
        save_dict = {'state_dict': ckpt_model.state_dict(),
                     'itos': self.train_dataset.itos,
                     'stoi': self.train_dataset.stoi,
                     'model_config': self.model.model_config,
                     'train_config': self.config}

        torch.save(save_dict, path)

    def train(self):
        model, config = self.model, self.config

        # create the optimizer
        optimizer = optim.Adam(
            params=model.parameters(),
            lr=config.learning_rate,
        )

        def run_epoch(split):
            model.train(True)

            data = self.train_dataset
            loader = DataLoader(data, 
                batch_size=config.batch_size, 
                num_workers=config.num_workers
            )

            pbar = tqdm(enumerate(loader), total=len(loader))
            for it, (x, y) in pbar:
                if it % 1000 == 0:
                    self.save_checkpoint('ckpt/model.iter.params')

                # place data on the correct device
                x = x.to(self.device)
                y = y.to(self.device)

                # forward the model
                with torch.set_grad_enabled(True):
                    logits, loss = model(x, y)
                    loss = loss.mean()

                # backprop and update the parameters
                model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                optimizer.step()

                # report progress
                pbar.set_description(f"epoch {epoch + 1} iter {it}: train loss {loss.item():.3f}")

        self.tokens = 0
        for epoch in range(config.max_epochs):
            run_epoch('train')
            self.save_checkpoint('ckpt/model.final.params')
