import numpy as np

from torch.autograd import Variable
from tqdm import tqdm


class TaskManager():
    def __init__(self, args, save_path):
        self.save_path = save_path
        self.cuda = args.cuda
        self.output_dim = args.output_dim

    def run(self, model, dataloader):

        model.eval()
        no_img_imgs = dataloader.dataset.__len__()
        embeddings = np.zeros(shape=(no_img_imgs, self.output_dim))

        with tqdm(total=len(dataloader), desc=f'Getting Embeddings') as t:
            for idx, batch in enumerate(dataloader):

                if self.cuda:
                    batch = batch.cuda()

                batch = Variable(batch)

                output = model.forward(batch)

                start_idx = idx * len(batch)
                end_idx = min(((idx + 1) * len(batch)), no_img_imgs)


                embeddings[start_idx:end_idx, :] = output.cpu().detach().numpy()

                t.update()

        return embeddings
