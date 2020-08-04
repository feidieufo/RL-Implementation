import pickle
from algos.vae.encoder import PixelEncoder
from algos.vae.decoder import PixelDecoder
from torch.utils.data import Dataset
from env.dmc_env import DMCFrameStack
import torch.nn.functional as F
import torch
import argparse
import dmc2gym
import os
from utils.logx import EpochLogger
from torch.utils.tensorboard import SummaryWriter
import numpy as np

class ExpertDataset(Dataset):
    def __init__(self, data, device):
        self.data = data
        self.device = device
    
    def __getitem__(self, item):
        expert = {}

        expert["obs"] = torch.tensor(self.data["obs"][item], dtype=torch.float32).to(self.device)
        expert["action"] = torch.tensor(self.data["action"][item], dtype=torch.float32).to(self.device)

        return expert

    def __len__(self):
        return len(self.data["obs"])

def preprocess_obs(obs, bits=5):
    """Preprocessing image, see https://arxiv.org/abs/1807.03039."""
    bins = 2**bits
    assert obs.dtype == torch.float32
    if bits < 8:
        obs = torch.floor(obs / 2**(8 - bits))
    obs = obs / bins
    obs = obs + torch.rand_like(obs) / bins
    obs = obs - 0.5
    return obs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain_name', default='cheetah')
    parser.add_argument('--task_name', default='run')
    parser.add_argument('--image_size', default=84, type=int)
    parser.add_argument('--action_repeat', default=1, type=int)
    parser.add_argument('--frame_stack', default=3, type=int)
    parser.add_argument('--encoder_type', default='pixel', type=str)

    parser.add_argument('--exp_name', default="ppo_cheetah_run_clipv_maxgrad_anneallr2.5e-3_stack3_normal_state01_maxkl0.03_gae")
    parser.add_argument('--expert_num', default=10, type=int)
    parser.add_argument('--seed', default=10, type=int)
    parser.add_argument('--gpu', default=0)
    parser.add_argument('--batch', default=50, type=int)
    parser.add_argument('--encoder_lr', default=1e-3, type=float)
    parser.add_argument('--decoder_lr', default=1e-3, type=float)
    parser.add_argument('--decoder_latent_lambda', default=1e-6, type=float)
    parser.add_argument('--epoch', default=1000, type=int)
    parser.add_argument('--out_dir', default="vae_test")
    parser.add_argument('--log_every', default=10, type=int)
    parser.add_argument('--feature_dim', default=50, type=int)
    args = parser.parse_args()

    device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")

    env = dmc2gym.make(
        domain_name=args.domain_name,
        task_name=args.task_name,
        seed=args.seed,
        visualize_reward=False,
        from_pixels=(args.encoder_type == 'pixel'),
        height=args.image_size,
        width=args.image_size,
        frame_skip=args.action_repeat
    )
    if args.encoder_type == 'pixel':
        env = DMCFrameStack(env, k=args.frame_stack)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env.seed(args.seed)
    state_dim = env.observation_space.shape

    from utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    expert_data_file = os.path.join(logger_kwargs["output_dir"], "experts")
    with open(os.path.join(expert_data_file, 
        args.domain_name + "_" + args.task_name + "_epoch" + str(args.expert_num) + ".pkl"), "rb") as f:
        expert_data = pickle.load(f)

    out_kwargs = setup_logger_kwargs(args.out_dir, args.seed)
    logger = EpochLogger(**out_kwargs)
    writer = SummaryWriter(os.path.join(logger.output_dir, "logs"))
    if not os.path.exists(os.path.join(logger.output_dir, "checkpoints")):
        os.makedirs(os.path.join(logger.output_dir, "checkpoints"))

    encoder = PixelEncoder(state_dim, args.feature_dim, num_layers=4).to(device)
    decoder = PixelDecoder(state_dim, args.feature_dim, num_layers=4).to(device)
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=args.encoder_lr)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.decoder_lr)

    expert_dataset = ExpertDataset(expert_data, device=device)
    expert_loader = torch.utils.data.DataLoader(expert_dataset, batch_size=args.batch, shuffle=True)
    
    for iter in range(args.epoch):
        for expert in expert_loader:
            obs = expert["obs"]
            h = encoder(obs)
            rec_obs = decoder(h)

            target_obs = obs.clone()
            target_obs = preprocess_obs(obs)

            rec_loss = F.mse_loss(target_obs, rec_obs)
            # add L2 penalty on latent representation
            # see https://arxiv.org/pdf/1903.12436.pdf
            latent_loss = (0.5 * h.pow(2).sum(1)).mean()
            loss = rec_loss + args.decoder_latent_lambda * latent_loss

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            loss.backward()

            encoder_optimizer.step()
            decoder_optimizer.step()

            logger.store(loss=loss)

        writer.add_scalar("loss", logger.get_stats("loss")[0], global_step=iter)
        logger.log_tabular('Epoch', iter)
        logger.log_tabular("loss", with_min_and_max=True)
        logger.dump_tabular()       

        if iter % args.log_every == 0:
            state = {
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
            }

            torch.save(state, os.path.join(logger.output_dir, "checkpoints", str(iter) + '.pth'))





    