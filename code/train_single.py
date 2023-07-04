import argparse
import os
import torch
from torch.utils import data
from dataset import HeadData
import torchvision
import torchvision.transforms as transforms
from trainer_rgb import Trainer
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import torch.multiprocessing as mp

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def data_sampler(dataset, shuffle):
    if shuffle:
        return data.RandomSampler(dataset)
    else:
        return data.SequentialSampler(dataset)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def display_img(idx, img, name, writer, args):
    img = img.clamp(-1, 1)
    img = ((img - img.min()) / (img.max() - img.min())).data
    torchvision.utils.save_image(img, args.exp_path + args.exp_name + '/display/'+str(idx)+name+'.png')

    writer.add_images(tag='%s' % (name), global_step=idx, img_tensor=img)


def display_bases(imgs, name, args):
    for idx in range(len(imgs)):
        img = imgs[idx]
        img = img.clamp(-1, 1)
        img = ((img - img.min()) / (img.max() - img.min())).data
        torchvision.utils.save_image(img, args.exp_path + args.exp_name + '/bases/'+str(idx)+name+'.png')



def write_loss(i, l2_loss, lpips_loss, writer):
    writer.add_scalar('l2_loss', l2_loss.item(), i)
    writer.add_scalar('lpips_loss', lpips_loss.item(), i)
    writer.flush()


def ddp_setup(args, rank, world_size):
    os.environ['MASTER_ADDR'] = args.addr
    os.environ['MASTER_PORT'] = args.port

    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def main(rank, world_size, args):
    # init distributed computing
    ddp_setup(args, rank, world_size)
    torch.cuda.set_device(rank)
    device = torch.device("cuda")

    # make logging folder
    log_path = os.path.join(args.exp_path, args.exp_name + '/log')
    checkpoint_path = os.path.join(args.exp_path, args.exp_name + '/checkpoint')
    display_path = os.path.join(args.exp_path, args.exp_name + '/display')
    bases_path = os.path.join(args.exp_path, args.exp_name + '/bases')
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(checkpoint_path, exist_ok=True)
    os.makedirs(display_path, exist_ok=True)
    os.makedirs(bases_path, exist_ok=True)
    writer = SummaryWriter(log_path)

    print('==> preparing dataset')
    transform = torchvision.transforms.Compose([
        transforms.Resize(args.size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    dataset = HeadData('train', transform, dataset = args.dataset , person = args.person)
    dataset_test = HeadData('test', transform, dataset = args.dataset , person = args.person )

    loader = data.DataLoader(
        dataset,

        batch_size=args.batch_size // world_size,
        sampler=data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True),
        pin_memory=True,
        drop_last=False,
    )

    loader_test = data.DataLoader(
        dataset_test,
        batch_size=1,
        sampler=data.distributed.DistributedSampler(dataset_test, num_replicas=world_size, rank=rank, shuffle=False),
        pin_memory=True,
        drop_last=False,
    )

    loader = sample_data(loader)
    loader_test = sample_data(loader_test)
    print('==> initializing trainer')
    # Trainer
    trainer = Trainer(args, device, rank)

    # resume
    if args.resume_ckpt is not None:
        args.start_iter = trainer.resume(args.resume_ckpt)
        print('==> resume from iteration %d' % (args.start_iter))

    print('==> training')
    pbar = range(args.iter)
    for idx in pbar:
        i = idx + args.start_iter

        # laoding data
        real_image, label = next(loader)
        real_image = real_image.to(rank, non_blocking=True)
        label = label.to(rank, non_blocking=True)


        # update generator
        l2_loss, loss_lpips, generated_image  = trainer.gen_update(real_image, label, person_2 = False)


        if rank == 0:
            # write to log
            write_loss(idx, l2_loss, loss_lpips, writer)

        if (i+1) >= args.tune_iter:
            # print('begin training nerf')
            trainer.tune_generator()
        # display
        if (i+1) % args.display_freq == 0 and rank == 0:
            print("[Iter %d/%d] [l2 loss: %f] [lpips loss: %f]"
                  % (i, args.iter, l2_loss.item(), loss_lpips.item()))

            if rank == 0:
                real_image_test, label_test = next(loader_test)
                real_image_test = real_image_test.to(rank, non_blocking=True)
                label_test = label_test.to(rank, non_blocking=True)

                img_recon = trainer.sample(real_image_test, label_test)
                bases_1 = trainer.sample_bases(person_2 = False)
                display_bases(bases_1, 'person_1', args)
                display_img(i, real_image_test, 'source', writer, args)
                display_img(i, img_recon, 'recon', writer, args)
                writer.flush()

        # save model
        if (i+1) % args.save_freq == 0 and rank == 0:
            trainer.save(i, checkpoint_path)

    return


if __name__ == "__main__":
    # training params
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, default=800000)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--dataset", type=str, default='nerface_dataset')
    parser.add_argument("--person", type=str, default='person_3')
    parser.add_argument("--person_2", type=str, default=None)
    parser.add_argument("--run_id", type=str, default='nerface2')
    parser.add_argument("--run_id_2", type=str, default=None)
    parser.add_argument("--emb_dir", type=str, default='./PTI/embeddings/')

    parser.add_argument("--d_reg_every", type=int, default=16)
    parser.add_argument("--g_reg_every", type=int, default=4)
    parser.add_argument("--resume_ckpt", type=str, default=None)
    parser.add_argument("--old", action='store_true', default=True)
    parser.add_argument("--tune", action='store_true', default=True)
    parser.add_argument("--init", action='store_true', default=False)
    parser.add_argument("--same_bases", action='store_true', default=False)
    parser.add_argument("--out_pose", action='store_true', default=False)
    parser.add_argument("--lr", type=float, default=3e-4)

    
    parser.add_argument("--channel_multiplier", type=int, default=1)
    parser.add_argument("--start_iter", type=int, default=0)
    parser.add_argument("--display_freq", type=int, default=5000)
    parser.add_argument("--save_freq", type=int, default=5000)
    parser.add_argument("--latent_dim_style", type=int, default=512)
    parser.add_argument("--latent_dim_shape", type=int, default=50)
    parser.add_argument("--exp_path", type=str, default='./code/exps/')
    parser.add_argument("--exp_name", type=str, default='v1')
    parser.add_argument("--addr", type=str, default='localhost')
    parser.add_argument("--port", type=str, default='12345')
    parser.add_argument("--tune_iter", type=int, default=50000)
    opts = parser.parse_args()

    n_gpus = torch.cuda.device_count()
    print('==> training on %d gpus' % n_gpus)
    world_size = n_gpus
    if world_size == 1:
        main(rank=0, world_size = world_size, args=opts)
    elif world_size > 1:
        mp.spawn(main, args=(world_size, opts,), nprocs=world_size, join=True)