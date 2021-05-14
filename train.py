from data_loader import parseArguments
import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt



def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help="File path to the CSV or NPY file that contains walking data.")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="Directory where to save outputs.")
    parser.add_argument('--lr', type = float, default = LEARNING_RATE,
                        help = 'Set the learning rate')
    args = parser.parse_args()
    return args


def train_GAN():
    for epoch in range(n_epochs):
    # Dataloader returns the batches and the labels
        for real, labels in tqdm(dataloader):
            cur_batch_size = len(real)
            # Flatten the batch of real images from the dataset
            real = real.to(device)

            one_hot_labels = get_one_hot_labels(labels.to(device), n_classes)
            image_one_hot_labels = one_hot_labels[:, :, None, None]
            image_one_hot_labels = image_one_hot_labels.repeat(1, 1, mnist_shape[1], mnist_shape[2])

            ### Update discriminator ###
            # Zero out the discriminator gradients
            disc_opt.zero_grad()
            # Get noise corresponding to the current batch_size 
            fake_noise = get_noise(cur_batch_size, z_dim, device=device)
            
            # Now you can get the images from the generator
            # Steps: 1) Combine the noise vectors and the one-hot labels for the generator
            #        2) Generate the conditioned fake images
        
            #### START CODE HERE ####
            noise_and_labels = combine_vectors(fake_noise, one_hot_labels)
            fake = gen(noise_and_labels)
            #### END CODE HERE ####
            
            # Make sure that enough images were generated
            assert len(fake) == len(real)
            # Check that correct tensors were combined
            assert tuple(noise_and_labels.shape) == (cur_batch_size, fake_noise.shape[1] + one_hot_labels.shape[1])
            # It comes from the correct generator
            assert tuple(fake.shape) == (len(real), 1, 28, 28)

            # Now you can get the predictions from the discriminator
            # Steps: 1) Create the input for the discriminator
            #           a) Combine the fake images with image_one_hot_labels, 
            #              remember to detach the generator (.detach()) so you do not backpropagate through it
            #           b) Combine the real images with image_one_hot_labels
            #        2) Get the discriminator's prediction on the fakes as disc_fake_pred
            #        3) Get the discriminator's prediction on the reals as disc_real_pred
            
            #### START CODE HERE ####
            fake_image_and_labels = combine_vectors(fake, image_one_hot_labels)
            real_image_and_labels = combine_vectors(real, image_one_hot_labels)
            disc_fake_pred = disc(fake_image_and_labels.detach())
            disc_real_pred = disc(real_image_and_labels)
            #### END CODE HERE ####
            
            # Make sure shapes are correct 
            assert tuple(fake_image_and_labels.shape) == (len(real), fake.detach().shape[1] + image_one_hot_labels.shape[1], 28 ,28)
            assert tuple(real_image_and_labels.shape) == (len(real), real.shape[1] + image_one_hot_labels.shape[1], 28 ,28)
            # Make sure that enough predictions were made
            assert len(disc_real_pred) == len(real)
            # Make sure that the inputs are different
            assert torch.any(fake_image_and_labels != real_image_and_labels)
            # Shapes must match
            assert tuple(fake_image_and_labels.shape) == tuple(real_image_and_labels.shape)
            assert tuple(disc_fake_pred.shape) == tuple(disc_real_pred.shape)
            
            
            disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
            disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
            disc_loss = (disc_fake_loss + disc_real_loss) / 2
            disc_loss.backward(retain_graph=True)
            disc_opt.step() 

            # Keep track of the average discriminator loss
            discriminator_losses += [disc_loss.item()]

            ### Update generator ###
            # Zero out the generator gradients
            gen_opt.zero_grad()

            fake_image_and_labels = combine_vectors(fake, image_one_hot_labels)
            # This will error if you didn't concatenate your labels to your image correctly
            disc_fake_pred = disc(fake_image_and_labels)
            gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
            gen_loss.backward()
            gen_opt.step()

            # Keep track of the generator losses
            generator_losses += [gen_loss.item()]
            #

            if cur_step % display_step == 0 and cur_step > 0:
                gen_mean = sum(generator_losses[-display_step:]) / display_step
                disc_mean = sum(discriminator_losses[-display_step:]) / display_step
                print(f"Step {cur_step}: Generator loss: {gen_mean}, discriminator loss: {disc_mean}")
                show_tensor_images(fake)
                show_tensor_images(real)
                step_bins = 20
                x_axis = sorted([i * step_bins for i in range(len(generator_losses) // step_bins)] * step_bins)
                num_examples = (len(generator_losses) // step_bins) * step_bins
                plt.plot(
                    range(num_examples // step_bins), 
                    torch.Tensor(generator_losses[:num_examples]).view(-1, step_bins).mean(1),
                    label="Generator Loss"
                )
                plt.plot(
                    range(num_examples // step_bins), 
                    torch.Tensor(discriminator_losses[:num_examples]).view(-1, step_bins).mean(1),
                    label="Discriminator Loss"
                )
                plt.legend()
                plt.show()
            elif cur_step == 0:
                print("Congratulations! If you've gotten here, it's working. Please let this train until you're happy with how the generated numbers look, and then go on to the exploration!")
            cur_step += 1


def main(args = sys.argv[1:]):
    args = parseArguments()
    train_GAN(args)


if __name__ == "__main__":
    main()