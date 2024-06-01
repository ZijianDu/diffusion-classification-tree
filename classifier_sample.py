"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import argparse
import os
from visualization.visualizer import visualizer
from PIL import Image
import pickle
import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    logger.configure()

    logger.log("reading imagenet images...")
    with np.load('imgnet_64x64_datasets/train_data_batch_1.npz') as data:
        shape = data["data"].shape
        logger.log("reshape clean images...")
        images = data["data"].reshape(shape[0], 3, args.image_size, args.image_size)
        # convert images into -1.0 ~ 1.0 range
        images = (images/127.5) - 1.0
        #sample_image = images[0, :, :, :]
        #Image.fromarray(sample_image.transpose(1, 2, 0)).save("test.png")

    images = th.tensor(images, dtype = th.float32).to("cuda")

    logger.log("creating model and diffusion...")
   
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("loading classifier...")
    classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
    classifier.load_state_dict(
        dist_util.load_state_dict(args.classifier_path, map_location="cpu")
    )
    classifier.to(dist_util.dev())
    if args.classifier_use_fp16:
        classifier.convert_to_fp16()
    classifier.eval()

    # classifier to output probability class for each input image/t combination
    # x: batchsize x 3 x imgsize x imgsize
    # probability: batchsize x 1000
    def classify(x, t):
        x_in = x.detach()
        logits = classifier(x_in, t)
        probability = F.softmax(logits, dim = -1)
        assert len(probability.shape) == 2
        assert probability.shape[0] == args.batch_size
        assert probability.shape[1] == 1000
        return probability

    def cond_fn(x, t, y=None):
        assert y is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            # x_in: batch_size x 3 x img_size x img_size
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            probability = F.softmax(logits, dim = -1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            return th.autograd.grad(selected.sum(), x_in)[0] * args.classifier_scale

    def model_fn(x, t, y=None):
        assert y is not None
        return model(x, t, y if args.class_cond else None)

    logger.log("sampling...")
    all_images = []
    all_probabilities = []

    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        classes = th.randint(
            low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
        )
        print("classes:", classes)
        model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        print("sample function: ", sample_fn)

        # probability has DDIMstep as first dimension
        one_batch_probability_vector = np.zeros(shape = (args.batch_size, 100, 1000))
        # provide clean image for reverse ddim
        # output images: batch x diffusionstep x 3 x imgsize x imgsize 
        # output probability vectors: batch x diffusionsteps x num_classes
        output_images = sample_fn(
            model_fn,
            (args.batch_size, 3, args.image_size, args.image_size),
            noise = images[:args.batch_size, :, :, :], 
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            device=dist_util.dev(),
        )
        output_images = ((output_images + 1) * 127.5).clamp(0, 255).to(th.uint8)
        #output_images = output_images.permute(0, 2, 3, 1)
        output_images = output_images.contiguous()

        print("output images shape: ", output_images.shape)
        
        for time_step in range(100):
            t = [time_step] * args.batch_size
            probability = classify(output_images, t)
            one_batch_probability_vector[:, time_step, :] = probability
                
        gathered_samples = [th.zeros_like(output_images) for _ in range(dist.get_world_size())]
        #dist.all_gather(gathered_samples, output_images)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        gathered_probabilities = [th.zeros_like(classes) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_probabilities, classes)
        all_probabilities.extend([probabilities.cpu().numpy() for probabilities in gathered_probabilities])
        logger.log(f"created {len(all_images) * args.batch_size} samples")
        

    # final image arr shape: num_samples x diffusionsteps x 3 x img_size x img_size
    output_images = np.concatenate(all_images, axis=0)
    output_images = output_images[: args.num_samples]
    print("final output image shape: ", output_images.shape)

    # final probability vector shape: num_samples x diffusionsteps x 1000
    output_probabilities = np.concatenate(all_probabilities, axis=0)
    output_probabilities = output_probabilities[: args.num_samples]
    print("final output probability shape: ", output_probabilities.shape)

    # generated batch of images and labels
    """
    images = list(arr[i][:, :, 0].squeeze() for i in range(arr.shape[0]))
    print("generated image inputs", images)
    print(images[0].shape)
    vis = visualizer(2, images, list(label_arr))
    vis.run_manifold_learning()
    """

    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in output_images.shape])
        out_path = os.path.join("samples/", f"{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, output_images, output_probabilities)
        #for i in range(arr.shape[0]):
        #    name = str(i+1) + ".jpg"
        #    Image.fromarray(arr[i, :, :, :]).save("samples/" + name)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=True,
        model_path="",
        classifier_path="",
        classifier_scale=1.0,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
