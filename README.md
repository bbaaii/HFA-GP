# HFA-GP: High-fidelity Facial Avatar Reconstruction from Monocular Video with Generative Priors
<p align="center">
<img src=assets/teaser.png />
</p>

## HFA-GP
**HFA-GP** is a framework for reconstructing high-fidelity facial avatar from monocular video.
This is the official implementation of *High-fidelity Facial Avatar Reconstruction from Monocular Video with Generative Priors*

## Abstract
High-fidelity facial avatar reconstruction from a monocular video is a significant research problem in computer graphics and computer vision. Recently, Neural Radiance Field (NeRF) has shown impressive novel view rendering results and has been considered for facial avatar reconstruction. However, the complex facial dynamics and missing 3D information in monocular videos raise significant challenges for faithful facial reconstruction. In this work, we propose a new method for NeRF-based facial avatar reconstruction that utilizes 3D-aware generative prior. Different from existing works that depend on a conditional deformation field for dynamic modeling, we propose to learn a personalized generative prior, which is formulated as a local and low dimensional subspace in the latent space of 3D-GAN. We propose an efficient method to construct the personalized generative prior based on a small set of facial images of a given individual. After learning, it allows for photo-realistic rendering with novel views, and the face reenactment can be realized by performing navigation in the latent space. Our proposed method is applicable for different driven signals, including RGB images, 3DMM coefficients, and audio. Compared with existing works, we obtain superior novel view synthesis results and faithfully face reenactment performance. 

## Preprocessing data

```
python3 ./eg3d-pose-detection/process_test_video.py --input_dir
```


## Training on monocular video
audio-driven：
```
python3 ./code/train_audio.py --dataset 'ad_dataset' --person_1 'english_m' --exp_path './code/exps/' --exp_name 'ad-english_w_e'
```

3DMM-driven:
```
python3 ./code/train_3dmm.py --dataset 'nerface_dataset' --person_1 'person_3' --exp_path './code/exps/' --exp_name '1-nerface3-3dmm2'
```

RGB-driven:
```
python3 ./code/train_rgb.py --dataset 'nerface_dataset' --person 'person_3' --exp_path './code/exps/' --exp_name '1-nerface-3-2' 
```

# Performing face reenactment
audio-driven：
```
python3 ./code/run_recon_video_audio.py --dataset 'ad_dataset' --person_1 'english_w' --demo_name 'english_w' --dataset_type 'val' --model_path './code/exps/ad-english_m/checkpoint/checkpoint.pt' --cat_video
```
For audio-driven experiments, deepspeech is required to extract features from the audio. This part uses AD-NeRF's code [AD-NeRF](https://github.com/YudongGuo/AD-NeRF). First, use ffmpeg to extract the audio in WAV format and then extract the features. The extracted feature file should be named aud.npy.

3DMM-driven：
```
python3 ./code/run_recon_video_3dmm.py --dataset 'nerface_dataset' --person_1 'person_3' --demo_name 'nerface3' --dataset_type 'test' --model_path './code/exps/1-nerface3/checkpoint/checkpoint.pt' --cat_video
```

RGB-driven：
```
python3 ./code/run_recon_video_rgb.py --dataset 'nerface_dataset' --person 'person_3' --demo_name '1-nerface3-2-new' --model_path './code/exps/1-nerface/checkpoint/checkpoint.pt' --cat_video --dataset_type 'test' --suffix '.png' --latent_dim_shape 50
```

## Citation ##
Please cite the following paper if you use this repository in your reseach.
```
@InProceedings{Bai_2023_CVPR,
    author    = {Bai, Yunpeng and Fan, Yanbo and Wang, Xuan and Zhang, Yong and Sun, Jingxiang and Yuan, Chun and Shan, Ying},
    title     = {High-Fidelity Facial Avatar Reconstruction From Monocular Video With Generative Priors},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {4541-4551}
}



