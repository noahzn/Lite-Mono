# Lite-Mono
This is the official PyTorch Implementation of **Lite-Mono: A Lightweight CNN and Transformer Architecture for Self-Supervised Monocular Depth Estimation**



If you find our work useful please cite our paper.
>@article{zhang2022lite,  
>title={Lite-Mono: A Lightweight CNN and Transformer Architecture for Self-Supervised Monocular Depth Estimation},  
>author={Zhang, Ning and Nex, Francesco and Vosselman, George and Kerle, Norman},  
>journal={arXiv preprint arXiv:2211.13202},
>year={2022}  
>}




## KITTI data
Please refer to [Monodepth2](https://github.com/nianticlabs/monodepth2) to preapre your KITTI data. 



## Results on KITTI
|     --model     | Params | ImageNet Pretrained | Input size |  Abs Rel  |   Sq Rel  |    RMSE   |  RMSE log | delta < 1.25 | delta < 1.25^2 | delta < 1.25^3 |
|:---------------:|:------:|:-------------------:|:----------:|:---------:|:---------:|:---------:|:---------:|:------------:|:--------------:|:--------------:|
|  [**lite-mono**](https://surfdrive.surf.nl/files/index.php/s/CUjiK221EFLyXDY)  |  3.1M  |         yes         |   640x192  | **0.107** | **0.765** | **4.561** | **0.183** |   **0.886**  |    **0.963**   |    **0.983**   |
| [lite-mono-small](https://surfdrive.surf.nl/files/index.php/s/8cuZNH1CkNtQwxQ) |  2.5M  |         yes         |   640x192  |   0.110   |   0.802   |   4.671   |   0.186   |     0.879    |      0.961     |      0.982     |
|  lite-mono-tiny |  2.2M  |         yes         |   640x192  |   0.110   |   0.837   |   4.710   |   0.187   |     0.880    |      0.960     |      0.982     |
|  [**lite-mono**](https://surfdrive.surf.nl/files/index.php/s/IK3VtPj6b5FkVnl)  |  3.1M  |         yes         |  1024x320  | **0.102** | **0.746** | **4.444** | **0.179** |   **0.896**  |    **0.965**   |    **0.983**   |
| [lite-mono-small](https://surfdrive.surf.nl/files/index.php/s/w8mvJMkB1dP15pu) |  2.5M  |         yes         |  1024x320  |   0.103   |   0.757   |   4.449   |   0.180   |     0.894    |      0.964     |      0.983     |
|  lite-mono-tiny |  2.2M  |         yes         |  1024x320  |   0.104   |   0.764   |   4.487   |   0.180   |     0.892    |      0.964     |      0.983     |


## Evaluation
You can evaluate on KITTI using the following command.  
`python evaluate_depth.py --load_weights_folder path/to/your/weights/folder --data_path path/to/kitti_data/ --model lite-mono`  

 
 
The training code will released soon.