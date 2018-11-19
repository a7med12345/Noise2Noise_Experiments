# Noise2Noise_Experiments

A-  Network Architecture:
![Alt text](images/Network.png?raw=true "Network architecture")


B- Text to Text Experiment:
    
B-1 Example of network output after 200 epochs:

     a)
     * Input image; Noisy target; Output of noise2noise network 
       output of noise2clean network; Clean target

![Alt text](images/t2t.png?raw=true "Network output")

     b)
     * Input image; Noisy target; Output of noise2noise network 
       output of noise2clean network; Clean target
![Alt text](images/t2t2.png?raw=true "Network output")


B-2 Training curves:

* PSNR_clean: psnr of network trained on clean target
* PSNR_n2n: psnr of network trained on noisy target
![Alt text](images/t2t.svg?raw=true "training curve")

B-3 Test images results:

* Noise 2 Noise

![Alt text](results/t2t/test_latest/images/1000_A_fake_B_n.png?raw=true "test image noise 2 noise")

* Noise 2 Clean

![Alt text](results/t2t/test_latest/images/1000_A_fake_B_c.png?raw=true "test image noise 2 clean")

* Input Image

![Alt text](results/t2t/test_latest/images/1000_A_real_A1.png?raw=true "test image noise 2 noise")


C- Rain to Rain Experiment:
    
C-1 Example of network output after 200 epochs:

     a)
     * Input image; Noisy target; Output of noise2noise network 
       output of noise2clean network; Clean target

![Alt text](images/r2n_example.png?raw=true "Network output")

     b)
     * Input image; Noisy target; Output of noise2noise network 
       output of noise2clean network; Clean target
![Alt text](images/r2r2.png?raw=true "Network output")


C-2 Training curves:

* PSNR_clean: psnr of network trained on clean target
* PSNR_n2n: psnr of network trained on noisy target
![Alt text](images/r2r_training.svg?raw=true "training curve")

C-3 Test images results:

* Noise 2 Noise

![Alt text](results/r2r/test_latest/images/1000_3_fake_B_n.png?raw=true "test image noise 2 noise")

* Noise 2 Clean

![Alt text](results/r2r/test_latest/images/1000_3_fake_B_c.png?raw=true "test image noise 2 clean")

* Input Image

![Alt text](results/r2r/test_latest/images/1000_3_real_A1.png?raw=true "test image noise 2 noise")




D- Testing the network:
