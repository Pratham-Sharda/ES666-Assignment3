In my code I am stitching left and right and then blending the image on left and right to create a Panorama image together,


# Note: When running the code one of the main issue is that only for image I1 ,the central image index = 2 works properly, while for others good image is obtained if we choose the refernce image = 3
This is because in image I1 there are 6 images thus it handles well when 2 is chosen while in others due to 5 images central image index = 3 works better


##  Due to Computation issues , output of image 2 is not visible in colab notebook and it is visible in trial2 python notebook

References:

https://kushalvyas.github.io/stitching.html
https://github.com/Avinash793/panoramic-image-stitching/blob/master/panorama.py
https://github.com/sajmaru/Image-Stitching-OpenCV
https://fpcv.cs.columbia.edu/
https://www.michelliao.com/content/files/2024/01/image_stitching_notes.pdf
