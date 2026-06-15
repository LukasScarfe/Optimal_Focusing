# Optimal_Focusing

The data/focusing_images folder has the images that were collected as data 

## Data from June 11, 2026

Each file collected is a tif file that was collected in step sizes of 0.5 cm. The step size can be decreased to have more data points, but for now the smaller step size is kept for nchoosing a method of data analysis. 

There is one folder for the measurement of each OAM mode, ranging from LG mode 0 to LG mode 1. The data set was collected for beam size 20 mm and focusing paramater n=8. 

The camera started in the image plane of the SLM (position 0000), then was moved in steps of 0.5 mm to final position, (position 0050), which is the focal point of the hologram. 

I chose the focusing paramater n=8 because the focal length was appropriate for the limited range of the actuator moving the stage (25mm). 

## Data from June 12, 2026

I collected 3 sets of data, none of them are perfect. 

18_n5: The beam size is too small so the resolution is very limited. It would be optimal to use the largest possible beam size (29) to get the highest resolution from the SLM. 

29_12: This set is nice, but its not the correct focusing paramater, the n12 is too close and it looks like we are passing the focal point. You can see the beam focus, then diverge again. 

29_n10: This focusing looks really nice, the problem is that the alignment will need some help. The 0th order of diffraction leaks in a lot from the SLM. In order to clean this up, I will have to spatially seperat the orders of diffraction by increasing the x angle of the LG modes. I was looking at increasing this to 15, in this data set its at 13. You will notice that due to the increased space, the experiment needs to be realigned because the 1st order of diffraction is cut by the mirrors. 

FOR REALIGNMENT: The beam coming off the SLM should not directly hit the center of the mirror, instead, it should prioritize the 1st order of diffraction coming off the SLM so that it doesnt get cut later in the experiment. 

TESTED WITH JAMES: James and I attempted to place a polarizing beam splitter to minimize the intensity of the 0th order since it comes in as mixed polarization. This did not work effectively, it dimmed the first order, but not enought to prevent the leakage. We also tried placing a half wave plate which also did not work. 

AN IDEA: To test the polarization control before continuing with spatial control, use the half wave plate to see if I can make the 0th order extinct, just before the final lens. This may also dim the 1st order, but then I can increase the eexposure of the camera for data collection. 


