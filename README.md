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

29_n12: This set is nice, but its not the correct focusing paramater, the n12 is too close and it looks like we are passing the focal point. You can see the beam focus, then diverge again. 

29_n10: This focusing looks really nice, the problem is that the alignment will need some help. The 0th order of diffraction leaks in a lot from the SLM. In order to clean this up, I will have to spatially seperat the orders of diffraction by increasing the x angle of the LG modes. I was looking at increasing this to 15, in this data set its at 13. You will notice that due to the increased space, the experiment needs to be realigned because the 1st order of diffraction is cut by the mirrors. 

FOR REALIGNMENT: The beam coming off the SLM should not directly hit the center of the mirror, instead, it should prioritize the 1st order of diffraction coming off the SLM so that it doesnt get cut later in the experiment. 

TESTED WITH JAMES: James and I attempted to place a polarizing beam splitter to minimize the intensity of the 0th order since it comes in as mixed polarization. This did not work effectively, it dimmed the first order, but not enought to prevent the leakage. We also tried placing a half wave plate which also did not work. 

AN IDEA: To test the polarization control before continuing with spatial control, use the half wave plate to see if I can make the 0th order extinct, just before the final lens. This may also dim the 1st order, but then I can increase the eexposure of the camera for data collection. 

OUTCOME: I was unable to get extinction of the zeroth order, so it seems the isloation has to be done spatially rather than through phase. 

## Data from June 15, 2026

29_n14: This looks pretty good, but there is definitely some leakage from the -1 order of diffraction now that I increased the xAngle. The angle is now 17 mRad. I think the outermost ring was also from maybe the 0th order. 

29_n14_pt2: I collected this set without moving the camera or lenses, everything was the same except I moved the iris closer to the first lens of the 4f system (400mm). It seems this is doing a better job of isolating the 1st order of diffraction than in the far field. I think this may be because the other orders dont get the opportunity to seperate and interfere, they are blocked sooner. The thing I noticed in this data set is that the exposure on the camera may be too high and saturating. 

CODE UPDATES: I noticed the figures were being generated to scale exposure based on its own image rather than all the images in general, this makes it visually look like the ring has the same brightness in each z plane, which is not true. I updated the plot code to ensure we can visually see the intensity growing as the ring focuses in the further z planes. 

INPUT FROM EBRAHIM: The center of the singularity is not perfectly aligned. I should work on moving the hologram in the x direction to make the donut shape of the OAM beams perfectly symmetric. The images seem saturated at the focus (this is why I updated the code, this is not the case when watching it live, it was just the way the images were generated with the inferno color scale). However, I will still collect another set of data with a lower exposure to ensure the camera is not saturated and we can measure the full affect of the focusing. 

MEET WITH FARID: Discuss the simmulation Farid has. We should compare the experimental paramaters with the simulation so that the simulation is an accurate representation of the expected data. 

NOTE ON PARAMETERS: I have been using a beam size of 29 since it is the largest generated hologram we have. This is essential to use the most of the SLM screen so we can use the most pixels and get the highest resolution for the images.  

## Data from June 16, 2026

29_n14_pt3: I collected this data set similar to 29_n14_pt2, but I lowered the exposure to 700ms from 4000ms. I suspect that if the camera was saturated it may have blocked true intensities in some frames. 

NEW SETS: I would like to collect another set with n15 to see if the focus is passed. This should tell me if n14 is actually the correct focusing hologram for the beam size 29, if the focus is truly reached, the beam should re diverge past that point. This is limited since I can only move 25 mm with the stage.

## Data from June 17, 2026 

29_n15: In progress
