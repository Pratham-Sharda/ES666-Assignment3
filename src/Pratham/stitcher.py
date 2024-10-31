import numpy as np
import random
import glob
import cv2
import os

class PanaromaStitcher:
    def __init__(self):
        # Configuration parameters
        self.resize_percentage = 30
        self.ransac_iterations = 100  # Increased from 50
        self.ransac_threshold = 4.0    # Refined threshold
        self.match_ratio = 0.7         # Slightly more strict matching
        self.min_matches = 10          # Minimum matches required
        self.blend_width = 50          # For feathering blend

    def make_panaroma_for_images_in(self, path):
        image_files = sorted(glob.glob(path + os.sep + '*'))
        print(f"Found {len(image_files)} images for panorama creation.")

        if len(image_files) < 2:
            raise ValueError("Stitching requires at least two images.")

        # Read and preprocess first image
        pano_result = cv2.imread(image_files[0])
        if pano_result is None:
            raise ValueError(f"Unable to load first image: {image_files[0]}")
            
        # Apply color correction
        pano_result = self.adjust_gamma(pano_result, 1.2)
        pano_result = cv2.resize(pano_result, None, 
                               fx=self.resize_percentage / 100, 
                               fy=self.resize_percentage / 100)

        homography_matrices = []
        for current_image in image_files[1:]:
            next_image = cv2.imread(current_image)
            if next_image is None:
                print(f"Warning: Unable to load image at {current_image}, skipping.")
                continue
            
            # Apply same preprocessing to next image
            next_image = self.adjust_gamma(next_image, 1.2)
            next_image = cv2.resize(next_image, None, 
                                  fx=self.resize_percentage / 100, 
                                  fy=self.resize_percentage / 100)
            
            try:
                pano_result, homography_matrix = self.Stitch_2_image_and_matrix_return(pano_result, next_image)
                homography_matrices.append(homography_matrix)
            except Exception as e:
                print(f"Warning: Stitching failed for image {current_image}: {str(e)}")
                continue

        # Final color balance and contrast adjustment
        pano_result = self.enhance_final_image(pano_result)
        
        cv2.imwrite('stitched_panorama_result.jpg', pano_result)
        print("Panorama image saved as 'stitched_panorama_result.jpg'.")

        return pano_result, homography_matrices

    def adjust_gamma(self, image, gamma=1.0):
        """Apply gamma correction for better exposure."""
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                         for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    def enhance_final_image(self, image):
        """Enhance the final panorama with better contrast and color."""
        # Convert to LAB color space for better color manipulation
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        # Merge channels and convert back to BGR
        lab = cv2.merge((l,a,b))
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Increase saturation slightly
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = cv2.multiply(s, 1.2)  # Increase saturation by 20%
        hsv = cv2.merge([h, s, v])
        enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return enhanced

    def obtain_the_key_points(self, left_image, right_image):
        # Convert to grayscale
        grayscale_left = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        grayscale_right = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

        # Apply contrast enhancement before feature detection
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        grayscale_left = clahe.apply(grayscale_left)
        grayscale_right = clahe.apply(grayscale_right)

        # Use SIFT with adjusted parameters for better feature detection
        sift_detector = cv2.SIFT_create(
           
            nOctaveLayers=3,  # increased from default 3
            contrastThreshold=0.04,  # decreased from default 0.04
            edgeThreshold=10,  # default is 10
            sigma=1.6  # default is 1.6
        )

        kp1, desc1 = sift_detector.detectAndCompute(grayscale_left, None)
        kp2, desc2 = sift_detector.detectAndCompute(grayscale_right, None)

        return kp1, desc1, kp2, desc2

    def align_and_match_feature_points(self, kp1, kp2, desc1, desc2):
        # Use better FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(desc1, desc2, k=2)
        filtered_matches = []

        # Apply ratio test with configurable ratio
        for m, n in matches:
            if m.distance < self.match_ratio * n.distance:
                left_coords = kp1[m.queryIdx].pt
                right_coords = kp2[m.trainIdx].pt
                filtered_matches.append([left_coords[0], left_coords[1], 
                                      right_coords[0], right_coords[1]])

        if len(filtered_matches) < self.min_matches:
            raise ValueError(f"Not enough matches found: {len(filtered_matches)}")

        return filtered_matches

    def implementated_Ransac(self, matched_points):
        if len(matched_points) < 4:
            raise ValueError("Not enough matches for RANSAC")

        max_inliers = []
        best_homography = None
        
        # Normalize points for better numerical stability
        pts_src = np.float32([[pt[0], pt[1]] for pt in matched_points])
        pts_dst = np.float32([[pt[2], pt[3]] for pt in matched_points])
        
        # Find homography using OpenCV's built-in RANSAC
        H, mask = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC, 
                                   ransacReprojThreshold=self.ransac_threshold,
                                   maxIters=self.ransac_iterations)
        
        if H is None:
            raise ValueError("Could not find valid homography")
            
        return H

    def apply_warp(self, source_img, homography_matrix, output_width, output_height):
        # Use OpenCV's warp perspective with border reflection
        warped = cv2.warpPerspective(
            source_img, 
            homography_matrix,
            (output_width, output_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )
        
        return warped

    def Stitch_2_image_and_matrix_return(self, left_image, right_image):
        # Get keypoints and descriptors
        kp1, desc1, kp2, desc2 = self.obtain_the_key_points(left_image, right_image)
        
        # Match features
        keypoint_matches = self.align_and_match_feature_points(kp1, kp2, desc1, desc2)
        
        # Find homography
        optimal_H = self.implementated_Ransac(keypoint_matches)
        
        # Calculate output dimensions
        h1, w1 = right_image.shape[:2]
        h2, w2 = left_image.shape[:2]
        
        # Calculate corners of warped image
        corners = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
        warped_corners = cv2.perspectiveTransform(corners, optimal_H)
        all_corners = np.concatenate((
            np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2),
            warped_corners
        ))
        
        # Calculate dimensions of output image
        [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
        
        # Adjust transformation matrix
        translation = np.array([
            [1, 0, -x_min],
            [0, 1, -y_min],
            [0, 0, 1]
        ])
        optimal_H = translation.dot(optimal_H)
        
        # Create output image
        output = self.apply_warp(left_image, optimal_H, x_max - x_min, y_max - y_min)
        
        # Create a mask for blending
        mask = np.zeros((y_max - y_min, x_max - x_min), dtype=np.float32)
        mask[(-y_min):h1 + (-y_min), (-x_min):w1 + (-x_min)] = 1
        
        # Apply feathering blend
        mask = cv2.GaussianBlur(mask, (self.blend_width * 2 + 1, self.blend_width * 2 + 1), 0)
        
        # Blend images
        warped_img = output.astype(np.float32)
        img2_warped = np.zeros_like(warped_img)
        img2_warped[(-y_min):h1 + (-y_min), (-x_min):w1 + (-x_min)] = right_image
        
        # Blend using the mask
        blended = img2_warped * mask[:, :, np.newaxis] + warped_img * (1 - mask[:, :, np.newaxis])
        
        return blended.astype(np.uint8), optimal_H