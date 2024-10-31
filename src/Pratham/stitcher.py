import numpy as np
import random
import glob
import cv2
import os

class PanoramaStitcher:
    def __init__(self):
        self.focal_length = 1000  # Default focal length for cylindrical projection

    def cylindrical_warp(self, img):
        h, w = img.shape[:2]
        K = np.array([[self.focal_length, 0, w/2], 
                     [0, self.focal_length, h/2],
                     [0, 0, 1]], dtype=np.float32)
        
        # Create meshgrid of coordinates
        y_i, x_i = np.indices((h, w))
        X = np.stack([x_i, y_i, np.ones_like(x_i)], axis=-1).reshape(h*w, 3)
        
        # Convert to cylindrical coordinates
        X = X.T
        x_normalized = X[0] - K[0,2]
        y_normalized = X[1] - K[1,2]
        theta = x_normalized / K[0,0]
        h_i = K[1,1] * np.tan(y_normalized / np.sqrt(K[0,0]**2 + x_normalized**2))
        x_proj = K[0,2] + K[0,0] * theta
        y_proj = K[1,2] + h_i
        
        # Reshape back to image
        x_proj = x_proj.reshape(h, w).astype(np.float32)
        y_proj = y_proj.reshape(h, w).astype(np.float32)
        
        # Remap image
        warped = cv2.remap(img, x_proj, y_proj, cv2.INTER_LINEAR)
        
        # Mask out invalid regions
        mask = (x_proj >= 0) & (x_proj < w) & (y_proj >= 0) & (y_proj < h)
        mask = mask.astype(np.uint8)
        warped = warped * mask[:,:,np.newaxis]
        
        return warped

    def make_panorama_for_images_in(self, path, reference_index=None):
        image_files = sorted(glob.glob(path + os.sep + '*'))
        print(f"Found {len(image_files)} images for panorama creation.")
        
        if len(image_files) < 2:
            raise ValueError("Stitching requires at least two images.")
            
        if reference_index is None:
            reference_index = len(image_files) // 2
            
        # Read and warp all images
        images = []
        for img_path in image_files:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Unable to load image at {img_path}, skipping.")
                continue
            img = cv2.resize(img, None, fx=0.5, fy=0.5)
            warped_img = self.cylindrical_warp(img)
            images.append(warped_img)
            
        # Start from reference image
        result = images[reference_index]
        homography_matrices = []
        
        # Stitch images to the right
        for i in range(reference_index + 1, len(images)):
            result, H = self.stitch_images(result, images[i], direction='right')
            homography_matrices.append(H)
            
        # Stitch images to the left
        left_result = images[reference_index]
        for i in range(reference_index - 1, -1, -1):
            left_result, H = self.stitch_images(images[i], left_result, direction='left')
            homography_matrices.append(H)
            
        # Combine left and right parts if necessary
        if reference_index > 0:
            result, H = self.stitch_images(left_result, result, direction='right')
            homography_matrices.append(H)
            
        cv2.imwrite('stitched_panorama_result.jpg', result)
        print("Panorama image saved as 'stitched_panorama_result.jpg'.")
        
        return result, homography_matrices

    def stitch_images(self, left_image, right_image, direction='right'):
        kp1, desc1, kp2, desc2 = self.obtain_the_key_points(left_image, right_image)
        keypoint_matches = self.align_and_match_feature_points(kp1, kp2, desc1, desc2)
        optimal_H = self.implementated_Ransac(keypoint_matches)
        
        img_height1, img_width1 = right_image.shape[:2]
        img_height2, img_width2 = left_image.shape[:2]
        
        corners_image1 = np.float32([[0, 0], [0, img_height1], [img_width1, img_height1], [img_width1, 0]]).reshape(-1, 1, 2)
        corners_image2 = np.float32([[0, 0], [0, img_height2], [img_width2, img_height2], [img_width2, 0]]).reshape(-1, 1, 2)
        
        transformed_corners = cv2.perspectiveTransform(corners_image2, optimal_H)
        combined_corners = np.concatenate((corners_image1, transformed_corners), axis=0)
        
        [x_min, y_min] = np.int32(combined_corners.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(combined_corners.max(axis=0).ravel() + 0.5)
        
        translation_matrix = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]]).dot(optimal_H)
        
        warped = self.apply_warp(left_image, translation_matrix, x_max - x_min, y_max - y_min)
        
        # Create masks for blending
        mask1 = np.zeros((y_max - y_min, x_max - x_min), dtype=np.float32)
        mask1[(-y_min):img_height1 + (-y_min), (-x_min):img_width1 + (-x_min)] = 1
        
        mask2 = np.ones_like(warped[:,:,0], dtype=np.float32)
        
        # Apply Gaussian blur to masks for smooth blending
        blur_size = 51  # Must be odd
        mask1 = cv2.GaussianBlur(mask1, (blur_size, blur_size), 0)
        mask2 = cv2.GaussianBlur(mask2, (blur_size, blur_size), 0)
        
        # Normalize masks
        mask1 = mask1 / (mask1 + mask2 + 1e-10)
        mask2 = mask2 / (mask1 + mask2 + 1e-10)
        
        # Apply masks and blend
        warped_with_mask = warped * mask2[:,:,np.newaxis]
        right_with_mask = np.zeros_like(warped)
        right_with_mask[(-y_min):img_height1 + (-y_min), (-x_min):img_width1 + (-x_min)] = \
            right_image * mask1[(-y_min):img_height1 + (-y_min), (-x_min):img_width1 + (-x_min), np.newaxis]
        
        blended = warped_with_mask + right_with_mask
        
        return blended, optimal_H

    def obtain_the_key_points(self, left_image, right_image):
        grayscale_left = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        grayscale_right = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
        
        sift_detector = cv2.SIFT_create()
        kp1, desc1 = sift_detector.detectAndCompute(grayscale_left, None)
        kp2, desc2 = sift_detector.detectAndCompute(grayscale_right, None)
        
        return kp1, desc1, kp2, desc2

    def align_and_match_feature_points(self, kp1, kp2, desc1, desc2):
        flann_index_params = dict(algorithm=1, trees=5)
        flann_search_params = dict(checks=50)
        flann_matcher = cv2.FlannBasedMatcher(flann_index_params, flann_search_params)
        
        knn_matches = flann_matcher.knnMatch(desc1, desc2, k=2)
        filtered_matches = []
        
        for match_1, match_2 in knn_matches:
            if match_1.distance < 0.75 * match_2.distance:
                left_coords = kp1[match_1.queryIdx].pt
                right_coords = kp2[match_1.trainIdx].pt
                filtered_matches.append([left_coords[0], left_coords[1], right_coords[0], right_coords[1]])
        
        return filtered_matches

    def calculate_homography(self, matched_points):
        equation_matrix = []
        for match in matched_points:
            x, y = match[0], match[1]
            X, Y = match[2], match[3]
            equation_matrix.append([x, y, 1, 0, 0, 0, -X * x, -X * y, -X])
            equation_matrix.append([0, 0, 0, x, y, 1, -Y * x, -Y * y, -Y])
        
        equation_matrix = np.array(equation_matrix)
        _, _, v_transpose = np.linalg.svd(equation_matrix)
        homography_matrix = (v_transpose[-1, :].reshape(3, 3))
        homography_matrix = homography_matrix / homography_matrix[2, 2]
        return homography_matrix

    def implementated_Ransac(self, matched_points):
        max_inliers = []
        best_homography = []
        threshold_dist = 5
        num_iterations = 50
        
        for _ in range(num_iterations):
            sample_points = random.sample(matched_points, k=4)
            H = self.calculate_homography(sample_points)
            current_inliers = []
            
            for pt in matched_points:
                origin_pt = np.array([pt[0], pt[1], 1]).reshape(3, 1)
                target_pt = np.array([pt[2], pt[3], 1]).reshape(3, 1)
                transformed_pt = np.dot(H, origin_pt)
                transformed_pt /= transformed_pt[2]
                point_distance = np.linalg.norm(target_pt - transformed_pt)
                
                if point_distance < threshold_dist:
                    current_inliers.append(pt)
            
            if len(current_inliers) > len(max_inliers):
                max_inliers, best_homography = current_inliers, H
        
        return best_homography