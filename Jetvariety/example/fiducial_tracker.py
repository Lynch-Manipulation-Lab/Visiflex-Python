import numpy as np
import cv2 as cv
import math

DEG2RAD = .01745329252
RAD2DEG = 57.29577951

# Camera Parameters
FIDUCIAL_RING_RADIUS = 11.8792
DOME_FRAME_OFFSET = 3.25
DOME_EXTERNAL_RADIUS = 12.7
PI = math.pi

def angle_diff(x,y):
	tmp = math.fmod(x-y,360.0)
	if tmp < 0.0:
		tmp+=360.0
	return min(tmp, 360.0-tmp)

class FiducialTracker:

	################################# Class Variables ##############################################
		
	historyFiducials = [] 
	seenModelPoints3D = []
	visibleIDs = []

	radAngle = 45*PI/180.0
	r = FIDUCIAL_RING_RADIUS

	fiducial_roi_size = 75

	#Reference Points
	c = math.cos(radAngle)
	s = math.sin(radAngle)
	z = DOME_FRAME_OFFSET
	model_points_3D = [[r,0,z],[r*c,r*s,z],[0,r,z],[-r*c,r*s,z],[-r,0,z],[-r*c,-r*s,z],[0,-r,z],[r*c,-r*s,z]]
	#cameraMatrix = cv.Mat()
	#cameraDistCoeffs = cv.Mat()
	gpuChannelBuff = cv.cuda_GpuMat()
	gpuRedMask = cv.cuda_GpuMat()
	#################################   Functions     ##############################################
	
	def gpuGetRedMask(self, hsvChannels):
		hsv_img = cv.cuda_GpuMat(hsvChannels)		
		hsv_img = cv.cuda.cvtColor(hsv_img, cv.COLOR_BGR2HSV)
		shsv = cv.cuda_GpuMat()
		h,s,v = cv.cuda.split(hsv_img)
		h1 = cv.cuda.threshold(h,20,180,cv.THRESH_BINARY_INV)
		h2 = cv.cuda.threshold(h,170,180,cv.THRESH_BINARY)
		h = cv.cuda.bitwise_or(h1[1],h2[1])
		s = cv.cuda.threshold(s,20,255,cv.THRESH_BINARY)
		v = cv.cuda.threshold(v,80,255,cv.THRESH_BINARY)
		temp = cv.cuda.bitwise_and(h,s[1])
		hsv = cv.cuda.bitwise_and(temp,v[1])
		hsv = hsv.download()
		return hsv

	def findFiducials(self,redMask,centroids):
		
		contours, hierarchy = cv.findContours(redMask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
		
		for contour in contours:
			if cv.contourArea(contour) > 1000:
				M = cv.moments(contour)
				xVal = int(M['m10']/M['m00'])
				yVal = int(M['m01']/M['m00'])
				centroids.append([xVal,yVal])
				
	def updateFiducials(self,hsvChannels):
		
		# Find red pixels in image and download to CPU
		self.gpuRedMask = self.gpuGetRedMask(hsvChannels)
		
		if len(self.historyFiducials) < 8:
			newFiducials = []
			self.findFiducials(self.gpuRedMask,newFiducials)

			if len(newFiducials) < 5:
				self.historyFiducials.clear()
				self.seenModelPoints3D.clear()
				return

			# If there is no history, or there are more points now than before
			print(self.historyFiducials)
			if not self.historyFiducials or len(newFiducials) > len(self.historyFiducials):
				print("Entered the if not part!")
				self.historyFiducials.clear()
				self.seenModelPoints3D.clear()
				
				domeEllipse = cv.fitEllipse(np.array(newFiducials,dtype=np.int32))
				domeCenter = domeEllipse[0]
				# I think the first parameter returned is center, need to check if it is second
				fiducialAngles = []
				for i in range(len(newFiducials)):
					fiducialAngles.append(RAD2DEG*(math.atan2(newFiducials[i][1]-domeCenter[1],newFiducials[i][0]-domeCenter[0])))

				firstFiducialID = -1
				minDeviation = 100  # Initialized large

				for i in range(len(newFiducials)):
					deviation = math.fabs(fiducialAngles[i] - round(fiducialAngles[i]/45.0)*45.0)
					if firstFiducialID == -1 or deviation < minDeviation:
						firstFiducialID = i
						firstModelPointID = round(fiducialAngles[i]/45.0)
						if firstModelPointID < 0:
							firstModelPointID += 8
						minDeviation = deviation

				fiducialIDs = [-1]*8 # Creates array of 8 values, all -1
				fiducialIDs[firstModelPointID] = firstFiducialID;
				closestModelPoints = [0]*len(newFiducials)

				for i in range(len(newFiducials)):
					closestModelPoints[i] = (round((fiducialAngles[i]-fiducialAngles[firstFiducialID])/45.0)+ firstModelPointID) % 8
					if closestModelPoints[i] < 0:
						closestModelPoints[i]+=8
					difference1 = angle_diff(fiducialAngles[i]-fiducialAngles[firstFiducialID],45.0*closestModelPoints[i])
					difference2 = angle_diff(fiducialAngles[fiducialIDs[closestModelPoints[i]]]-fiducialAngles[firstFiducialID],45.0*closestModelPoints[i])

					if fiducialIDs[closestModelPoints[i]]==-1 or difference1 < difference2:
						fiducialIDs[closestModelPoints[i]] = i
				
				for i in range(8):
					if fiducialIDs[i]==-1:
						continue
					self.historyFiducials.append(newFiducials[fiducialIDs[i]])
					self.seenModelPoints3D.append(self.model_points_3D[i])
				print(self.historyFiducials)
			else:
				domeEllipse = cv.fitEllipse(np.array(newFiducials,dtype=np.int32))
				domeCenter = domeEllipse[0]

				counter = 0
				while counter < len(self.historyFiducials) and len(newFiducials)>0:
					nearest = -1
					minDistance = 1000
					tempFiducials = []
					for k in range(len(newFiducials)):
						distance = cv.norm(newFiducials[k]-self.historyFiducials[counter])
						if nearest==-1 or distance < minDistance:
							nearest = k
							minDistance = distance
					tempFiducials.append(newFiducials[k])
					counter += 1
				newFiducials = tempFiducials

				if len(newFiducials)==0:
					return
		
				firstFiducialAngle = math.atan2(self.historyFiducials[0][1]-domeCenter[1],historyFiducials[0][0]-domeCenter[0])
				firstModelPointID = model_points_3D.index(seenModelPoints3D[0])
				
				fiducialAngles = [-1]*len(newFiducials)
				for i in range(len(newFiducials)):
					fiducialAngles[i] = RAD2DEG*(math.atan2(newFiducials[i][1]-domeCenter[1],newFiducials[i][0]-domeCenter[0]))
				
				closestModelPoints = []*len(newFiducials)
				for i in range(len(newFiducials)):
					closestModelPoints[i] = (round((fiducialAngles[i]-firstFiducialAngle)/45.0)+firstModelPointID)%8
					if closestModelPoints[i]<0:
						closestModelPoints[i]+=8
	
			
				self.historyFiducials.clear()
				self.historyFiducials = newFiducials

				self.seenModelPoints3D.clear()
				
				for i in range(len(closestModelPoints)):
					self.seenModelPoints3D.append(model_points_3D[closestModelPoints[i]])

			old_history_size = len(self.historyFiducials)
		else: # If there are 8 history fiducials
			
			tempHistory = []
			current = 0
			for dot in self.historyFiducials:
				centerX = dot[0]
				centerY = dot[1]
				x1 = centerX-self.fiducial_roi_size/2
				x2 = centerX+self.fiducial_roi_size/2
				y1 = centerY+self.fiducial_roi_size/2
				y2 = centerY-self.fiducial_roi_size/2

				newFiducials = []
				self.findFiducials(self.gpuRedMask[x1:x2,y1:y2],newFiducials)

				if len(newFiducials)!=0:
					tempHistory.append(dot)
					nearest = -1
					minDistance = 1000
					for point in newFiducials:
						distance = cv.norm(point-dot)
						if nearest == -1 or distance < minDistance:
							nearest = newFiducials.index(point)
							minDistance = distance
					self.historyFiducials[current] = newFiducials[nearest]
					current += 1
				else:
					current+= 1
			self.historyFiducials = tempHistory

		if len(self.historyFiducials) < 5: #Too few fiducials, might be errors
			self.historyFiducials.clear()
			self.seenModelPoints3D.clear()
			self.visibleIDs.clear()
			return

		if 
