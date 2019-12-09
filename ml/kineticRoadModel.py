import numpy as np
import matplotlib.pyplot as plt

pi = np.pi

class Vehicle:
	""" spawn vehicle with all the specifications """

	def __init__(self, length=4, width=2, speed=2, heading=pi/2, lane=1, y_co=0):
		""" this will create a vehicle with specified features """

		self.lane_width = 3
		self.lane_change= 0 
		self.length 	= length
		self.width 		= width
		self.speed  	= speed
		self.heading	= heading
		self.y_co 		= y_co
		self.x_co 		= self.lane_width * (lane - 0.5)
		self.n_hood 	= np.array([])

	def move(self, time):
		""" it will change the vehicle location """

		if self.lane_change == 0:
			dx = speed * np.cos(self.heading) * time
			dy = speed * np.sin(self.heading) * time
			self.x_co += dx
			self.y_co += dy

		elif self.lane_change == 1:
			self.speed = 0

class Environment:
	""" this will create environment of traffic """

	def __init__(self, lane_n=4, inner_r=20, width=3):
		""" ajsdlfja;lsdj"""

		self.lane_no = lane_n
		self.inner_r = inner_r
		self.lane_width = width
		self.waypoints_x = None
		self.waypoints_y = None
		self.points = 100

	def makeRoad(self):
		""" create all the waypoints of road """

		indx = np.linspace(0, pi, self.points)
		for lane in range(self.lane_no + 1):
			radius = self.inner_r + lane * self.lane_width
			x_co = radius * np.cos(indx)
			y_co = radius * np.sin(indx)
			if self.waypoints_x is None:
				self.waypoints_x = x_co
				self.waypoints_y = y_co
			else:
				self.waypoints_x = np.vstack((self.waypoints_x, x_co))
				self.waypoints_y = np.vstack((self.waypoints_y, y_co))

	def vehicleWaypoints(self, lane, speed):
		""" vehicle path waypoints """
		radius = self.inner_r + self.lane_width * (lane - 0.5)
		points = self.points / speed
		indx = np.linspace(0, pi, points)
		x_co = radius * np.cos(indx)
		y_co = radius * np.sin(indx)
		heading = indx + pi/2
		self.waypoint = np.vstack((x_co, y_co, heading))
		#plt.plot(x_co, y_co, 'r.')

	def laneChange(self, lane, speed):
		""" creats waypoints of new path of vehicle """
		radius = self.inner_r + self.lane_width * (lane - 0.5)
		points = self.points / speed
		indx = np.linspace(0, pi, points)
		heading = indx + pi/2
		x_co = radius * np.cos(indx)
		y_co = radius * np.sin(indx)
		waypoint = np.vstack((x_co, y_co, heading))
		self.lane_change_time = 10
		current_time = 25
		new_point1 = self.waypoint[:, 0:25]
		new_point2 = waypoint[:, 35:]
		disp_x = new_point2[0, 0] - new_point1[0, 24]
		disp_y = new_point2[1, 0] - new_point1[1, 24]
		new_heading = 1.5 * np.arctan(disp_y/disp_x)
		heading1 = np.linspace(new_point1[2, 24], new_heading, 5)
		heading2 = np.linspace(new_heading, new_point2[0, 2], 5)
		new_x_co = np.linspace(new_point1[0, 24], new_point2[0, 0], 10)
		new_y_co = np.linspace(new_point1[1, 24], new_point2[1, 0], 10)
		lane_change_point = np.vstack((new_x_co, new_y_co, np.hstack((heading1, heading2))))
		self.new_waypoint = np.hstack((new_point1, lane_change_point, new_point2))
		print(self.new_waypoint.shape, self.waypoint.shape, waypoint.shape)
		#plt.plot(self.waypoint[0], self.waypoint[1], 'b')
		#plt.plot(self.new_waypoint[0], self.new_waypoint[1], 'r--')
		#plt.plot(waypoint[0], waypoint[1], 'k')
		r = np.sqrt(new_x_co**2 + new_y_co**2)
		x = r * np.cos(np.hstack((heading1, heading2)))
		y = r * np.cos(np.hstack((heading1, heading2)))
		#plt.plot(x, y, 'r.')
		#plt.show()

	def animateVehicle(self, time):
		""" cars animation """
		center = self.new_waypoint[:, time]
		length = 2.0
		width = 1.0
		t_matrix = np.matrix([[np.cos(center[2]), np.sin(center[2])], [-np.sin(center[2]), np.cos(center[2])]])
		x1 = -length/2
		x2 = length/2
		y1 = -width/2
		y2 = width/2

		xtrans = center[0] * np.ones((1, 4))
		ytrans = center[1] * np.ones((1, 4))
		trans = np.vstack((xtrans, ytrans))
		print(trans)
		rect = np.array([[x1, x1, x2, x2], [y1, y2, y2, y1]])
		vehicle = t_matrix.dot(rect) + trans
		x_co = list(vehicle[0].flat)
		y_co = list(vehicle[1].flat)
		plt.fill(x_co, y_co)
		plt.show()

	def plotRoad(self):
		""" plot the road """
		for i in range(self.lane_no + 1):
			if i == 0 or i == self.lane_no:
				plt.plot(self.waypoints_x[i], self.waypoints_y[i], 'b', linewidth=5)
			else:
				plt.plot(self.waypoints_x[i], self.waypoints_y[i], 'k--')

		#plt.axes().set_aspect('equal')
		plt.show()

def main():
	plt.ion()
	env = Environment()
	env.makeRoad()
	env.plotRoad()
	env.vehicleWaypoints(2, 2)
	env.laneChange(1, 2)
	env.animateVehicle(10)


if __name__ == "__main__":
	main()
