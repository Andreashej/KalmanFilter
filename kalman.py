import numpy as np
import matplotlib.pyplot as plt

from pykalman import KalmanFilter as KF

class KalmanFilter:
    def __init__(self, dim, dt, measurements, controls, process_err, obs_err):
        self.X_k = np.zeros((1,3))
        self.dt = dt
        self.measurements = measurements
        self.controls = controls
        self.obs_err = obs_err
        self.process_err = process_err
        self.A = self.A(dim)
        self.B = self.B(dim)
        self.C = self.C(dim)
        self.R = self.covariance_matrix(obs_err)

        self.P_k = self.covariance_matrix(process_err)

    def kalman_gain(self, P_kp, H):
        '''
        Compute the kalman gain matrix. R is the measurement covariance matrix.
        '''

        n = P_kp.dot(H.T)
        D = H.dot(P_kp).dot(H.T) + self.R

        K = np.dot(n, np.linalg.inv(D))
        return K

    def predict(self, X, u):
        '''
        X is the previous state
        u is the control matrix
        '''

        X_k = self.A.dot(X) + self.B.dot(u)
        
        P_kp = self.A.dot(self.P_k).dot(self.A.T)
        return X_k, P_kp

    def update(self, X, Y, P_kp):
        '''
        Update the predicted state X based on a measurement Y. H is a matrix to convert process covanriance matrix into the correct format, in this case identity, since it is already correct.
        '''
        H = np.identity(X.shape[0])

        K = self.kalman_gain(P_kp, H)

        Y_k = self.get_observation(Y)

        X_k = X + K.dot(Y_k - H.dot(X))

        I = np.identity(K.shape[0])

        P_k = np.dot(I - K.dot(H), P_kp)

        return X_k, P_k
        

    def get_observation(self, Y):
        '''
        Return the measurement in the correct format for updating the state.
        '''
        return self.C.dot(Y)

    @staticmethod
    def covariance_matrix(err):
        errors = list()
        for error in err:
            errors.append(error**2)

        return np.diag(errors)
    
    def A(self, dim):
        if dim == 1:
            return np.array([
                [1, self.dt],
                [0, 1]
            ])
        elif dim == 2:
            return np.array([
                [1, 0, self.dt, 0],
                [0, 1, 0, self.dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])

    def B(self, dim):
        if dim == 1:
            return np.array([
                        [(1/2)*self.dt**2],
                        [self.dt]
                    ])
        elif dim == 2:
            return np.array([
                        [(self.dt**2) / 2, 0],
                        [0, (self.dt**2) / 2],
                        [self.dt, 0],
                        [0, self.dt]
                    ])
    
    def C(self, dim):
        '''
            In the case where the all the predicted values are observed, the C-matrix is simply an identity matrix.
        '''
        if dim == 1:
            dim = 2
        elif dim == 2:
            dim = 4
        elif dim == 3:
            dim = 9
        return np.identity(dim)

    
    # def covariance_matrix(self):
    #     dev_mat = self.deviation_matrix(self.measurements)
    #     return np.diag(np.diag(np.transpose(dev_mat).dot(dev_mat)))

    # def deviation_matrix(self, measurements):
    #     return measurements - np.ones((len(measurements), len(measurements))).dot(measurements * (1 / len(measurements)))
    
    def run(self):
        values = list()
        #Extract the initial value
        X_k, self.measurements, self.controls = self.measurements[0], self.measurements[1:], self.controls[1:]

        for i, obs in enumerate(self.measurements):
            X_k, P_kp = self.predict(X_k, self.controls[i])
            # print("Prediction: {}".format(X_k))
            
            X_k, self.P_k = self.update(X_k, obs, P_kp)
            # print("Update: {}".format(X_k))
            values.append(X_k)
        
        return np.array(values)

def load_dataset(observation_err):
    data = np.loadtxt("data.csv", delimiter=',')

    with_velocity = list()
    for i in range(len(data) - 1):
        x_velocity = (data[i + 1][0] - data[i][0]) / .1
        y_velocity = (data[i + 1][1] - data[i][1]) / .1

        with_velocity.append([data[i][0], data[i][1], x_velocity, y_velocity])
    
    with_noise = list()

    for row in with_velocity:
        x_noise = np.random.normal(row[0], observation_err[0])
        y_noise = np.random.normal(row[1], observation_err[1])
        x_velocity_noise = np.random.normal(row[2], observation_err[2])
        y_velocity_noise = np.random.normal(row[3], observation_err[3])

        with_noise.append([x_noise, y_noise, x_velocity_noise, y_velocity_noise])

    ground_truth = np.array(with_velocity)

    observations = np.array(with_noise)

    controls = np.zeros((len(observations), 2, 1))
    
    return ground_truth, observations.reshape((observations.shape[0], 4, 1)), controls


dt = .1 #time between measurements

observation_err = [25, 25, 14, 14]
process_err = [12, 12, 6, 6]

ground_truth, observations, controls = load_dataset(observation_err)

# observation_err = [100, 75, 100, 100]

kf = KalmanFilter(2, dt, observations, controls, process_err, observation_err)

values = kf.run()

# A = np.array([
#     [1, 0, dt, 0],
#     [0, 1, 0, dt],
#     [0, 0, 1, 0],
#     [0, 0, 0, 1]
# ])

# B = np.array([
#     [dt**2 / 2, 0],
#     [0, dt**2 / 2],
#     [dt, 0],
#     [0 ,dt]
# ])

# kf = KF(transition_matrices=A, observation_covariance=KalmanFilter.covariance_matrix(observation_err), transition_covariance=KalmanFilter.covariance_matrix(process_err))

# kf = kf.em(observations, n_iter=len(observations)-1)

# (values, filtered_state_covariances) = kf.filter(observations)

# for value in values:
#     print("X: {}, Y: {}".format(value[0], value[1]))

plt.plot(ground_truth[:,0], ground_truth[:,1], label="Ground Truth")
plt.plot(observations[:,0], observations[:,1], 'ro', label="Noisy measuments")
plt.plot(values[:,0], values[:,1], 'g-', label="Kalman Filtered")
plt.ylim(0,650)
plt.xlim(0,1500)
plt.legend()
plt.show()

'''
State matrix X contains state variables (position, velocity)
Ex: X = [x, y, x', y']

The matrix A determines the relationship between the variables in X
Ex: A = [
    1   dt
    0   1
]

A and B comes from law of motion
x = x_0 + x*t + 1/2*x*t^2

By multiplying A and X we get a new position based on the old position and velocity.
AX = [
    1   dt
    0   1
] * [x, x']
AX = [
    x + dt*x'
    0 + x'
]

Adding B * u takes acceleration into account:
B = [
    1/2 dt^2
    dt
]

u matrix is defined by the case, can be 0. In case of falling object it can be gravity in y-direction. The dimensions are freedom of movement x 1, so the results Bu gives the same format as the state vector.

Process covariance matrix contains the error in the estimate

u is control variable matrix and contains all external forces
w is predicted state noise matrix and contains the predicted noise
Q is process noise covariance matrix.

'''
