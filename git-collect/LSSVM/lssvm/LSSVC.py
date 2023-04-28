import numpy as np
from numpy import dot
import time
import torch
import math

from utils.kernel import get_kernel
from utils.import_export import dump_model, load_model
from utils.conversion import numpy_json_encoder


class LSSVC():
    """A class that implements the Least Squares Support Vector Machine 
    for classification tasks.

    It uses Numpy pseudo-inverse function to solve the dual optimization 
    problem with ordinary least squares. In multiclass classification 
    problems the approach used is one-vs-all, so, a model is fit for each 
    class while considering the others a single set of the same class.
    
    # Parameters:
    - gamma: float, default = 1.0
        Constant that control the regularization of the model, it may vary 
        in the set (0, +infinity). The closer gamma is to zero, the more 
        regularized the model will be.
    - kernel: {'linear', 'poly', 'rbf'}, default = 'rbf'
        The kernel used for the model, if set to 'linear' the model 
        will not take advantage of the kernel trick, and the LSSVC maybe only
        useful for linearly separable problems.
    - kernel_params: **kwargs, default = depends on 'kernel' choice
        If kernel = 'linear', these parameters are ignored. If kernel = 'poly',
        'd' is accepted to set the degree of the polynomial, with default = 3. 
        If kernel = 'rbf', 'sigma' is accepted to set the radius of the 
        gaussian function, with default = 1. 
     
    # Attributes:
    - All hyperparameters of section "Parameters".
    - alpha: ndarray of shape (1, n_support_vectors) if in binary 
             classification and (n_classes, n_support_vectors) for 
             multiclass problems
        Each column is the optimum value of the dual variable for each model
        (using the one-vs-all approach we have n_classes == n_classifiers), 
        it can be seen as the weight given to the support vectors 
        (sv_x, sv_y). As usually there is no alpha == 0, we have 
        n_support_vectors == n_train_samples.
    - b: ndarray of shape (1,) if in binary classification and (n_classes,) 
         for multiclass problems 
        The optimum value of the bias of the model.
    - sv_x: ndarray of shape (n_support_vectors, n_features)
        The set of the supporting vectors attributes, it has the shape 
        of the training data.
    - sv_y: ndarray of shape (n_support_vectors, n)
        The set of the supporting vectors labels. If the label is represented 
        by an array of n elements, the sv_y attribute will have n columns.
    - y_labels: ndarray of shape (n_classes, n)
        The set of unique labels. If the label is represented by an array 
        of n elements, the y_label attribute will have n columns.
    - K: function, default = rbf()
        Kernel function.
    """
    

    def __init__(self, gamma=1, kernel='rbf', **kernel_params): 
        # Hyperparameters
        self.gamma = gamma
        self.kernel_ = kernel
        self.kernel_params = kernel_params
        
        # Model parameters
        self.alpha = None
        self.b = None
        self.sv_x = None
        self.sv_y = None
        self.y_labels = None
        
        self.K = get_kernel(kernel, **kernel_params)

        
    
    def _optimize_parameters(self, X, y_values,method):
        """Help function that optimizes the dual variables through the 
        use of the kernel matrix pseudo-inverse.
        """
        sigma = np.multiply(y_values*y_values.T, self.K(X,X))
        #print(y_values.shape)
        print(self.K(X,X).shape)
        A = np.block([
            [0, y_values.T],
            [y_values, sigma + self.gamma**-1 * np.eye(len(y_values))]
        ])
        B = np.array([0]+[1]*len(y_values))
        
        Aa=np.copy(A)
      
        
    
        
        
        
        #v=vh.mH
        #print(v.shape,(torch.transpose(u)).shape,s.shape)
        #C=torch.matmul(vh,torch.matmul(torch.diag(s),torch.transpose(u,0,1)))
        #C= (v@(np.linalg.inv(np.diag(s))@np.transpose(u)))

        ## This is rsvd on torch(gpu)

        if(method==3):
       
            print('i DOing gpu')
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            at = torch.from_numpy(A)
            
            at=at.float()
            bb=torch.from_numpy(B)
            
            bb=bb.float()
            
            at=at.cuda()
            bb=bb.cuda()
            tic1=time.perf_counter()
            with torch.no_grad():
                
                rank1 = math.ceil(0.5*Aa.shape[1])
                Omega1 = torch.rand(at.shape[1],rank1,device=device)
                Y1=torch.matmul(at,Omega1)
                Q1, _ = torch.linalg.qr(at)
                Bb1=torch.matmul(torch.transpose(Q1,0,1),at)
                U_t,S,Vt=torch.linalg.svd(Bb1,full_matrices=False)
                U=torch.matmul(Q1,U_t)

                solution = torch.mv(U.mH,bb)
                solution = torch.div(solution,S)
                solution = torch.mv(Vt.mH,solution)

            
            
            toc1=time.perf_counter()
            tim1=toc1-tic1
            print("time gpu rsvd = ", tim1)
            solution=solution.detach().cpu().numpy() 









        ## THis is RSVD on numpy(cpu)
        if(method==2):
            print('cpu_rsvd')
            ## output matrix for checking stuff
            np.savetxt('result.txt', Aa)
            np.savetxt('bvec.txt',B)
            tic2=time.perf_counter()
            rank = math.ceil(0.5*Aa.shape[1])
            Omega = np.random.randn(Aa.shape[1], rank)
            Y = Aa @ Omega

            Q, _ = np.linalg.qr(Y)
            Bb = Q.T @ Aa
         
            u_tilde, s, v = np.linalg.svd(Bb, full_matrices = 0)
            u = Q @ u_tilde 
            print(u.shape)
            print(B.shape)
            solution = np.dot(u.T, B)
            print(solution.shape)
            solution=np.divide(solution,s)
            solution=np.dot(v.T,solution)
            toc2=time.perf_counter()
            tim2=toc2-tic2
            print("time cpu rsvd = ", tim2)
        #print(u.shape,(1/s).shape,v.shape)
        #C= (np.transpose(v)@((np.diag(1/s))@np.transpose(u)))

        
        
## this is original cpu implementation with pinv
        if (method ==1):
            print('i Doing CPU')
            tic=time.perf_counter()
            u,s,vt = np.linalg.svd(A)

            A_cross=vt.T@(np.linalg.inv(np.diag(s))@u.T)

            solution = dot(A_cross, B)
            toc=time.perf_counter()
            tim=toc-tic 
            print("time = ", tim)  
        
    
        #solution = dot(v.T,dot(np.diag(1/s),dot(u.T,B)))
        #solution = dot(u.T,B)
        #solution= dot((1/s),solution)
        #solution = dot(v.T,solution)

        
        
        b = solution[0]
        alpha = solution[1:]
        
        return (b, alpha)
    
    def fit(self, X, y,method):
        """Fits the model given the set of X attribute vectors and y labels.
        - X: ndarray of shape (n_samples, n_attributes)
        - y: ndarray of shape (n_samples,) or (n_samples, n)
            If the label is represented by an array of n elements, the y 
            parameter must have n columns.
        """
        y_reshaped = y.reshape(-1,1) if y.ndim==1 else y

        self.sv_x = X
        self.sv_y = y_reshaped
        self.y_labels = np.unique(y_reshaped, axis=0)
        
        if len(self.y_labels) == 2: # binary classification
            # converting to -1/+1
            y_values = np.where(
                (y_reshaped == self.y_labels[0]).all(axis=1)
                ,-1,+1)[:,np.newaxis] # making it a column vector
            
            self.b, self.alpha = self._optimize_parameters(X, y_values,method)
        
        else: # multiclass classification, one-vs-all approach
            n_classes = len(self.y_labels)
            self.b = np.zeros(n_classes)
            self.alpha = np.zeros((n_classes, len(y_reshaped)))
            
            for i in range(n_classes):
                # converting to +1 for the desired class and -1 for all 
                # other classes
                y_values = np.where(
                    (y_reshaped == self.y_labels[i]).all(axis=1)
                    ,+1,-1)[:,np.newaxis]
  
                self.b[i], self.alpha[i] = self._optimize_parameters(X, y_values,method)
        
    def predict(self, X):
        """Predicts the labels of data X given a trained model.
        - X: ndarray of shape (n_samples, n_attributes)
        """
        if self.alpha is None:
            raise Exception(
                "The model doesn't see to be fitted, try running .fit() method first"
            )

        X_reshaped = X.reshape(1,-1) if X.ndim==1 else X
        KxX = self.K(self.sv_x, X_reshaped)
        
        if len(self.y_labels)==2: # binary classification
            y_values = np.where(
                (self.sv_y == self.y_labels[0]).all(axis=1),
                -1,+1)[:,np.newaxis]

            y = np.sign(dot(np.multiply(self.alpha, y_values.flatten()), KxX) + self.b)
            
            y_pred_labels = np.where(y==-1, self.y_labels[0], self.y_labels[1])
        
        else: # multiclass classification, one-vs-all approach
            y = np.zeros((len(self.y_labels), len(X)))
            for i in range(len(self.y_labels)):
                y_values = np.where(
                    (self.sv_y == self.y_labels[i]).all(axis=1),
                    +1, -1)[:,np.newaxis]
                y[i] = dot(np.multiply(self.alpha[i], y_values.flatten()), KxX) + self.b[i]
            
            predictions = np.argmax(y, axis=0)
            y_pred_labels = np.array([self.y_labels[i] for i in predictions])
            
        return y_pred_labels

    def dump(self, filepath='model', only_hyperparams=False):
        """This method saves the model in a JSON format.
        - filepath: string, default = 'model'
            File path to save the model's json.
        - only_hyperparams: boolean, default = False
            To either save only the model's hyperparameters or not, it 
            only affects trained/fitted models.
        """
        model_json = {
            'type': 'LSSVC',
            'hyperparameters': {
                'gamma': self.gamma,
                'kernel': self.kernel_,
                'kernel_params': self.kernel_params
            }           
        }

        if (self.alpha is not None) and (not only_hyperparams):
            model_json['parameters'] = {
                'alpha': self.alpha,
                'b': self.b,
                'sv_x': self.sv_x,
                'sv_y': self.sv_y,
                'y_labels': self.y_labels
            }
        
        dump_model(model_dict=model_json, file_encoder=numpy_json_encoder, filepath=filepath)
        
    @classmethod
    def load(cls, filepath, only_hyperparams=False):
        """This class method loads a model from a .json file.
        - filepath: string
            The model's .json file path.
        - only_hyperparams: boolean, default = False
            To either load only the model's hyperparameters or not, it 
            only has effects when the dump of the model as done with the
            model's parameters.
        """
        model_json = load_model(filepath=filepath)

        if model_json['type'] != 'LSSVC':
            raise Exception(
                f"Model type '{model_json['type']}' doesn't match 'LSSVC'"
            )

        lssvc = LSSVC(
            gamma = model_json['hyperparameters']['gamma'],
            kernel = model_json['hyperparameters']['kernel'],
            **model_json['hyperparameters']['kernel_params']
        )

        if (model_json.get('parameters') is not None) and (not only_hyperparams):
            lssvc.alpha = np.array(model_json['parameters']['alpha'])
            lssvc.b = np.array(model_json['parameters']['b'])
            lssvc.sv_x = np.array(model_json['parameters']['sv_x'])
            lssvc.sv_y = np.array(model_json['parameters']['sv_y'])
            lssvc.y_labels = np.array(model_json['parameters']['y_labels'])

        return lssvc
        
