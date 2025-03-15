# standard libraries
import math, os, time
import numpy as np

# plotting
import matplotlib.pyplot as plt

# progress bars
from tqdm.notebook import trange, tqdm

# PyTorch
import torch

from IPython.display import clear_output # for animating the plot

# Handy line for using a CUDA device if available but default to the CPU if not
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class PINN_solution:
    """This class saves the results of the training"""
    def __init__(self, hidden: int):
        self.w_list = np.array([])
        self.a_list = np.array([])
        self.b_list = np.array([])
        self.w_grad_list = np.array([])
        self.a_grad_list = np.array([])
        self.b_grad_list = np.array([])
        self.infl_list = np.array([])
        self.loss_list = np.array([])
        self.loss = 10**12
        self.hidden = hidden
        
    def update(self, network, loss):
        """Add the parameter values to the lists"""
        self.w_list = np.append(self.w_list,network.hidden_layer1.weight.cpu().detach().numpy())
        self.b_list = np.append(self.b_list,network.hidden_layer1.bias.cpu().detach().numpy())
        self.a_list = np.append(self.a_list,network.output_layer.weight.cpu().detach().numpy())

        self.w_grad_list = np.append(self.w_grad_list,network.hidden_layer1.weight.grad.cpu().detach().numpy())
        self.b_grad_list = np.append(self.b_grad_list,network.hidden_layer1.bias.grad.cpu().detach().numpy())
        self.a_grad_list = np.append(self.a_grad_list,network.output_layer.weight.grad.cpu().detach().numpy())

        Xinflection = - network.hidden_layer1.bias.detach() / network.hidden_layer1.weight.detach().squeeze()
        self.infl_list=np.append(self.infl_list,Xinflection.cpu())
        self.loss = loss.cpu().detach().numpy()
        self.loss_list = np.append(self.loss_list, self.loss)
        
    def evol_plot(self, savepath=None):
        """Make plots of the evolution of the parameters"""
        fig, axs = plt.subplots(ncols=3,figsize=(10, 3))
        
        axs[0].grid()
        axs[0].set_xlabel('Epochs')
        axs[0].set_ylabel('w')
        axs[0].set_title('Evolution of w')
        axs[0].plot(np.reshape(self.w_list,(-1,self.hidden)));

        axs[1].grid()
        axs[1].set_xlabel('Epochs')
        axs[1].set_ylabel('a')
        axs[1].set_title('Evolution of a')
        axs[1].plot(np.reshape(self.a_list,(-1,self.hidden)));
        
        axs[2].grid()
        axs[2].set_xlabel('Epochs')
        axs[2].set_ylabel('b')
        axs[2].set_title('Evolution of b')
        axs[2].plot(np.reshape(self.b_list,(-1,self.hidden)));
        plt.tight_layout()
        if savepath != None:
            plt.savefig(savepath)
        plt.show()

    def evol_plot_grad(self, savepath=None):
        """Make plots of the evolution of the gradient of the parameters"""
        fig, axs = plt.subplots(ncols=3,figsize=(10, 3))
        
        axs[0].grid()
        axs[0].set_xlabel('Epochs')
        axs[0].set_ylabel('w\'')
        axs[0].set_title('Evolution of w\'')
        axs[0].plot(np.reshape(self.w_grad_list,(-1,self.hidden)));

        axs[1].grid()
        axs[1].set_xlabel('Epochs')
        axs[1].set_ylabel('a\'')
        axs[1].set_title('Evolution of a\'')
        axs[1].plot(np.reshape(self.a_grad_list,(-1,self.hidden)));
        
        axs[2].grid()
        axs[2].set_xlabel('Epochs')
        axs[2].set_ylabel('b\'')
        axs[2].set_title('Evolution of b\'')
        axs[2].plot(np.reshape(self.b_grad_list,(-1,self.hidden)));
        plt.tight_layout()
        if savepath != None:
            plt.savefig(savepath)
        plt.show()

    def loss_infl_plots(self, savepath=None):
        """Make plots of the evolution of the loss and inflection points"""
        fig, axs = plt.subplots(ncols=2,figsize=(10, 4))
        
        axs[0].grid()
        axs[0].plot(np.reshape(self.infl_list,(-1,self.hidden)));
        axs[0].set_xlabel('Epochs')
        axs[0].set_ylabel('-b/w')
        axs[0].set_title('Inflection points')
        axs[0].set_ylim(-1,3)

        axs[1].grid()
        axs[1].semilogy(self.loss_list)
        axs[1].set_xlabel('Epochs')
        axs[1].set_ylabel('Loss')
        axs[1].set_title('Loss evolution')        
        plt.tight_layout()
        if savepath != None:
            plt.savefig(savepath)

        plt.show()

    def loss_plot(self, savepath=None):
        """Make plots of the evolution of the loss and inflection points"""
        plt.figure(figsize=[4,3])
        plt.grid()
        plt.semilogy(self.loss_list)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        #plt.title('Loss evolution')
        plt.tight_layout()
        if savepath != None:
            plt.savefig(savepath)
        plt.show()

        
    def sol(self, savepath=None):
        x= np.arange(0,1.0001,0.001)
        y= np.zeros(1001)
        for i in range(self.hidden):
            y+=self.a_list[-i-1]*np.tanh(self.w_list[-i-1]*x+self.b_list[-i-1])
        return y

    def show_sol(self, savepath=None):
        x= np.arange(0,1.0001,0.001)
        self.sol()
        plt.figure(figsize=[4, 3])
        plt.plot(x,y)
        plt.grid()
        plt.xlim([0,1])
        plt.xlabel('x')
        plt.ylabel('u(x)')
        plt.tight_layout()
        if savepath != None:
            plt.savefig(savepath)
        plt.show()
        
        
def real_sol(alpha: float, kappa: float, F: float, f: float, epsil: float, g0: float, g1: float, N: int):
    """A function to calculate the analtyical solution"""
    if epsil<0.025:
        x_coords = np.arange(0,1+1/N,1/N)
        return x_coords, x_coords, 0
    else:
        A = np.array([[kappa,kappa-alpha*F/epsil],[kappa,kappa*np.exp(F/epsil)+alpha*F/epsil*np.exp(F/epsil)]])
        b = np.array([[g0+alpha*f/F],[g1-f/F*(kappa+alpha)]])
        [c1,c2] = np.linalg.solve(A,b)
        #Coordinates for plotting:
        x_coords = np.arange(0,1+1/N,1/N)
        y_coords = c1+c2*np.exp(F*x_coords/epsil)+f/F*x_coords
        return x_coords, y_coords, [A,b,c1,c2]

## Calculates the values of the PDE
def pde(x: torch.Tensor, network, epsil: float):
    """Given a point x_i and a network, it calculates the predicted residual of the pde for a given epsilon."""
    u = network(x) #Calculate the value at this point
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0] #calculate the derivative at this point
    u_xx = torch.autograd.grad(u_x.sum(), x , create_graph=True)[0] #calculate the second derivative
    pde_val = -epsil*u_xx + u_x - 1
    return pde_val

def initializeNetwork(LR: float, MOMENTUM: float, hidden: int, params=0, optim='SGD') :
    """Initializes the neural network to pytorch standard"""
    class Network(torch.nn.Module):
        def __init__(self):
            super(Network, self).__init__()
            self.hidden_layer1 = torch.nn.Linear(1,hidden)
            self.output_layer = torch.nn.Linear(hidden,1, bias=False)

        def forward(self, x):
            inputs = x
            layer1_out = torch.tanh(self.hidden_layer1(inputs))
            output = self.output_layer(layer1_out)
            return output
    
    network = Network()
    network = network.to(device)
    if optim == 'SGD':
        optimizer = torch.optim.SGD(network.parameters(),lr=LR, momentum=MOMENTUM)
    elif optim == 'LBFGS':
        optimizer = torch.optim.LBFGS(network.parameters(), lr=LR)

    return network, optimizer

# def initializeNetwork_guess_1(LR: float, MOMENTUM: float, hidden: int, params: list, optim='SGD') :
#     """Initialize the network with one neuron, with specific
#     starting values for w, a and b. THIS FUNCTION SHOULD BE
#     DELETED EVENTUALLY SINCE IT IS ENCLOSED IN THE NEXT
#     FUNCTION!"""
#     class Network(torch.nn.Module):
#         def __init__(self):
#             super(Network, self).__init__()
#             self.hidden_layer1 = torch.nn.Linear(1,hidden)
#             self.output_layer = torch.nn.Linear(hidden,1, bias=False)

#         def forward(self, x):
#             inputs = x
#             layer1_out = torch.tanh(self.hidden_layer1(inputs))
#             output = self.output_layer(layer1_out)
#             return output
    
#     network = Network()
#     network = network.to(device)
    
#     if optim == 'SGD':
#         optimizer = torch.optim.SGD(network.parameters(),lr=LR, momentum=MOMENTUM)
#     elif optim == 'LBFGS':
#         optimizer = torch.optim.LBFGS(network.parameters(), lr=LR)
        
#     with torch.no_grad():
#         w = torch.tensor([[params[0]]]).to(device)
#         a = torch.tensor([[params[1]]]).to(device)
#         b = torch.tensor([params[2]]).to(device)

#         network.hidden_layer1.weight.data = w
#         network.hidden_layer1.bias.data = b
#         network.output_layer.weight.data = a

#     return network, optimizer

def initializeNetwork_guess(LR: float, MOMENTUM: float, hidden: int, params: list[torch.Tensor], optim='SGD'):
    """Initialize the network with multiple neurons, with specific starting values
    for w, a and b."""
    class Network(torch.nn.Module):
        def __init__(self):
            super(Network, self).__init__()
            self.hidden_layer1 = torch.nn.Linear(1,hidden)
            self.output_layer = torch.nn.Linear(hidden,1, bias=False)

        def forward(self, x):
            inputs = x
            layer1_out = torch.tanh(self.hidden_layer1(inputs))
            output = self.output_layer(layer1_out)
            return output
    
    network = Network()
    network = network.to(device)
    if optim == 'SGD':
        optimizer = torch.optim.SGD(network.parameters(),lr=LR, momentum=MOMENTUM)
    elif optim == 'LBFGS':
        optimizer = torch.optim.LBFGS(network.parameters(), lr=LR)

    with torch.no_grad():
        w = params[0].to(device)
        a = params[1].to(device)
        b = params[2].to(device)

        network.hidden_layer1.weight.data = w
        network.hidden_layer1.bias.data = b
        network.output_layer.weight.data = a

    return network, optimizer

# def train_network(LR: float, HIDDEN: int, NR_EPOCHS: int, h_SAMPLE: float, LAMBDA: float, PLOT_INTERVAL: int, EPSIL: float, Initializer=initializeNetwork, optim='SGD', params=0):
#     """Train the network given:
#         *) a learning rate, 
#         *) the number of hidden layers, 
#         *) the number of epochs, 
#         *) the space between the samples x_i, 
#         *) the parameter lambda that balances the inner and boundary terms,  
#         *) after how many epochs we want to animate the plot, choosing -1 will result in no plots at all,
#         *) the value of epsilon,
#         *) a function on how we want to initialize the network.
#     Returns a list with the values for w, b and a. Also returns a list with the losses and a list with the inflection points."""
    
#     Result = PINN_solution(HIDDEN)
    
#     save = False
#     #Default values of the PDE:
#     F = 1 #1
#     f = 1 #1
#     alpha = 10**-3 #10**-3
#     kappa = 1 #1
#     g0 = 0 #0
#     g1 = 0 #0
#     #Calculate the real solution:
#     xcoords,ycoords,abc = real_sol(alpha,kappa,F,f,EPSIL,g0,g1,1000)
    
#     #Loss function:
#     mse_cost_function = torch.nn.MSELoss() # Mean squared error

#     #Initialize the network
#     #if params == 0:
#     #    network, optimizer = Initializer(LR, HIDDEN)
#     #else:
#     network, optimizer = Initializer(LR, HIDDEN, params, optim)
#     #Tensor needed to calculate loss at boundary values
#     alph = torch.Tensor([[-alpha],[alpha]]).to(device)

#     #start training:
    
#     x_bc = np.array([[0],[1]]) #x-values of boundary conditions:
#     pt_x_bc = torch.autograd.Variable(torch.from_numpy(x_bc).float(), requires_grad=True).to(device) #make it to a Tensor
#     bc = np.array([[0],[0]]) #what the boundary conditions should really be
#     pt_bc = torch.autograd.Variable(torch.from_numpy(bc).float(), requires_grad=False).to(device)
    
#     x_0 = np.arange(h_SAMPLE/2,1,h_SAMPLE) # for now, we take the points to calculate the loss evenly spaced
#     x_0 = x_0.reshape((-1,1))
#     all_zeros = np.zeros((100,1)) # this is what the value of -eps*u_xx+u_x-1 should be  
#     pt_x0 = torch.autograd.Variable(torch.from_numpy(x_0).float(), requires_grad=True).to(device)
#     pt_all_zeros = torch.autograd.Variable(torch.from_numpy(all_zeros).float(), requires_grad=False).to(device)

#     for epoch in range(NR_EPOCHS):

#         optimizer.zero_grad()
        
#         pred_bc = network(pt_x_bc) #prediction at BC's
#         u_x_bc = torch.autograd.grad(pred_bc.sum(), pt_x_bc, create_graph=True)[0]

#         bc1 = alph*u_x_bc+pred_bc # calculate -alpha*u'(0)+1*u(0) and alpha*u'(1)+1*u(1) 

#         mse_bc = mse_cost_function(bc1, pt_bc) # calculate 0.5*(-alpha*u'(0)+1*u(0))^2+0.5*(alpha*u'(1)+1*u(1))^2

#         #x_0 = np.random.uniform(low=0.0, high=1, size=(100,1)) # we can use this if we want to sample uniformly

#         f_out = pde(pt_x0, network,EPSIL) # this is the predicted values of -eps*u_xx+u_x-1
#         mse_f = mse_cost_function(f_out, pt_all_zeros) # mean squared error

#         # Combining the loss functions
#         loss = (1-LAMBDA)*mse_bc + LAMBDA*mse_f
#         #loss_list = np.append(loss_list, loss.cpu().detach().numpy())

#         loss.backward() # This is for computing gradients using backward propagation
#         optimizer.step() # This is equivalent to : theta_new = theta_old - alpha * derivative of J w.r.t theta
        
#         Result.update(network,loss)
        
#         # Make an animated plot:
#         if epoch%PLOT_INTERVAL == 0 and PLOT_INTERVAL != -1:
#             with torch.autograd.no_grad():
#                 #print(mse_bc,mse_f)
#                 h = 0.01
#                 x = x_0
#                 pt_x = torch.autograd.Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
#                 y = network(pt_x)
#                 y = y.cpu().detach().numpy()
#                 clear_output(wait=True)
#                 print(epoch,"Training Loss:",loss.data)
#                 plt.figure(figsize=[6, 6])
#                 plt.title(f"Epoch {epoch}/{NR_EPOCHS}, {loss.data}")
#                 plt.xlim(0,1)
#                 #maxy = 1.5*max(ycoords)
#                 maxy = 1
#                 plt.ylim(-0.5*maxy,1*maxy)
#                 plt.plot(x,y)
#                 plt.plot(xcoords,ycoords)
#                 Xinflection = - network.hidden_layer1.bias.detach() / network.hidden_layer1.weight.detach().squeeze()
#                 Yinflection = network(Xinflection.unsqueeze(-1)).squeeze().detach()
#                 plt.plot(Xinflection.cpu(), Yinflection.cpu(), 'r+', label="inflection points")
#                 #plt.savefig('Eps0_1.png')                
#                 if save==True:
#                     dummyepoch = int(epoch+1000000)
#                     filename = "Filename"+str(dummyepoch)+".png"
#                     plt.savefig(filename)
#                 plt.show()
    
#     return Result

def train_network(LR: float, HIDDEN: int, NR_EPOCHS: int, h_SAMPLE: float, LAMBDA: float, PLOT_INTERVAL: int, EPSIL: float, Initializer=initializeNetwork, optim='SGD', params=0, momentum=0, M=-1):
    """Train the network given:
        *) a learning rate, 
        *) the number of hidden layers, 
        *) the number of epochs, 
        *) the space between the samples x_i, 
        *) the parameter lambda that balances the inner and boundary terms,  
        *) after how many epochs we want to animate the plot, choosing -1 will result in no plots at all,
        *) the value of epsilon,
        *) a function on how we want to initialize the network (optional),
        *) the initialization function (optional),
        *) eventually initial parameters, only if the optimizer is not default (optional),
        *) momentum parameter (optional),
        *) bound on parameter size, if negative, there is no bound (optional).
    Returns a list with the values for w, b and a. Also returns a list with the losses and a list with the inflection points."""
    
    Result = PINN_solution(HIDDEN)
    
    save = False
    #Default values of the PDE:
    F, f, g0, g1 = 1, 1, 0, 0
    alpha, kappa = 0, 1

    #Calculate the real solution:
    xcoords,ycoords,abc = real_sol(alpha,kappa,F,f,EPSIL,g0,g1,1000)
    
    #Loss function:
    mse_cost_function = torch.nn.MSELoss() # Mean squared error

    #Initialize the network
    network, optimizer = Initializer(LR, momentum, HIDDEN, params, optim)
    
    #Tensor needed to calculate loss at boundary values
    alph = torch.Tensor([[-alpha],[alpha]]).to(device)
    
    x_bc = np.array([[0],[1]]) #x-values of boundary conditions:
    pt_x_bc = torch.autograd.Variable(torch.from_numpy(x_bc).float(), requires_grad=True).to(device) #make it to a Tensor
    bc = np.array([[0],[0]]) #what the boundary conditions should really be
    pt_bc = torch.autograd.Variable(torch.from_numpy(bc).float(), requires_grad=False).to(device)
    
    x_0 = np.arange(h_SAMPLE/2,1,h_SAMPLE) # for now, we take the points to calculate the loss evenly spaced
    x_0 = x_0.reshape((-1,1))
    all_zeros = np.zeros((int(1/h_SAMPLE),1))
    #all_zeros = np.zeros((100,1)) # this is what the value of -eps*u_xx+u_x-1 should be  
    pt_x0 = torch.autograd.Variable(torch.from_numpy(x_0).float(), requires_grad=True).to(device)
    pt_all_zeros = torch.autograd.Variable(torch.from_numpy(all_zeros).float(), requires_grad=False).to(device)
    
    def calc_loss():
        """Calculate the loss"""
        optimizer.zero_grad()
        
        pred_bc = network(pt_x_bc) #prediction at BC's
        u_x_bc = torch.autograd.grad(pred_bc.sum(), pt_x_bc, create_graph=True)[0]

        bc1 = alph*u_x_bc+pred_bc # calculate -alpha*u'(0)+1*u(0) and alpha*u'(1)+1*u(1) 

        mse_bc = mse_cost_function(bc1, pt_bc) # calculate 0.5*(-alpha*u'(0)+1*u(0))^2+0.5*(alpha*u'(1)+1*u(1))^2

        f_out = pde(pt_x0, network, EPSIL) # this is the predicted values of -eps*u_xx+u_x-1
        mse_f = mse_cost_function(f_out, pt_all_zeros) # mean squared error

        # Combining the loss functions
        loss = (1-LAMBDA)*mse_bc + LAMBDA*mse_f

        loss.backward() # This is for computing gradients using backward propagation
        return loss

    for epoch in range(NR_EPOCHS):

        optimizer.step(calc_loss) # Opimizing step

        if M>0:
            for p in network.parameters():
                p.data.clamp_(-M, M)
        
        loss = calc_loss() #This is inefficient? => Look into this?
        
        #print(network.hidden_layer1.weight.grad)
        #print(network.hidden_layer1.bias.grad)
        #print(network.output_layer.weight.grad)
        #print('-------------')

        Result.update(network,loss)
        
        # Make an animated plot:
        if epoch%PLOT_INTERVAL == 0 and PLOT_INTERVAL != -1:
            with torch.autograd.no_grad():
                #print(mse_bc,mse_f)
                h = h_SAMPLE
                #h = 0.01
                x = x_0
                pt_x = torch.autograd.Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
                y = network(pt_x)
                y = y.cpu().detach().numpy()
                clear_output(wait=True)
                print(epoch,"Training Loss:",loss.data)
                plt.figure(figsize=[6, 6])
                plt.title(f"Epoch {epoch}/{NR_EPOCHS}, {loss.data}")
                plt.xlim(0,1)
                #maxy = 1.5*max(ycoords)
                maxy = 1
                plt.ylim(-0.5*maxy,1*maxy)
                plt.plot(x,y)
                plt.plot(xcoords,ycoords)
                Xinflection = - network.hidden_layer1.bias.detach() / network.hidden_layer1.weight.detach().squeeze()
                Yinflection = network(Xinflection.unsqueeze(-1)).squeeze().detach()
                plt.plot(Xinflection.cpu(), Yinflection.cpu(), 'r+', label="inflection points")
                #plt.savefig('Eps0_1.png')                
                if save==True:
                    dummyepoch = int(epoch+1000000)
                    filename = "Filename"+str(dummyepoch)+".png"
                    plt.savefig(filename)
                plt.show()
    
    return Result