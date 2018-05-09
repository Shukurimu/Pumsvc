'''
class RMSProp():
    def __init__():
        self.b
        self.a
    def fun(x):
        pass

class Sigmoid():
    def fun(x):
        expNegX = np.exp(-x)
        return 1/(1+np.exp(-x))
    def derFun(x):
        funValue = fun(x)
        return funValue * (1 - funValue)

class Tanh():
    def fun(x):
        expX = np.exp(x)
        expNegX = np.exp(-x)
        return (expX - expNegX) / (expX + expNegX)
    def derFun(x):
        funValue = fun(x)
        return 1 - funValue ** 2

class GRU():
    #variable as same as https://colah.github.io/posts/2015-08-Understanding-LSTMs/
    def __init__( inputDim, outputDim, activeFun = Sigmoid(), activeFun2 = Tanh(), updateFun = RMSProp()):
        low, high = -0.05, 0.05
        totalDim = inputDim + outputDim
        self.weightR = np.random.uniform(low, high, (outputDim, totalDim))
        self.weightZ = np.random.uniform(low, high, (outputDim, totalDim))
        self.weight = np.random.uniform(low, high, (outputDim, totalDim))
        self.h = np.zeros((outputDim))
        self.activeFun = activeFun
        self.activeFun2 = activeFun2
        self.updateFun = RMSProp()

    def forward( x):
        self.a = np.hstack( self.h, x)
        r = self.activeFun.fun(self.weightR.dot(a))
        z = self.activeFun.fun(self.weightZ.dot(a))
        hLoss = self.activeFun2.fun(self.weight.dot(a))
        self.hNew = (1-z) * self.h + z * hLoss
        return self.hNew
    
    def backprogation( hTrue):
        if self.hNew is None:
            perror("error: Not Forward")
        diff = hTrue - self.hNew
        self.h = self.hNew
        del self.hNew 
'''