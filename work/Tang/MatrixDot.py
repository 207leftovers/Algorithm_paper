import time
import numpy as np

class Matrix(object):
    
    def __init__(self, array):
        """
        Something like Matrix([[1,2],[3,4]])
        """
        self.shape = self._getShape(array)
        self.matrix = array
        
    # Get shape from 2D array
    # Exception when it is not generated 2D. eg. [1,2,3] should be [[1,2,3]] or [[1], [2], [3]]
    def _getShape(self, array):
        try:
            l1 = len(array)
            l2 = [len(e) for e in array]
            
            try:
                len(array[0][0])
                raise ValueError, 'Matrix size error.'
            except:
                
                if all([i==l2[0] for i in l2]):
                    return (l1,l2[0])
                else:
                    raise ValueError, 'Matrix size error.'
        except:
            raise ValueError, 'Matrix size error.'
            
            
    def dot_elementwise(self, matB):
        """
        matrixA.dot(matrixB)
        """
        
        assert self.shape[1] == matB.shape[0]
        
        result = []
        
        for i in xrange(self.shape[0]):
            thisrow = []
            for j in xrange(matB.shape[1]):
                element = 0
                for k in xrange(self.shape[1]):
                    element += self.matrix[i][k] * matB.matrix[k][j]
                thisrow.append(element)
            result.append(thisrow)
            
        return result
    
    

def timeit(method):

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print '%r %d nanoseconds' % \
              (method.__name__, int((te-ts)*1e9))
        return result

    return timed

if __name__ == '__main__':
    A_numpy = np.random.uniform(-200000,200000, size = (100,200))
    B_numpy = np.random.uniform(-200000,200000, size = (200,300))
    A = list(A_numpy)
    B = list(B_numpy)
    
    @timeit
    def naive_multiplication(A, B):
        A_mat = Matrix(A)
        B_mat = Matrix(B)

        return A_mat.dot_elementwise(B_mat)
    
    @timeit
    def numpy_multiplication(A, B):
        return np.array(A).dot(np.array(B))
    
    naive_result = naive_multiplication(A,B)
    numpy_result = numpy_multiplication(A,B)
    
    print 'Approximately equal: ', np.allclose(naive_result, numpy_result)