import matplotlib.pyplot as plt
import numpy as np

def GP_1d(x, beta=0.005):
    k = lambda x, y: np.exp(-beta*(x-y)*(x-y))

    n = len(x)

    # Construct covariance matrix
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            C[i,j] = k(x[i],x[j])


    # Sample from Gaussian Process at points
    u = np.random.randn(n,1)     #sample from normal distribution
    [A,S,B] = np.linalg.svd(C)   #factor C with diagonal S and unitary matrices A, B
    S = np.diag(S)
    f = np.matmul(np.matmul(A, np.sqrt(S)), u)    #resulting output

    return f

def GP_2d(x, y):

    # % %Choose kernel
    # % kern = 1
    # % switch kern
    # %     case 1
    # % end
    # k = @(x,y) exp(-100*(x-y)'*(x-y));
    #
    # %Choose sample
    # points = (0:.05:1)';
    # [U,V] = meshgrid(points,points);
    # x = [U(:) V(:)]';
    # n = size(x,2);
    #
    # %Construct covariance matrix
    # C = zeros(n,n);
    # for i = 1:n
    #     for j = 1:n
    #         C(i,j) = k(x(i),x(j));
    #     end
    # end
    #
    # %Sample from Gaussian Process at points
    # u = randn(n,1);     %sample from normal distribution
    # [A,S,B] = svd(C);   %factor C
    # f = A*sqrt(S)*u;    %resulting output
    #
    # %Plot
    # figure(2);
    # clf;
    # Z = reshape(f,sqrt(n),sqrt(n));
    # surf(U,V,Z);

    return f


if __name__ == '__main__':

    #Plot
    fig,ax = plt.subplots(1)
    # plt.hold()
    # x = np.arange(0, 1, .005)
    x = np.arange(250)
    f = GP_1d(x)
    plt.plot(x,f,'.','LineWidth',2)
    # plt.axis([0,1,-2,2])
    plt.show()
    print('')