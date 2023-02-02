"""
    Phillip Cussen Burke - 11/05/19

    Using the lanczos algorithm to approximate the time evolution of the form
    psi(t) = exp(-iHt)*psi(0)
"""
#

using SparseArrays

"""
    Returns optimal dimension to use in the algorithm

    Using lanczos algorithm to approximate the time evolution of the form
    psi(t) = exp(-iHt)*psi(0)

    This version will check every iteration whether or not the expm(...) converges
    Thus the minimum required Krylov Dimension is used

    kD_min = D/2, kD_max = D
"""
function lanczos_evolve_adaptive(H::Matrix, psi::Vector, dt::Float64, kD_min::Integer=2, kD_max::Integer=D)
    # kD is m from wiki
    D =  length(psi) # Basis dimension (n x n matrix )
    Vmat = spzeros(Complex{Float64},D,kD_max) # Projector onto Krylov, will be matrix with columns Vi
    psi = psi / norm(psi) # (re)normalize psi
    Vmat[:,1] = psi # v1 is an arbitrary vector of length n, with norm 1

    alpha = spzeros(Complex{Float64},kD_max)
    beta = spzeros(Complex{Float64},kD_max) # storage for T matrix diagonals

    uvec = spzeros(kD_max); uvec[1] = 1

    # Outside Loop first
    wp = H*psi # w1' = H*v1 (v1 = psi)
    alpha[1] = (wp'*psi)[1,1]
    w = wp - (alpha[1]*Vmat[:,1])
    beta[2] = norm(w)

    Vmat[:,2] = w / beta[2]

    # now loop up to the minimum value of kD min first
    for j in 2:(kD_min)
        w = H*Vmat[:,j] - beta[j]*Vmat[:,j-1]
        alpha[j] = (conj(transpose(w))*Vmat[:,j])
        w = w - alpha[j]*Vmat[:,j]
        beta[j+1] = norm(w)
        Vmat[:,j+1] = w / beta[j+1]
    end

    # now we move onto finding where the matrix exponential converges
    kD = kD_max ; subspacefinal = spzeros(Complex{Float64},kD_max)
    # these are default values as julia doesnt like making new varibales inside of loops
    for j in (kD_min+1):kD_max
        w = H*Vmat[:,j] - beta[j]*Vmat[:,j-1]
        alpha[j] = w' * Vmat[:,j]
        # make unit vector of correct dimension
        temp_uvec = uvec[1:j]
        # now make T matrix
        Tmat = diagm(0 => alpha[1:j]) + diagm(1 => beta[2:j]) +diagm(-1 => beta[2:j])
        # do matrix exponential
        subspacefinal = exp(-(1im)*dt*Tmat)*temp_uvec
        #check if converges
        if ( abs(subspacefinal[j]) < 1e-16)
            # Save the neccessary dimension
            kD = j
            #print(kD)
            break
        end
        # if it doesnt, then continue as if in the previous loop
        w = w - alpha[j]*Vmat[:,j]
        beta[j+1] = norm(w)
        Vmat[:,j+1] = w / beta[j+1]
    end
    return kD
    #=
    # Projection of inital vector onto Krylov subspace

    #subspacefinal = expm(-(1im)*dt*Tmat)*uvec
    Vmat =  Vmat[:,1:kD]

    psi_final = Vmat*subspacefinal

    return psi_final
    =#
end

"""
    Using lanczos algorithm to approximate the time evolution of the form
    psi(t) = exp(-iHt)*psi(0)
"""
function lanczos_evolve(H::Matrix, psi::Vector, dt::Float64, kD::Integer=D, step::Integer=0)

    # kD is m from wiki
    D =  length(psi) # Basis dimension (n x n matrix )
    Vmat = spzeros(Complex{Float64},D,kD) # Projector onto Krylov, matrix with columns Vi
    psi = psi / norm(psi) # (re)normalize psi
    Vmat[:,1] = psi # v1 is an arbitrary vector of length n, with norm 1

    alpha = spzeros(Complex{Float64},kD)
    beta = spzeros(Complex{Float64},kD) # storage for T matrix diagonals

    # Outside Loop first
    wp = H*psi # w1' = H*v1 (v1 = psi)
    alpha[1] = (wp'*psi)[1,1]
    w = wp - (alpha[1]*Vmat[:,1])
    beta[2] = norm(w)

    Vmat[:,2] = w / beta[2]

    for j in 2:(kD-1)
        w = H*Vmat[:,j] - beta[j]*Vmat[:,j-1]
        alpha[j] = (wp' *Vmat[:,j])[1]
        w = w - alpha[j]*Vmat[:,j]
        beta[j+1] = norm(w)
        Vmat[:,j+1] = w / beta[j+1]
    end

    w = H*Vmat[:,kD] - beta[kD]*Vmat[:,kD-1]
    alpha[kD] = (wp' * Vmat[:,kD])[1]

    Tmat = diagm(0 => alpha) + diagm(1 => beta[2:kD]) +diagm(-1 => beta[2:kD])
    #Tmat = sparse(Tmat)

    # Projection of inital vector onto Krylov subspace
    uvec = zeros(kD)
    uvec[1] = 1

    #=
        tempm = zeros(Complex{Float64},kD,kD)
        for x in 1:Int64(sqrt(length(Tmat)))
            for y in 1:Int64(sqrt(length(Tmat)))
                if abs(Tmat[x,y]) < 1e-15
                    tempm[x,y] = 0+0im
        
                    elseif isnan.(Tmat[x,y])
                        tempm[x,y] = 0+0im
        
                    else
                        tempm[x,y] = Tmat[x,y]
                    end
                end
            end
            Tmat = tempm
        end
    =#
    
    #=
        println(Tmat)
        if step > 15990
            println(Tmat)
            println("--")
        end
    =#

    subspacefinal = exp(-(0+1im)*dt*Tmat)*uvec
    psi_final = Vmat*subspacefinal

    return psi_final

end

function test_lanczos()
    # make a Hamiltonian (simple hopping in 1D)
    L = 100 # Size of matrix (number of sites in 1D lattice)
    Ls = collect(1:L)
    
    H = -1.0 .* (diagm(1 => ones(L - 1)) + diagm(-1 => ones(L - 1)))
    # make a wavepacket to evolve using H
    sigma = 5; k = pi/2
    psi = zeros(Complex{Float64},L,1) # empty vector
    j0 = Int(floor(L/6)) # where to start the packet
    # wavepacket structure:
    psi = exp.( (-1.0 .* (Ls .- j0).^2) ./ (2*sigma^2)) .* exp.((1im*k).*Ls)
    psi = psi / norm(psi) # normalize

    min_d = 2
    max_d = L  # some parameters for lanczos algorithm
    kD = max_d

    dt = 0.1
    ts = collect(0:dt:100) # time steps
    P_Density = zeros(Complex{Float64},L,length(ts)) # storage of wavepacket evolution
    Norm = zeros(length(ts)) # norm

    for j in eachindex(ts)
        if (100*(j/(length(ts)-1)) % 10) == 0
            println("$(100*(j/(length(ts)-1))) % done")
        end

        if j == 1
            kD = lanczos_evolve_adaptive(H, psi, dt, min_d, max_d) # get neccessary dimension to evolve efficiently
            println("Optimal dimension for evolution : D = $kD\n")
        else
            psi = lanczos_evolve(H, psi, dt, kD) # lanczos evolution
        end

        Norm[j] = (norm(psi))^2
        P_Density[:,j] = psi
    end

    Data_P = zeros(length(ts),L)
    P_Density = transpose(P_Density)
    for j in eachindex(ts)
        for k in 1:L
            Data_P[j,k] = real(norm(P_Density[j,k]))
        end
    end

    return Data_P,Norm,ts
end
Data_P,Norm,ts = test_lanczos();

# #---------------------------------------------------------
function plot_evolution_density(Data_P)
    cm = pyimport("matplotlib.cm"); ## needs pycall
    fz = 20
    PyPlot.rc("mathtext", fontset = "stix")
    PyPlot.rc("font", family = "STIXGeneral", size = fz)
    fig, ax = subplots(figsize=(7,6),constrained_layout=true)
    #PyPlot.rc("font",size=12)
    asp = 0.1 # Aspect Ratio for plot
    imshow(Data_P,aspect=asp,cmap=cm.gist_heat_r)
    colorbar()
    xlabel("Site")
    ylabel("Time")
end
plot_evolution_density(Data_P);
