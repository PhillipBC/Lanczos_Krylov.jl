"""
        Using the lanczos algorithm to approximate the time evolution of the form
        psi(t) = exp(-iHt)*psi(0)
"""
function lanczos_evolve(H::Matrix, psi::Vector, dt::Float64, kD::Integer)
    # kD is m from wiki
    D =  length(psi) # Basis dimension (n x n matrix )
    Vmat = zeros(Complex{Float64},D,kD) # Projector onto Krylov, matrix with columns Vi
    psi = psi / norm(psi) # (re)normalize psi
    Vmat[:,1] = psi # v1 is an arbitrary vector of length n, with norm 1

    alpha = zeros(Complex{Float64},kD)
    beta = zeros(Complex{Float64},kD) # storage for T matrix diagonals

    # Outside Loop first
    wp = H*psi # w1' = H*v1 (v1 = psi)
    alpha[1] = (ctranspose(wp)*psi)[1,1]
    w = wp - (alpha[1]*Vmat[:,1])
    beta[2] = norm(w)

    Vmat[:,2] = w / beta[2]

    for j in 2:(kD-1)
        w = H*Vmat[:,j] - beta[j]*Vmat[:,j-1]
        alpha[j] = (ctranspose(w)*Vmat[:,j])
        w = w - alpha[j]*Vmat[:,j]
        beta[j+1] = norm(w)
        Vmat[:,j+1] = w / beta[j+1]
    end

    w = H*Vmat[:,kD] - beta[kD]*Vmat[:,kD-1]
    alpha[kD] = ctranspose(w) * Vmat[:,kD]

    Tmat = diagm(alpha,0) + diagm(beta[2:kD],1) +diagm(beta[2:kD],-1)

    # Projection of inital vector onto Krylov subspace
    uvec = zeros(kD)
    uvec[1] = 1

    subspacefinal = expm(-(0+1im)*dt*Tmat)*uvec
    psi_final = Vmat*subspacefinal

    return psi_final

end
